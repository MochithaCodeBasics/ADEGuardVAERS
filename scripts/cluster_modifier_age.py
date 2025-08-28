
"""
Modifier-aware, age-wise clustering with HDBSCAN (preferred) or DBSCAN fallback.

Adds cluster summaries:
- For each age_group & cluster, computes top-3 symptoms (from SYMPTOM1..SYMPTOM5 if available,
  else from SYMPTOM_TEXT*_ split), plus top modifiers and cluster size.
- Writes one combined CSV: cluster_summary_all.csv

Noise (cluster == -1) is ALWAYS plotted in light grey.
"""

import argparse
from pathlib import Path
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- UMAP fallback to PCA ---
try:
    import umap.umap_ as umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False
    from sklearn.decomposition import PCA

# ---- Prefer HDBSCAN, else fallback to DBSCAN (Python 3.13 friendly) ----
try:
    import hdbscan  # type: ignore
    HAVE_HDBSCAN = True
except Exception:
    HAVE_HDBSCAN = False
    from sklearn.cluster import DBSCAN


# Optional plotting (only imported if --plot is used)
def _lazy_import_plotting():
    import matplotlib.pyplot as plt
    import seaborn as sns
    return plt, sns


# ---- Modifiers we look for in text ----
MODIFIERS = ["mild", "moderate", "severe", "acute", "chronic", "extreme", "persistent"]


def extract_modifier(text: str) -> str:
    """Return first matching modifier in text, else 'none'."""
    text = str(text or "").lower()
    for m in MODIFIERS:
        if m in text:
            return m
    return "none"


def modifier_features(text: str):
    """One-hot vector for MODIFIERS list."""
    tokens = str(text or "").lower().split()
    return [1 if m in tokens else 0 for m in MODIFIERS]


def age_group(age):
    """Bin numeric age -> child/young_adult/adult/elderly; unknown if NA."""
    if pd.isna(age):
        return "unknown"
    try:
        a = float(age)
    except Exception:
        return "unknown"
    if a < 18:
        return "child"
    elif a < 40:
        return "young_adult"
    elif a < 65:
        return "adult"
    else:
        return "elderly"


def choose_text_col(df: pd.DataFrame) -> str:
    """Prefer SYMPTOM_TEXT_CLEAN; else fall back to SYMPTOM_TEXT."""
    if "SYMPTOM_TEXT_CLEAN" in df.columns:
        return "SYMPTOM_TEXT_CLEAN"
    if "SYMPTOM_TEXT" in df.columns:
        return "SYMPTOM_TEXT"
    raise ValueError("CSV must have SYMPTOM_TEXT_CLEAN or SYMPTOM_TEXT column.")


def make_cluster_palette(labels, gray_hex="#D3D3D3"):
    """
    Return a dict palette mapping each cluster label (as str) to a color,
    with noise (-1) fixed to light grey.
    """
    import matplotlib.pyplot as plt

    uniq = sorted(set(labels))
    non_noise = [u for u in uniq if u != -1]

    base_cmap = plt.get_cmap("tab20")
    colors = [base_cmap(i % 20) for i in range(len(non_noise))]

    palette = {}
    if -1 in uniq:
        palette["-1"] = gray_hex
    for i, c in enumerate(non_noise):
        palette[str(c)] = colors[i]
    return palette


# ---------- Symptom extraction helpers ----------
_SPLIT_RX = re.compile(r"[;,]| and | with ", flags=re.IGNORECASE)

STOP_PHRASES = {
    "", "na", "n/a", "none", "normal", "unknown", "unspecified", "other",
    "no reaction", "no adverse event", "nil",
}

def pick_symptom_terms(row: pd.Series, text_col: str) -> list[str]:
    """
    Prefer SYMPTOM1..SYMPTOM5 columns if present; else extract phrases from text.
    Returns a list of normalized symptom phrases for that row.
    """
    terms = []
    symptom_cols = [c for c in ["SYMPTOM1", "SYMPTOM2", "SYMPTOM3", "SYMPTOM4", "SYMPTOM5"] if c in row.index]
    if symptom_cols:
        for c in symptom_cols:
            val = str(row.get(c, "") or "").strip().lower()
            if val and val not in STOP_PHRASES:
                terms.append(val)
        if terms:
            return terms

    # Fallback from free text
    txt = str(row.get(text_col, "") or "")
    parts = [p.strip().lower() for p in _SPLIT_RX.split(txt)]
    # simple cleanup
    parts = [re.sub(r"\s+", " ", p) for p in parts]
    parts = [p for p in parts if p and p not in STOP_PHRASES and not p.isdigit()]
    return parts[:10]  # cap per-row additions


def summarize_cluster(group_df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Build per-cluster summary with top-3 symptoms and top modifiers.
    """
    rows = []
    for cl, dfc in group_df.groupby("cluster", dropna=False):
        # Count symptoms
        counter = Counter()
        for _, r in dfc.iterrows():
            counter.update(pick_symptom_terms(r, text_col))
        top_symptoms = [t for t, _ in counter.most_common(3)]

        # Top modifiers (mild/moderate/severe/…)
        mod_counts = dfc["modifier"].value_counts(dropna=False).to_dict()
        # Keep top 3 non-'none' plus possibly 'none' if dominant
        top_mods = [m for m, _ in sorted(mod_counts.items(), key=lambda kv: (-kv[1], kv[0]))][:3]
        rows.append({
            "cluster": int(cl) if isinstance(cl, (int, np.integer)) else cl,
            "n_points": int(len(dfc)),
            "top_symptoms": ", ".join(top_symptoms) if top_symptoms else "",
            "top_modifiers": ", ".join(top_mods),
        })
    return pd.DataFrame(rows).sort_values(by=["cluster"]).reset_index(drop=True)


def cluster_age_modifier(
    input_csv: str,
    output_csv: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    neighbors: int = 15,
    umap_dim: int = 50,
    seed: int = 42,
    min_cluster_size: int = 10,  # for HDBSCAN
    min_samples: int = 5,        # for both HDBSCAN/DBSCAN
    eps: float = 0.7,            # for DBSCAN only
    plot: bool = True,
    save_plots: bool = False,
    label_clusters: bool = False,
    plot_dir: str = "plots/age_groups",
    min_group_size: int = 20,    # skip clusters for tiny age groups
):
    df = pd.read_csv(input_csv, low_memory=False)

    # Ensure columns exist
    text_col = choose_text_col(df)
    if "AGE_YRS" not in df.columns:
        df["AGE_YRS"] = np.nan

    # Derived columns
    df["age_group"] = df["AGE_YRS"].apply(age_group)
    df["modifier"] = df[text_col].apply(extract_modifier)

    # Load sentence-transformer once
    model = SentenceTransformer(model_name)

    results = []
    all_summaries = []

    for group, group_df in df.groupby("age_group", dropna=False):
        group_df = group_df.copy()
        if len(group_df) < min_group_size:
            continue

        texts = group_df[text_col].fillna("").astype(str).tolist()

        # --- Embeddings ---
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        # Append modifier one-hot features
        mod_vecs = np.array([modifier_features(t) for t in texts], dtype=np.float32)
        enhanced = np.hstack([embeddings, mod_vecs]).astype(np.float32)

        # --- Dimensionality reduction for clustering ---
        if HAVE_UMAP:
            reducer = umap.UMAP(n_neighbors=neighbors, n_components=umap_dim, random_state=seed)
            reduced = reducer.fit_transform(enhanced)
        else:
            reducer = PCA(n_components=min(umap_dim, enhanced.shape[1]))
            reduced = reducer.fit_transform(enhanced)

        # --- Clustering ---
        if HAVE_HDBSCAN:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            labels = clusterer.fit_predict(reduced)
            algo_name = "HDBSCAN"
        else:
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(reduced)
            algo_name = f"DBSCAN (eps={eps})"

        group_df["cluster"] = labels

        # 2D viz coordinates (optional)
        if HAVE_UMAP:
            vis2d = umap.UMAP(n_neighbors=neighbors, n_components=2, random_state=seed).fit_transform(reduced)
        else:
            vis2 = PCA(n_components=2)
            vis2d = vis2.fit_transform(reduced)

        group_df["x"], group_df["y"] = vis2d[:, 0], vis2d[:, 1]

        # --- Per-group summary (top-3 symptoms & modifiers) ---
        summary = summarize_cluster(group_df, text_col)
        summary.insert(0, "age_group", group)
        all_summaries.append(summary)

        # --- Plot per age group (optional) ---
        if plot:
            plt, sns = _lazy_import_plotting()

            group_df["cluster_str"] = group_df["cluster"].astype(str)
            pal = make_cluster_palette(group_df["cluster"].tolist(), gray_hex="#D3D3D3")

            plt.figure(figsize=(10, 7))
            ax = sns.scatterplot(
                data=group_df,
                x="x",
                y="y",
                hue="cluster_str",
                style="modifier",
                palette=pal,
                s=40,
                alpha=0.85,
                edgecolor="none",
            )
            plt.title(f"{algo_name} clusters for age group: {group}")

            # Optionally label each cluster at its centroid
            if label_clusters:
                for cl, dfc in group_df.groupby("cluster"):
                    cx, cy = dfc["x"].mean(), dfc["y"].mean()
                    label = "noise" if cl == -1 else str(cl)
                    ax.text(cx, cy, label, fontsize=10, weight="bold")

            # Legends
            handles, labels_text = ax.get_legend_handles_labels()
            cluster_handles, cluster_labels = [], []
            modifier_handles, modifier_labels = [], []
            pal_keys = set(pal.keys())
            for h, l in zip(handles, labels_text):
                if l in pal_keys:
                    cluster_handles.append(h)
                    cluster_labels.append("noise" if l == "-1" else l)
                elif l != "modifier":
                    modifier_handles.append(h)
                    modifier_labels.append(l)
            leg1 = ax.legend(cluster_handles, cluster_labels, title="cluster",
                             bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
            ax.add_artist(leg1)
            if modifier_handles:
                ax.legend(modifier_handles, modifier_labels, title="modifier",
                          bbox_to_anchor=(1.02, 0.45), loc="upper left", frameon=True)

            plt.tight_layout()
            if save_plots:
                outdir = Path(plot_dir); outdir.mkdir(parents=True, exist_ok=True)
                plt.savefig(outdir / f"{group}_{algo_name}_clusters.png", dpi=180)
                plt.close()
            else:
                plt.show()

        results.append(group_df)

    if not results:
        raise RuntimeError("No age groups met the minimum group size for clustering.")

    final = pd.concat(results).sort_index()
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_csv, index=False)

    # Write combined cluster summary
    summary_all = pd.concat(all_summaries, ignore_index=True)
    summary_path = Path(output_csv).with_name("cluster_summary_all.csv")
    summary_all.to_csv(summary_path, index=False)

    print(f"✅ Saved clustered file: {output_csv}")
    print(f"✅ Saved summary file:  {summary_path}")


def main():
    ap = argparse.ArgumentParser(description="Modifier-aware, age-wise clustering (HDBSCAN or DBSCAN fallback) + summaries.")
    ap.add_argument("--input", default="data/sample_1k_truncated_symptom_text.csv", help="Input CSV path")
    ap.add_argument("--output", default="data/clustered_age_modifier.csv", help="Output CSV path")
    ap.add_argument("--model_name", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--neighbors", type=int, default=15, help="UMAP/vis n_neighbors (ignored if PCA fallback)")
    ap.add_argument("--umap_dim", type=int, default=50, help="Dimensionality for clustering step (UMAP or PCA)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_group_size", type=int, default=20, help="Skip age groups smaller than this")
    # HDBSCAN params
    ap.add_argument("--min_cluster_size", type=int, default=10, help="HDBSCAN only")
    # Both HDBSCAN/DBSCAN
    ap.add_argument("--min_samples", type=int, default=5, help="Used by HDBSCAN and DBSCAN")
    # DBSCAN-only params
    ap.add_argument("--eps", type=float, default=0.7, help="DBSCAN eps (ignored if HDBSCAN is available)")
    ap.add_argument("--no_plot", action="store_true", help="Disable plots")
    ap.add_argument("--save_plots", action="store_true", help="Save plots to disk instead of showing")
    ap.add_argument("--label_clusters", action="store_true", help="Overlay cluster ids at centroids")
    ap.add_argument("--plot_dir", default="plots/age_groups", help="Directory to save plots when --save_plots is set")
    args = ap.parse_args()

    cluster_age_modifier(
        input_csv=args.input,
        output_csv=args.output,
        model_name=args.model_name,
        batch_size=args.batch_size,
        neighbors=args.neighbors,
        umap_dim=args.umap_dim,
        seed=args.seed,
        min_group_size=args.min_group_size,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        eps=args.eps,
        plot=not args.no_plot,
        save_plots=args.save_plots,
        label_clusters=args.label_clusters,
        plot_dir=args.plot_dir,
    )


if __name__ == "__main__":
    main()
