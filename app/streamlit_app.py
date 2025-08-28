# ADEGaurd — NER (ADE/DRUG) + Modifier-colored, Cluster-shaped Clustering (Streamlit)

import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ===== UMAP → PCA fallback =====
try:
    import umap.umap_ as umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False
    from sklearn.decomposition import PCA

# ===== HDBSCAN → DBSCAN fallback =====
try:
    import hdbscan  # type: ignore
    HAVE_HDBSCAN = True
except Exception:
    HAVE_HDBSCAN = False
    from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# NER config (no sliders)
# =========================
APP_TITLE = "ADEGaurd"
MODEL_DIR_DEFAULT = "models/biobert_ner"
MAX_LEN = 256
CONF_THRESH = 0.60

LABELS = ["O", "B-ADE", "I-ADE", "B-DRUG", "I-DRUG"]
id2label = {i: l for i, l in enumerate(LABELS)}
SEV_ORDER = ["mild", "moderate", "severe"]

# highlight colors for NER spans
ADE_COLOR  = "#ffd6d6"   # light red
DRUG_COLOR = "#d6e4ff"   # light blue

# =========================
# NER helpers
# =========================
def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )

def infer_severity(text: str) -> str:
    t = text.lower()
    hits = [w for w in SEV_ORDER if w in t]
    return hits[-1] if hits else "none"

@st.cache_resource
def load_model(model_dir: str = MODEL_DIR_DEFAULT):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    mdl = AutoModelForTokenClassification.from_pretrained(model_dir)
    mdl.eval()
    return tok, mdl

def bio_to_spans(text: str, tags: List[str], offsets: List[List[int]], probs: List[float]) -> List[Dict[str, Any]]:
    """Merge BIO tags to spans; DRUG wins overlaps."""
    spans = []
    cur = None
    cur_probs: List[float] = []
    for tag, (s, e), p in zip(tags, offsets, probs):
        if s == e:
            continue
        if tag.startswith("B-"):
            if cur:
                spans.append({**cur, "score": float(sum(cur_probs) / len(cur_probs))})
            cur = {"label": tag[2:], "start": s, "end": e, "text": text[s:e]}
            cur_probs = [p]
        elif tag.startswith("I-") and cur and tag[2:] == cur["label"]:
            cur["end"] = e
            cur["text"] = text[cur["start"]:e]
            cur_probs.append(p)
        else:
            if cur:
                spans.append({**cur, "score": float(sum(cur_probs) / len(cur_probs))})
            cur = None
            cur_probs = []
    if cur:
        spans.append({**cur, "score": float(sum(cur_probs) / len(cur_probs))})

    # Prefer DRUG on overlaps
    spans.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    kept = []
    for s in spans:
        if any(not (s["end"] <= k["start"] or k["end"] <= s["start"]) and k["label"] == "DRUG" for k in kept):
            continue
        kept.append(s)
    return kept

def ner_infer(text: str, tok, mdl) -> List[Dict[str, Any]]:
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        out = mdl(
            input_ids=torch.tensor([enc["input_ids"]]),
            attention_mask=torch.tensor([enc["attention_mask"]]),
        )
    logits = out.logits[0].softmax(-1).cpu().numpy()
    pred_ids = logits.argmax(-1)
    offsets = enc["offset_mapping"]

    tags, token_probs = [], []
    for i, off in enumerate(offsets):
        s, e = off
        if s == e:
            continue
        tag = id2label[pred_ids[i]]
        tags.append(tag)
        token_probs.append(float(logits[i, pred_ids[i]]))

    spans = bio_to_spans(text, tags, offsets, token_probs)
    return [s for s in spans if s["score"] >= CONF_THRESH]

def highlight_text(text: str, spans: List[Dict[str, Any]]) -> str:
    spans = sorted(spans, key=lambda s: s["start"])
    html = []
    last = 0
    for sp in spans:
        s, e, label = sp["start"], sp["end"], sp["label"]
        html.append(_escape_html(text[last:s]))
        color = ADE_COLOR if label == "ADE" else DRUG_COLOR
        title = f'{label} • {sp["score"]:.2f}'
        content = _escape_html(text[s:e])
        html.append(
            f'<span title="{title}" style="background:{color}; padding:0 2px; border-radius:3px;">{content}</span>'
        )
        last = e
    html.append(_escape_html(text[last:]))
    return "".join(html)

# =========================
# Clustering helpers
# =========================
MODIFIERS = ["mild", "moderate", "severe", "acute", "chronic", "extreme", "persistent"]

# Colors by modifier (hue)
MODIFIER_PALETTE = {
    "mild": "#43a047",        # green
    "moderate": "#fdd835",    # yellow
    "severe": "#e53935",      # red
    "acute": "#fb8c00",       # orange
    "chronic": "#8e24aa",     # purple
    "extreme": "#6d4c41",     # brown
    "persistent": "#00897b",  # teal
    "none": "#9e9e9e",        # grey
}

# Marker shapes by cluster (style)
MARKERS_CYCLE = ["o", "s", "^", "P", "X", "D", "v", "<", ">", "*", "h", "H"]

def extract_modifier(text: str) -> str:
    t = str(text or "").lower()
    for m in MODIFIERS:
        if m in t:
            return m
    return "none"

def modifier_features(text: str):
    tokens = str(text or "").lower().split()
    return [1 if m in tokens else 0 for m in MODIFIERS]

def age_group(age):
    if pd.isna(age):
        return "unknown"
    try:
        a = float(age)
    except Exception:
        return "unknown"
    if a < 18: return "child"
    if a < 40: return "young_adult"
    if a < 65: return "adult"
    return "elderly"

def choose_text_col(df: pd.DataFrame) -> str:
    if "SYMPTOM_TEXT_CLEAN" in df.columns:
        return "SYMPTOM_TEXT_CLEAN"
    if "SYMPTOM_TEXT" in df.columns:
        return "SYMPTOM_TEXT"
    raise ValueError("CSV must have SYMPTOM_TEXT_CLEAN or SYMPTOM_TEXT.")

def make_cluster_markers(labels: list) -> dict:
    """Return dict mapping cluster_str -> marker symbol (cycled)."""
    uniq = [str(u) for u in sorted(set(labels), key=lambda x: (x == -1, x))]
    mapping = {}
    j = 0
    for u in uniq:
        mapping[u] = MARKERS_CYCLE[j % len(MARKERS_CYCLE)]
        j += 1
    return mapping

def cluster_df(df: pd.DataFrame, batch_size=64, neighbors=15, umap_dim=50, seed=42,
               min_group=20, min_samples=5, eps=0.7):
    """Returns df with age_group, modifier, cluster, x, y; draws one plot per age group."""
    from sentence_transformers import SentenceTransformer
    text_col = choose_text_col(df)
    if "AGE_YRS" not in df.columns:
        df["AGE_YRS"] = np.nan
    df["age_group"] = df["AGE_YRS"].apply(age_group)
    df["modifier"] = df[text_col].apply(extract_modifier)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    blocks = []
    for group, group_df in df.groupby("age_group", dropna=False):
        gdf = group_df.copy()
        if len(gdf) < min_group:
            continue

        texts = gdf[text_col].fillna("").astype(str).tolist()
        emb = model.encode(texts, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
        mod = np.array([modifier_features(t) for t in texts], dtype=np.float32)
        enhanced = np.hstack([emb, mod]).astype(np.float32)

        # Dimensionality reduction
        if HAVE_UMAP:
            reducer = umap.UMAP(n_neighbors=neighbors, n_components=umap_dim, random_state=seed)
            reduced = reducer.fit_transform(enhanced)
        else:
            reducer = PCA(n_components=min(umap_dim, enhanced.shape[1]))
            reduced = reducer.fit_transform(enhanced)

        # Clustering
        if HAVE_HDBSCAN:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=min_samples)
            labels = clusterer.fit_predict(reduced)
            algo = "HDBSCAN"
        else:
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(reduced)
            algo = f"DBSCAN (eps={eps})"

        gdf["cluster"] = labels

        # 2D projection
        if HAVE_UMAP:
            vis2d = umap.UMAP(n_neighbors=neighbors, n_components=2, random_state=seed).fit_transform(reduced)
        else:
            vis2 = PCA(n_components=2)
            vis2d = vis2.fit_transform(reduced)
        gdf["x"], gdf["y"] = vis2d[:, 0], vis2d[:, 1]

        # ---- Plot: color by modifier, shape by cluster; noise in light grey ----
        fig, ax = plt.subplots(figsize=(9, 6))
        gdf["cluster_str"] = gdf["cluster"].astype(str)
        cluster_markers = make_cluster_markers(gdf["cluster"].tolist())

        # split noise vs non-noise
        noise_mask = gdf["cluster"] == -1
        nonnoise = gdf[~noise_mask].copy()
        noise = gdf[noise_mask].copy()

        # non-noise layer (hue=modifier, style=cluster_str)
        if not nonnoise.empty:
            sns.scatterplot(
                data=nonnoise,
                x="x", y="y",
                hue="modifier",
                style="cluster_str",
                palette=MODIFIER_PALETTE,
                markers=cluster_markers,
                s=40, alpha=0.9, edgecolor="none",
                ax=ax,
            )

        # noise layer (always light grey)
        if not noise.empty:
            sns.scatterplot(
                data=noise,
                x="x", y="y",
                hue=None, style=None,
                color="#D3D3D3",
                marker=cluster_markers.get("-1", "o"),
                s=38, alpha=0.8, edgecolor="none",
                ax=ax,
                legend=False,
            )

        ax.set_title(f"{algo} clusters — age group: {group}")

        # --- Legends: modifiers (colors) and clusters (shapes)
        from matplotlib.lines import Line2D

        # Modifiers legend
        mod_items = sorted(nonnoise["modifier"].dropna().unique()) if not nonnoise.empty else []
        mod_handles = [
            Line2D([0], [0], marker="o", linestyle="", color=MODIFIER_PALETTE.get(m, "#9e9e9e"),
                   markersize=8, label=m)
            for m in mod_items
        ]
        if not noise.empty:
            mod_handles.append(
                Line2D([0], [0], marker="o", linestyle="", color="#D3D3D3", markersize=8, label="noise (color)")
            )
        leg1 = ax.legend(handles=mod_handles, title="modifier (color)",
                         bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=8)
        ax.add_artist(leg1)

        # Clusters legend
        cl_items = sorted(gdf["cluster_str"].unique(), key=lambda x: (x == "-1", int(x) if x.lstrip('-').isdigit() else 1e9))
        cl_handles = [
            Line2D([0], [0], marker=cluster_markers[c], linestyle="", color="#666666", markerfacecolor="#666666",
                   markersize=8, label=("noise" if c == "-1" else f"cluster {c}"))
            for c in cl_items
        ]
        ax.legend(handles=cl_handles, title="cluster (shape)",
                  bbox_to_anchor=(1.02, 0.45), loc="upper left", frameon=True, fontsize=8)

        st.pyplot(fig)
        blocks.append(gdf)

    if not blocks:
        st.warning("No age groups met the minimum group size for clustering. Try a smaller dataset threshold.")
        return df
    return pd.concat(blocks).sort_index()

# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="centered")
st.title(f"🩺 {APP_TITLE}")

tab1, tab2 = st.tabs(["NER (ADE/DRUG)", "Clustering"])

# -------- Tab 1: NER --------
with tab1:
    model_dir = st.text_input("Model directory", MODEL_DIR_DEFAULT)
    tok, mdl = load_model(model_dir)

    st.caption(f"Confidence threshold is fixed at {CONF_THRESH:.2f}. Max sequence length fixed at {MAX_LEN}.")
    sample = "After second dose of Comirnaty, patient developed severe chest pain and dizziness."
    text = st.text_area("Paste a symptom narrative:", sample, height=140)

    if st.button("Extract entities"):
        severity = infer_severity(text)
        spans = ner_infer(text, tok, mdl)

        sev_color = {"none": "#999", "mild": "#43a047", "moderate": "#fdd835", "severe": "#e53935"}.get(severity, "#999")
        st.markdown(
            f"**Severity (heuristic):** "
            f"<span style='background:{sev_color}; color:white; padding:2px 8px; border-radius:10px;'>{severity}</span>",
            unsafe_allow_html=True
        )

        html = highlight_text(text, spans)
        st.markdown(
            f"<div style='border:1px solid #eee; padding:10px; border-radius:6px; font-family:ui-monospace;'>{html}</div>",
            unsafe_allow_html=True
        )

        if spans:
            st.markdown("**Detected spans**")
            st.dataframe(
                [
                    {"label": s["label"], "text": s["text"], "start": s["start"], "end": s["end"], "score": round(s["score"], 3)}
                    for s in spans
                ],
                use_container_width=True,
            )
        else:
            st.info("No entities above the fixed confidence threshold.")

# -------- Tab 2: Clustering --------
with tab2:
    st.write("Upload a CSV with at least **SYMPTOM_TEXT_CLEAN** (or **SYMPTOM_TEXT**) and optionally **AGE_YRS**.")
    up = st.file_uploader("CSV file", type=["csv"])
    min_group = st.number_input("Minimum rows per age group", value=20, min_value=1, step=1)

    if st.button("Run clustering"):
        if up is None:
            st.warning("Please upload a CSV first.")
        else:
            df = pd.read_csv(up, low_memory=False)
            clustered = cluster_df(df, min_group=min_group)
            st.success("Clustering complete.")
            st.write("Preview of clustered data:")
            st.dataframe(clustered.head(30), use_container_width=True)

            # download
            csv_bytes = clustered.to_csv(index=False).encode("utf-8")
            st.download_button("Download clustered CSV", data=csv_bytes, file_name="clustered_age_modifier.csv")
