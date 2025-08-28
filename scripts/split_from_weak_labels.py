#Train, Val, Test Split
import argparse, json, os
from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict
import random

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            yield json.loads(line)

def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def stratified_split(ids, labels, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    ids: list of record IDs
    labels: list of stratification labels (same length as ids), e.g. severity
    Returns: dict id->split ("train"|"val"|"test")
    """
    random.seed(seed)
    by_label = defaultdict(list)
    for i, lab in zip(ids, labels):
        by_label[str(lab).lower()].append(i)

    assign = {}
    for lab, lab_ids in by_label.items():
        random.shuffle(lab_ids)
        n = len(lab_ids)
        n_train = int(train_ratio * n)
        n_val   = int(val_ratio * n)
        tr = lab_ids[:n_train]
        va = lab_ids[n_train:n_train+n_val]
        te = lab_ids[n_train+n_val:]
        for rid in tr: assign[rid] = "train"
        for rid in va: assign[rid] = "val"
        for rid in te: assign[rid] = "test"
    return assign

def main():
    ap = argparse.ArgumentParser(description="Split existing weak labels (BIO + JSONL) into train/val/test, stratified by severity.")
    ap.add_argument("--weak_dir", default="data/weak_labels", help="Folder with all_weak_bio.csv and all_weak_spans.jsonl")
    ap.add_argument("--out_dir",  default="data/weak_labels_splits", help="Output folder for split weak-label files")
    ap.add_argument("--train", type=float, default=0.70, help="Train ratio (default 0.70)")
    ap.add_argument("--val",   type=float, default=0.15, help="Val ratio (default 0.15; test is remainder)")
    ap.add_argument("--seed",  type=int, default=42, help="Random seed")
    ap.add_argument("--base_csv", default="", help="(Optional) Base CSV to also split (must contain VAERS_ID or RID matching weak 'id')")
    ap.add_argument("--id_column", default="", help="(Optional) Column in base CSV that matches weak 'id' (e.g., VAERS_ID or RID). If empty, tries VAERS_ID then RID.")
    args = ap.parse_args()

    weak_dir = Path(args.weak_dir)
    out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    bio_path   = weak_dir / "all_weak_bio.csv"
    spans_path = weak_dir / "all_weak_spans.jsonl"

    if not bio_path.exists() or not spans_path.exists():
        raise FileNotFoundError(f"Missing weak files. Expected {bio_path} and {spans_path}")

    # --- Load BIO (token-level) ---
    bio = pd.read_csv(bio_path)
    if not {"id","severity"}.issubset(bio.columns):
        raise ValueError("BIO file must have columns: 'id' and 'severity'")

    # Derive per-record severity (mode of token rows; usually all same)
    rec_sev = (
        bio.groupby("id")["severity"]
           .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
           .reset_index()
    )
    rec_sev.columns = ["id", "severity"]

    # Stratified split by severity
    assign = stratified_split(
        ids=rec_sev["id"].tolist(),
        labels=rec_sev["severity"].tolist(),
        train_ratio=args.train,
        val_ratio=args.val,
        seed=args.seed
    )
    print(f"[INFO] Records by split: {Counter(assign.values())}")

    # --- Split BIO by id ---
    bio["__split__"] = bio["id"].map(assign)
    for split in ["train","val","test"]:
        out = bio[bio["__split__"]==split].drop(columns="__split__")
        out_file = out_dir / f"{split}_weak_bio.csv"
        out.to_csv(out_file, index=False)
        print(f"[OK] wrote {out_file}  ({len(out)} token rows)")

    # --- Split JSONL by id ---
    spans = list(read_jsonl(spans_path))
    spans_by_split = {"train":[], "val":[], "test":[]}
    missing_ids = 0
    for rec in spans:
        rid = rec.get("id")
        which = assign.get(rid)
        if which is None:
            missing_ids += 1
            continue
        spans_by_split[which].append(rec)
    for split in ["train","val","test"]:
        out_file = out_dir / f"{split}_weak_spans.jsonl"
        write_jsonl(out_file, spans_by_split[split])
        print(f"[OK] wrote {out_file}  ({len(spans_by_split[split])} records)")
    if missing_ids:
        print(f"[WARN] {missing_ids} JSONL records had ids not found in BIO-derived assignment.")

    # --- Optional: also split the base CSV if provided ---
    if args.base_csv:
        base_path = Path(args.base_csv)
        if not base_path.exists():
            print(f"[WARN] base CSV not found: {base_path} (skipping base CSV split)")
        else:
            base = pd.read_csv(base_path, low_memory=False)
            # Decide which column matches 'id'
            id_col = args.id_column
            if not id_col:
                if "VAERS_ID" in base.columns: id_col = "VAERS_ID"
                elif "RID" in base.columns:    id_col = "RID"
                else:
                    print("[WARN] No VAERS_ID or RID in base CSV; skipping base CSV split.")
                    id_col = ""
            if id_col:
                # Ensure same dtype as in BIO (ids may be int, cast both to str for safe matching)
                assign_str = {str(k):v for k,v in assign.items()}
                base["_ID_STR_"] = base[id_col].astype(str)
                base["__split__"] = base["_ID_STR_"].map(assign_str)

                for split in ["train","val","test"]:
                    out = base[base["__split__"]==split].drop(columns=["__split__","_ID_STR_"])
                    out_file = out_dir / f"{split}.csv"
                    out.to_csv(out_file, index=False)
                    print(f"[OK] wrote base CSV split {out_file}  ({len(out)} rows)")

if __name__ == "__main__":
    main()
