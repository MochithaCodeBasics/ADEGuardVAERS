#Create jsonl for LabelStudio
import json
import argparse
from pathlib import Path

# ---- Label Studio control names (match your project config) ----
TO_NAME   = "text"             # Text node name
FROM_NAME = "label"            # Labels control for ADE/DRUG
SEV_NAME  = "severity_label"   # Choices control for severity
MODEL_VER = "weak_supervision_v2"

def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def to_ls_span_results(text: str, entities):
    """Convert span dicts to Label Studio 'labels' results."""
    results = []
    for e in entities or []:
        start = e.get("start")
        end   = e.get("end")
        if start is None or end is None:
            continue
        start, end = int(start), int(end)
        if not (0 <= start < end <= len(text)):
            continue

        label = e.get("label")
        if not label:
            labels = e.get("labels")
            if isinstance(labels, list) and labels:
                label = labels[0]
        if not label:
            continue

        results.append({
            "from_name": FROM_NAME,
            "to_name": TO_NAME,
            "type": "labels",
            "value": {
                "start": start,
                "end": end,
                "text": text[start:end],
                "labels": [str(label)]
            }
        })
    return results

def to_ls_severity_result(severity):
    """Convert severity string to LS 'choices' result (if valid)."""
    if not isinstance(severity, str):
        return []
    sev = severity.strip().lower()
    if sev not in {"mild", "moderate", "severe"}:
        return []
    return [{
        "from_name": SEV_NAME,
        "to_name": TO_NAME,
        "type": "choices",
        "value": {"choices": [sev]}
    }]

def main():
    ap = argparse.ArgumentParser(
        description="Convert split weak JSONL (ADE/DRUG + severity) to Label Studio import JSON."
    )
    ap.add_argument("--weak_root", default="data/weak_labels_splits",
                    help="Folder containing {train,val,test}_weak_spans.jsonl or all_weak_spans.jsonl")
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "all"],
                    help="Which split to convert (default: val)")
    ap.add_argument("--out_json", default="data/gold/labelstudio_val_bootstrap.json",
                    help="Output Label Studio import JSON path")
    ap.add_argument("--include_meta", action="store_true",
                    help="Include record id and severity in the 'data' block for filtering")
    ap.add_argument("--id_field", default="id",
                    help="Record id field name to copy into 'data' if include_meta is set (default: id)")
    args = ap.parse_args()

    weak_root = Path(args.weak_root)
    if args.split == "all":
        in_path = weak_root / "all_weak_spans.jsonl"
    else:
        # accept either flat files under weak_root or nested per-split dirs
        candidate_flat   = weak_root / f"{args.split}_weak_spans.jsonl"
        candidate_nested = weak_root / args.split / f"{args.split}_weak_spans.jsonl"
        in_path = candidate_flat if candidate_flat.exists() else candidate_nested

    if not in_path.exists():
        raise FileNotFoundError(f"Could not find weak JSONL for split='{args.split}': {in_path}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items, total, with_spans, with_sev = [], 0, 0, 0
    for rec in load_jsonl(in_path):
        total += 1
        text = rec.get("text", "") or ""
        entities = rec.get("entities") or rec.get("spans") or []
        severity = rec.get("severity") or rec.get("severity_label")
        rid = rec.get(args.id_field, rec.get("id"))

        span_results = to_ls_span_results(text, entities)
        sev_result   = to_ls_severity_result(severity)

        if span_results: with_spans += 1
        if sev_result:   with_sev   += 1

        data = {"text": text}
        if args.include_meta:
            if rid is not None:
                data["record_id"] = rid
            if isinstance(severity, str) and severity.strip():
                data["severity_label"] = severity.strip().lower()

        items.append({
            "data": data,
            "predictions": [{
                "model_version": MODEL_VER,
                "result": span_results + sev_result
            }]
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Split: {args.split}")
    print(f"[IN ] {in_path}")
    print(f"[OUT] {out_path}")
    print(f"Tasks: {total} | With spans: {with_spans} | With severity choices: {with_sev}")

if __name__ == "__main__":
    main()
