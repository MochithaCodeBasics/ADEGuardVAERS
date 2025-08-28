#!/usr/bin/env python3
"""
Train BioBERT NER (ADE, DRUG) from:
  - data/train.jsonl  (each line: {"text": "...", "entities":[{"start":..,"end":..,"label":"ADE|DRUG"}, ...]})
  - data/valid.json   (Label Studio export: [{"data":{"text":"..."}, "annotations":[{"result":[...]}]}, ...])

Outputs to: models/biobert_ner (by default)
"""

import json
import re
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, Trainer, TrainingArguments
)
from transformers.modeling_utils import unwrap_model
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

# ----------------------- Config -----------------------
MODEL = "dmis-lab/biobert-base-cased-v1.1"
LABELS = ["O", "B-ADE", "I-ADE", "B-DRUG", "I-DRUG"]
id2label = {i: l for i, l in enumerate(LABELS)}
label2id = {l: i for i, l in id2label.items()}
ADE, DRUG = "ADE", "DRUG"
VALID_ENTITY_LABELS = {ADE, DRUG}

# ------------------- Helpers & EDA --------------------
WS_RE = re.compile(r"\S+")

def word_spans(text: str):
    """Simple whitespace tokenization with character spans."""
    return [(m.group(0), m.start(), m.end()) for m in WS_RE.finditer(text or "")]

def deoverlap_drug_wins(spans):
    """If ADE/DRUG spans overlap, keep DRUG (ties broken by longer span)."""
    spans = sorted(spans, key=lambda x: (x["start"], -(x["end"] - x["start"])))
    kept = []
    for s in spans:
        if any(not (s["end"] <= k["start"] or k["end"] <= s["start"]) and k["label"] == DRUG for k in kept):
            continue
        kept.append(s)
    return kept

def parse_entities_generic(rec):
    """Support both {"entities":[{start,end,label}]} and Label Studio-style {"spans":[...]}."""
    ents = rec.get("entities") or rec.get("spans") or []
    out = []
    for e in ents:
        if "start" in e and "end" in e:
            lab = e.get("label") or (e.get("labels", [None])[0])
            if lab:
                L = str(lab).upper()
                if L in VALID_ENTITY_LABELS:
                    out.append({"start": int(e["start"]), "end": int(e["end"]), "label": L})
        elif "value" in e:
            v = e["value"]; labs = v.get("labels") or []
            if labs:
                L = str(labs[0]).upper()
                if L in VALID_ENTITY_LABELS:
                    out.append({"start": int(v["start"]), "end": int(v["end"]), "label": L})
    return out

def words_and_bio(text: str, spans):
    """Map char-level spans to BIO tags over whitespace tokens."""
    wsp = word_spans(text)
    labels = ["O"] * len(wsp)
    spans = deoverlap_drug_wins(spans)
    for sp in spans:
        first = True
        for i, (_, ws, we) in enumerate(wsp):
            if not (we <= sp["start"] or sp["end"] <= ws):
                base = sp["label"]
                labels[i] = f"B-{base}" if first and ws == sp["start"] else f"I-{base}"
                first = False
    return [w for (w, _, __) in wsp], labels

# ---------------------- Loaders -----------------------
def load_train_jsonl_as_bio(path: str):
    tokens, tags = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text") or rec.get("data", {}).get("text") or ""
            spans = parse_entities_generic(rec)
            w, y = words_and_bio(text, spans)
            tokens.append(w); tags.append(y)
    return tokens, tags

def load_valid_ls_json_as_bio(path: str):
    """Label Studio JSON (array)."""
    tasks = json.loads(Path(path).read_text(encoding="utf-8"))
    tokens, tags = [], []
    for t in tasks:
        text = (t.get("data") or {}).get("text") or t.get("text") or ""
        spans = []
        anns = t.get("annotations") or []
        if anns:
            results = anns[0].get("result") or []
            for r in results:
                if r.get("type") != "labels":
                    continue
                labs = r.get("value", {}).get("labels") or []
                if not labs: 
                    continue
                lab = labs[0].upper()
                if lab not in VALID_ENTITY_LABELS:
                    continue
                v = r["value"]
                spans.append({"start": int(v["start"]), "end": int(v["end"]), "label": lab})
        w, y = words_and_bio(text, spans)
        tokens.append(w); tags.append(y)
    return tokens, tags

# --------------- Tokenization & Alignment ---------------
def align_labels_with_tokens(labels, word_ids):
    aligned, prev = [], None
    for wid in word_ids:
        if wid is None:
            aligned.append(-100)
        elif wid != prev:
            aligned.append(label2id[labels[wid]])
        else:
            tag = labels[wid]
            if tag.startswith("B-"): 
                tag = "I-" + tag[2:]
            aligned.append(label2id[tag])
        prev = wid
    return aligned

def tokenize_and_align(examples, tokenizer):
    tok = tokenizer(
        examples["tokens"], is_split_into_words=True,
        truncation=True, padding="max_length", max_length=256
    )
    aligned = []
    for i, labs in enumerate(examples["ner_tags"]):
        word_ids = tok.word_ids(batch_index=i)
        aligned.append(align_labels_with_tokens(labs, word_ids))
    tok["labels"] = aligned
    return tok

# --------------- Weight "O" down, ADE/DRUG up ---------------
def auto_class_weights(tokenized_train, min_w=1.0, max_w=8.0):
    counts = np.zeros(len(LABELS), dtype=np.float64)
    for arr in tokenized_train["labels"]:
        for lab in arr:
            if lab != -100:
                counts[lab] += 1
    counts = np.where(counts > 0, counts, 1.0)
    w = (counts.mean() / counts)              # inverse frequency
    w = np.clip(w, min_w, max_w)
    w = w / w[label2id["O"]]                  # normalize so O ≈ 1.0
    return torch.tensor(w, dtype=torch.float)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor = None, ignore_index: int = -100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weights = class_weights
        self._ignore = ignore_index
        self._loss   = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels  = inputs.get("labels")
        outputs = model(**{k:v for k,v in inputs.items() if k!="labels"})
        logits  = outputs.get("logits")
        if self._weights is not None:
            w = self._weights.to(logits.device, dtype=logits.dtype)
            self._loss = nn.CrossEntropyLoss(weight=w, ignore_index=self._ignore)
        else:
            self._loss = nn.CrossEntropyLoss(ignore_index=self._ignore)
        loss = self._loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ---------------------- Metrics ------------------------
def compute_metrics(pred):
    predictions, labels = pred
    preds = predictions.argmax(-1)
    true_labels = [[id2label[l] for l in lab if l != -100] for lab in labels]
    true_preds  = [[id2label[p] for p,l in zip(pr, lab) if l != -100] for pr, lab in zip(preds, labels)]
    return {
        "precision": precision_score(true_labels, true_preds),
        "recall":    recall_score(true_labels, true_preds),
        "f1":        f1_score(true_labels, true_preds)
    }

# ---------------------- Main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="data/train.jsonl", help="train.jsonl (line-delimited)")
    ap.add_argument("--valid_json",  default="data/valid.json",   help="valid.json (Label Studio array)")
    ap.add_argument("--outdir", default="models/biobert_ner")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--train_bs", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--min_w", type=float, default=1.0)
    ap.add_argument("--max_w", type=float, default=8.0)
    ap.add_argument("--seed",  type=int, default=42)
    args = ap.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load & tokenize
    train_tokens, train_tags = load_train_jsonl_as_bio(args.train_jsonl)
    valid_tokens, valid_tags = load_valid_ls_json_as_bio(args.valid_json)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ds_train  = Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_tags})
    ds_valid  = Dataset.from_dict({"tokens": valid_tokens, "ner_tags": valid_tags})
    tok_train = ds_train.map(lambda x: tokenize_and_align(x, tokenizer), batched=True)
    tok_valid = ds_valid.map(lambda x: tokenize_and_align(x, tokenizer), batched=True)

    # Model + weights
    model   = AutoModelForTokenClassification.from_pretrained(
        MODEL, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    )
    weights = auto_class_weights(tok_train, min_w=args.min_w, max_w=args.max_w)
    print("[INFO] class_weights (O, B-ADE, I-ADE, B-DRUG, I-DRUG):", weights.tolist())
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Make state_dict contiguous to avoid safetensors issues anywhere
    base = unwrap_model(model)
    orig_state_dict = base.state_dict
    def contiguous_state_dict(*a, **kw):
        sd = orig_state_dict(*a, **kw)
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and not v.is_contiguous():
                sd[k] = v.contiguous()
        return sd
    base.state_dict = contiguous_state_dict

    # Training args: save every epoch (no safetensors), keep last 2
    try:
        targs = TrainingArguments(
            output_dir=args.outdir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            save_safetensors=False,      # ensure .bin for checkpoints
            learning_rate=args.lr,
            per_device_train_batch_size=args.train_bs,
            per_device_eval_batch_size=args.eval_bs,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            logging_steps=50,
            report_to=[],
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=args.seed
        )
    except TypeError:
        # very old transformers fallback
        targs = TrainingArguments(
            output_dir=args.outdir,
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=args.lr,
            per_device_train_batch_size=args.train_bs,
            per_device_eval_batch_size=args.eval_bs,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            logging_steps=50,
            report_to=[],
            seed=args.seed
        )

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model, args=targs,
        train_dataset=tok_train, eval_dataset=tok_valid,
        tokenizer=tokenizer, data_collator=collator,
        compute_metrics=compute_metrics
    )

    # Train & eval
    print("[INFO] Training …")
    trainer.train()
    print("[INFO] Validation metrics:", trainer.evaluate())

    # Per-class report
    preds = trainer.predict(tok_valid)
    p = preds.predictions.argmax(-1)
    true_labels = [[id2label[l] for l in lab if l != -100] for lab in preds.label_ids]
    true_preds  = [[id2label[x] for x,l in zip(pi, lab) if l != -100] for pi, lab in zip(p, preds.label_ids)]
    print("\n=== Validation report ===")
    print(classification_report(true_labels, true_preds, digits=4))

    # FINAL SAVE (version-proof, no safetensors):
    # 1) trainer.save_model saves config etc.
    trainer.save_model(args.outdir)
    # 2) force .bin weights (even if trainer used something else)
    torch.save(model.state_dict(), f"{args.outdir}/pytorch_model.bin")
    # 3) tokenizer & labels
    tokenizer.save_pretrained(args.outdir)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(f"{args.outdir}/labels.txt").write_text("\n".join(LABELS), encoding="utf-8")

    print(f"✅ Saved model to {args.outdir}")

if __name__ == "__main__":
    main()
