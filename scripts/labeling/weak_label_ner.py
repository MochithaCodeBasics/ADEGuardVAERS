import os, re, json, argparse
from pathlib import Path
from collections import Counter
import pandas as pd

ADE_LABEL, DRUG_LABEL = "ADE", "DRUG"

# ---------------- ADE blacklist ----------------
BLACKLIST_TERMS = {
    "expired product administered","product storage error","incorrect dose administered",
    "product administered to patient of inappropriate age","inappropriate schedule of product administration",
    "vaccination error","accidental exposure",
    "no adverse event","normal test result","test negative","unevaluable event",
    "blood test","laboratory test","x-ray","ct scan",
    "covid-19","covid 19","covid19","sars-cov-2","sars cov 2",
    "medical history","family history", "immunisation", "immunization",

    # also keep these admin/placeholder tokens out of ADE entirely
    "unknown","outcome unknown","condition unknown","not specified","unspecified","n/a","na"
}

# ---------------- Post-vax COVID gating ----------------
POS_PATTERNS = [
    r"tested\s+positive", r"pcr\s+positive", r"antigen\s+positive",
    r"diagnosed\s+with\s+covid", r"developed\s+covid",
    r"symptomatic\s+covid", r"breakthrough\s+infection",
    r"(?:\b\d{1,3}\b)\s*(?:day|week|month)s?\s*(?:after|post)\s*(?:vaccine|vaccination|shot|dose)",
    r"post[-\s]*vaccination", r"following\s+vaccination"
]
NEG_PATTERNS = [
    r"prior\s+to\s+vaccination", r"before\s+vaccination",
    r"history\s+of\s+covid", r"previous\s+covid",
    r"vaccinated\s+after\s+testing\s+positive", r"pre[-\s]*existing\s+covid",
    r"tested\s+negative"
]
POS_RE = re.compile("|".join(POS_PATTERNS), re.I)
NEG_RE = re.compile("|".join(NEG_PATTERNS), re.I)
COVID_RE = re.compile(r"\b(covid[-\s]*19|sars[-\s]*cov[-\s]*2|covid)\b", re.I)

def norm(s: str) -> str:
    s = str(s or "").strip().lower()
    return re.sub(r"\s+", " ", s)

def is_post_vax_covid(text: str) -> bool:
    if not COVID_RE.search(text or ""): return False
    if NEG_RE.search(text or ""):       return False
    return bool(POS_RE.search(text or ""))

# ================= DRUG LEXICON =================
# 🚫 Never treat these as DRUG (fixes “unknown” etc.)
DRUG_EXCLUDE = {
    "unknown", "unknown manufacturer", "outcome unknown", "condition unknown",
    "unspecified", "not specified", "not known", "n/a", "na",
    "intramuscular", "subcutaneous", "intradermal", "im", "sc", "id",
    "dose", "dosage form", "lot", "batch", "route of administration",
    "vaccine", "vaccination", "covid", "covid-19"
}

# Base aliases (you asked for these exact items)
VAX_ALIASES_BASE = [
    # brands / trade names
    "comirnaty", "spikevax", "vaxzevria", "nuvaxovid",

    # companies / shorthand
    "pfizer", "biontech", "pfizer-biontech", "moderna",
    "janssen", "johnson & johnson", "astrazeneca", "novavax",

    # codes / INNs
    "bnt162b2", "mrna-1273", "elasomeran", "tozinameran",
    "ad26.cov2.s", "chadox1-s",

    # phrases
    "covid-19 vaccine", "covid 19 vaccine", "covid vaccine",
    "covid-19 immunization", "covid 19 immunization", "mrna vaccine",
    "moderna covid-19 vaccine", "pfizer-biontech covid-19 vaccine",
    "janssen covid-19 vaccine", "covid 19 vaccination", "covid-19 vaccination"
]

def _variants(token: str):
    """Expand hyphen/dot/space variants (AD26.COV2.S / AD26-COV2-S / AD26 COV2 S)."""
    t = token
    out = {t}
    combos = {
        t.replace("-", " "), t.replace("-", "."),
        t.replace(".", " "), t.replace(".", "-"),
        t.replace(" ", "-"), t.replace(" ", ".")
    }
    out |= {c for c in combos if c.strip()}
    base = re.sub(r"[-.\s]+", "", t)
    if base: out.add(base)
    return {x for x in out if len(x) >= 3}

PAREN_CONTENT = re.compile(r"\(([^)]+)\)")

def build_drug_terms_from_vax_name(df: pd.DataFrame):
    """
    Use only manufacturer/product strings inside parentheses from VAX_NAME,
    drop 'covid' substrings and excluded placeholders.
    Example: 'COVID19 (PFIZER-BIONTECH)' -> 'pfizer-biontech'
    """
    terms = set()
    if "VAX_NAME" not in df.columns:
        return terms
    for raw in df["VAX_NAME"].dropna().astype(str):
        for p in PAREN_CONTENT.findall(raw):     # only (...) content
            p_norm = norm(p)
            if not p_norm:
                continue
            for chunk in re.split(r"[;/,]", p_norm):
                c = chunk.strip()
                if not c:
                    continue
                if "covid" in c or c in DRUG_EXCLUDE:
                    continue
                if len(c) >= 3:
                    terms.add(c)
    return terms

def build_drug_lexicon(df: pd.DataFrame):
    table_terms = build_drug_terms_from_vax_name(df)
    alias_terms = set()
    for a in VAX_ALIASES_BASE:
        alias_terms |= _variants(a.lower())

    combined = (table_terms | alias_terms)
    generic_drop = {"vaccine", "vaccination", "booster", "covid", "covid-19"}
    all_terms = {t for t in combined if t not in DRUG_EXCLUDE and t not in generic_drop}

    # longest-first for greedy matching
    return sorted(all_terms, key=lambda s: (-len(s), s))

def compile_drug_regex(terms):
    if not terms:
        return None
    escaped = [re.escape(t) for t in terms]
    # non-alnum boundaries; tolerate hyphen/dot/space inside terms
    return re.compile(r"(?i)(?<![A-Za-z0-9])(" + "|".join(escaped) + r")(?![A-Za-z0-9])")

# ================= ADE LEXICON =================
def build_ade_regex(df: pd.DataFrame, drug_terms_set):
    cols = [c for c in ["SYMPTOM1","SYMPTOM2","SYMPTOM3","SYMPTOM4","SYMPTOM5"] if c in df.columns]
    terms = []
    for c in cols:
        terms.extend(df[c].dropna().astype(str).tolist())
    terms = [norm(x) for x in terms if str(x).strip()]
    ctr = Counter(terms)

    def is_unknownish(s: str) -> bool:
        return ("unknown" in s) or (s in {"not specified","unspecified","n/a","na"})

    keep = [
        t for t in ctr
        if t not in BLACKLIST_TERMS
        and not is_unknownish(t)
        and t not in drug_terms_set
    ]

    keep.sort(key=lambda s: (-ctr[s], -len(s), s))
    keep_len = [t for t in sorted(keep, key=lambda s: (-len(s), s)) if len(t) >= 3]
    if not keep_len:
        return None
    return re.compile(r"(?i)\b(" + "|".join(re.escape(t) for t in keep_len) + r")\b")

# ================= Span helpers =================
def find_spans(text, regex):
    if not regex:
        return []
    spans = [(m.start(), m.end(), m.group(0)) for m in regex.finditer(text)]
    # longest-first per start, keep non-overlapping greedily
    spans.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    out, last_end = [], -1
    for s, e, tok in spans:
        if s >= last_end:
            out.append((s, e, tok))
            last_end = e
    return out

def bio_from_spans(text, spans_labeled):
    rows = []
    for m in re.finditer(r"\S+|\s+", text):
        tok = m.group(0)
        s, e = m.start(), m.end()
        if tok.isspace():
            continue
        tag = "O"
        for (ss, ee, lbl) in spans_labeled:
            if s >= ss and e <= ee:
                tag = "B-"+lbl if s == ss else "I-"+lbl
                break
        rows.append((tok, s, e, tag))
    return rows

# ================= Add spans with NO ADE/DRUG overlap =================
def add_spans_with_rules(row, ade_re, drug_re):
    text = str(row["SYMPTOM_TEXT"] or "")

    def overlaps(a, b):
        # a=(s,e), b=(s,e)
        return not (a[1] <= b[0] or b[1] <= a[0])

    # 1) DRUG first
    drug_spans = [(s, e, DRUG_LABEL) for s, e, _ in find_spans(text, drug_re)]

    # 2) ADE next, forbid ANY overlap with any DRUG span
    ade_spans = []
    for s, e, tok in find_spans(text, ade_re):
        tnorm = norm(tok)
        # COVID mentions require post-vax evidence
        if tnorm in {"covid-19","covid 19","covid19","covid","sars-cov-2","sars cov 2"}:
            if not is_post_vax_covid(text):
                continue
        if any(overlaps((s, e), (ss, ee)) for (ss, ee, _) in drug_spans):
            continue
        ade_spans.append((s, e, ADE_LABEL))

    # 3) Merge (DRUG wins by construction; still sort by start)
    spans = sorted(drug_spans + ade_spans, key=lambda x: x[0])
    return spans

# ================= Writers =================
def write_outputs(df, outdir, ade_re, drug_re):
    jsonl_path = outdir / "all_weak_spans.jsonl"
    bio_path   = outdir / "all_weak_bio.csv"

    jsonl_rows, bio_rows = [], []
    for idx, row in df.iterrows():
        text = str(row["SYMPTOM_TEXT"] or "")
        rid  = int(row["VAERS_ID"]) if "VAERS_ID" in row and pd.notna(row["VAERS_ID"]) else int(idx)
        sev  = str(row.get("severity_label", "unknown")).lower()

        spans = add_spans_with_rules(row, ade_re, drug_re)

        # JSONL per record
        entities = [{"start": int(s), "end": int(e), "label": lbl} for (s, e, lbl) in spans]
        jsonl_rows.append({"id": rid, "text": text, "severity": sev, "entities": entities})

        # BIO (token rows)
        for tok, s, e, tag in bio_from_spans(text, spans):
            bio_rows.append({"id": rid, "token": tok, "start": s, "end": e, "tag": tag, "severity": sev})

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in jsonl_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame(bio_rows).to_csv(bio_path, index=False)
    return jsonl_path, bio_path

# ================= Main =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",   default="data/processed")
    ap.add_argument("--csv",    default="sample_1k_truncated_symptom_text.csv")
    ap.add_argument("--outdir", default="data/weak_labels")
    args = ap.parse_args()

    root = Path(args.root); outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    csv_path = root / args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    if "SYMPTOM_TEXT" not in df.columns:  raise ValueError("missing SYMPTOM_TEXT")
    if "severity_label" not in df.columns: raise ValueError("missing severity_label")

    # Build lexicons
    drug_terms = build_drug_lexicon(df)
    drug_re    = compile_drug_regex(drug_terms)
    ade_re     = build_ade_regex(df, set(drug_terms))

    print(f"[LEX] DRUG terms (after exclude): {len(drug_terms)}")
    print(f"[LEX] ADE regex ready: {'yes' if ade_re else 'no'}")

    jpath, bpath = write_outputs(df, outdir, ade_re, drug_re)
    print(f"[OK] JSONL -> {jpath.name} | BIO -> {bpath.name}")

if __name__ == "__main__":
    main()
