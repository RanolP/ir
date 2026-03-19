#!/usr/bin/env -S uv run --script
# Generate 50 compound-targeting BM25 benchmark queries from the Korean corpus.
#
# Compounds: Hangul surface tokens that lindera (Mode::Decompose) splits into 2+
# content morphemes whose concatenation equals the original surface form.
# Clean compounds: none of the sub-parts appear independently in the corpus
# (ensures BM25-without-decompounding returns zero hits for the query terms).
#
# Output: test-data/ko-compound/{corpus,queries}.jsonl + qrels/test.tsv
# Usage:  uv run scripts/gen-compound-bench.py

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
LINDERA = ROOT / "preprocessors/ko/lindera-tokenize/target/release/lindera-tokenize"
KO_MIRACL = ROOT / "test-data/ko-miracl"
KO_STRATEGYQA = ROOT / "test-data/ko-strategyqa"
OUT = ROOT / "test-data/ko-compound"

HANGUL_RE = re.compile(r"[가-힣]+")
MIN_LEN = 4
TARGET = 50
# Context word: must appear in at least this many docs (not discriminative alone)
MIN_CTX_DOCS = 5
# Relaxed filter threshold: sub-part appears in at most this many docs
RELAX_THRESHOLD = 2


def load_corpus(path: Path) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def extract_hangul(text: str) -> list[str]:
    return HANGUL_RE.findall(text)


def lindera_batch(tokens: list[str]) -> list[str]:
    """Pipe tokens through lindera (one per line), return output lines."""
    if not tokens:
        return []
    proc = subprocess.run(
        [str(LINDERA)],
        input="\n".join(tokens) + "\n",
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"lindera error: {proc.stderr[:200]}", file=sys.stderr)
        sys.exit(1)
    lines = proc.stdout.splitlines()
    # Pad with empty strings if output is short (should not happen)
    while len(lines) < len(tokens):
        lines.append("")
    return lines[: len(tokens)]


def build_vocab(docs: list[dict]) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """Return (token→doc_ids, doc_id→tokens)."""
    token_docs: dict[str, set[str]] = defaultdict(set)
    doc_tokens: dict[str, list[str]] = {}
    for doc in docs:
        did = doc["_id"]
        text = (doc.get("title", "") or "") + " " + (doc.get("text", "") or "")
        toks = extract_hangul(text)
        doc_tokens[did] = toks
        for t in toks:
            token_docs[t].add(did)
    return token_docs, doc_tokens


def find_compounds(
    candidates: list[str],
) -> list[tuple[str, list[str]]]:
    """Return (surface, parts) pairs where lindera splits surface into 2+ content morphemes."""
    outputs = lindera_batch(candidates)
    compounds = []
    for surface, out in zip(candidates, outputs):
        parts = out.strip().split()
        if len(parts) >= 2 and "".join(parts) == surface:
            compounds.append((surface, parts))
    return compounds


def pick_context_word(
    doc_tokens: list[str],
    token_docs: dict[str, set[str]],
    exclude: set[str],
) -> str | None:
    """Pick a Hangul word from doc_tokens that appears in MIN_CTX_DOCS+ docs."""
    seen = set()
    for tok in doc_tokens:
        if tok in seen or tok in exclude or len(tok) < 2:
            continue
        seen.add(tok)
        if len(token_docs.get(tok, set())) >= MIN_CTX_DOCS:
            return tok
    return None


def select_compounds(
    compounds: list[tuple[str, list[str]]],
    token_docs: dict[str, set[str]],
    doc_tokens: dict[str, list[str]],
    all_docs: list[dict],
    raw_vocab: set[str],
    strict: bool,
) -> list[dict]:
    """Filter compounds and build query records; returns up to TARGET items."""
    doc_index = {d["_id"]: d for d in all_docs}
    used_docs: set[str] = set()
    results = []

    for surface, parts in compounds:
        if len(results) >= TARGET:
            break

        # Strict: no sub-part in raw_vocab at all
        # Relaxed: sub-part appears in ≤ RELAX_THRESHOLD docs
        if strict:
            if any(p in raw_vocab for p in parts):
                continue
        else:
            if any(len(token_docs.get(p, set())) > RELAX_THRESHOLD for p in parts):
                continue

        source_docs = sorted(token_docs.get(surface, set()))
        if not source_docs:
            continue

        # Prefer documents not yet used (diversity)
        chosen_did = next((d for d in source_docs if d not in used_docs), source_docs[0])
        if chosen_did not in doc_index:
            continue

        ctx = pick_context_word(
            doc_tokens.get(chosen_did, []),
            token_docs,
            exclude=set(parts) | {surface},
        )
        if ctx is None:
            continue

        used_docs.add(chosen_did)
        results.append(
            {
                "surface": surface,
                "parts": parts,
                "doc_id": chosen_did,
                "ctx": ctx,
            }
        )

    return results


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):,} rows → {path}")


def main() -> None:
    if not LINDERA.exists():
        print(f"error: lindera binary not found at {LINDERA}", file=sys.stderr)
        print("  cd preprocessors/ko/lindera-tokenize && cargo build --release", file=sys.stderr)
        sys.exit(1)

    # ── Load primary corpus ────────────────────────────────────────────────────

    miracl_corpus_path = KO_MIRACL / "corpus.jsonl"
    if not miracl_corpus_path.exists():
        print(f"error: {miracl_corpus_path} not found. Run: uv run scripts/download-ko-miracl.py", file=sys.stderr)
        sys.exit(1)

    print("loading ko-miracl corpus...")
    docs = load_corpus(miracl_corpus_path)
    print(f"  {len(docs):,} documents")

    print("building vocabulary...")
    token_docs, doc_tokens = build_vocab(docs)
    raw_vocab = set(token_docs.keys())
    print(f"  {len(raw_vocab):,} unique Hangul tokens")

    candidates = [t for t in raw_vocab if len(t) >= MIN_LEN]
    print(f"  {len(candidates):,} tokens >= {MIN_LEN} chars")

    print(f"decomposing {len(candidates):,} tokens through lindera...")
    compounds = find_compounds(candidates)
    print(f"  {len(compounds):,} compound tokens")

    # ── Select with strict filter ──────────────────────────────────────────────

    selected = select_compounds(compounds, token_docs, doc_tokens, docs, raw_vocab, strict=True)
    print(f"  {len(selected):,} clean compounds (strict: no sub-part in raw_vocab)")

    # ── Fallback: add ko-strategyqa ────────────────────────────────────────────

    all_docs = list(docs)
    all_token_docs = dict(token_docs)
    all_doc_tokens = dict(doc_tokens)
    all_compounds = list(compounds)

    if len(selected) < TARGET:
        strat_path = KO_STRATEGYQA / "corpus.jsonl"
        if strat_path.exists():
            print(f"  only {len(selected)} compounds — loading ko-strategyqa as fallback...")
            strat_docs = load_corpus(strat_path)
            print(f"  {len(strat_docs):,} strategyqa documents")

            strat_token_docs, strat_doc_tokens = build_vocab(strat_docs)

            # Merge into combined view
            combined_token_docs: dict[str, set[str]] = defaultdict(set)
            for t, ds in token_docs.items():
                combined_token_docs[t] |= ds
            for t, ds in strat_token_docs.items():
                combined_token_docs[t] |= ds

            combined_raw_vocab = raw_vocab | set(strat_token_docs.keys())
            combined_doc_tokens = {**doc_tokens, **strat_doc_tokens}
            combined_docs = all_docs + strat_docs

            # New candidates: tokens only in strategyqa (avoid re-processing)
            new_candidates = [
                t for t in strat_token_docs
                if len(t) >= MIN_LEN and t not in raw_vocab
            ]
            print(f"  decomposing {len(new_candidates):,} new strategyqa tokens...")
            new_compounds = find_compounds(new_candidates)

            extra = select_compounds(
                new_compounds,
                combined_token_docs,
                combined_doc_tokens,
                combined_docs,
                combined_raw_vocab,
                strict=True,
            )
            selected += extra[: TARGET - len(selected)]
            all_docs = combined_docs
            all_token_docs = dict(combined_token_docs)
            all_doc_tokens = combined_doc_tokens
            all_compounds += new_compounds
            print(f"  after fallback: {len(selected):,} compounds")
        else:
            print(f"  warning: ko-strategyqa not found at {strat_path}")

    # ── Relaxed filter if still < TARGET ──────────────────────────────────────

    if len(selected) < TARGET:
        print(f"  only {len(selected)} — relaxing filter (sub-part in ≤ {RELAX_THRESHOLD} docs)...")
        # all_compounds accumulates from primary + strategyqa fallback
        selected_surfaces = {r["surface"] for r in selected}
        remaining = [(s, p) for s, p in all_compounds if s not in selected_surfaces]
        extra = select_compounds(
            remaining,
            all_token_docs,
            all_doc_tokens,
            all_docs,
            set(all_token_docs.keys()),
            strict=False,
        )
        selected += extra[: TARGET - len(selected)]
        print(f"  after relaxed filter: {len(selected):,} compounds")

    if len(selected) < TARGET:
        print(f"warning: only found {len(selected)} compounds (target {TARGET})")

    # ── Print examples ─────────────────────────────────────────────────────────

    print(f"\ncompound examples:")
    for r in selected[:10]:
        print(f"  {r['surface']} → {' + '.join(r['parts'])}  (ctx: {r['ctx']}, doc: {r['doc_id']})")

    # ── Write BEIR dataset ─────────────────────────────────────────────────────

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qrels").mkdir(exist_ok=True)

    # Corpus: reuse full ko-miracl corpus (plus any strategyqa docs if added)
    miracl_docs = load_corpus(miracl_corpus_path)
    needed_doc_ids = {r["doc_id"] for r in selected}
    # Add strategyqa docs that were selected as source docs
    extra_docs = [d for d in all_docs if d["_id"] in needed_doc_ids and d not in miracl_docs]
    corpus_out = miracl_docs + extra_docs

    write_jsonl(OUT / "corpus.jsonl", corpus_out)

    queries = [
        {
            "_id": f"q{i+1:03d}",
            "text": " ".join(r["parts"]) + " " + r["ctx"],
        }
        for i, r in enumerate(selected)
    ]
    write_jsonl(OUT / "queries.jsonl", queries)

    qrels_path = OUT / "qrels" / "test.tsv"
    with open(qrels_path, "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for i, r in enumerate(selected):
            f.write(f"q{i+1:03d}\t{r['doc_id']}\t1\n")
    print(f"  wrote {len(selected):,} qrels → {qrels_path}")

    print(f"\ndone. dataset at {OUT}")
    print(f"queries use sub-components of compound nouns — BM25 without decompounding should score ~0")


if __name__ == "__main__":
    main()
