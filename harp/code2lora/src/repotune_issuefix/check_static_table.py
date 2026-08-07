#!/usr/bin/env python3
"""Smoke-check a static issue-fix table before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-name", default="google/gemma-4-E2B-it")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--target-column", default="target_text")
    parser.add_argument("--fail-on-overlength", action="store_true")
    args = parser.parse_args()

    table = pq.read_table(args.input)
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    target_column = args.target_column if args.target_column in cols else "target_patch"
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
    n = table.num_rows if not args.limit else min(args.limit, table.num_rows)
    bad = []
    overlength = []
    target_toks = []
    prompt_toks = []
    total_toks = []
    emb_dims = []
    for i in range(n):
        prompt = cols["prompt"][i] or ""
        target = cols[target_column][i] or ""
        emb = cols["repo_state_embedding"][i]
        emb_dims.append(len(emb) if emb is not None else 0)
        p_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
        t_len = len(tok(target + (tok.eos_token or ""), add_special_tokens=False)["input_ids"])
        prompt_toks.append(p_len)
        target_toks.append(t_len)
        total_toks.append(p_len + t_len)
        if not prompt or not target or len(emb) != 2048:
            bad.append({
                "row": i,
                "prompt": bool(prompt),
                "target": bool(target),
                "emb_dim": len(emb),
                "prompt_tokens": p_len,
                "target_tokens": t_len,
                "total_tokens": p_len + t_len,
            })
        if p_len + t_len > args.max_seq_len:
            overlength.append({
                "row": i,
                "prompt_tokens": p_len,
                "target_tokens": t_len,
                "total_tokens": p_len + t_len,
            })
    summary = {
        "rows_checked": n,
        "bad_rows": len(bad),
        "bad_examples": bad[:10],
        "overlength_rows": len(overlength),
        "overlength_examples": overlength[:10],
        "retained_at_max_seq_len": n - len(overlength),
        "retained_pct_at_max_seq_len": 100.0 * (n - len(overlength)) / max(1, n),
        "embedding_dims": sorted(set(emb_dims)),
        "prompt_tokens_max": max(prompt_toks) if prompt_toks else 0,
        "target_tokens_max": max(target_toks) if target_toks else 0,
        "total_tokens_max": max(total_toks) if total_toks else 0,
        "target_tokens_mean": sum(target_toks) / max(1, len(target_toks)),
        "total_tokens_mean": sum(total_toks) / max(1, len(total_toks)),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if bad or (args.fail_on_overlength and overlength):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
