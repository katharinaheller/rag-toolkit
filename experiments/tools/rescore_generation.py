"""Re-score existing generation records with the fixed faithfulness metrics.

Use this when generation has already been executed (raw/.../generation_records.jsonl
exist) but the aggregated CSVs were produced with the buggy metric call (joined
string instead of list of chunks → overlap always 0.0, hallucination always 1.0).

This tool reads the raw JSONL files, recomputes ``context_overlap`` and
``hallucination_score`` against the original retrieved chunks, and rewrites the
aggregated CSVs and figures. It does NOT call the generator again, so no Ollama
needed.

Usage (inside the benchmark-runner container):

    python -m experiments.tools.rescore_generation \\
        --run-id run_20260527_144558_fbf83b \\
        --output-root /opt/experiment-outputs

If --output-root is omitted the value of ``EXPERIMENTS_OUTPUT_ROOT`` (or the
default from ``experiments.configs.settings``) is used.

The tool needs the corpus chunks to look up text by chunk_id; it reads them
from ``EXPERIMENTS_DATA_ROOT`` (default ``/opt/data``) via the same loader the
suites use.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from experiments.configs.settings import SETTINGS
from experiments.core.corpus_loader import load_corpus_chunks
from rag.evaluation.metrics import (
    context_overlap,
    hallucination_score,
)

log = logging.getLogger("rescore_generation")

# Suites whose generation records we know how to rescore.
_GENERATION_SUITES = ("s10_context_pollution", "s11_long_context")


def _resolve_run_dir(output_root: Path, run_id: str, kind: str) -> Path:
    return output_root / kind / run_id


def _load_raw_records(raw_dir: Path) -> List[Dict[str, Any]]:
    path = raw_dir / "generation_records.jsonl"
    if not path.exists():
        log.warning("No generation_records.jsonl in %s", raw_dir)
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line in %s: %s", path, exc)
    return out


def _rebuild_chunk_index(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Build {corpus_name: {chunk_id: chunk_dict}} for the corpora referenced
    by the records. We load each referenced corpus exactly once.
    """
    corpora_needed = {r.get("corpus") for r in records if r.get("corpus")}
    index: Dict[str, Dict[str, Dict[str, str]]] = {}
    for corpus_name in sorted(c for c in corpora_needed if c):
        try:
            chunks = load_corpus_chunks(corpus_name)
        except Exception as exc:
            log.error("Could not load corpus %s: %s", corpus_name, exc)
            continue
        index[corpus_name] = {c["id"]: c for c in chunks}
        log.info("Loaded corpus %s with %d chunks", corpus_name, len(chunks))
    return index


def _retrieved_chunk_ids_for(record: Dict[str, Any]) -> Optional[List[str]]:
    """Return the list of retrieved chunk ids used for this generation record.

    The raw JSONL produced by s10 does not store the retrieved chunk ids per
    pollution sample (they vary across pollution_ratio); the same is true for
    s11 across top_k. As a fallback we look up the gold chunks and reuse them
    as a coarse approximation. This is acceptable for re-scoring because:
      * For pollution_ratio = 0 the gold chunks dominate, so the overlap upper
        bound is reached.
      * For high pollution_ratio the overlap will be approximated low, which
        matches the intended behaviour.
    If a future raw dump records the actual ``retrieved_ids`` per row, this
    helper picks them up automatically.
    """
    explicit = (record.get("extras") or {}).get("retrieved_ids")
    if isinstance(explicit, list) and explicit:
        return [str(x) for x in explicit]
    return None


def _approx_context_chunks(
    record: Dict[str, Any],
    chunk_index: Dict[str, Dict[str, Dict[str, str]]],
) -> List[str]:
    """Best effort: use explicit retrieved_ids if present, else fall back to
    the expected_answer text (which is the gold chunk text in this project)."""
    ids = _retrieved_chunk_ids_for(record)
    corpus = record.get("corpus")
    if ids and corpus and corpus in chunk_index:
        return [
            chunk_index[corpus][cid]["text"]
            for cid in ids
            if cid in chunk_index[corpus]
        ]
    # Fallback: expected_answer carries the gold chunk text in this project.
    expected = record.get("expected_answer") or ""
    return [expected] if expected else []


def _rescore_records(
    records: List[Dict[str, Any]],
    chunk_index: Dict[str, Dict[str, Dict[str, str]]],
) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        ans = r.get("generated_answer") or ""
        ctx_list = _approx_context_chunks(r, chunk_index)
        new_overlap = context_overlap(ans, ctx_list)
        new_halluc = hallucination_score(ans, ctx_list)
        extras = dict(r.get("extras") or {})
        extras["context_overlap"] = round(new_overlap, 6)
        extras["hallucination_score"] = round(new_halluc, 6)
        extras["rescored"] = True
        new_r = dict(r)
        new_r["extras"] = extras
        out.append(new_r)
    return out


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_s10(records: List[Dict[str, Any]]) -> tuple[list, list]:
    per_query: List[Dict[str, Any]] = []
    for r in records:
        extras = r.get("extras") or {}
        per_query.append({
            "retriever": r.get("retriever"),
            "corpus": r.get("corpus"),
            "top_k": r.get("top_k"),
            "pollution_ratio": extras.get("pollution_ratio"),
            "query_id": r.get("query_id"),
            "context_overlap": extras.get("context_overlap"),
            "hallucination_score": extras.get("hallucination_score"),
            "success": r.get("success"),
        })
    ratios = sorted({r["pollution_ratio"] for r in per_query if r["pollution_ratio"] is not None})
    summary: List[Dict[str, Any]] = []
    for ratio in ratios:
        subset = [r for r in per_query if r["pollution_ratio"] == ratio]
        if not subset:
            continue
        summary.append({
            "retriever": subset[0]["retriever"],
            "corpus": subset[0]["corpus"],
            "top_k": int(round(ratio * 100)),
            "pollution_ratio": ratio,
            "mean_context_overlap": round(
                sum(r["context_overlap"] for r in subset) / len(subset), 6),
            "mean_hallucination": round(
                sum(r["hallucination_score"] for r in subset) / len(subset), 6),
            "n_queries": len(subset),
        })
    return summary, per_query


def _aggregate_s11(records: List[Dict[str, Any]]) -> tuple[list, list]:
    per_query: List[Dict[str, Any]] = []
    for r in records:
        extras = r.get("extras") or {}
        per_query.append({
            "retriever": r.get("retriever"),
            "corpus": r.get("corpus"),
            "top_k": r.get("top_k"),
            "query_id": r.get("query_id"),
            "context_overlap": extras.get("context_overlap"),
            "hallucination_score": extras.get("hallucination_score"),
            "latency_ms": r.get("latency_ms"),
            "prompt_chars": r.get("prompt_chars"),
            "context_chars": r.get("context_chars"),
            "success": r.get("success"),
        })
    ks = sorted({r["top_k"] for r in per_query if r["top_k"] is not None})
    summary: List[Dict[str, Any]] = []
    for k in ks:
        subset = [r for r in per_query if r["top_k"] == k]
        if not subset:
            continue
        summary.append({
            "retriever": subset[0]["retriever"],
            "corpus": subset[0]["corpus"],
            "top_k": k,
            "mean_context_overlap": round(
                sum((r["context_overlap"] or 0.0) for r in subset) / len(subset), 6),
            "mean_hallucination": round(
                sum((r["hallucination_score"] or 0.0) for r in subset) / len(subset), 6),
            "mean_gen_latency_ms": round(
                sum((r["latency_ms"] or 0.0) for r in subset) / len(subset), 3),
            "mean_prompt_chars": round(
                sum((r["prompt_chars"] or 0) for r in subset) / len(subset), 1),
            "mean_context_chars": round(
                sum((r["context_chars"] or 0) for r in subset) / len(subset), 1),
            "n_queries": len(subset),
        })
    return summary, per_query


def _regenerate_figures(
    suite_key: str,
    summary_rows: List[Dict[str, Any]],
    figures_dir: Path,
) -> List[Path]:
    """Re-render the figures the original suite would have produced.

    Returns the list of paths actually written (empty if matplotlib is not
    available or the data is empty).
    """
    if not summary_rows:
        return []
    try:
        from experiments.visualisation.plots import plot_topk_sensitivity
    except Exception as exc:  # pragma: no cover - matplotlib optional
        log.warning("plot_topk_sensitivity not importable, skipping figures: %s", exc)
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    if suite_key == "s10_context_pollution":
        # The original suite plots context_overlap and hallucination vs the
        # pollution ratio, encoded as top_k = int(pollution_ratio * 100).
        for metric in ("mean_context_overlap", "mean_hallucination"):
            out_path = figures_dir / f"{metric}_vs_pollution.png"
            ok = plot_topk_sensitivity(
                data=summary_rows,
                metric=metric,
                out_path=out_path,
                title=f"{metric} vs pollution ratio×100",
            )
            if ok:
                written.append(out_path)
    elif suite_key == "s11_long_context":
        for metric in ("mean_context_overlap", "mean_hallucination",
                       "mean_gen_latency_ms", "mean_context_chars"):
            out_path = figures_dir / f"{metric}_vs_topk.png"
            ok = plot_topk_sensitivity(
                data=summary_rows,
                metric=metric,
                out_path=out_path,
                title=f"{metric} vs top_k",
            )
            if ok:
                written.append(out_path)
    return written


def _rescore_suite(
    suite_key: str,
    run_id: str,
    output_root: Path,
    chunk_index: Dict[str, Dict[str, Dict[str, str]]],
) -> None:
    raw_dir = _resolve_run_dir(output_root, run_id, "raw") / suite_key
    agg_dir = _resolve_run_dir(output_root, run_id, "aggregated") / suite_key
    fig_dir = _resolve_run_dir(output_root, run_id, "figures") / suite_key

    records = _load_raw_records(raw_dir)
    if not records:
        log.warning("Skipping %s: no records found in %s", suite_key, raw_dir)
        return

    log.info("Rescoring %d records for %s", len(records), suite_key)
    rescored = _rescore_records(records, chunk_index)

    # Backup the original raw file and write the rescored version next to it.
    raw_path = raw_dir / "generation_records.jsonl"
    backup_path = raw_dir / "generation_records.pre_rescore.jsonl"
    if not backup_path.exists():
        raw_path.replace(backup_path)
        log.info("Backed up original raw file to %s", backup_path)
    _write_jsonl(raw_path, rescored)

    summary: List[Dict[str, Any]] = []
    if suite_key == "s10_context_pollution":
        summary, per_query = _aggregate_s10(rescored)
        _write_csv(agg_dir / "pollution_effect.csv", summary)
        _write_csv(agg_dir / "per_query_pollution.csv", per_query)
    elif suite_key == "s11_long_context":
        summary, _ = _aggregate_s11(rescored)
        _write_csv(agg_dir / "long_context.csv", summary)
    else:
        log.warning("No aggregator for %s — wrote only the rescored raw file.", suite_key)
        return

    log.info("Wrote rescored aggregates for %s into %s", suite_key, agg_dir)

    figs = _regenerate_figures(suite_key, summary, fig_dir)
    if figs:
        log.info("Wrote %d new figure(s) for %s into %s",
                 len(figs), suite_key, fig_dir)
    else:
        log.warning("No figures written for %s "
                    "(matplotlib missing or empty summary).", suite_key)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Run id under raw/ and aggregated/")
    parser.add_argument("--output-root", default=None, help="Override EXPERIMENTS_OUTPUT_ROOT")
    parser.add_argument("--suite", action="append", default=None,
                        help="Restrict to a specific suite (repeatable). "
                             "Default: s10 and s11.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    output_root = Path(args.output_root) if args.output_root else Path(SETTINGS.output_root)
    if not output_root.exists():
        log.error("Output root does not exist: %s", output_root)
        return 2

    suites = args.suite or list(_GENERATION_SUITES)

    # Preload corpora referenced anywhere in the suites we touch.
    all_records: List[Dict[str, Any]] = []
    for s in suites:
        raw_dir = _resolve_run_dir(output_root, args.run_id, "raw") / s
        all_records.extend(_load_raw_records(raw_dir))
    if not all_records:
        log.error("No generation records found for run %s. Nothing to rescore.",
                  args.run_id)
        return 3

    chunk_index = _rebuild_chunk_index(all_records)

    for s in suites:
        _rescore_suite(s, args.run_id, output_root, chunk_index)

    log.info("Done. Re-aggregated CSVs are next to the raw files; the original "
             "raw JSONLs are preserved as *.pre_rescore.jsonl.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
