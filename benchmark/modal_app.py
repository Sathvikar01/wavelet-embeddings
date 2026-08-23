"""Reproducible head-pruning benchmark on Modal.

Run with::

    modal run benchmark/modal_app.py

Optional CLI flags pass straight through to ``run_benchmark``::

    modal run benchmark/modal_app.py \\
        --models roberta-base deberta-base \\
        --datasets wikitext2 penn_treebank \\
        --seeds 0 1 2 \\
        --max-sentences 3000

Outputs (per-input correlations CSV, aggregate CSV, seed-variance CSV,
comparisons JSON plus a tar bundle copy of the project source) are written
to a Modal *Volume* named ``wavelet-bench`` under ``runs/<timestamp>/``.
After the run prints the in-container path so you can copy it down with
``modal volume get wavelet-bench <path> ./bench_out``.
"""

import sys
import argparse
import time
import tarfile
import io
import os
from typing import List, Optional

import modal
import numpy as np


APP_NAME = "wavelet-bench"


# --------------------------------------------------------------------------- #
# Modal resources
# --------------------------------------------------------------------------- #

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch==2.3.0",
        "transformers==4.42.2",
        "numpy>=1.24,<2.0",
        "PyWavelets>=1.5",
        "scikit-learn>=1.4",
        "matplotlib>=3.7",
        "scipy>=1.11",
        "datasets>=2.14",
    )
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1"})
    # Mount the local project source into the image so the in-container
    # process imports the *same* code the developer is editing.
    .add_local_dir(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        remote_path="/root/wavelet_embeddings",
        ignore=lambda p: (
            "__pycache__" in str(p) or "results" in str(p)
            or ".git" in str(p)
            or str(p).endswith((".pyc", ".npz"))
        ),
        copy=False,
    )
)

# Persistent storage for outputs across runs.
vol = modal.Volume.from_name("wavelet-bench", create_if_missing=True)

app = modal.App(APP_NAME, image=image, volumes={"/vol": vol})


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

DEFAULT_MODELS = [
    "bert-base", "distilbert", "gpt2", "roberta-base", "tinyllama",
]
DEFAULT_DATASETS = ["wikitext2", "penn_treebank", "glue_subset"]
DEFAULT_SEEDS = [0, 1, 2]


def _bundle_source(out_tar: str) -> None:
    """Tar the project source (excluding results / caches) into ``out_tar``
    so the run is fully reproducible from the artefacts alone."""
    root = "/root/wavelet_embeddings"
    with tarfile.open(out_tar, "w:gz") as tar:
        for dirpath, _dirs, files in os.walk(root):
            if "__pycache__" in dirpath or "results" in dirpath \
                  or ".git" in dirpath:
                continue
            for f in files:
                if f.endswith((".pyc", ".npz")):
                    continue
                full = os.path.join(dirpath, f)
                tar.add(full, arcname=os.path.relpath(full, root))


@app.function(gpu="A10G", timeout=60 * 60 * 6)
def bench_cell(model_key, ds_name, seed, max_sentences, max_length, metric,
               timestamp=None):
    """Run ONE (model, dataset, seed) cell on its own A10G container.

    Writes the cell's ``BenchRecord`` as a per-cell JSON file to the
    persistent Volume under ``/vol/runs/<timestamp>/cells/`` so per-cell
    results survive even if the dispatch driver or the aggregation step
    is interrupted. Also returns the record as a JSON-able dict so the
    driver can aggregate in-process when it stays alive.
    """
    sys.path.insert(0, "/root/wavelet_embeddings")
    os.chdir("/root/wavelet_embeddings")
    import json
    from benchmark.runner import run_cell

    ts = timestamp or time.strftime("%Y%m%d-%H%M%S")
    rec = run_cell(model_key, ds_name, seed,
                    max_sentences=max_sentences, max_len=max_length,
                    cache_dir=None, metric=metric, device="cuda",
                    verbose=True, keep_per_input_features=True)
    rd = rec.to_dict()

    # Persist per-cell JSON to the Volume so a dead driver cannot
    # lose computed work. Commit only the new cell file.
    cells_dir = f"/vol/runs/{ts}/cells"
    os.makedirs(cells_dir, exist_ok=True)
    base = f"cell_{model_key.replace('/', '_')}_{ds_name}_{seed}"
    fname = base + ".json"
    with open(os.path.join(cells_dir, fname), "w", encoding="utf-8") as f:
        json.dump(rd, f)

    # Phase-6 LOOO ridge support: persist per-input feature matrices +
    # per-input measured losses as a .npz alongside the per-cell JSON.
    # The aggregator loads these (instead of just the JSON dict) to run
    # the leave-one-cell-out ridge post-pass. Per-feature matrices are
    # cell-internal numpy arrays (already standardised within the cell
    # by feature_matrix.build_feature_matrix) and safe to dump.
    try:
        if rec.per_input_features:
            npz_path = os.path.join(cells_dir, base + ".npz")
            # Stack inputs: arrays of shape (n_inputs, n_heads, n_features)
            # and (n_inputs, n_heads) -- ragged across cells of different
            # n_heads, but each cell's array is rectangular.
            Xin = np.stack(rec.per_input_features, axis=0)
            yin = np.stack(rec.per_input_loss, axis=0)
            np.savez_compressed(npz_path, X=Xin, y=yin)
            print(f"[cell] persisted {base}.npz "
                  f"({Xin.shape[0]} inputs x {Xin.shape[1]} heads x "
                  f"{Xin.shape[2]} features)", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"[cell] npz persist warning: {e}", flush=True)
    try:
        vol.commit()
    except Exception as e:  # pragma: no cover
        print(f"[cell] vol.commit() warning: {e}", flush=True)
    print(f"[cell] persisted {fname} ({len(rd.get('per_input', {}).get('wavelet', []))} inputs)",
          flush=True)
    return rd


@app.function(cpu=2, timeout=60 * 60 * 8)
def aggregate_run(timestamp, expected_cells, deadline_sec=8 * 3600):
    """Scan ``/vol/runs/<timestamp>/cells/`` until all ``expected_cells``
    per-cell JSONs have landed (or until ``deadline_sec`` elapses), then
    build the per-input CSV, aggregate CSV, seed-variance CSV and the
    comparisons JSON via ``benchmark.runner.write_results`` and commit
    them to ``/vol/runs/<timestamp>/``.

    Runs on a small CPU container (cheap) and is meant to be invoked
    server-side by :func:`run_bench` after it has dispatched every cell
    via ``bench_cell.spawn``. Because the aggregator itself is a
    Modal function, it keeps running even if the local entrypoint that
    started the run is disconnected.
    """
    sys.path.insert(0, "/root/wavelet_embeddings")
    os.chdir("/root/wavelet_embeddings")
    import json
    import time as _time
    from benchmark.runner import write_results

    ts = timestamp
    cells_dir = f"/vol/runs/{ts}/cells"
    out_dir_on_vol = f"/vol/runs/{ts}"

    deadline = _time.time() + deadline_sec
    done = []
    while True:
        vol.reload()
        if os.path.isdir(cells_dir):
            done = sorted(f for f in os.listdir(cells_dir)
                            if f.startswith("cell_") and f.endswith(".json"))
            print(f"[aggregate] {len(done)}/{expected_cells} cells present",
                  flush=True)
            if len(done) >= expected_cells:
                break
        if _time.time() >= deadline:
            print(f"[aggregate] deadline reached with {len(done)}"
                  f"/{expected_cells} cells; aggregating partial run",
                  flush=True)
            break
        _time.sleep(120)

    cells_dir_ok = os.path.isdir(cells_dir)
    recs = []
    records_with_features = []
    if cells_dir_ok:
        for f in sorted(os.listdir(cells_dir)):
            if not (f.startswith("cell_") and f.endswith(".json")):
                continue
            try:
                with open(os.path.join(cells_dir, f), encoding="utf-8") as fh:
                    rd = json.load(fh)
                if not rd.get("n_heads"):
                    continue
                recs.append(rd)

                # Phase-6 LOOO ridge: if a matching .npz exists, build a
                # full BenchRecord carrying per_input_features / loss so
                # the post-pass can fit leave-one-cell-out. We construct
                # BenchRecords here (not dicts) for the LOOO call only;
                # the JSON-dict path (write_results) is unaffected.
                npz_path = os.path.join(cells_dir, f[:-5] + ".npz")
                if os.path.exists(npz_path):
                    try:
                        from benchmark.runner import BenchRecord
                        br = BenchRecord(
                            model=rd.get("model", ""),
                            dataset=rd.get("dataset", ""),
                            seed=int(rd.get("seed", 0)),
                            n_heads=int(rd.get("n_heads", 0)),
                            metric_used=rd.get("metric_used", ""),
                            per_input=dict(rd.get("per_input", {})),
                        )
                        with np.load(npz_path) as z:
                            Xin = z["X"]   # (n_inputs, n_heads, n_features)
                            yin = z["y"]   # (n_inputs, n_heads)
                            br.per_input_features = list(Xin)
                            br.per_input_loss = list(yin)
                        records_with_features.append(br)
                    except Exception as e:  # pragma: no cover
                        print(f"[aggregate] npz load {f} warning: {e}",
                              flush=True)
            except Exception as e:  # pragma: no cover
                print(f"[aggregate] skip {f}: {e}", flush=True)

        print(f"[aggregate] collected {len(recs)} valid cells "
              f"({len(records_with_features)} with Feature npz)",
              flush=True)

    if not recs:
        print("[aggregate] no valid records to aggregate; aborting commit",
              flush=True)
        return out_dir_on_vol, []

    # Phase-6 leave-one-cell-out ridge post-pass. Run before
    # write_results so the populated ridge per-input correlations land in
    # the per-input CSV / summary.json / aggregate.csv. If even one cell
    # lacks an npz its row's ridge_* correlations are set to length-0 (a
    # missing-row placeholder) so partial runs still aggregate cleanly.
    if records_with_features:
        try:
            from benchmark.runner import BenchResult
            from benchmark.ridge_looo import apply_ridge_looo
            br_result = BenchResult(records=records_with_features)
            coeffs_art = apply_ridge_looo(br_result, verbose=True,
                                            return_coeffs=True)
            br_result.ridge_folds_coeffs = coeffs_art
            # Pull the populated ridge per-input correlations back into
            # the JSON dicts (the same column names, list form).
            for r in records_with_features:
                key = (r.model, r.dataset, r.seed)
                for rd in recs:
                    if (rd.get("model"), rd.get("dataset"),
                            rd.get("seed")) == key:
                        for rid_pred, vals in r.per_input.items():
                            if rid_pred.startswith("ridge_") or \
                                    rid_pred.endswith("_model"):
                                rd.setdefault("per_input", {})[rid_pred] = \
                                    [float(v) for v in vals]
                        # Stash the spearman / kendall / cosine_pres /
                        # pruning_loss extras so the aggregator's
                        # write_results can emit predictor_metrics.csv.
                        for fld in ("per_input_spearman",
                                     "per_input_kendall",
                                     "per_input_cosine_pres",
                                     "per_input_pruning_loss"):
                            extra = getattr(r, fld)
                            for rid_pred, vals in extra.items():
                                if rid_pred.endswith("_model"):
                                    rd.setdefault(fld, {})[rid_pred] = \
                                        [float(v) for v in vals]
                        break
            # Stash the per-fold coeffs on the aggregate so write_results
            # can emit feature_importance.csv from the aggregator path too.
            # We wrap coeffs_art in a JSON-friendly form: it becomes
            # returned_csv_path/feature_importance.csv on write.
            import_outfeat = getattr(br_result, "ridge_folds_coeffs", None)
            at_local_dataset = {"_ridge_folds_coeffs": import_outfeat}
            # Make the last record-on-disk absorb this metadata for the
            # writer pass below (write_results is BenchResult/dict-tolerant
            # but our list-of-dicts path can't see the artefact otherwise).
            if recs and "_ridge_folds_coeffs" not in recs[-1]:
                recs[-1]["_ridge_folds_coeffs"] = import_outfeat
        except Exception as e:  # pragma: no cover
            print(f"[aggregate] ridge LOOO post-pass warning: {e}",
              flush=True)

    local_out = "/tmp/bench_out"
    import shutil
    if os.path.isdir(local_out):
        shutil.rmtree(local_out)
    os.makedirs(local_out, exist_ok=True)
    doc = write_results(recs, local_out, test="wilcoxon")

    os.makedirs(out_dir_on_vol, exist_ok=True)
    for f in os.listdir(local_out):
        shutil.copy(os.path.join(local_out, f),
                    os.path.join(out_dir_on_vol, f))
    _bundle_source(os.path.join(out_dir_on_vol, "source.tar.gz"))
    vol.commit()
    print(f"[aggregate] committed outputs to {out_dir_on_vol}", flush=True)
    return out_dir_on_vol, _summaries(recs)


@app.function(cpu=2, timeout=60 * 60 * 6)
def run_bench(models, datasets, seeds, max_sentences, max_length, metric):
    """Driver: fans out every (model, dataset, seed) cell to parallel A10G
    containers via ``bench_cell.spawn`` (fire-and-forget) and then
    schedules ``aggregate_run`` (also a server-side Modal function) to wait
    for every cell to land on the Volume and write the aggregate outputs.

    Cells persist their own per-cell JSON to the Volume as soon as they
    finish, so driver disconnects / client cancellations do not lose work;
    the aggregator later reads the per-cell JSONs and writes the canonical
    CSV + comparison bundle.

    The local entrypoint returns immediately after launching this function
    server-side -- everything important runs remotely.
    """
    sys.path.insert(0, "/root/wavelet_embeddings")
    os.chdir("/root/wavelet_embeddings")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir_on_vol = f"/vol/runs/{timestamp}"
    os.makedirs(out_dir_on_vol, exist_ok=True)

    jobs = [
        (m, d, s, max_sentences, max_length, metric, timestamp)
        for m in models for d in datasets for s in seeds
    ]
    print(f"[driver] dispatching {len(jobs)} cells to parallel A10Gs "
          f"(run={timestamp})", flush=True)
    n_dispatched = 0
    for job in jobs:
        bench_cell.spawn(*job)
        n_dispatched += 1
    print(f"[driver] spawn returned for {n_dispatched}/{len(jobs)} cells; "
          f"scheduling aggregator (deadline 8h)", flush=True)
    aggregate_run.spawn(timestamp, len(jobs), 8 * 3600)
    return out_dir_on_vol, []


def _summaries(result):
    """Return a flat list of (model, dataset, seed, predictor,
    mean, std, n) rows for the local-entrypoint pretty-printer. Accepts
    either a BenchResult or a list of plain dicts (the JSON-able form
    returned by the parallel cells)."""
    import numpy as np
    if hasattr(result, "records"):
        recs = result.records
    else:
        recs = list(result)
    s = []
    for rec in recs:
        model = getattr(rec, "model", None) or rec.get("model")
        dataset = getattr(rec, "dataset", None) or rec.get("dataset")
        seed = getattr(rec, "seed", None) or rec.get("seed")
        per_input = (getattr(rec, "per_input", None)
                     or rec.get("per_input", {}))
        for pname, vals in per_input.items():
            if not vals:
                continue
            s.append((model, dataset, seed, pname,
                       float(np.mean(vals)), float(np.std(vals, ddof=1))
                       if len(vals) > 1 else 0.0, len(vals)))
    return s


def _parse_args():
    p = argparse.ArgumentParser(description="Modal head-pruning benchmark.")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--max-sentences", type=int, default=2000)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--metric", default="cosine_drop",
                    choices=["cosine_drop", "attention_drift",
                             "kl_div_next_token"])
    return p.parse_args()


@app.local_entrypoint()
def main(
    models: Optional[str] = None,
    datasets: Optional[str] = None,
    seeds: Optional[str] = None,
    max_sentences: int = 2000,
    max_length: int = 64,
    metric: str = "cosine_drop",
):
    """Run the head-pruning benchmark on Modal.

    Pass lists as comma-separated values, e.g.
    ``--models bert-base,distilbert``  ``--seeds 0,1,2``.  Empty values fall
    back to the documented defaults.

    The driver fans out every cell via ``bench_cell.spawn`` (fire &
    forget) and schedules a separate ``aggregate_run`` Modal function to
    wait for every per-cell JSON file to land on the Volume and then write
    the canonical CSVs + ``summary.json``.  Both run server-side, so even
    if this local entrypoint exits the run keeps going.

    Poll the run via::

        modal volume ls wavelet-bench runs
        modal volume ls wavelet-bench runs/<timestamp>
        modal volume ls wavelet-bench runs/<timestamp>/cells

    And download when ``summary.json`` and ``per_input_scores.csv`` exist
    in ``runs/<timestamp>/`` (a marker that ``aggregate_run`` completed)::

        modal volume get wavelet-bench runs/<timestamp> ./bench_out
    """
    m = _parse_csv(models) if models else DEFAULT_MODELS
    d = _parse_csv(datasets) if datasets else DEFAULT_DATASETS
    s = [int(x) for x in _parse_csv(seeds)] if seeds else DEFAULT_SEEDS
    n_cells = len(m) * len(d) * len(s)
    out_dir, _summary = run_bench.remote(
        m, d, s, max_sentences, max_length, metric,
    )
    print("\nModal benchmark dispatched.")
    print(f"   cells expected     : {n_cells}")
    print(f"   volume run path    : {out_dir}")
    print(f"   deadline (max wait): 8h")
    print("\nThe run keeps going on Modal after this local process exits:")
    print("    Per-cell JSON  -> run/cells/cell_<model>_<dataset>_<seed>.json")
    print("    Aggregate CSVs  -> run/per_input_scores.csv, run/aggregate.csv,")
    print("                       run/seed_variance.csv, run/summary.json")
    print("Run 'modal volume ls wavelet-bench runs' to poll for the run dir,")
    print("then 'modal volume ls wavelet-bench runs/<timestamp>' to verify")
    print("aggregation has finished before downloading with:")
    print(f"    modal volume get wavelet-bench {out_dir} ./bench_out")


def _parse_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _entry():
    """Direct-invocation fallback; some workflows still import this."""
    args = _parse_args()
    run_bench.remote(args.models, args.datasets, args.seeds,
                       args.max_sentences, args.max_length, args.metric)
