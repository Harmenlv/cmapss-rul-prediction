#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chronos-2 + BiLSTM hybrid RUL adaptation for NASA C-MAPSS (V6, bug-fixed).

Reviewer-facing experiments:
  E1: FD001-FD004, repeated seeds
  E2: multi-seed mean/std
  E3: reduced-data learning curves
  E4: cross-subset transfer (train on one subset, evaluate another)
  E5: frozen Chronos-2 feature linear probe
  E6: LSTM-backbone replacement with the same adapter/GAT/fusion/head
  E7: module ablations
  E8: parameters, training/inference time, peak CUDA memory

Fixes relative to v5_fast (see CHANGELOG in the report):
  * training-split statistics are re-fitted after low-data subsampling
    (no leakage from the 90% of engines that are dropped at fraction<1)
  * Chronos cache is validated against expected (N, 24, ...) shape before reuse
  * cache name carries cond_norm / pool tags only when they are not default,
    so existing v5 caches are still reused
  * `--resume` skips finished runs (metrics.json) and reuses caches
  * optional `--cond_norm` per-operating-condition scaling (FD002/FD004)
  * LSTM backbone is vectorised (one batched call instead of 24 python-loop calls)
  * `--merge_from` rebuilds the paper table without re-running anything

V7 throughput pass (this file), on top of the v6 correctness fixes:
  * txt parsing via np.loadtxt: pandas' C engine rejects a regex separator and
    silently falls back to the python engine, roughly 10x slower on 60k rows
  * dataset frames are memoised in-process; prepared windows/graph are memoised
    on disk keyed by (subset, split_seed, window, rmax, fraction, cond_norm, topk),
    so the same split is not rebuilt once per seed (v6 rebuilt it 3x per fraction)
  * rolling windows use sliding_window_view instead of one python append per end
  * the test split builds only the final window per engine (~248) instead of the
    ~50k windows of the whole test set that were immediately discarded
  * the frozen encoder runs under autocast on CUDA and features are stored as f16,
    halving both peak RAM and cache IO
  * feature tensors are pinned on the GPU when small enough (--gpu_budget_gb) and
    indexed with index_select, instead of copying a fresh numpy slice to the GPU
    every mini-batch (v6 moved ~7.7 GB per run that way on FD004)
  * the (window, window+1) pair index is built once with a single lexsort
  * E8 inference timing reuses the single evaluation pass

Route-A evaluation pass (V8), on top of the V7/V7K pass:
  * metric_dict reports the diagnostics behind an asymmetric-cost advantage:
    NASA per engine (the raw score scales with engine count), signed bias,
    late/early ratio, p90 and worst absolute error
  * every run writes per_engine_metrics.csv; FD002/FD004 additionally write
    per_condition_metrics.csv (test snapshots grouped by their os3 regime)
    and metrics.json gains cond_RMSE_gap / cond_RMSE_worst
  * compare_modes.csv ranks every metric inside the same protocol group and
    marks which metrics each configuration leads (best_on), optionally merging
    a baseline_csv of published numbers

Kaggle notebook variant (V7K), on top of the V7 pass:
  * ALL writable state is redirected with --cache_dir, because /kaggle/input is
    read-only: the prep cache and the Chronos feature cache used to be built
    under --data_dir and would fail with PermissionError on Kaggle
  * --data_dir is auto-detected (CMAPSS_DATA_DIR env, /kaggle/input, cwd)
  * main() takes an argv list and returns the run table; run_config(**kwargs)
    lets a notebook call one configuration per cell instead of shelling out,
    and the loaded Chronos pipeline is memoised across those calls
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    class _DummyBar:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self.iterable)

        def __len__(self):
            return len(self.iterable)

        def set_postfix(self, **kwargs):
            pass

        def write(self, *a, **k):
            print(*a)

    def tqdm(iterable=None, **kwargs):
        return _DummyBar(iterable, **kwargs)

    tqdm.write = lambda *a, **k: print(*a)

COLS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
FEATURES = [f"s{i}" for i in range(1, 22)] + ["os1", "os2", "os3"]
N_FEATURES = len(FEATURES)  # 24
POOLS = ["none", "token", "both"]


# --------------------------------------------------------------------------- #
# args
# --------------------------------------------------------------------------- #
def autodetect_data_dir():
    """Find a directory that holds train_FD001.txt (Kaggle: /kaggle/input)."""
    env = os.environ.get("CMAPSS_DATA_DIR")
    if env and Path(env).is_dir() and (Path(env) / "train_FD001.txt").exists():
        return env
    for root in ("/kaggle/input", str(Path.cwd())):
        rp = Path(root)
        if not rp.is_dir():
            continue
        for p in rp.rglob("train_FD001.txt"):
            return str(p.parent)
    return None


def args_parser(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None,
                   help="Directory with train_/test_/RUL_ *.txt. Only required for real runs. "
                        "If omitted it is auto-detected (CMAPSS_DATA_DIR, /kaggle/input, cwd).")
    p.add_argument("--cache_dir", default=None,
                   help="Writable directory for the prep cache and the Chronos feature cache. "
                        "Defaults to --data_dir. On Kaggle this must NOT be /kaggle/input "
                        "(read-only), so pass e.g. /kaggle/working/cache.")
    p.add_argument("--output_dir", default="chronos2_rul_revision")
    p.add_argument("--subsets", nargs="+", default=["FD001"],
                   choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    p.add_argument("--mode", choices=["full", "linear_probe", "lstm_backbone", "chronos_bilstm"],
                   default="full")
    p.add_argument("--ablation", default="full",
                   choices=["full", "no_temporal_adapter", "no_graph",
                            "no_fusion_gate", "no_monotonic", "no_smoothness",
                            "no_stage_weighting", "all"])
    p.add_argument("--cross_train", default=None,
                   choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--cross_test", default=None,
                   choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--data_fractions", nargs="+", type=float, default=None)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--rmax", type=float, default=125.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--embed_batch_size", type=int, default=16,
                   help="Number of windows per Chronos forward pass.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_mono", type=float, default=0.05)
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--graph_topk", type=int, default=6)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--chronos_model", default="amazon/chronos-2")
    p.add_argument("--context_length", type=int, default=30)
    p.add_argument("--force_reembed", action="store_true")
    p.add_argument("--cache_dtype", choices=["f16", "f32"], default="f16",
                   help="Storage dtype of the Chronos feature cache "
                        "(f16 halves RAM and disk; loaded and used as float32).")
    p.add_argument("--amp", choices=["auto", "on", "off"], default="auto",
                   help="autocast fp16 for the frozen encoder on CUDA (auto = on when cuda).")
    p.add_argument("--gpu_budget_gb", type=float, default=2.0,
                   help="Feature tensors larger than this stay in host memory and are "
                        "copied per mini-batch instead of being pinned on the GPU.")
    p.add_argument("--no_prep_cache", action="store_true",
                   help="Disable the on-disk window/graph cache (slow, for debugging).")
    p.add_argument("--split_seed", type=int, default=42,
                   help="Fixed engine split for comparable/reusable multi-seed runs.")
    p.add_argument("--torch_threads", type=int, default=0,
                   help="CPU intra-op threads; 0 = leave PyTorch default (all cores).")
    p.add_argument("--cond_norm", action="store_true",
                   help="Per-operating-condition (os3) standardisation for FD002/FD004.")
    p.add_argument("--embed_pool", choices=POOLS, default="none",
                   help="Chronos token pooling: none=B,S,T,D (token-preserving), "
                        "token=B,S,D (memory saver), both=B,D (linear probe only).")
    p.add_argument("--resume", action="store_true",
                   help="Skip runs whose metrics.json already exists.")
    p.add_argument("--merge_from", nargs="+", default=None,
                   help="Merge all_runs.csv from the given dirs into output_dir and exit.")
    p.add_argument("--quick_test", action="store_true",
                   help="Fast smoke test: FD001, one seed, 10%% data, 5 epochs.")
    p.add_argument("--quick_hybrid", action="store_true",
                   help="Quick Chronos-2+BiLSTM test: FD001, one seed, full data, 8 epochs.")
    p.add_argument("--no_report_cond", action="store_true",
                   help="Skip the per-operating-condition breakdown (FD002/FD004). "
                        "On by default; it writes per_condition_metrics.csv.")
    p.add_argument("--baseline_csv", default=None,
                   help="Optional CSV of prior baselines (columns: subset, model, RMSE, "
                        "MAE, NASA_Score, ...) merged into compare_modes.csv, so the "
                        "new numbers sit next to the published ones in one table.")
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args(argv)

    if args.quick_hybrid:
        args.subsets = ["FD001"]
        args.seeds = [42]
        args.mode = "chronos_bilstm"
        args.ablation = "full"
        args.data_fractions = [1.0]
        args.epochs = 8
        args.patience = 3
        args.embed_batch_size = max(args.embed_batch_size, 32)
        args.batch_size = max(args.batch_size, 256)
    if args.quick_test:
        args.subsets = ["FD001"]
        args.seeds = [42]
        args.mode = "full"
        args.ablation = "full"
        args.data_fractions = [0.1]
        args.epochs = 5
        args.patience = 2
        args.embed_batch_size = max(args.embed_batch_size, 32)
        args.batch_size = max(args.batch_size, 256)

    if args.hidden % 2 != 0:
        p.error("--hidden must be even (bidirectional BiLSTM splits it in half).")
    if args.hidden % args.heads != 0:
        p.error("--hidden must be divisible by --heads.")
    if args.embed_pool == "both" and args.mode != "linear_probe":
        p.error("--embed_pool both collapses the sensor axis and only works with "
                "--mode linear_probe. Use none or token otherwise.")
    if bool(args.cross_train) != bool(args.cross_test):
        p.error("--cross_train and --cross_test must be given together.")
    if not args.merge_from:
        if not args.data_dir:
            auto = autodetect_data_dir()
            if auto is None:
                p.error("--data_dir is required (unless --merge_from is used). "
                        "Set CMAPSS_DATA_DIR or add a C-MAPSS dataset under /kaggle/input.")
            args.data_dir = auto
            print(f"[data] auto-detected --data_dir={auto}")
        missing = []
        for s in set(args.subsets + [x for x in (args.cross_train, args.cross_test) if x]):
            for name in (f"train_{s}.txt", f"test_{s}.txt", f"RUL_{s}.txt"):
                if not (Path(args.data_dir) / name).exists():
                    missing.append(name)
        if missing:
            p.error("missing data files in %s: %s" % (args.data_dir, ", ".join(sorted(missing))))
    if args.context_length != args.window:
        print(f"[warn] --context_length={args.context_length} != --window={args.window}: "
              f"Chronos will be told {args.context_length} steps about a window that "
              f"actually holds {args.window}. Keep them equal unless you know why.")
    return args


def seed_all(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
_DATA_CACHE = {}


def _read_whitespace(path, names):
    """C-speed whitespace parse.

    pd.read_csv(..., sep=r"\\s+") looks tidy but the C engine rejects a regex
    separator, so pandas falls back to the python engine with a ParserWarning
    and runs several times slower. np.loadtxt is the same parse at C speed.
    """
    arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if arr.shape[1] != len(names):
        raise ValueError(f"{Path(path).name}: expected {len(names)} columns, "
                         f"got {arr.shape[1]}")
    df = pd.DataFrame(arr, columns=names)
    # RUL_*.txt has a single "RUL" column: only cast the columns that exist
    for c in ("unit", "cycle"):
        if c in df.columns:
            df[c] = df[c].astype(np.int64)
    float_cols = [c for c in names if c not in ("unit", "cycle")]
    if float_cols:
        df[float_cols] = df[float_cols].astype(np.float32)
    return df


def load_cmapss(root, subset):
    key = (str(Path(root).resolve()), subset)
    hit = _DATA_CACHE.get(key)
    if hit is not None:
        return hit
    root = Path(root)
    tr = _read_whitespace(root / f"train_{subset}.txt", COLS)
    te = _read_whitespace(root / f"test_{subset}.txt", COLS)
    rul = _read_whitespace(root / f"RUL_{subset}.txt", ["RUL"])
    _DATA_CACHE[key] = (tr, te, rul)
    return tr, te, rul


def attach_train_rul(df, cap):
    z = df.copy()
    last = z.groupby("unit")["cycle"].transform("max")
    z["RUL"] = (last - z["cycle"]).clip(upper=cap)
    return z


def attach_test_rul(df, rul, cap):
    z = df.copy().reset_index(drop=True)
    ids = np.sort(z.unit.unique())
    if len(ids) != len(rul):
        raise ValueError(f"Test engine count ({len(ids)}) != RUL label count ({len(rul)}).")
    rmap = {int(i): float(v) for i, v in zip(ids, rul.RUL.to_numpy())}
    last = z.groupby("unit")["cycle"].transform("max")
    z["RUL"] = (z.unit.map(rmap) + last - z.cycle).clip(upper=cap)
    return z


def make_windows(df, window):
    """All rolling windows per engine.

    sliding_window_view builds them as a view per engine and we concatenate
    once: same order as the v5 append loop, without 54k python iterations.
    """
    xs, ys, us, cs = [], [], [], []
    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        if len(g) < window:
            continue
        x = np.ascontiguousarray(g[FEATURES].to_numpy(np.float32))
        nw = len(g) - window + 1
        # (L-w+1, 24, w) -> (L-w+1, w, 24)
        xs.append(sliding_window_view(x, window, axis=0).transpose(0, 2, 1))
        ys.append(g["RUL"].to_numpy(np.float32)[window - 1:])
        cs.append(g["cycle"].to_numpy(np.int64)[window - 1:])
        us.append(np.full(nw, int(unit), dtype=np.int64))
    if not xs:
        return (np.zeros((0, window, N_FEATURES), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.int64), np.zeros(0, np.int64))
    return (np.ascontiguousarray(np.concatenate(xs, axis=0), dtype=np.float32),
            np.ascontiguousarray(np.concatenate(ys), dtype=np.float32),
            np.concatenate(us).astype(np.int64), np.concatenate(cs).astype(np.int64))


def make_last_windows(df, window):
    """Only the final window of each engine - the official C-MAPSS test target.

    v5/v6 built every test window (~50k on FD004) and then kept one per engine.

    Returns (X, y, unit, cycle, cond). `cond` is the os3 value of the final row,
    i.e. the operating regime the prediction is actually made in: on FD002/FD004
    this is what the per-condition breakdown is built on.
    """
    xs, ys, us, cs, os3 = [], [], [], [], []
    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        if len(g) < window:
            continue
        x = g[FEATURES].to_numpy(np.float32)
        xs.append(x[-window:])
        ys.append(np.float32(g["RUL"].to_numpy(np.float32)[-1]))
        cs.append(np.int64(g["cycle"].to_numpy(np.int64)[-1]))
        us.append(np.int64(int(unit)))
        os3.append(np.float32(g["os3"].to_numpy(np.float32)[-1]))
    if not xs:
        return (np.zeros((0, window, N_FEATURES), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))
    return (np.ascontiguousarray(np.stack(xs, axis=0), dtype=np.float32),
            np.asarray(ys, np.float32), np.asarray(us, np.int64),
            np.asarray(cs, np.int64), np.asarray(os3, np.float32))


def make_pseudotest(df, window, cap, fractions=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """Simulated in-service snapshots from validation engines (last cycle = failure)."""
    xs, ys, ids, cycles = [], [], [], []
    sid = 0
    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        if len(g) < window:      # v5 used <=, which silently dropped len == window engines
            continue
        x = g[FEATURES].to_numpy(np.float32)
        c = g["cycle"].to_numpy(np.int64)
        failure_cycle = int(c[-1])
        used = set()
        for frac in fractions:
            end = int(round((len(g) - 1) * frac))
            end = max(window - 1, min(end, len(g) - 1))
            if end in used:
                continue
            used.add(end)
            xs.append(x[end - window + 1:end + 1])
            ys.append(float(np.clip(failure_cycle - int(c[end]), 0, cap)))
            ids.append(sid)
            cycles.append(int(c[end]))
            sid += 1
    return (np.asarray(xs, np.float32), np.asarray(ys, np.float32),
            np.asarray(ids, np.int64), np.asarray(cycles, np.int64))


def build_next_index(units, cycles):
    """next_idx[i] = j when j is i's successor (same engine, cycle+1), else -1.

    One lexsort + two vector comparisons replace the per-engine python loop
    that v5 ran inside ArrayDataset.__init__ on every run.
    """
    units = np.asarray(units, np.int64)
    cycles = np.asarray(cycles, np.int64)
    n = len(cycles)
    nxt = np.full(n, -1, np.int64)
    if n < 2:
        return nxt
    order = np.lexsort((cycles, units))
    su, sc = units[order], cycles[order]
    link = (su[:-1] == su[1:]) & (sc[1:] == sc[:-1] + 1)
    src = order[:-1][link]
    dst = order[1:][link]
    nxt[src] = dst
    return nxt


def metric_dict(y, p):
    """Point metrics plus the diagnostics needed to argue about *where* the gain is.

    The extra fields are not decoration; each one backs a claim a reviewer can
    (and did) ask for:
      mean_error  - signed bias. Positive = over-estimates RUL = predicts late =
                    the dangerous direction on a real engine (you keep running it).
      late_ratio  - share of engines predicted later than the truth. NASA score
                    weights late predictions by exp(e/10) vs exp(-e/13), so this
                    ratio is the mechanism behind any NASA-score advantage.
      NASA_per_engine - the raw NASA score is a sum, so it scales with engine
                    count; only the per-engine value is comparable across subsets
                    (FD001 has 100 engines, FD002 has 259).
      worst_abs_error / p90_abs_error - tail behaviour, where a "good average"
                    model can still be unacceptable in service.
    """
    e = np.asarray(p, np.float64) - np.asarray(y, np.float64)
    score = np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1).sum()
    ss = np.sum((y - np.mean(y)) ** 2)
    ae = np.abs(e)
    n = max(len(e), 1)
    late = e > 0
    return {
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAE": float(np.mean(ae)),
        "NASA_Score": float(score),
        "NASA_per_engine": float(score / n),
        "R2": float(1 - np.sum(e ** 2) / ss) if ss > 0 else float("nan"),
        "mean_error": float(np.mean(e)),
        "late_ratio": float(late.mean()),
        "early_ratio": float((e < 0).mean()),
        "p90_abs_error": float(np.percentile(ae, 90)) if len(ae) else float("nan"),
        "worst_abs_error": float(ae.max()) if len(ae) else float("nan"),
        "n_eval": int(n),
    }


def per_engine_frame(units, cycles, y, p):
    """Engine-level breakdown: who is being predicted badly, not just how many."""
    df = pd.DataFrame({"unit": np.asarray(units).astype(int),
                       "cycle": np.asarray(cycles).astype(int),
                       "true_RUL": np.asarray(y, np.float64),
                       "pred_RUL": np.asarray(p, np.float64)})
    df["error"] = df.pred_RUL - df.true_RUL
    df["abs_error"] = df.error.abs()
    df["nasa"] = np.where(df.error < 0, np.exp(-df.error / 13) - 1,
                          np.exp(df.error / 10) - 1)
    return df


def per_condition_metrics(y, p, cond, tol=0.5):
    """Break the test set down by operating regime (os3 at the predicted snapshot).

    On FD001/FD003 os3 is constant, so this collapses to one row and says nothing;
    on FD002/FD004 it shows which regimes the model handles and which it does not,
    which is the concrete evidence for a cross-condition claim.
    """
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    cond = np.asarray(cond, np.float64)
    if len(cond) != len(y):
        return None
    keys = np.round(cond / tol) * tol
    rows = []
    for key in sorted(set(keys.tolist())):
        m = keys == key
        if m.sum() < 3:
            continue
        met = metric_dict(y[m], p[m])
        rows.append({"os3": float(key), "n": int(m.sum()),
                     "RMSE": met["RMSE"], "MAE": met["MAE"],
                     "NASA_per_engine": met["NASA_per_engine"],
                     "mean_error": met["mean_error"], "late_ratio": met["late_ratio"]})
    if len(rows) < 2:
        return None
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def summarize_conditions(cond_df):
    if cond_df is None or len(cond_df) < 2:
        return {}
    return {
        "n_conditions": int(len(cond_df)),
        "cond_RMSE_best": float(cond_df.RMSE.min()),
        "cond_RMSE_worst": float(cond_df.RMSE.max()),
        "cond_RMSE_gap": float(cond_df.RMSE.max() - cond_df.RMSE.min()),
        "cond_NASA_worst": float(cond_df.NASA_per_engine.max()),
    }


# --------------------------------------------------------------------------- #
# scaling
# --------------------------------------------------------------------------- #
def probe_input(Z):
    """Collapse whatever is left of the (sensor, token) axes for the linear probe."""
    # dtype=float32 matters once caches are stored as f16: numpy would otherwise
    # accumulate 24 * T terms in half precision.
    if Z.ndim == 4:
        return Z.mean(axis=(1, 2), dtype=np.float32)
    if Z.ndim == 3:
        return Z.mean(axis=1, dtype=np.float32)
    return np.asarray(Z, dtype=np.float32)


class FeatureScaler:
    """Train-split-only standardisation; optional per-operating-condition stats."""

    def __init__(self, train_df, cond_norm=False, clip=10.0):
        self.cond_norm = bool(cond_norm) and train_df["os3"].nunique() > 1
        self.clip = clip
        g_mu = train_df[FEATURES].mean().astype(np.float64)
        g_sd = train_df[FEATURES].std().replace(0, 1.0).fillna(1.0).astype(np.float64)
        g_sd[g_sd < 1e-8] = 1.0
        self.g_mu, self.g_sd = g_mu, g_sd
        if self.cond_norm:
            grp = train_df.groupby("os3")
            mu = grp[FEATURES].mean()
            sd = grp[FEATURES].std().replace(0, 1.0)
            sd = sd.fillna(1.0)
            sd[sd < 1e-8] = 1.0
            # conditions with too few rows give unstable stats -> fall back to global
            counts = grp.size()
            good = counts[counts >= 50].index
            self.mu = mu.loc[mu.index.isin(good)].astype(np.float64)
            self.sd = sd.loc[sd.index.isin(good)].astype(np.float64)
            if len(self.mu) == 0:
                self.cond_norm = False
            else:
                # os3 identifies the operating condition itself: keeping it raw
                # per condition would collapse it to a constant 0 and destroy
                # the only explicit regime signal. Scale it globally instead.
                self.mu["os3"] = self.g_mu["os3"]
                self.sd["os3"] = self.g_sd["os3"]
        else:
            self.mu, self.sd = None, None

    def transform(self, frame):
        z = frame.copy()
        if self.cond_norm:
            mu = self.mu.reindex(frame["os3"].to_numpy()).fillna(self.g_mu)
            sd = self.sd.reindex(frame["os3"].to_numpy()).fillna(self.g_sd)
            mu_v, sd_v = mu.to_numpy(np.float64), sd.to_numpy(np.float64)
        else:
            mu_v = np.broadcast_to(self.g_mu.to_numpy(np.float64), (len(frame), N_FEATURES))
            sd_v = np.broadcast_to(self.g_sd.to_numpy(np.float64), (len(frame), N_FEATURES))
        vals = (frame[FEATURES].to_numpy(np.float64) - mu_v) / sd_v
        if self.clip is not None:
            vals = np.clip(vals, -self.clip, self.clip)
        z[FEATURES] = vals.astype(np.float32)
        return z


def normalize_frames(train_df, other_frames, cond_norm=False):
    scaler = FeatureScaler(train_df, cond_norm=cond_norm)
    out = [scaler.transform(train_df)]
    out += [scaler.transform(f) for f in other_frames]
    return out


# --------------------------------------------------------------------------- #
# Chronos-2 embeddings
# --------------------------------------------------------------------------- #
_PIPE_CACHE = {}


def load_chronos(model_id, device):
    """Load the frozen pipeline once per (model, device) and reuse it.

    A notebook calls main() several times (one per subset / mode); reloading a
    120M checkpoint on every call would waste minutes and fragment the GPU cache.
    """
    key = (model_id, str(device))
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]
    try:
        from chronos import Chronos2Pipeline
    except Exception as exc:
        raise ImportError(
            "Chronos-2 package missing. Install with: "
            'pip install "chronos-forecasting>=2.0"'
        ) from exc
    pipe = Chronos2Pipeline.from_pretrained(model_id, device_map=str(device))
    pipe.model.eval()
    for p in pipe.model.parameters():
        p.requires_grad_(False)
    _PIPE_CACHE[key] = pipe
    return pipe


def _expected_ndim(pool):
    return {"none": 4, "token": 3, "both": 2}[pool]


def _try_load_cache(cache_file, n_expected, pool, force):
    """Return a validated feature array or None (=> rebuild)."""
    if force or cache_file is None or not Path(cache_file).exists():
        return None
    try:
        with np.load(cache_file) as z:
            if "features" not in z:
                return None
            feats = z["features"]
    except Exception as exc:
        print(f"[cache] unreadable {cache_file} ({exc}); rebuilding")
        return None

    ndim = _expected_ndim(pool)
    if pool == "both":                       # (N, d_model): sensor axis already collapsed
        ok = feats.ndim == 2 and feats.shape[0] == n_expected
    else:
        ok = (feats.ndim == ndim and feats.shape[0] == n_expected
              and feats.shape[1] == N_FEATURES)
    if not ok or feats.dtype not in (np.float16, np.float32):
        print(f"[cache] shape mismatch for {Path(cache_file).name}: "
              f"got {feats.shape}/{feats.dtype}, expected ({n_expected}, {N_FEATURES}, ...) "
              f"as float16/float32; rebuilding")
        return None
    print(f"[cache] hit {Path(cache_file).name} {feats.shape} {feats.dtype} "
          f"({feats.nbytes / 1e9:.2f} GB)")
    return feats


def _pool_batch(batch_array, pool):
    if pool == "token":
        return batch_array.mean(axis=2)          # (B,S,D)
    if pool == "both":
        return batch_array.mean(axis=(2, 1))     # (B,D)
    return batch_array                            # (B,S,T,D)


@torch.no_grad()
def chronos_embed_windows(pipe, X, batch_size, context_length, cache_file=None,
                          force=False, pool="none", cache_dtype="f16", amp=False):
    """
    X: [N, window, 24] -> Chronos-2 encoder features.

    Chronos2Pipeline.embed(inputs, batch_size=..., context_length=...) returns
    (list_of_embeddings, list_of_loc_scale), one tensor (n_variates, num_patches+2, d_model)
    per input series. We stack them to [N, 24, T, D] (or pooled forms).
    """
    n = len(X)
    store_dtype = np.float16 if cache_dtype == "f16" else np.float32
    cached = _try_load_cache(cache_file, n, pool, force)
    if cached is not None:
        return cached

    feats = []
    total_batches = (n + batch_size - 1) // batch_size
    progress = tqdm(range(0, n, batch_size), total=total_batches,
                    desc=f"Chronos-2 embedding ({Path(cache_file).stem if cache_file else 'features'})",
                    unit="batch", dynamic_ncols=True)
    t_start = time.perf_counter()
    for bi, start in enumerate(progress):
        xb = X[start:start + batch_size]
        # (window, 24) -> (24, window) = (n_variates, history)
        inputs = [torch.as_tensor(np.ascontiguousarray(a.T), dtype=torch.float32) for a in xb]
        if amp:
            # The frozen encoder is where all the FLOPs live; autocast roughly
            # halves its cost and its activation footprint, with no effect on
            # the trained modules (they stay fp32).
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                res = pipe.embed(inputs, batch_size=max(batch_size * N_FEATURES, 1),
                                 context_length=context_length)
        else:
            res = pipe.embed(inputs, batch_size=max(batch_size * N_FEATURES, 1),
                             context_length=context_length)
        embeddings = res[0] if isinstance(res, (tuple, list)) and len(res) == 2 else res
        if isinstance(embeddings, torch.Tensor):
            embeddings = list(embeddings)

        if len(embeddings) != len(inputs):
            raise RuntimeError(
                f"Chronos embed returned {len(embeddings)} items for {len(inputs)} windows. "
                "The pipeline API changed; check chronos-forecasting version."
            )

        batch_feats = []
        for emb in embeddings:
            e = torch.as_tensor(emb).detach()
            if e.ndim == 2:
                if e.shape[0] == N_FEATURES:
                    e = e.unsqueeze(1)                       # (S,1,D)
                elif e.shape[1] == N_FEATURES:
                    e = e.T.unsqueeze(1)                     # (S,1,D)
                else:
                    raise RuntimeError(f"Unexpected Chronos embedding shape {tuple(e.shape)}")
            elif e.ndim == 3:
                if e.shape[0] == N_FEATURES:
                    pass
                elif e.shape[1] == N_FEATURES:
                    e = e.transpose(0, 1)
                else:
                    raise RuntimeError(
                        f"Cannot identify the {N_FEATURES}-variable axis in {tuple(e.shape)}")
            else:
                raise RuntimeError(f"Unexpected Chronos embedding rank {e.ndim}")
            batch_feats.append(e.contiguous())

        shapes = {tuple(f.shape[1:]) for f in batch_feats}
        if len(shapes) != 1:
            raise RuntimeError(f"Inconsistent per-window embedding shapes inside one batch: {shapes}")

        batch_array = torch.stack(batch_feats, dim=0).to(device="cpu", dtype=torch.float32).numpy()
        # Cast per batch, so the array is never held as float32 and float16 at
        # the same time (at 15 GB that difference is the whole point).
        feats.append(np.ascontiguousarray(_pool_batch(batch_array, pool), dtype=store_dtype))

    out = np.concatenate(feats, axis=0)
    dt = time.perf_counter() - t_start
    gb = out.nbytes / 1e9
    print(f"[embed] shape={out.shape} {out.dtype} {gb:.2f} GB  {dt:.1f}s "
          f"({1000 * dt / max(n, 1):.1f} ms/window)")
    if gb > 6.0:
        print(f"[embed] WARNING: {gb:.1f} GB of features are held in RAM and on disk. "
              f"Use --embed_pool token to cut this by ~T x, or --cache_dtype f16 (current).")
    if cache_file is not None:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        # atomic: an interrupt must not leave a truncated .npz behind
        tmp = Path(str(cache_file) + f".tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            np.savez(f, features=out)
        os.replace(tmp, cache_file)
    return out


# --------------------------------------------------------------------------- #
# graph + models
# --------------------------------------------------------------------------- #
def build_graph(X_train, topk):
    """Train-split-only absolute Pearson correlation graph with self loops."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    std = flat.std(axis=0)
    valid = std > 1e-12
    corr = np.zeros((flat.shape[1], flat.shape[1]), dtype=np.float32)
    if valid.sum() > 1:
        xv = flat[:, valid].astype(np.float64)
        xv = xv - xv.mean(axis=0, keepdims=True)
        denom = xv.std(axis=0, keepdims=True)
        denom[denom < 1e-12] = 1.0
        xv = xv / denom
        corr_valid = (xv.T @ xv) / max(len(xv), 1)
        corr_valid = np.nan_to_num(np.abs(corr_valid), nan=0.0, posinf=0.0, neginf=0.0)
        corr[np.ix_(valid, valid)] = corr_valid.astype(np.float32)
    np.fill_diagonal(corr, 1.0)
    n = corr.shape[0]
    adj = np.eye(n, dtype=np.float32)
    k = min(max(1, topk), n - 1)
    for i in range(n):
        order = np.argsort(-corr[i])          # descending, deterministic
        ids = [j for j in order[: k + 1] if j != i][:k]
        adj[i, ids] = 1.0
    adj = np.maximum(adj, adj.T)
    return torch.tensor(adj, dtype=torch.bool)


class DenseGraphAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden dimension must be divisible by attention heads")
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, adj):
        b, n, d = x.shape
        q = self.q(x).view(b, n, self.heads, self.dh).transpose(1, 2)
        k = self.k(x).view(b, n, self.heads, self.dh).transpose(1, 2)
        v = self.v(x).view(b, n, self.heads, self.dh).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) / (self.dh ** 0.5)
        logits = logits.masked_fill(~adj[None, None, :, :], -1e4)
        a = self.drop(torch.softmax(logits, dim=-1))
        h = (a @ v).transpose(1, 2).contiguous().view(b, n, d)
        return self.norm(x + self.out(h))


class ChronosRULAdapter(nn.Module):
    """Token-preserving temporal adapter followed by sensor-relation learning."""

    def __init__(self, d_in, hidden=128, heads=4, dropout=0.1,
                 use_temporal=True, use_graph=True, use_fusion=True):
        super().__init__()
        self.temporal_adapter = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.LayerNorm(hidden), nn.Dropout(dropout)
        )
        self.use_temporal = use_temporal
        self.temporal_score = (
            nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(),
                          nn.Linear(hidden // 2, 1, bias=False))
            if use_temporal else None
        )
        self.graph = DenseGraphAttention(hidden, heads, dropout) if use_graph else None
        self.fusion_gate = (
            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(),
                          nn.Linear(hidden, hidden), nn.Sigmoid())
            if use_fusion and use_graph else None
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, z, adj):
        if z.ndim == 3:                     # (B,S,D) -> single "token"
            z = z.unsqueeze(2)
        if z.ndim != 4:
            raise ValueError(f"Expected [B,S,T,D] Chronos features, got {tuple(z.shape)}")
        h = self.temporal_adapter(z)
        if self.temporal_score is not None:
            scores = self.temporal_score(h).squeeze(-1)
            weights = torch.softmax(scores, dim=2).unsqueeze(-1)
            temporal = (weights * h).sum(dim=2)
        else:
            temporal = h.mean(dim=2)

        if self.graph is not None:
            graph = self.graph(temporal, adj)
            if self.fusion_gate is not None:
                g = self.fusion_gate(torch.cat([temporal, graph], dim=-1))
                fused = g * temporal + (1 - g) * graph
            else:
                fused = temporal + graph
        else:
            fused = temporal
        rep = fused.mean(dim=1)
        return self.regressor(rep).squeeze(-1)


class ChronosBiLSTMHybrid(nn.Module):
    """Frozen Chronos token features + trainable multivariate BiLSTM fusion."""

    def __init__(self, d_in, n_features=N_FEATURES, hidden=128, heads=4, dropout=0.1):
        super().__init__()
        self.chronos_proj = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout)
        )
        self.token_score = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1, bias=False)
        )
        self.graph = DenseGraphAttention(hidden, heads, dropout)
        self.bilstm = nn.LSTM(input_size=n_features, hidden_size=hidden // 2,
                              num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_norm = nn.LayerNorm(hidden)
        self.lstm_score = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1, bias=False)
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.Sigmoid()
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, z, adj, raw_x):
        if z.ndim == 3:                     # (B,S,D) from --embed_pool token
            z = z.unsqueeze(2)
        if z.ndim != 4:
            raise ValueError(f"Expected [B,S,T,D] Chronos features, got {tuple(z.shape)}")
        h = self.chronos_proj(z)
        tw = torch.softmax(self.token_score(h).squeeze(-1), dim=2).unsqueeze(-1)
        sensor = (tw * h).sum(dim=2)
        sensor = self.graph(sensor, adj)
        chronos_vec = sensor.mean(dim=1)

        lh, _ = self.bilstm(raw_x)
        lh = self.lstm_norm(lh)
        aw = torch.softmax(self.lstm_score(lh).squeeze(-1), dim=1).unsqueeze(-1)
        lstm_vec = (aw * lh).sum(dim=1)

        gate = self.fusion_gate(torch.cat([chronos_vec, lstm_vec], dim=-1))
        fused = gate * chronos_vec + (1.0 - gate) * lstm_vec
        return self.regressor(fused).squeeze(-1)


class LSTMFeatureBackbone(nn.Module):
    """Trainable per-sensor LSTM replacement (vectorised, same maths as the loop)."""

    def __init__(self, n_features=N_FEATURES, hidden=128, dropout=0.1):
        super().__init__()
        if hidden % 2:
            raise ValueError("hidden must be even for the bidirectional backbone")
        self.n_features = n_features
        self.lstm = nn.LSTM(1, hidden // 2, batch_first=True,
                            num_layers=2, bidirectional=True, dropout=dropout)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        b, t, f = x.shape
        if f != self.n_features:
            raise ValueError(f"expected {self.n_features} sensor channels, got {f}")
        # (B,T,F) -> (B*F,T,1): one batched call instead of a 24-iteration python loop.
        seq = x.permute(0, 2, 1).reshape(b * f, t, 1)
        h, _ = self.lstm(seq)
        node = self.proj(h.mean(dim=1))
        return node.view(b, f, -1)


class RULModelLSTM(nn.Module):
    def __init__(self, hidden, heads, dropout, use_graph=True, use_fusion=True):
        super().__init__()
        self.backbone = LSTMFeatureBackbone(hidden=hidden, dropout=dropout)
        self.adapter = ChronosRULAdapter(
            d_in=hidden, hidden=hidden, heads=heads, dropout=dropout,
            use_temporal=False, use_graph=use_graph, use_fusion=use_fusion
        )

    def forward(self, x, adj):
        return self.adapter(self.backbone(x), adj)


# --------------------------------------------------------------------------- #
# training helpers
# --------------------------------------------------------------------------- #
def select_engine_split(df, seed, val_ratio=0.2):
    engines = np.sort(df.unit.unique())
    tr_ids, va_ids = train_test_split(engines, test_size=val_ratio, random_state=seed)
    return df[df.unit.isin(tr_ids)].copy(), df[df.unit.isin(va_ids)].copy()


def make_cache_path(root, subset, split, window, seed, tag=""):
    return Path(root) / f"chronos2_cache_{subset}_{split}_w{window}_s{seed}{tag}.npz"


def prepare_features(pipe, X, cache, args):
    if len(X) == 0:
        raise RuntimeError(f"no feature windows to embed for {cache}")
    return chronos_embed_windows(
        pipe, X, args.embed_batch_size, args.context_length,
        cache_file=cache, force=args.force_reembed, pool=args.embed_pool,
        cache_dtype=args.cache_dtype, amp=bool(getattr(args, "amp_resolved", False))
    )


# --------------------------------------------------------------------------- #
# device feeds
# --------------------------------------------------------------------------- #
def _host_tensor(arr):
    """Zero-copy host tensor for a contiguous numpy array."""
    return torch.from_numpy(np.ascontiguousarray(arr))


def _place(t, device, budget_bytes):
    """Pin the tensor on the GPU only when it fits the budget."""
    if t is None:
        return None
    if device.type == "cuda" and t.numel() * t.element_size() <= budget_bytes:
        return t.to(device)
    return t


def _fetch(t, ix, device, dtype=torch.float32):
    """Batch fetch by index array or slice; converts to float32 for the model."""
    if t is None:
        return None
    if isinstance(ix, slice):
        out = t[ix]
    else:
        out = t.index_select(0, torch.as_tensor(ix, dtype=torch.long, device=t.device))
    return out.to(device=device, dtype=dtype)


def batch_pairs(next_idx, ids, device):
    ids = np.asarray(ids, dtype=np.int64)
    partner = next_idx[ids]
    valid = partner >= 0
    ai, bi = ids[valid], partner[valid]
    all_ids = np.unique(np.concatenate((ids, bi)))
    pi_np = np.searchsorted(all_ids, ai)
    pj_np = np.searchsorted(all_ids, bi)
    pi = torch.as_tensor(pi_np, dtype=torch.long, device=device)
    pj = torch.as_tensor(pj_np, dtype=torch.long, device=device)
    return all_ids, pi, pj


def evaluate_features(model, X, y, adj, device, batch_size, raw_x=None):
    model.eval()
    if len(y) == 0:
        raise RuntimeError("empty evaluation set")
    # One host tensor per input, sliced per batch: no numpy re-wrap per call.
    X_t = None if X is None else _host_tensor(X)
    raw_t = None if raw_x is None else _host_tensor(raw_x)
    pred = []
    with torch.no_grad():
        for st in range(0, len(y), batch_size):
            sl = slice(st, st + batch_size)
            if isinstance(model, ChronosBiLSTMHybrid):
                out = model(_fetch(X_t, sl, device), adj, _fetch(raw_t, sl, device))
            elif raw_t is not None:
                out = model(_fetch(raw_t, sl, device), adj)
            else:
                out = model(_fetch(X_t, sl, device), adj)
            pred.append(out.detach().cpu().numpy())
    return np.concatenate(pred)


def train_model(model, Xtr, ytr, utr, ctr, Xva, yva, adj, args,
                seed, use_mono=True, use_smooth=True, use_stage=True, data_tag="",
                raw_train=None, raw_val=None):
    device = next(model.parameters()).device
    nxt = build_next_index(utr, ctr)
    budget = args.gpu_budget_gb * 1e9
    # One host tensor per input, optionally pinned on the GPU. v5 did
    # torch.from_numpy(Xtr[ix]) per mini-batch, i.e. a fresh copy every step.
    Xtr_t = _place(_host_tensor(Xtr), device, budget)
    Xva_t = _place(_host_tensor(Xva), device, budget)
    raw_tr_t = _place(_host_tensor(raw_train), device, budget) if raw_train is not None else None
    raw_va_t = _place(_host_tensor(raw_val), device, budget) if raw_val is not None else None
    ytr_t = torch.as_tensor(np.asarray(ytr, np.float32), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay
    )
    best, best_epoch, bad = float("inf"), 0, 0
    best_state = None
    history = []
    start_time = time.perf_counter()

    stage_edges = torch.tensor([args.rmax / 3.0, 2.0 * args.rmax / 3.0], device=device)
    train_stage_id = torch.bucketize(ytr_t, stage_edges)
    stage_counts = torch.bincount(train_stage_id, minlength=3).float().clamp_min(1.0)
    fixed_stage_weights = (stage_counts.sum() / (3.0 * stage_counts)).sqrt()
    fixed_stage_weights = fixed_stage_weights / fixed_stage_weights.mean()

    epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"Training {data_tag}",
                     unit="epoch", dynamic_ncols=True)
    for ep in epoch_bar:
        model.train()
        order = np.random.permutation(len(ytr))
        stat_sums = torch.zeros(4, dtype=torch.float64, device=device)
        stat_batches = 0
        for st in range(0, len(order), args.batch_size):
            anchor = order[st:st + args.batch_size]
            ix, pi, pj = batch_pairs(nxt, anchor, device)
            yb = ytr_t.index_select(0, torch.as_tensor(ix, dtype=torch.long, device=device))
            opt.zero_grad(set_to_none=True)
            if isinstance(model, ChronosBiLSTMHybrid):
                pred = model(_fetch(Xtr_t, ix, device), adj, _fetch(raw_tr_t, ix, device))
            else:
                pred = model(_fetch(Xtr_t, ix, device), adj)
            per_sample = (pred - yb).pow(2)
            if use_stage:
                stage_id = torch.bucketize(yb.detach(), stage_edges)
                mse = (per_sample * fixed_stage_weights[stage_id]).mean()
            else:
                mse = per_sample.mean()
            mono = (F.relu(pred[pj] - pred[pi]).mean()
                    if (use_mono and len(pi)) else pred.new_tensor(0.))
            if use_smooth and len(pi):
                uncapped = yb[pi] < (args.rmax - 1e-5)
                smooth = (((pred[pj[uncapped]] - pred[pi[uncapped]]) + 1.0).pow(2).mean()
                          if uncapped.any() else pred.new_tensor(0.))
            else:
                smooth = pred.new_tensor(0.)
            loss = mse + args.lambda_mono * mono + args.lambda_smooth * smooth
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            stat_sums += torch.stack([
                loss.detach().double(), mse.detach().double(),
                mono.detach().double(), smooth.detach().double()
            ])
            stat_batches += 1

        epoch_stats = (stat_sums / max(stat_batches, 1)).cpu().numpy()
        mean_loss, mean_mse, mean_mono, mean_smooth = map(float, epoch_stats)

        model.eval()
        vp = []
        with torch.no_grad():
            for st in range(0, len(yva), args.batch_size):
                sl = slice(st, st + args.batch_size)
                if isinstance(model, ChronosBiLSTMHybrid):
                    vp.append(model(_fetch(Xva_t, sl, device), adj,
                                    _fetch(raw_va_t, sl, device)).detach().cpu().numpy())
                else:
                    vp.append(model(_fetch(Xva_t, sl, device), adj).detach().cpu().numpy())
        vm = metric_dict(yva, np.concatenate(vp))
        history.append({
            "epoch": ep, "train_loss": mean_loss, "train_mse": mean_mse,
            "mono_raw": mean_mono, "smooth_raw": mean_smooth,
            "val_pseudotest_RMSE": vm["RMSE"], "val_pseudotest_MAE": vm["MAE"],
            "val_pseudotest_NASA": vm["NASA_Score"],
        })
        epoch_bar.set_postfix(
            train_mse=f"{mean_mse:.3f}",
            val_rmse=f"{vm['RMSE']:.3f}",
            best=f"{best:.3f}" if np.isfinite(best) else "inf"
        )
        if ep == 1 or ep % 5 == 0:
            tqdm.write(f"{data_tag} epoch {ep:03d} | train MSE {mean_mse:.3f} "
                       f"| val pseudo-test RMSE {vm['RMSE']:.3f}")
        if vm["RMSE"] < best - 1e-4:
            best, best_epoch, bad = vm["RMSE"], ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, time.perf_counter() - start_time, best, best_epoch


def ablation_flags(name):
    return {
        "use_temporal": name != "no_temporal_adapter",
        "use_graph": name != "no_graph",
        "use_fusion": name != "no_fusion_gate",
        "use_stage": name != "no_stage_weighting",
        "use_mono": name != "no_monotonic",
        "use_smooth": name != "no_smoothness",
    }


# --------------------------------------------------------------------------- #
# prepared splits (windows + graph), memoised on disk
# --------------------------------------------------------------------------- #
PREP_KEYS = ("Xtr", "ytr", "utr", "ctr", "Xva", "yva", "adj",
             "Xte", "yte", "ute", "cte", "ote", "n_train_engines")


def cache_root(args):
    """Where writable caches live: --cache_dir, else --data_dir (backwards compatible)."""
    return Path(args.cache_dir) if args.cache_dir else Path(args.data_dir)


def prep_cache_path(args, subset, fraction, seed):
    # The low-data subset is drawn with default_rng(seed), so both its windows
    # and its scaler statistics depend on the seed. fraction 1.0 does not.
    seed_tag = f"_seed{seed}" if fraction < 1.0 else ""
    cond_tag = "_cn" if args.cond_norm else ""
    name = (f"prep_{subset}_ss{args.split_seed}_w{args.window}_r{args.rmax:g}"
            f"_f{fraction:g}{seed_tag}{cond_tag}_k{args.graph_topk}.npz")
    return cache_root(args) / "_prep_cache_v7" / name


def prepare_split(args, subset, seed, fraction):
    """Frames -> windows -> graph, memoised on disk.

    Depends only on (subset, split_seed, window, rmax, fraction, cond_norm,
    graph_topk) plus seed when fraction < 1. With a fixed split_seed, three
    seeds at fraction 1.0 therefore share one prepared split instead of
    rebuilding it three times (v6 rebuilt it on every run).
    """
    p = prep_cache_path(args, subset, fraction, seed)
    if not args.no_prep_cache and p.exists():
        try:
            with np.load(p) as z:
                prep = {k: z[k] for k in PREP_KEYS}
            prep["adj"] = prep["adj"].astype(bool)
            prep["n_train_engines"] = int(prep["n_train_engines"])
            print(f"[prep] hit {p.name}")
            return prep
        except Exception as exc:
            print(f"[prep] unreadable {p.name} ({exc}); rebuilding")

    tr0, te0, rul = load_cmapss(args.data_dir, subset)
    tr0 = attach_train_rul(tr0, args.rmax)
    te0 = attach_test_rul(te0, rul, args.rmax)
    tr_df, va_df = select_engine_split(tr0, args.split_seed)

    # Low-data experiment drops whole engines BEFORE scaling, so the scaler
    # statistics never come from engines that are then discarded.
    if fraction < 1.0:
        eng = np.sort(tr_df.unit.unique())
        rng = np.random.default_rng(seed)
        keep_n = max(2, int(round(len(eng) * fraction)))
        keep = rng.choice(eng, size=min(keep_n, len(eng)), replace=False)
        tr_df = tr_df[tr_df.unit.isin(keep)].copy()

    tr_df, va_df, te_df = normalize_frames(tr_df, [va_df, te0], cond_norm=args.cond_norm)

    Xtr, ytr, utr, ctr = make_windows(tr_df, args.window)
    Xva, yva, _, _ = make_pseudotest(va_df, args.window, args.rmax)
    Xte, yte, ute, cte, ote = make_last_windows(te_df, args.window)

    if len(ytr) == 0 or len(yva) == 0 or len(yte) == 0:
        raise RuntimeError(
            f"{subset} frac={fraction:g}: empty split "
            f"(train={len(ytr)}, val={len(yva)}, test={len(yte)}). "
            f"Check --window against engine lengths.")

    adj = build_graph(Xtr, args.graph_topk).numpy()
    prep = {"Xtr": Xtr, "ytr": ytr, "utr": utr, "ctr": ctr,
            "Xva": Xva, "yva": yva, "adj": adj,
            "Xte": Xte, "yte": yte, "ute": ute, "cte": cte,
            "ote": ote,
            "n_train_engines": np.int64(tr_df.unit.nunique())}
    if not args.no_prep_cache:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(p) + f".tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            np.savez(f, **prep)
        os.replace(tmp, p)
        print(f"[prep] built {p.name} train={len(ytr)} val={len(yva)} test={len(yte)}")
    return prep


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def run_one(args, subset, seed, ablation, pipe, device, fraction=1.0):
    run_name = f"{subset}_{args.mode}_{ablation}_seed{seed}_frac{fraction:g}"
    out = Path(args.output_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"

    if args.resume and metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                cached = json.load(f)
            print(f"RESULT {run_name}: SKIP (already done) RMSE={cached.get('RMSE', float('nan')):.4f}")
            return cached
        except Exception:
            print(f"[resume] unreadable {metrics_path}; re-running")

    seed_all(seed, args.deterministic)

    prep = prepare_split(args, subset, seed, fraction)
    Xtr_raw, ytr, utr, ctr = prep["Xtr"], prep["ytr"], prep["utr"], prep["ctr"]
    Xva_raw, yva = prep["Xva"], prep["yva"]
    Xte_final_raw, yte = prep["Xte"], prep["yte"]
    ute, cte = prep["ute"], prep["cte"]
    ote = prep.get("ote")
    n_train_engines = int(prep["n_train_engines"])

    print(f"[data] {run_name}: train={len(ytr)} val={len(yva)} test engines={len(yte)} "
          f"({time.strftime('%H:%M:%S')})")

    adj = torch.as_tensor(np.ascontiguousarray(prep["adj"])).to(device)

    if args.mode == "lstm_backbone":
        flags = ablation_flags(ablation)
        model = RULModelLSTM(
            hidden=args.hidden, heads=args.heads, dropout=args.dropout,
            use_graph=flags["use_graph"], use_fusion=flags["use_fusion"]
        ).to(device)
        # Raw windows only: this branch never needs the pair index, which v5 built
        # inside ArrayDataset and then never read.
        budget = args.gpu_budget_gb * 1e9
        x_train_tensor = _place(_host_tensor(Xtr_raw), device, budget)
        x_val_tensor = _place(_host_tensor(Xva_raw), device, budget)
        y_train_tensor = torch.as_tensor(np.asarray(ytr, np.float32), device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        best, bad, best_state, best_epoch = float("inf"), 0, None, 0
        history = []
        t0 = time.perf_counter()
        for ep in range(1, args.epochs + 1):
            model.train()
            order = np.random.permutation(len(ytr))
            ep_loss, ep_n = 0.0, 0
            for st in range(0, len(order), args.batch_size):
                ix = order[st:st + args.batch_size]
                xb = _fetch(x_train_tensor, ix, device)
                yb = y_train_tensor.index_select(
                    0, torch.as_tensor(ix, dtype=torch.long, device=device))
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb, adj)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                ep_loss += float(loss.detach()) * len(ix)
                ep_n += len(ix)
            model.eval()
            vp = []
            with torch.no_grad():
                for st in range(0, len(yva), args.batch_size):
                    vp.append(model(_fetch(x_val_tensor, slice(st, st + args.batch_size), device),
                                    adj).detach().cpu().numpy())
            vm = metric_dict(yva, np.concatenate(vp))
            val = vm["RMSE"]
            history.append({"epoch": ep, "train_mse": ep_loss / max(ep_n, 1),
                            "val_pseudotest_RMSE": val, "val_pseudotest_MAE": vm["MAE"],
                            "val_pseudotest_NASA": vm["NASA_Score"]})
            if ep == 1 or ep % 5 == 0:
                print(f"{run_name} epoch {ep:03d} | train MSE {ep_loss / max(ep_n, 1):.3f} "
                      f"| val pseudo-test RMSE {val:.3f}")
            if val < best - 1e-4:
                best, bad, best_epoch = val, 0, ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        train_sec = time.perf_counter() - t0
        if best_state is not None:
            model.load_state_dict(best_state)

        t0 = time.perf_counter()
        test_pred = evaluate_features(model, None, yte, adj, device, args.batch_size,
                                      raw_x=Xte_final_raw)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        infer_sec = time.perf_counter() - t0
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        best_val = best
    else:
        # Shared cache across Chronos-based modes (linear probe / adapter / hybrid).
        croot = cache_root(args) / "_chronos2_shared_cache_v5"
        # Keep legacy v5 filenames when defaults are used, so existing caches are reused.
        tag = ""
        if args.cond_norm:
            tag += "_condnorm"
        if args.embed_pool != "none":
            tag += f"_{args.embed_pool}"
        train_cache_tag = (f"train_f{fraction:g}" if fraction >= 1.0
                           else f"train_f{fraction:g}_seed{seed}")
        Ztr = prepare_features(pipe, Xtr_raw,
                               make_cache_path(croot, subset, train_cache_tag, args.window,
                                               args.split_seed, tag), args)
        Zva = prepare_features(pipe, Xva_raw,
                               make_cache_path(croot, subset, "val", args.window,
                                               args.split_seed, tag), args)
        Zte = prepare_features(pipe, Xte_final_raw,
                               make_cache_path(croot, subset, "test", args.window, 0, tag), args)

        flags = ablation_flags(ablation)
        if args.mode == "linear_probe":
            Ztr_p = probe_input(Ztr)
            Zva_p = probe_input(Zva)
            Zte_p = probe_input(Zte)
            for name, a, b in (("train", Ztr_p, ytr), ("val", Zva_p, yva), ("test", Zte_p, yte)):
                if len(a) != len(b):
                    raise RuntimeError(
                        f"{run_name}: {name} embedding/label mismatch "
                        f"({len(a)} vs {len(b)}). Remove the stale cache entry and re-embed."
                    )
            dim = Ztr_p.shape[1]
            model = nn.Linear(dim, 1).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                          weight_decay=args.weight_decay)
            budget = args.gpu_budget_gb * 1e9
            Ztr_t = _place(_host_tensor(Ztr_p), device, budget)
            Zva_t = _place(_host_tensor(Zva_p), device, budget)
            Zte_t = _place(_host_tensor(Zte_p), device, budget)
            ytr_t = torch.as_tensor(np.asarray(ytr, np.float32), device=device)
            best, bad, state, best_epoch = float("inf"), 0, None, 0
            history = []
            t0 = time.perf_counter()
            for ep in range(1, args.epochs + 1):
                model.train()
                order = np.random.permutation(len(ytr))
                for st in range(0, len(order), args.batch_size):
                    ix = order[st:st + args.batch_size]
                    xb = _fetch(Ztr_t, ix, device)
                    yb = ytr_t.index_select(
                        0, torch.as_tensor(ix, dtype=torch.long, device=device))
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.mse_loss(model(xb).squeeze(-1), yb)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    pv = model(_fetch(Zva_t, slice(0, len(yva)), device)).squeeze(-1).cpu().numpy()
                vm = metric_dict(yva, pv)
                val = vm["RMSE"]
                history.append({"epoch": ep, "val_pseudotest_RMSE": val,
                                "val_pseudotest_MAE": vm["MAE"],
                                "val_pseudotest_NASA": vm["NASA_Score"]})
                if val < best - 1e-4:
                    best, bad, best_epoch = val, 0, ep
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= args.patience:
                        break
            train_sec = time.perf_counter() - t0
            if state is not None:
                model.load_state_dict(state)
            t0 = time.perf_counter()
            with torch.no_grad():
                test_pred = model(_fetch(Zte_t, slice(0, len(yte)), device)).squeeze(-1).cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_sec = time.perf_counter() - t0
            # The probe itself is tiny, but it is only meaningful on top of the
            # frozen encoder; report the encoder so the table is comparable with
            # the adapter / hybrid rows (they already include it).
            n_train = sum(p.numel() for p in model.parameters())
            n_total = n_train + sum(p.numel() for p in pipe.model.parameters())
            best_val = best
        else:
            if args.mode == "chronos_bilstm":
                model = ChronosBiLSTMHybrid(
                    d_in=Ztr.shape[-1], n_features=Xtr_raw.shape[-1],
                    hidden=args.hidden, heads=args.heads, dropout=args.dropout
                ).to(device)
            else:
                model = ChronosRULAdapter(
                    d_in=Ztr.shape[-1], hidden=args.hidden, heads=args.heads,
                    dropout=args.dropout, use_temporal=flags["use_temporal"],
                    use_graph=flags["use_graph"], use_fusion=flags["use_fusion"]
                ).to(device)
            n_total = (sum(p.numel() for p in model.parameters())
                       + sum(p.numel() for p in pipe.model.parameters()))
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model, history, train_sec, best_val, best_epoch = train_model(
                model, Ztr, ytr, utr, ctr, Zva, yva, adj, args, seed,
                use_mono=flags["use_mono"], use_smooth=flags["use_smooth"],
                use_stage=flags["use_stage"], data_tag=run_name,
                raw_train=Xtr_raw if args.mode == "chronos_bilstm" else None,
                raw_val=Xva_raw if args.mode == "chronos_bilstm" else None
            )
            t0 = time.perf_counter()
            test_pred = evaluate_features(
                model, Zte, yte, adj, device, args.batch_size,
                raw_x=Xte_final_raw if args.mode == "chronos_bilstm" else None
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_sec = time.perf_counter() - t0

    metrics = metric_dict(yte, test_pred)
    eng = per_engine_frame(ute, cte, yte, test_pred)
    cond_df = None
    if not args.no_report_cond and ote is not None:
        cond_df = per_condition_metrics(yte, test_pred, ote)
    cond_sum = summarize_conditions(cond_df)
    result = {
        "subset": subset, "mode": args.mode, "ablation": ablation,
        "seed": seed, "data_fraction": fraction,
        "best_epoch": int(best_epoch), "best_val_pseudotest_RMSE": float(best_val),
        **metrics,
        "total_params_including_backbone": int(n_total),
        "trainable_params": int(n_train),
        "trainable_fraction_of_total": float(n_train / max(n_total, 1)),
        "train_time_sec": float(train_sec),
        "inference_time_sec": float(infer_sec),
        "inference_ms_per_engine": float(infer_sec / max(len(yte), 1) * 1000),
        "peak_cuda_memory_MB": (float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
                                if device.type == "cuda" else float("nan")),
        "device": str(device),
        "n_train_windows": len(ytr),
        "n_train_engines": n_train_engines,
        "n_val_pseudotest": len(yva),
        "n_test_engines": len(yte),
        "chronos_model": (args.chronos_model + "+BiLSTM") if args.mode == "chronos_bilstm"
        else (args.chronos_model if args.mode != "lstm_backbone" else "LSTM"),
        "window": args.window, "rul_cap": args.rmax,
        "split_seed": args.split_seed,
        "cond_norm": bool(args.cond_norm),
        "embed_pool": args.embed_pool,
        "quick_test": bool(args.quick_test),
        "engine_RMSE_mean": float(np.sqrt(np.mean(eng.error ** 2))),
        "engine_RMSE_std": float(eng.groupby("unit").apply(
            lambda g: np.sqrt(np.mean(g.error ** 2))).std()),
        "engine_abs_error_median": float(eng.abs_error.median()),
        **cond_sum,
    }
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    pd.DataFrame({"unit": ute, "cycle": cte, "true_RUL": yte, "pred_RUL": test_pred,
                  "error": test_pred - yte}).to_csv(out / "test_predictions.csv", index=False)
    eng.to_csv(out / "per_engine_metrics.csv", index=False)
    if cond_df is not None:
        cond_df.to_csv(out / "per_condition_metrics.csv", index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    extra = ""
    if cond_df is not None:
        extra = (f" | cond gap={cond_sum['cond_RMSE_gap']:.3f} "
                 f"(worst {cond_sum['cond_RMSE_worst']:.2f})")
    print(f"RESULT {run_name}: RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f} "
          f"NASA={metrics['NASA_Score']:.2f} (per-engine {metrics['NASA_per_engine']:.3f}) "
          f"late={metrics['late_ratio'] * 100:.0f}%{extra} ({time.strftime('%H:%M:%S')})")
    return result


# --------------------------------------------------------------------------- #
# cross-subset transfer
# --------------------------------------------------------------------------- #
def run_cross_subset_transfer(args, source_subset, target_subset, seed, pipe, device):
    """Train on source-subset training engines; evaluate target-subset test engines."""
    run_name = f"TRANSFER_{source_subset}_to_{target_subset}_seed{seed}"
    out = Path(args.output_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    if args.resume and metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                cached = json.load(f)
            print(f"RESULT {run_name}: SKIP (already done)")
            return cached
        except Exception:
            pass

    seed_all(seed, args.deterministic)

    src_train, _, _ = load_cmapss(args.data_dir, source_subset)
    _, tgt_test, tgt_rul = load_cmapss(args.data_dir, target_subset)
    src_train = attach_train_rul(src_train, args.rmax)
    src_tr, src_va = select_engine_split(src_train, args.split_seed)

    # Source-only scaler; the target subset never contributes statistics.
    scaler = FeatureScaler(src_tr, cond_norm=args.cond_norm)
    src_tr = scaler.transform(src_tr)
    src_va = scaler.transform(src_va)
    tgt_test = scaler.transform(attach_test_rul(tgt_test, tgt_rul, args.rmax))

    Xtr, ytr, utr, ctr = make_windows(src_tr, args.window)
    Xva, yva, _, _ = make_pseudotest(src_va, args.window, args.rmax)
    # Only the final window per engine: the official test evaluates the last
    # snapshot, not every intermediate one (v5 built all of them and kept one).
    Xte, yte, ute, cte, ote = make_last_windows(tgt_test, args.window)

    adj = build_graph(Xtr, args.graph_topk).to(device)
    xfer_cache = Path(args.output_dir) / "_feature_cache"
    tag = ("_condnorm" if args.cond_norm else "") + ("" if args.embed_pool == "none" else f"_{args.embed_pool}")
    Ztr = prepare_features(pipe, Xtr, make_cache_path(xfer_cache, source_subset,
                                                      f"xfer_train_{target_subset}",
                                                      args.window, seed, tag), args)
    Zva = prepare_features(pipe, Xva, make_cache_path(xfer_cache, source_subset,
                                                      f"xfer_val_{target_subset}",
                                                      args.window, seed, tag), args)
    Zte = prepare_features(pipe, Xte, make_cache_path(xfer_cache, target_subset,
                                                      f"xfer_test_from_{source_subset}",
                                                      args.window, seed, tag), args)

    model = ChronosRULAdapter(
        d_in=Ztr.shape[-1], hidden=args.hidden, heads=args.heads,
        dropout=args.dropout, use_temporal=True, use_graph=True, use_fusion=True
    ).to(device)
    model, hist, train_sec, best_val, best_epoch = train_model(
        model, Ztr, ytr, utr, ctr, Zva, yva, adj, args, seed,
        use_mono=True, use_smooth=True, use_stage=True, data_tag=run_name
    )
    pred = evaluate_features(model, Zte, yte, adj, device, args.batch_size)
    met = metric_dict(yte, pred)
    eng = per_engine_frame(ute, cte, yte, pred)
    cond_df = None
    if not args.no_report_cond:
        cond_df = per_condition_metrics(yte, pred, ote)
    cond_sum = summarize_conditions(cond_df)
    result = {
        "source_subset": source_subset, "target_subset": target_subset, "seed": seed,
        "best_epoch": int(best_epoch), "best_val_RMSE": float(best_val), **met,
        "train_time_sec": float(train_sec), "n_source_train_windows": len(ytr),
        "n_target_test_engines": len(yte),
        "transfer_protocol": "source-only scaler/graph/training; target official test",
        "chronos_model": args.chronos_model,
        "cond_norm": bool(args.cond_norm),
        **cond_sum,
    }
    pd.DataFrame(hist).to_csv(out / "training_history.csv", index=False)
    pd.DataFrame({"unit": ute, "cycle": cte, "true_RUL": yte, "pred_RUL": pred,
                  "error": pred - yte}).to_csv(out / "target_predictions.csv", index=False)
    eng.to_csv(out / "per_engine_metrics.csv", index=False)
    if cond_df is not None:
        cond_df.to_csv(out / "per_condition_metrics.csv", index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"RESULT {run_name}: RMSE={met['RMSE']:.4f} MAE={met['MAE']:.4f} "
          f"NASA/eng={met['NASA_per_engine']:.3f} late={met['late_ratio'] * 100:.0f}%")
    return result


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
SUMMARY_COLS = ["RMSE", "MAE", "NASA_Score", "R2", "total_params_including_backbone",
                "trainable_params", "trainable_fraction_of_total", "train_time_sec",
                "inference_ms_per_engine", "peak_cuda_memory_MB"]


TABLE_KEY = ["subset", "mode", "ablation", "seed", "data_fraction"]
TABLE_CFG = ["window", "rul_cap", "cond_norm", "embed_pool", "split_seed"]


def write_tables(raw_new, out_dir, verbose=True):
    """Upsert into all_runs.csv so several invocations can share one output dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "all_runs.csv"
    if len(raw_new) == 0:
        return raw_new
    key = [k for k in TABLE_KEY if k in raw_new.columns]
    if table_path.exists():
        try:
            old = pd.read_csv(table_path)
        except Exception:
            old = None
        if old is not None and len(old):
            # warn when a row is replaced by a run with different preprocessing
            cfg = [c for c in TABLE_CFG if c in old.columns and c in raw_new.columns]
            if cfg and key:
                merged_probe = pd.concat([old, raw_new], ignore_index=True)
                dup = merged_probe.duplicated(subset=key, keep=False)
                diff = merged_probe[dup].groupby(key)[cfg].nunique(dropna=False)
                bad = diff[(diff > 1).any(axis=1)]
                if len(bad):
                    print(f"[table] WARNING: {len(bad)} run(s) in {table_path.name} were produced "
                          f"with different {cfg}; the newest row replaces the old one. "
                          f"Use a separate --output_dir per configuration.")
            raw_new = pd.concat([old, raw_new], ignore_index=True)
    if key:
        raw_new = raw_new.drop_duplicates(subset=key, keep="last")
        raw_new = raw_new.sort_values(key).reset_index(drop=True)
    raw_new.to_csv(table_path, index=False)
    cols = [c for c in SUMMARY_COLS if c in raw_new.columns]
    gkeys = [k for k in ["subset", "mode", "ablation", "data_fraction"] if k in raw_new.columns]
    if cols and gkeys:
        summary = raw_new.groupby(gkeys)[cols].agg(["mean", "std", "count"])
        summary.to_csv(out_dir / "mean_std_summary.csv")
        if verbose:
            print(summary.to_string())
    return raw_new


CMP_METRICS = ["RMSE", "MAE", "NASA_Score", "NASA_per_engine", "R2",
               "late_ratio", "mean_error", "p90_abs_error",
               "trainable_params", "trainable_fraction_of_total",
               "train_time_sec", "inference_ms_per_engine", "peak_cuda_memory_MB"]

LOWER_BETTER = ["RMSE", "MAE", "late_ratio", "abs_mean_error",
                "p90_abs_error", "train_time_sec", "inference_ms_per_engine",
                "trainable_params"]
HIGHER_BETTER = ["NASA_Score", "NASA_per_engine", "R2"]
GROUP_KEYS = ["subset", "data_fraction", "ablation", "seed"]


def compare_modes(raw, out_dir, baseline_csv=None):
    """One row per (subset, fraction, mode) with per-group ranks and a 'best_on' tag.

    This is the table that answers "where does the model actually win": instead of
    a single headline number it ranks every metric inside the same protocol group
    and names the metrics each configuration leads. Nothing is filtered out -
    a losing metric shows as rank > 1, which is exactly what a reviewer will check.
    """
    if raw is None or len(raw) == 0:
        return None
    df = raw.copy()
    if baseline_csv and Path(baseline_csv).exists():
        b = pd.read_csv(baseline_csv)
        if "mode" not in b.columns and "model" in b.columns:
            b = b.rename(columns={"model": "mode"})
        b["ablation"] = b.get("ablation", "full")
        b["seed"] = b.get("seed", 0)
        b["data_fraction"] = b.get("data_fraction", 1.0)
        df = pd.concat([df, b], ignore_index=True, sort=False)

    df["abs_mean_error"] = df["mean_error"].abs() if "mean_error" in df.columns else np.nan
    cols = [c for c in CMP_METRICS + ["abs_mean_error"] if c in df.columns]
    gkeys = [k for k in GROUP_KEYS if k in df.columns]

    flags = {}
    for m in cols:
        if m in LOWER_BETTER:
            asc = True
        elif m in HIGHER_BETTER:
            asc = False
        else:
            continue
        r = df.groupby(gkeys)[m].rank(method="min", ascending=asc)
        df[m + "_rank"] = r
        # a metric nobody differentiates (all tied) must not be advertised as a win
        varies = df.groupby(gkeys)[m].transform(lambda x: x.nunique(dropna=True) > 1)
        flags[m] = (r == 1) & varies.fillna(False).astype(bool)
    flags_df = pd.DataFrame(flags)
    df["best_on"] = flags_df.apply(
        lambda row: ",".join(sorted(row.index[row.values])), axis=1)

    rank_cols = [c for c in df.columns if c.endswith("_rank")]

    base = df[df.get("mode") == "lstm_backbone"]
    if len(base):
        b = base.set_index(gkeys)[["RMSE", "NASA_Score", "MAE"]].rename(
            columns={"RMSE": "LSTM_RMSE", "NASA_Score": "LSTM_NASA", "MAE": "LSTM_MAE"})
        df = df.merge(b, left_on=gkeys, right_index=True, how="left")
        df["dRMSE_vs_LSTM"] = df["RMSE"] - df["LSTM_RMSE"]
        df["dNASA_vs_LSTM"] = df["NASA_Score"] - df["LSTM_NASA"]

    keep = gkeys + ["mode"] + [c for c in cols if c in df.columns] + rank_cols +            ["best_on"] + [c for c in ("dRMSE_vs_LSTM", "dNASA_vs_LSTM") if c in df.columns]
    out = df[[c for c in keep if c in df.columns]].sort_values(
        gkeys + ["mode"]).reset_index(drop=True)
    path = Path(out_dir) / "compare_modes.csv"
    out.to_csv(path, index=False)
    print(f"[compare] {len(out)} rows -> {path}")
    show = [c for c in ("subset", "data_fraction", "mode", "RMSE", "MAE", "NASA_Score",
                        "NASA_per_engine", "late_ratio", "best_on") if c in out.columns]
    print(out[show].to_string(index=False))
    return out


def merge_runs(dirs, out_dir):
    frames = []
    for d in dirs:
        p = Path(d) / "all_runs.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            print(f"[merge] skip missing {p}")
    if not frames:
        print("[merge] no all_runs.csv found")
        return None
    raw = pd.concat(frames, ignore_index=True)
    key = [k for k in ["subset", "mode", "ablation", "seed", "data_fraction"] if k in raw.columns]
    n_in = len(raw)
    raw = raw.drop_duplicates(subset=key, keep="last").sort_values(key).reset_index(drop=True)
    if len(raw) != n_in:
        print(f"[merge] WARNING: dropped {n_in - len(raw)} duplicate (subset,mode,ablation,seed,fraction) "
              f"rows; the last one written wins. Merge dirs that hold the same configuration "
              f"will hide the earlier copy.")
    write_tables(raw, Path(out_dir))
    print(f"[merge] {len(raw)} runs -> {Path(out_dir) / 'all_runs.csv'}")
    compare_modes(raw, Path(out_dir))
    return raw


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = args_parser(argv)
    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.merge_from:
        return merge_runs(args.merge_from, args.output_dir)

    device = choose_device(args.device)
    # autocast is CUDA-only; resolve it once so every embed call agrees
    args.amp_resolved = (device.type == "cuda") and (args.amp != "off")
    if args.amp == "on" and device.type != "cuda":
        print("[warn] --amp on but the device is cpu; autocast disabled")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device} threads={torch.get_num_threads()} "
          f"mode={args.mode} pool={args.embed_pool} cond_norm={args.cond_norm} "
          f"cache={args.cache_dtype} amp={args.amp_resolved} resume={args.resume}")

    pipe = None
    if args.mode != "lstm_backbone" or args.cross_train:
        pipe = load_chronos(args.chronos_model, device)

    if args.cross_train and args.cross_test:
        rows = [run_cross_subset_transfer(args, args.cross_train, args.cross_test,
                                          seed, pipe, device) for seed in args.seeds]
        df = pd.DataFrame(rows)
        df.to_csv(Path(args.output_dir) / "cross_subset_transfer.csv", index=False)
        if {"source_subset", "target_subset", "seed"}.issubset(df.columns):
            print(df.groupby(["source_subset", "target_subset"])[
                ["RMSE", "MAE", "NASA_Score"]].agg(["mean", "std"]).to_string())
        return df

    ablations = (["full", "no_temporal_adapter", "no_graph", "no_fusion_gate",
                  "no_monotonic", "no_smoothness", "no_stage_weighting"]
                 if args.ablation == "all" else [args.ablation])
    fractions = args.data_fractions or [1.0]
    jobs = [(s, f, ab, sd) for s in args.subsets for f in fractions
            for ab in ablations for sd in args.seeds]
    print(f"[plan] {len(jobs)} runs: {len(args.subsets)} subsets x {len(fractions)} fractions "
          f"x {len(ablations)} ablations x {len(args.seeds)} seeds")

    results, t_all = [], time.perf_counter()
    job_bar = tqdm(jobs, desc="Experiment runs", unit="run", dynamic_ncols=True)
    for subset, frac, ab, seed in job_bar:
        job_bar.set_postfix(subset=subset, seed=seed, ablation=ab, fraction=frac)
        try:
            results.append(run_one(args, subset, seed, ab, pipe, device, frac))
            write_tables(pd.DataFrame(results), args.output_dir, verbose=False)  # crash-safe partial table
        except Exception as exc:
            print(f"[ERROR] {subset} frac={frac} seed={seed} ablation={ab}: "
                  f"{type(exc).__name__}: {exc}")
            raise

    raw = write_tables(pd.DataFrame(results), args.output_dir)
    compare_modes(raw, args.output_dir, args.baseline_csv)
    print(f"[done] {len(raw)} runs in {(time.perf_counter() - t_all) / 60:.1f} min")
    return raw


def run_config(**kwargs):
    """Notebook entry point: run_config(data_dir=..., subsets=[...], mode=...) -> table.

    Keyword names are the CLI flag names without the leading dashes:
      mode="chronos_bilstm"   ->  --mode chronos_bilstm
      subsets=["FD001"]       ->  --subsets FD001
      resume=True             ->  --resume
      cond_norm=False         ->  (flag omitted)
    """
    argv = []
    for k, v in kwargs.items():
        if v is None or v is False:
            continue
        flag = "--" + k
        if v is True:
            argv.append(flag)
        elif isinstance(v, (list, tuple)):
            argv += [flag] + [str(x) for x in v]
        else:
            argv += [flag, str(v)]
    print("[run_config] " + " ".join(argv))
    return main(argv)


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 0)
#（注：内容由AI生成）
