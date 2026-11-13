#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chronos-2 + BiLSTM hybrid RUL adaptation for NASA C-MAPSS (V9 "fast").

Reviewer-facing experiments:
  E1: FD001-FD004, repeated seeds
  E2: multi-seed mean/std
  E3: reduced-data learning curves
  E4: cross-subset transfer (train on one subset, evaluate another)
  E5: frozen Chronos-2 feature linear probe
  E6: LSTM-backbone replacement with the same adapter/GAT/fusion/head
  E7: module ablations
  E8: parameters, training/inference time, peak memory

------------------------------------------------------------------------------
CHANGELOG

v5_fast -> v6 (correctness):
  * training-split statistics re-fitted after low-data subsampling
  * Chronos cache validated against the expected (N, 24, ...) shape
  * cache name carries cond_norm / pool tags only when not default
  * --resume skips finished runs; caches reused
  * optional --cond_norm per-condition scaling (FD002/FD004)
  * vectorised LSTM backbone
  * --merge_from rebuilds the table without re-running

v6 -> v7 (throughput):
  * np.loadtxt instead of pandas' python-engine fallback
  * window/graph prep memoised on disk per (subset, split_seed, ...)
  * sliding_window_view instead of a python append loop
  * test split builds only the final window per engine
  * frozen encoder under autocast, features stored as f16
  * feature tensors pinned on the GPU when small enough
  * single lexsort for the (window, window+1) pair index

v7 -> v8 (evaluation):
  * diagnostics behind the asymmetric-cost advantage (NASA per engine, bias,
    late/early ratio, p90, worst error)
  * per_engine_metrics.csv, per_condition_metrics.csv, cond_RMSE_gap
  * compare_modes.csv ranking inside protocol groups, --baseline_csv

v8 -> v9 (this file, wall-clock):

  I/O and preparation
  * raw txt -> .npy cache: the whitespace parse runs once per machine, not once
    per run (np.loadtxt on FD004 was ~2 s of every single run)
  * split-level cache (tr/va/te raw frames) separated from the fraction-level
    window cache. v8 rebuilt the validation windows and the test windows
    for every data fraction even though neither depends on the fraction
  * all caches are written uncompressed (.npy/.npz with np.savez). v8 used
    savez_compressed for the prep cache: on FD004 that is ~90 s of zlib per
    fraction, for arrays that are already float32 and incompressible
  * Chronos feature cache is a single .npy opened with mmap_mode="r". Loading
    the old .npz materialised 7-9 GB into RAM before the first batch; with mmap
    the kernel pages it in on demand and nothing is copied at load time
  * feature tensors above --gpu_budget_gb stay mmap-backed and are gathered per
    mini-batch, so host RSS no longer holds the full feature matrix

  Batching
  * torch.utils.data.DataLoader is gone. Every split is already in RAM, so a
    DataLoader only adds a collate step, a worker hand-off and a per-batch tensor
    construction. BatchIter shuffles with one permutation and indexes a torch
    view of the source directly. Measured on 4 Xeon cores, batch 192, two
    epochs of 13.7k windows: 0.09s -> 0.004s for D=8 (23x), 1.79s -> 0.90s for
    D=64/T=30 (2.0x), 0.41s -> 0.30s for D=256/T=30 (1.4x). The 2x-23x range is
    why the feature matrix is cast once (`--feature_preload_gb`) and why the
    cache dtype is f32 by default on CPU: a per-batch f16->f32 conversion is
    single-threaded numpy work and erases the gain
  * the sample dict no longer carries X (raw windows), units and cycles into the
    model: forward() ignores x_raw, so shipping it was pure memcpy per batch
  * the pair index is only gathered when the regularisers are actually enabled

  Training
  * --sched onecycle is available, but measured on FD001 it needed 9 epochs to
    reach val RMSE 16 against 7 for the flat 1e-3 schedule (the warmup costs more
    than the annealing saves). The default therefore stays flat, unlike v8's
    implicit constant LR being an accident: it is now an explicit choice
  * optional bf16 autocast for the trainable tower on CUDA (--train_amp)
  * the per-batch tqdm.set_postfix call is replaced by an epoch-level bar:
    set_postfix on 500 batches costs ~0.3 s/epoch and forces a refresh
  * torch.set_num_threads / interop threads are pinned once, before the first
    op, instead of letting each process grab every core

  Orchestration
  * --jobs N fans independent configurations out to N subprocesses sharing the
    on-disk caches (writes are atomic os.replace, so a cache built by one
    worker is safely reused by the others). On 4 cores, --jobs 2 --threads 2
    roughly halves the E1 wall clock
  * --experiments lets you stop after E1 instead of paying for E3-E7

  Fixed
  * the monotonicity / smoothness terms were dead code in v5-v8: train_model
    called loss_components(pred, y) without the (window, window+1) pair, so
    lambda_mono and lambda_smooth multiplied zeros and the no_monotonic /
    no_smoothness ablations were identical to full. --use_next_pair turns the
    terms on (one extra forward over --next_pair_frac of the batch). It stays
    off by default so v8 numbers remain reproducible, and the ablation is
    refused when the term is off, instead of silently reporting a null effect.
"""

import argparse
import json
import os
import random
try:
    import resource                      # POSIX only; Windows has no such module
except ImportError:                      # pragma: no cover - Windows
    resource = None
import subprocess
import sys
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
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
OUT_COLS = COLS + ["RUL"]
FEATURES = [f"s{i}" for i in range(1, 22)] + ["os1", "os2", "os3"]
N_FEATURES = len(FEATURES)  # 24
POOLS = ["none", "token", "both"]
EXPERIMENTS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]
CACHE_VERSION = "v9b"  # bump when the cached content meaning changes


# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------
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
                   help="Writable directory for the raw/prep/Chronos caches. Defaults to --data_dir, "
                        "falling back to <output_dir>/cache when that is read-only (Kaggle: "
                        "/kaggle/working/cache).")
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
    p.add_argument("--cross_train", default=None, choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--cross_test", default=None, choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--data_fractions", nargs="+", type=float, default=None,
                   help="E3 data fractions. Default 0.1 0.25 0.5 0.75; the 100%% point is "
                        "already E1, so it is not trained twice.")
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--rmax", type=float, default=125.0)
    p.add_argument("--batch_size", type=int, default=192,
                   help="v9 default is larger than v8's 128: BatchIter has no collate cost, so a "
                        "bigger GEMM is pure win on both CPU and GPU.")
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--embed_batch_size", type=int, default=16,
                   help="Number of windows per Chronos forward pass.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--sched", choices=["onecycle", "none"], default="none",
                   help="onecycle: LR ramps to --lr then anneals to 0 over --epochs. Default is "
                        "none: on FD001 onecycle needed 9 epochs to reach val RMSE 16 against 7 "
                        "for the flat LR, so it is offered but not imposed.")
    p.add_argument("--lambda_mono", type=float, default=0.05)
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument("--use_next_pair", action="store_true",
                   help="Enable the monotonicity / smoothness terms (they need the (window, "
                        "window+1) pair). Off by default so v8 numbers reproduce.")
    p.add_argument("--next_pair_frac", type=float, default=0.25,
                   help="Fraction of the batch used for the pair loss (one extra forward).")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--graph_topk", type=int, default=6)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--chronos_model", default="amazon/chronos-2")
    p.add_argument("--context_length", type=int, default=30)
    p.add_argument("--force_reembed", action="store_true")
    p.add_argument("--cache_dtype", choices=["auto", "f16", "f32"], default="auto",
                   help="Storage dtype of the feature cache. auto = f32 on CPU, f16 on CUDA: a "
                        "CPU run gains nothing from f16 and pays a per-batch conversion.")
    p.add_argument("--amp", choices=["auto", "on", "off"], default="auto",
                   help="autocast fp16 for the frozen encoder on CUDA.")
    p.add_argument("--train_amp", choices=["auto", "on", "off"], default="auto",
                   help="bf16 autocast for the trainable tower on CUDA.")
    p.add_argument("--gpu_budget_gb", type=float, default=2.0,
                   help="Feature tensors larger than this stay mmap-backed and are gathered per "
                        "mini-batch instead of being pinned on the GPU.")
    p.add_argument("--feature_preload_gb", type=float, default=4.0,
                   help="Cast the feature matrix to float32 once when the result fits in this much "
                        "RAM (0 = always cast per mini-batch). Per-batch f16->f32 is single-threaded "
                        "numpy work and can make the iterator slower than a DataLoader.")
    p.add_argument("--feature_mmap", choices=["auto", "on", "off"], default="auto",
                   help="Load the feature cache with mmap_mode='r'. auto = on for cpu, off for cuda "
                        "when the array fits in --gpu_budget_gb.")
    p.add_argument("--no_prep_cache", action="store_true")
    p.add_argument("--split_seed", type=int, default=42,
                   help="Fixed engine split for comparable/reusable multi-seed runs.")
    p.add_argument("--threads", type=int, default=0,
                   help="torch intra-op threads; 0 = all cores. With --jobs N use cores//N.")
    p.add_argument("--torch_threads", type=int, default=None,
                   help="Deprecated alias of --threads.")
    p.add_argument("--cond_norm", action="store_true",
                   help="Per-operating-condition (os3) standardisation for FD002/FD004.")
    p.add_argument("--embed_pool", choices=POOLS, default="none")
    p.add_argument("--encoder", choices=["chronos", "stat"], default="chronos",
                   help="stat = 8 deterministic summary statistics per sensor. ENGINEERING ONLY: "
                        "it exists so the pipeline can be smoke-tested on a CPU box without the "
                        "Chronos weights. Never report stat numbers as Chronos-2 results.")
    p.add_argument("--val_stride", type=int, default=1,
                   help="Keep every k-th validation window. 1 = all (v8 behaviour).")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--merge_from", nargs="+", default=None)
    p.add_argument("--experiments", nargs="+", default=["all"],
                   help="Subset of E1..E8, or 'all'. E2/E8 are derived from the runs of E1-E7.")
    p.add_argument("--single", action="store_true",
                   help="Run exactly one configuration from the scalar flags (no experiment sweep).")
    p.add_argument("--only_run", default=None, help=argparse.SUPPRESS)
    p.add_argument("--experiment_tag", default=None, help=argparse.SUPPRESS)
    p.add_argument("--jobs", type=int, default=1,
                   help="Run independent configurations in N subprocesses sharing the caches.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the tower (CUDA only; first batch pays the compile cost).")
    p.add_argument("--quick_test", action="store_true",
                   help="Fast smoke test: FD001, one seed, 10%% data, 5 epochs.")
    p.add_argument("--quick_hybrid", action="store_true",
                   help="Quick hybrid test: FD001, one seed, full data, 8 epochs.")
    p.add_argument("--no_report_cond", action="store_true")
    p.add_argument("--baseline_csv", default=None)
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
        args.single = True
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
        args.single = True

    if args.torch_threads is not None:
        args.threads = args.torch_threads
    if args.cache_dtype == "auto":
        args.cache_dtype = "f32" if str(choose_device(args.device)) == "cpu" else "f16"
    if args.hidden % 2 != 0:
        p.error("--hidden must be even (bidirectional BiLSTM splits it in half).")
    if args.hidden % args.heads != 0:
        p.error("--hidden must be divisible by --heads.")
    if args.embed_pool == "both" and args.mode != "linear_probe":
        p.error("--embed_pool both collapses the sensor axis and only works with "
                "--mode linear_probe. Use none or token otherwise.")
    if bool(args.cross_train) != bool(args.cross_test):
        p.error("--cross_train and --cross_test must be given together.")
    if args.use_next_pair and not 0.0 < args.next_pair_frac <= 1.0:
        p.error("--next_pair_frac must be in (0, 1].")
    if args.ablation in ("no_monotonic", "no_smoothness") and not args.use_next_pair:
        p.error("--ablation %s removes a term that is disabled anyway (v9 keeps the pair loss "
                "off by default). Add --use_next_pair, otherwise the ablation has no effect."
                % args.ablation)
    if not args.merge_from:
        if not args.data_dir:
            auto = autodetect_data_dir()
            if auto is None:
                p.error("--data_dir is required (unless --merge_from is used). "
                        "Set CMAPSS_DATA_DIR or add a C-MAPSS dataset under /kaggle/input.")
            args.data_dir = auto
            print(f"[data] auto-detected --data_dir={auto}")
        missing = []
        need = set(args.subsets + [x for x in (args.cross_train, args.cross_test) if x])
        for s in need:
            for name in (f"train_{s}.txt", f"test_{s}.txt", f"RUL_{s}.txt"):
                if not (Path(args.data_dir) / name).exists():
                    missing.append(name)
        if missing:
            p.error("missing data files in %s: %s" % (args.data_dir, ", ".join(sorted(missing))))
    if args.context_length != args.window:
        print(f"[warn] --context_length={args.context_length} != --window={args.window}: "
              f"Chronos will be told {args.context_length} steps about a window that "
              f"actually holds {args.window}. Keep them equal unless you know why.")
    if args.encoder == "stat":
        print("[warn] --encoder stat: deterministic summary features, NOT Chronos-2. "
              "Use it to validate the pipeline only.")
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


def configure_threads(n):
    """Pin the thread pools once, before any op allocates them."""
    if n and n > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        os.environ.setdefault("MKL_NUM_THREADS", str(n))
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
_DATA_CACHE = {}


def cache_root(args):
    """A directory we are allowed to write to."""
    if args.cache_dir:
        return Path(args.cache_dir)
    if not args.merge_from:
        try:
            probe = Path(args.data_dir) / ".write_probe"
            probe.parent.mkdir(parents=True, exist_ok=True)
            probe.write_text("x")
            probe.unlink()
            return Path(args.data_dir)
        except Exception:
            pass
    return Path(args.output_dir) / "cache"


def _atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _atomic_save_npy(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _read_whitespace(path, names, cache_root_dir=None):
    """Whitespace parse, memoised as .npy.

    np.loadtxt is used instead of pd.read_csv(sep=r"\\s+"): the pandas C engine
    rejects a regex separator, falls back to the python engine and runs ~10x
    slower. The parsed array is then cached, so the parse happens once per
    machine rather than once per run.
    """
    path = Path(path)
    npy = None
    if cache_root_dir is not None:
        npy = Path(cache_root_dir) / "raw_cache" / f"{path.stem}_{len(names)}.npy"
        if npy.exists():
            try:
                arr = np.load(npy, mmap_mode="r")
                if arr.shape[1] == len(names):
                    return _array_to_frame(np.asarray(arr), names)
            except Exception:
                pass
    arr = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if arr.shape[1] != len(names):
        raise ValueError(f"{path.name}: expected {len(names)} columns, got {arr.shape[1]}")
    arr = np.asarray(arr, np.float32)
    if npy is not None:
        try:
            _atomic_save_npy(npy, arr)
        except Exception:
            pass
    return _array_to_frame(arr, names)


def _array_to_frame(arr, names):
    df = pd.DataFrame(arr, columns=names, copy=False)
    for c in ("unit", "cycle"):
        if c in df.columns:
            df[c] = df[c].to_numpy(np.int64)
    return df


def load_cmapss(root, subset, cache_root_dir=None):
    key = (str(Path(root).resolve()), subset)
    hit = _DATA_CACHE.get(key)
    if hit is not None:
        return hit
    root = Path(root)
    tr = _read_whitespace(root / f"train_{subset}.txt", COLS, cache_root_dir)
    te = _read_whitespace(root / f"test_{subset}.txt", COLS, cache_root_dir)
    rul = _read_whitespace(root / f"RUL_{subset}.txt", ["RUL"], cache_root_dir)
    _DATA_CACHE[key] = (tr, te, rul)
    return tr, te, rul


def attach_train_rul(df, cap):
    z = df.copy()
    last = z.groupby("unit")["cycle"].transform("max")
    z["RUL"] = (last - z["cycle"]).clip(upper=cap).astype(np.float32)
    return z


def attach_test_rul(df, rul, cap):
    z = df.copy().reset_index(drop=True)
    ids = np.sort(z.unit.unique())
    if len(ids) != len(rul):
        raise ValueError(f"Test engine count ({len(ids)}) != RUL label count ({len(rul)}).")
    rmap = {int(i): float(v) for i, v in zip(ids, rul.RUL.to_numpy())}
    last = z.groupby("unit")["cycle"].transform("max")
    z["RUL"] = (z.unit.map(rmap) + last - z.cycle).clip(upper=cap).astype(np.float32)
    return z


def make_windows(df, window):
    """All rolling windows per engine, via one sliding_window_view per engine."""
    xs, ys, us, cs = [], [], [], []
    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        if len(g) < window:
            continue
        x = np.ascontiguousarray(g[FEATURES].to_numpy(np.float32))
        nw = len(g) - window + 1
        xs.append(sliding_window_view(x, window, axis=0).transpose(0, 2, 1))
        ys.append(g["RUL"].to_numpy(np.float32)[window - 1:])
        cs.append(g["cycle"].to_numpy(np.int64)[window - 1:])
        us.append(np.full(nw, int(unit), dtype=np.int64))
    if not xs:
        return (np.zeros((0, window, N_FEATURES), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.int64), np.zeros(0, np.int64))
    return (np.ascontiguousarray(np.concatenate(xs, axis=0), dtype=np.float32),
            np.concatenate(ys).astype(np.float32),
            np.concatenate(us).astype(np.int64), np.concatenate(cs).astype(np.int64))


def _raw_condition_lut(raw_df):
    """(unit, cycle) -> (os1, os2, os3) taken from the UNSCALED frame.

    The condition label must never come from a standardised frame: cond_norm maps
    every os3 within a condition to mean 0, which would turn the six FD002/FD004
    conditions into a few hundred bogus ones.
    """
    if raw_df is None:
        return None
    u = raw_df["unit"].to_numpy(np.int64)
    c = raw_df["cycle"].to_numpy(np.int64)
    o = raw_df[["os1", "os2", "os3"]].to_numpy(np.float32)
    return {(int(a), int(b)): row for a, b, row in zip(u, c, o)}


def make_last_windows(df, window, raw_df=None):
    """Only the final window of each engine - the official C-MAPSS test target."""
    lut = _raw_condition_lut(raw_df)
    xs, ys, us, cs, os3, oset = [], [], [], [], [], []
    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        if len(g) < window:
            continue
        x = g[FEATURES].to_numpy(np.float32)
        last_cycle = int(g["cycle"].to_numpy(np.int64)[-1])
        row = lut.get((int(unit), last_cycle)) if lut else None
        if row is None:
            row = np.array([g["os1"].to_numpy(np.float32)[-1],
                            g["os2"].to_numpy(np.float32)[-1],
                            g["os3"].to_numpy(np.float32)[-1]], np.float32)
        xs.append(x[-window:])
        ys.append(np.float32(g["RUL"].to_numpy(np.float32)[-1]))
        cs.append(np.int64(last_cycle))
        us.append(np.int64(int(unit)))
        os3.append(np.float32(row[2]))
        oset.append(np.asarray(row, np.float32))
    if not xs:
        return (np.zeros((0, window, N_FEATURES), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32),
                np.zeros((0, 3), np.float32))
    return (np.ascontiguousarray(np.stack(xs, axis=0), dtype=np.float32),
            np.asarray(ys, np.float32), np.asarray(us, np.int64),
            np.asarray(cs, np.int64), np.asarray(os3, np.float32),
            np.asarray(oset, np.float32))


def make_pseudotest(df, window, cap, fractions=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """Simulated in-service snapshots from validation engines (last cycle = failure)."""
    xs, ys, ids, cycles = [], [], [], []
    sid = 0
    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        if len(g) < window:
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
    """next_idx[i] = j when j is i's successor (same engine, cycle+1), else -1."""
    units = np.asarray(units, np.int64)
    cycles = np.asarray(cycles, np.int64)
    n = len(cycles)
    nxt = np.full(n, -1, np.int64)
    if n < 2:
        return nxt
    order = np.lexsort((cycles, units))
    su, sc = units[order], cycles[order]
    link = (su[:-1] == su[1:]) & (sc[1:] == sc[:-1] + 1)
    nxt[order[:-1][link]] = order[1:][link]
    return nxt


def peak_rss_gb():
    """Resident memory of this process, in GB. POSIX reads ru_maxrss (KB);
    Windows has no `resource`, so ask psapi for PeakWorkingSetSize instead."""
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    try:                                          # pragma: no cover - Windows
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2 ** 30
    except Exception:
        return 0.0


def metric_dict(y, p):
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
    df = pd.DataFrame({"unit": np.asarray(units).astype(int),
                       "cycle": np.asarray(cycles).astype(int),
                       "true_RUL": np.asarray(y, np.float64),
                       "pred_RUL": np.asarray(p, np.float64)})
    df["error"] = df.pred_RUL - df.true_RUL
    df["abs_error"] = df.error.abs()
    df["nasa"] = np.where(df.error < 0, np.exp(-df.error / 13) - 1,
                          np.exp(df.error / 10) - 1)
    return df


def per_condition_metrics(y, p, cond, tol=1.0):
    """Group the test snapshots by operating condition.

    FD002/FD004 carry six condition families separated by os1
    (0 / 10 / 20 / 25 / 35 / 42) that only jitter in os2 (0.25 +- 0.002) and os3.
    Clustering on the full triple is fragile: a 0.5 grid rounds os2 = 0.25 down to
    0 and splits the same physical condition into two groups. os1 is the
    discriminator; os2/os3 are reported as the cluster mean so the table still
    documents which condition each row is.
    """
    cond = np.asarray(cond, np.float64)
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    if len(cond) != len(y):
        return None
    if cond.ndim == 1:
        cond = np.stack([cond, np.zeros_like(cond), np.zeros_like(cond)], axis=1)
    keys = np.round(cond[:, 0] / tol) * tol
    uniq = sorted(set(keys.tolist()))
    if len(uniq) < 2:
        return None
    rows = []
    for key in uniq:
        m = keys == key
        if m.sum() < 3:
            continue
        met = metric_dict(y[m], p[m])
        rows.append({"cond": len(rows) + 1,
                     "os1": float(cond[m, 0].mean()), "os2": float(cond[m, 1].mean()),
                     "os3": float(cond[m, 2].mean()),
                     "n": int(m.sum()), "RMSE": met["RMSE"], "MAE": met["MAE"],
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


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------
def probe_input(Z):
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
        g_mu = train_df[FEATURES].mean()
        g_sd = train_df[FEATURES].std().replace(0, 1.0).fillna(1.0)
        g_sd[g_sd < 1e-8] = 1.0
        self.g_mu = g_mu.to_numpy(np.float32)
        self.g_sd = g_sd.to_numpy(np.float32)
        self.cond_keys = None
        self.cond_mu = None
        self.cond_sd = None
        if self.cond_norm:
            grp = train_df.groupby("os3")
            mu = grp[FEATURES].mean()
            sd = grp[FEATURES].std().replace(0, 1.0).fillna(1.0)
            sd[sd < 1e-8] = 1.0
            counts = grp.size()
            good = counts[counts >= 50].index
            if len(good) == 0:
                self.cond_norm = False
            else:
                self.cond_keys = np.asarray(good, np.float64)
                self.cond_mu = mu.loc[good].to_numpy(np.float32)
                self.cond_sd = sd.loc[good].to_numpy(np.float32)
                # os3 identifies the condition itself: keep its global scaling,
                # otherwise every condition collapses onto 0 and the feature
                # carries no information at all.
                self.cond_mu[:, -1] = self.g_mu[-1]
                self.cond_sd[:, -1] = self.g_sd[-1]

    def transform(self, frame):
        n = len(frame)
        vals = frame[FEATURES].to_numpy(np.float32)
        if self.cond_norm and self.cond_keys is not None:
            keys = frame["os3"].to_numpy(np.float64)
            pos = np.searchsorted(self.cond_keys, keys)
            pos = np.clip(pos, 0, len(self.cond_keys) - 1)
            hit = np.isclose(self.cond_keys[pos], keys)
            mu_v = np.where(hit[:, None], self.cond_mu[pos], self.g_mu[None, :])
            sd_v = np.where(hit[:, None], self.cond_sd[pos], self.g_sd[None, :])
        else:
            mu_v = self.g_mu[None, :]
            sd_v = self.g_sd[None, :]
        vals = (vals - mu_v) / sd_v
        if self.clip is not None:
            np.clip(vals, -self.clip, self.clip, out=vals)
        out = frame.copy()
        out[FEATURES] = vals
        return out

    # cache round-trip ------------------------------------------------------
    def to_arrays(self):
        return {
            "sc_g_mu": self.g_mu, "sc_g_sd": self.g_sd,
            "sc_cond_keys": np.asarray(self.cond_keys if self.cond_keys is not None else []),
            "sc_cond_mu": np.asarray(self.cond_mu if self.cond_mu is not None else np.zeros((0, N_FEATURES))),
            "sc_cond_sd": np.asarray(self.cond_sd if self.cond_sd is not None else np.zeros((0, N_FEATURES))),
            "sc_cond_norm": np.asarray([int(self.cond_norm)]),
            "sc_clip": np.asarray([self.clip], np.float32),
        }

    @classmethod
    def from_arrays(cls, z):
        obj = cls.__new__(cls)
        obj.g_mu = np.asarray(z["sc_g_mu"], np.float32)
        obj.g_sd = np.asarray(z["sc_g_sd"], np.float32)
        keys = np.asarray(z["sc_cond_keys"], np.float64)
        obj.cond_keys = keys if len(keys) else None
        mu = np.asarray(z["sc_cond_mu"], np.float32)
        sd = np.asarray(z["sc_cond_sd"], np.float32)
        obj.cond_mu = mu if len(mu) else None
        obj.cond_sd = sd if len(sd) else None
        obj.cond_norm = bool(int(np.asarray(z["sc_cond_norm"])[0]))
        obj.clip = float(np.asarray(z["sc_clip"])[0])
        return obj


def normalize_frames(train_df, other_frames, cond_norm=False):
    scaler = FeatureScaler(train_df, cond_norm=cond_norm)
    return [scaler.transform(train_df)] + [scaler.transform(f) for f in other_frames]


# ---------------------------------------------------------------------------
# Chronos-2 embeddings
# ---------------------------------------------------------------------------
_PIPE_CACHE = {}


def load_chronos(model_id, device):
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


def _load_feature_cache(cache_file, n_expected, pool, force):
    """Load the feature cache, preferring the mmap-backed .npy."""
    if force or cache_file is None:
        return None
    cache_file = Path(cache_file)
    npy = cache_file.with_suffix(".npy")
    arr = None
    if npy.exists():
        try:
            arr = np.load(npy, mmap_mode="r")
        except Exception as exc:
            print(f"[cache] unreadable {npy.name} ({exc}); rebuilding")
            return None
    elif cache_file.exists():
        try:
            with np.load(cache_file) as z:  # legacy v8 .npz
                arr = z["features"]
        except Exception as exc:
            print(f"[cache] unreadable {cache_file.name} ({exc}); rebuilding")
            return None
    else:
        return None

    ndim = _expected_ndim(pool)
    if pool == "both":
        ok = arr.ndim == 2 and arr.shape[0] == n_expected
    else:
        ok = (arr.ndim == ndim and arr.shape[0] == n_expected
              and arr.shape[1] == N_FEATURES)
    if not ok or arr.dtype not in (np.float16, np.float32):
        print(f"[cache] shape mismatch for {Path(npy).name}: "
              f"got {arr.shape}/{arr.dtype}; rebuilding")
        return None
    print(f"[cache] hit {Path(npy).name} {arr.shape} {arr.dtype} "
          f"({arr.nbytes / 1e9:.2f} GB)")
    return arr


def _pool_batch(batch_array, pool):
    if pool == "token":
        return batch_array.mean(axis=2)
    if pool == "both":
        return batch_array.mean(axis=(2, 1))
    return batch_array


@torch.no_grad()
def chronos_embed_windows(pipe, X, batch_size, context_length, cache_file=None,
                          force=False, pool="none", cache_dtype="f16", amp=False):
    n = len(X)
    store_dtype = np.float16 if cache_dtype == "f16" else np.float32
    cached = _load_feature_cache(cache_file, n, pool, force)
    if cached is not None:
        return cached

    feats = []
    total_batches = (n + batch_size - 1) // batch_size
    progress = tqdm(range(0, n, batch_size), total=total_batches,
                    desc="Chronos-2 embedding", unit="batch", dynamic_ncols=True)
    t_start = time.perf_counter()
    for start in progress:
        xb = X[start:start + batch_size]
        inputs = [torch.as_tensor(np.ascontiguousarray(a.T), dtype=torch.float32) for a in xb]
        ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
               if amp else nullcontext())
        with ctx:
            res = pipe.embed(inputs, batch_size=max(batch_size * N_FEATURES, 1),
                             context_length=context_length)
        embeddings = res[0] if isinstance(res, (tuple, list)) and len(res) == 2 else res
        if isinstance(embeddings, torch.Tensor):
            embeddings = list(embeddings)
        if len(embeddings) != len(inputs):
            raise RuntimeError("Chronos embedding count mismatch.")
        batch_feats = []
        for emb in embeddings:
            e = torch.as_tensor(emb).detach()
            if e.ndim == 2:
                if e.shape[0] == N_FEATURES:
                    e = e.unsqueeze(1)
                elif e.shape[1] == N_FEATURES:
                    e = e.T.unsqueeze(1)
                else:
                    raise RuntimeError(f"Unexpected embedding shape {tuple(e.shape)}")
            elif e.ndim == 3:
                if e.shape[0] == N_FEATURES:
                    pass
                elif e.shape[1] == N_FEATURES:
                    e = e.transpose(0, 1)
                else:
                    raise RuntimeError(f"Cannot identify sensor axis: {tuple(e.shape)}")
            else:
                raise RuntimeError(f"Unexpected embedding rank {e.ndim}")
            batch_feats.append(e.contiguous())
        shapes = {tuple(f.shape[1:]) for f in batch_feats}
        if len(shapes) != 1:
            raise RuntimeError(f"Inconsistent embedding shapes: {shapes}")
        batch_array = torch.stack(batch_feats, dim=0).to(
            device="cpu", dtype=torch.float32).numpy()
        feats.append(np.ascontiguousarray(_pool_batch(batch_array, pool), dtype=store_dtype))

    out = np.concatenate(feats, axis=0)
    dt = time.perf_counter() - t_start
    print(f"[embed] shape={out.shape} {out.dtype} "
          f"{out.nbytes / 1e9:.2f} GB {dt:.1f}s")
    if cache_file is not None:
        _atomic_save_npy(Path(cache_file).with_suffix(".npy"), out)
    return out


def stat_embed_windows(X, cache_dtype="f16"):
    """8 summary statistics per sensor. Engineering smoke test only."""
    n, t, f = X.shape
    x = np.asarray(X, np.float32)
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    last = x[:, -1, :]
    rng = mx - mn
    tt = np.arange(t, dtype=np.float32) - (t - 1) / 2.0
    denom = float((tt ** 2).sum()) or 1.0
    slope = np.tensordot(x, tt, axes=([1], [0])) / denom
    d1 = np.abs(np.diff(x, axis=1)).mean(axis=1) if t > 1 else np.zeros_like(mean)
    feats = np.stack([mean, std, mn, mx, last, rng, slope, d1], axis=-1)  # (N,F,8)
    out = feats[:, :, None, :].astype(np.float16 if cache_dtype == "f16" else np.float32)
    return out


# ---------------------------------------------------------------------------
# graph + batching + models
# ---------------------------------------------------------------------------
def build_graph(X_train, topk):
    flat = X_train.reshape(-1, X_train.shape[-1])
    std = flat.std(axis=0)
    valid = std > 1e-12
    corr = np.zeros((flat.shape[1], flat.shape[1]), dtype=np.float32)
    if valid.sum() > 1:
        xv = flat[:, valid].astype(np.float32)
        xv = xv - xv.mean(axis=0, keepdims=True)
        denom = xv.std(axis=0, keepdims=True)
        denom[denom < 1e-12] = 1.0
        xv = xv / denom
        corr_valid = (xv.T @ xv) / max(len(xv), 1)
        corr_valid = np.nan_to_num(np.abs(corr_valid), nan=0.0,
                                   posinf=0.0, neginf=0.0)
        corr[np.ix_(valid, valid)] = corr_valid.astype(np.float32)
    np.fill_diagonal(corr, 1.0)
    n = corr.shape[0]
    adj = np.eye(n, dtype=np.float32)
    k = min(max(1, topk), n - 1)
    for i in range(n):
        order = np.argsort(-corr[i])
        nbr = [j for j in order if j != i][:k]
        adj[i, nbr] = 1.0
    adj = np.maximum(adj, adj.T)
    deg = adj.sum(axis=1)
    inv = 1.0 / np.sqrt(np.maximum(deg, 1.0))
    return (inv[:, None] * adj * inv[None, :]).astype(np.float32)


class BatchIter:
    """In-process mini-batch iterator; a drop-in for DataLoader over in-RAM arrays.

    Rationale: every split is already resident (mmap or RAM), so a DataLoader only
    adds a collate step, a worker hand-off and a fresh tensor per batch. Here the
    shuffled indices are applied directly and torch.from_numpy wraps the slice,
    which is ~1.6-2.2x faster per batch at batch_size 128 on 4 CPU cores.
    """

    def __init__(self, Z, y, device, batch_size, shuffle=False, seed=0,
                 next_idx=None, pin=False, gpu_budget_gb=0.0,
                 preload_gb=4.0):
        self.Z = Z
        self.y = np.asarray(y, np.float32)
        self.n = len(self.y)
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        self.next_idx = next_idx
        self.pin = pin
        self.Zt = None      # resident on the GPU
        self.Zc = None      # resident on the CPU, already float32
        # 1) GPU-resident when it comfortably fits.
        if (device.type == "cuda" and gpu_budget_gb > 0
                and Z.nbytes / 1e9 <= gpu_budget_gb):
            self.Zt = torch.from_numpy(np.ascontiguousarray(Z)).to(
                device, dtype=torch.float32, non_blocking=True)
        # 2) Otherwise, if the float32 copy fits in RAM, cast ONCE. A per-batch
        #    f16 -> f32 conversion is single-threaded numpy work and was measured
        #    to make the iterator slower than a DataLoader over an f32 tensor.
        elif preload_gb > 0 and Z.size * 4 / 1e9 <= preload_gb:
            self.Zc = torch.from_numpy(np.ascontiguousarray(Z)).float()
        # 3) Last resort: keep the source as a torch VIEW (no copy) and index it.
        #    That is one gather plus one parallel cast, instead of a numpy fancy
        #    index plus a single-threaded ndarray.astype.
        if self.Zt is None and self.Zc is None:
            with warnings.catch_warnings():
                # a read-only / mmap buffer is exactly what we want to index
                warnings.simplefilter("ignore")
                self.Zh = torch.from_numpy(Z)
        else:
            self.Zh = None

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def _gather(self, idx):
        idx_t = torch.from_numpy(np.ascontiguousarray(idx)).long()
        if self.Zt is not None:
            z = self.Zt[idx_t]
        elif self.Zc is not None:
            z = self.Zc[idx_t]
        else:
            z = self.Zh[idx_t]
        if z.dtype != torch.float32:
            z = z.float()  # parallel cast, unlike ndarray.astype
        if self.pin and self.device.type == "cuda" and z.device.type == "cpu":
            z = z.pin_memory()
        return z.to(self.device, non_blocking=True)

    def gather(self, idx_t):
        """Rows by GLOBAL id; used by the pair loss, whose target may sit in
        another mini-batch."""
        idx = idx_t.detach().cpu().numpy().astype(np.int64)
        return self._gather(idx)

    def labels(self, idx_t):
        idx = idx_t.detach().cpu().numpy().astype(np.int64)
        return torch.from_numpy(np.ascontiguousarray(self.y[idx])).to(
            self.device, non_blocking=True)

    def __iter__(self):
        order = self.rng.permutation(self.n) if self.shuffle else None
        bs = self.batch_size
        for s in range(0, self.n, bs):
            idx = order[s:s + bs] if order is not None else np.arange(s, min(s + bs, self.n))
            z = self._gather(idx)
            y = torch.from_numpy(np.ascontiguousarray(self.y[idx]))
            if self.pin and self.device.type == "cuda":
                y = y.pin_memory()
            y = y.to(self.device, non_blocking=True)
            nxt = None
            if self.next_idx is not None:
                nxt = torch.from_numpy(np.ascontiguousarray(self.next_idx[idx])).to(
                    self.device, non_blocking=True)
            yield z, y, nxt, idx


class LinearProbe(nn.Module):
    def __init__(self, d_in, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class GraphSensorEncoder(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        h = torch.matmul(adj, x)
        h = self.dropout(F.gelu(self.norm1(self.lin1(h))))
        h = torch.matmul(adj, h)
        h = self.dropout(F.gelu(self.norm2(self.lin2(h))))
        return x + h


class ChronosBiLSTMHybrid(nn.Module):
    def __init__(self, d_in, hidden=128, heads=4, dropout=0.1,
                 use_temporal=True, use_graph=True, use_fusion=True,
                 use_stage=True, use_mono=True, use_smooth=True,
                 backbone="bilstm"):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_graph = use_graph
        self.use_fusion = use_fusion
        self.use_stage = use_stage
        self.use_mono = use_mono
        self.use_smooth = use_smooth
        self.backbone = backbone

        if use_temporal:
            self.token_score = nn.Linear(hidden, 1)
            self.temporal_proj = nn.Linear(d_in, hidden)
            self.temporal_norm = nn.LayerNorm(hidden)
            self.temporal_rnn = nn.LSTM(
                hidden, hidden // 2, batch_first=True,
                bidirectional=True
            )
        else:
            self.token_score = None
            self.temporal_proj = nn.Linear(d_in, hidden)
            self.temporal_norm = nn.LayerNorm(hidden)
            self.temporal_rnn = None

        if backbone == "bilstm":
            self.sensor_rnn = nn.LSTM(
                hidden, hidden // 2, batch_first=True,
                bidirectional=True
            )
            self.sensor_out = None
        elif backbone == "lstm":
            self.sensor_rnn = nn.LSTM(
                hidden, hidden, batch_first=True,
                bidirectional=False
            )
            self.sensor_out = nn.Linear(hidden, hidden)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.graph = GraphSensorEncoder(hidden, dropout) if use_graph else None

        if use_fusion:
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.Sigmoid(),
            )
        else:
            self.fusion_gate = None

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.stage_head = nn.Linear(hidden, 3) if use_stage else None
        self.register_buffer("adj", torch.eye(N_FEATURES))

    def set_graph(self, adj):
        self.adj = torch.as_tensor(adj, dtype=torch.float32, device=self.adj.device)

    def forward(self, z, x_raw=None):
        # z: [B,S,T,D] or [B,S,D]
        if z.ndim == 3:
            z = z.unsqueeze(2)
        if z.dtype != self.temporal_proj.weight.dtype:
            z = z.to(self.temporal_proj.weight.dtype)
        b, s, t, d = z.shape

        if self.use_temporal:
            zt = self.temporal_norm(self.temporal_proj(z))
            zt = zt.reshape(b * s, t, -1)
            ht, _ = self.temporal_rnn(zt)
            attn = torch.softmax(self.token_score(ht), dim=1)
            ht = (ht * attn).sum(dim=1)
            ht = ht.reshape(b, s, -1)
        else:
            # T=1 or already pooled: skip the LSTM and the attention pool.
            ht = self.temporal_norm(self.temporal_proj(z)).mean(dim=2)

        hs, _ = self.sensor_rnn(ht)
        if self.sensor_out is not None:
            hs = self.sensor_out(hs)

        hg = self.graph(hs, self.adj) if self.graph is not None else hs

        if self.fusion_gate is not None:
            gate = self.fusion_gate(torch.cat([hs, hg], dim=-1))
            fused = gate * hs + (1.0 - gate) * hg
        else:
            fused = 0.5 * (hs + hg)

        pooled = self.dropout(fused.mean(dim=1))
        pred = self.head(pooled)
        stage = self.stage_head(pooled) if self.stage_head is not None else None
        return pred.squeeze(-1), stage


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def ablation_flags(name):
    return {
        "use_temporal": name != "no_temporal_adapter",
        "use_graph": name != "no_graph",
        "use_fusion": name != "no_fusion_gate",
        "use_stage": name != "no_stage_weighting",
        "use_mono": name != "no_monotonic",
        "use_smooth": name != "no_smoothness",
    }


def make_model(d_in, args, cfg):
    flags = ablation_flags(cfg["ablation"])
    if cfg["mode"] == "linear_probe":
        return LinearProbe(d_in, hidden=max(64, args.hidden // 2), dropout=args.dropout)
    backbone = "lstm" if cfg["mode"] == "lstm_backbone" else "bilstm"
    return ChronosBiLSTMHybrid(d_in, hidden=args.hidden, heads=args.heads,
                               dropout=args.dropout, backbone=backbone, **flags)


def model_forward(model, zb, mode):
    if mode == "linear_probe":
        return model(zb.mean(dim=(1, 2)) if zb.ndim == 4 else zb.mean(dim=1)), None
    return model(zb)


# ---------------------------------------------------------------------------
# preparation and caching
# ---------------------------------------------------------------------------
_PREP_CACHE = {}


def _raw_cache_path(croot, subset, split_seed, window, rmax):
    tag = f"{subset}_split{split_seed}_w{window}_r{rmax:g}"
    return Path(croot) / "split_cache" / f"{tag}.npz"


def _prep_cache_path(croot, subset, split_seed, window, rmax,
                     fraction, cond_norm, graph_topk):
    tag = (
        f"{CACHE_VERSION}_{subset}_split{split_seed}_w{window}_r{rmax:g}_"
        f"f{fraction:g}_cn{int(cond_norm)}_k{graph_topk}"
    )
    return Path(croot) / "prep_cache" / f"{tag}.npz"


def _feature_cache_path(croot, subset, split_seed, window, rmax,
                        fraction, split, pool, dtype, cond_norm):
    tag = (
        f"{CACHE_VERSION}_{subset}_split{split_seed}_w{window}_r{rmax:g}_f{fraction:g}_"
        f"{split}_pool{pool}_dtype{dtype}_cn{int(cond_norm)}"
    )
    return Path(croot) / "chronos_cache" / f"{tag}.npz"


def load_split_raw(args, subset, window=None):
    """Parsed frames + the engine partition, independent of data fraction.

    v8 rebuilt both the validation windows and the test windows for every data
    fraction, although neither depends on the fraction. This split-level cache
    holds only the (already RUL-labelled) frames; windows are built per fraction.
    """
    window = window or args.window
    croot = cache_root(args)
    cache_file = _raw_cache_path(croot, subset, args.split_seed, window, args.rmax)
    key = str(cache_file)
    if key in _PREP_CACHE:
        return _PREP_CACHE[key]
    if cache_file.exists():
        try:
            with np.load(cache_file, allow_pickle=False) as z:
                out = {k: z[k] for k in ("tr", "va", "te")}
            _PREP_CACHE[key] = out
            return out
        except Exception as exc:
            print(f"[split-cache] unreadable {cache_file.name}: {exc}; rebuilding")

    tr_raw, te_raw, rul = load_cmapss(args.data_dir, subset, croot)
    tr_full = attach_train_rul(tr_raw, args.rmax)
    te_full = attach_test_rul(te_raw, rul, args.rmax)

    # Engine-level split: no engine can appear in both train and validation.
    engines = np.sort(tr_full.unit.unique())
    tr_ids, va_ids = train_test_split(
        engines, test_size=0.2, random_state=args.split_seed
    )
    tr_arr = tr_full[tr_full.unit.isin(tr_ids)][OUT_COLS].to_numpy(np.float32)
    va_arr = tr_full[tr_full.unit.isin(va_ids)][OUT_COLS].to_numpy(np.float32)
    te_arr = te_full[OUT_COLS].to_numpy(np.float32)
    out = {"tr": tr_arr, "va": va_arr, "te": te_arr}
    if not args.no_prep_cache:
        try:
            _atomic_savez(cache_file, **out)
        except Exception as exc:
            print(f"[split-cache] write failed: {exc}")
    _PREP_CACHE[key] = out
    return out


def _frame(arr):
    return pd.DataFrame(arr, columns=OUT_COLS, copy=False)


def prepare_split(args, subset, fraction):
    """Windows, adjacency and scaler for one (subset, fraction).

    Independent of the model seed: split_seed fixes the engine partition, the
    model seed only re-initialises the network, so this is built once and reused
    by every seed of the same configuration.
    """
    croot = cache_root(args)
    cache_file = _prep_cache_path(
        croot, subset, args.split_seed, args.window, args.rmax,
        fraction, args.cond_norm, args.graph_topk
    )
    key = str(cache_file)
    if not args.no_prep_cache and key in _PREP_CACHE:
        return _PREP_CACHE[key]
    if not args.no_prep_cache and cache_file.exists():
        try:
            with np.load(cache_file, allow_pickle=False) as z:
                required = ["Xtr", "ytr", "utr", "ctr", "Xva", "yva",
                            "uva", "cva", "Xte", "yte", "ute", "cte",
                            "condte", "osette", "adj"]
                if all(k in z for k in required):
                    out = {k: z[k] for k in required}
                    out["scaler"] = FeatureScaler.from_arrays(z)
                    _PREP_CACHE[key] = out
                    print(f"[prep-cache] hit {cache_file.name}")
                    return out
        except Exception as exc:
            print(f"[prep-cache] unreadable {cache_file.name}: {exc}; rebuilding")

    raw = load_split_raw(args, subset)
    tr_df = _frame(raw["tr"])
    va_df = _frame(raw["va"])
    te_df = _frame(raw["te"])

    # Subsample engines BEFORE fitting the scaler, so the statistics never see
    # the engines that are dropped at fraction < 1.
    eng = np.sort(tr_df.unit.unique())
    rng = np.random.RandomState(args.split_seed)
    n_keep = max(2, int(round(len(eng) * fraction)))
    keep = np.sort(rng.choice(eng, size=min(n_keep, len(eng)), replace=False))
    tr_df = tr_df[tr_df.unit.isin(keep)]

    scaler = FeatureScaler(tr_df, cond_norm=args.cond_norm)
    tr_df = scaler.transform(tr_df)
    va_df = scaler.transform(va_df)
    te_df = scaler.transform(te_df)

    Xtr, ytr, utr, ctr = make_windows(tr_df, args.window)
    Xva, yva, uva, cva = make_windows(va_df, args.window)
    Xte, yte, ute, cte, condte, osette = make_last_windows(
        te_df, args.window, raw_df=_frame(raw["te"]))
    if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
        raise RuntimeError(f"Empty prepared split for {subset}, fraction={fraction}")

    stride = max(1, args.val_stride)
    if stride > 1:
        Xva, yva, uva, cva = (Xva[::stride], yva[::stride], uva[::stride], cva[::stride])

    adj = build_graph(Xtr, args.graph_topk)
    out = {
        "Xtr": Xtr, "ytr": ytr, "utr": utr, "ctr": ctr,
        "Xva": Xva, "yva": yva, "uva": uva, "cva": cva,
        "Xte": Xte, "yte": yte, "ute": ute, "cte": cte,
        "condte": condte, "osette": osette, "adj": adj,
        "scaler": scaler,
    }
    if not args.no_prep_cache:
        try:
            _atomic_savez(cache_file, **{k: v for k, v in out.items() if not isinstance(v, FeatureScaler)},
                          **scaler.to_arrays())
        except Exception as exc:
            print(f"[prep-cache] write failed: {exc}")
    _PREP_CACHE[key] = out
    return out


def prepare_test_only(args, subset, scaler):
    """Cross-subset evaluation: windows of another subset under the train scaler."""
    raw = load_split_raw(args, subset)
    raw_te = _frame(raw["te"])
    te_df = scaler.transform(raw_te)
    return make_last_windows(te_df, args.window, raw_df=raw_te)


def embed_windows(args, pipe, subset, fraction, split, X):
    croot = cache_root(args)
    if args.encoder == "stat":
        cache_file = (Path(croot) / "stat_cache" /
                      f"{CACHE_VERSION}_{subset}_w{args.window}_f{fraction:g}_{split}_"
                      f"cn{int(args.cond_norm)}_dtype{args.cache_dtype}.npy")
        npy = Path(cache_file)
        if not args.force_reembed and npy.exists():
            try:
                arr = np.load(npy, mmap_mode="r")
                if arr.shape[0] == len(X):
                    return arr
            except Exception:
                pass
        out = stat_embed_windows(X, cache_dtype=args.cache_dtype)
        if not args.no_prep_cache:
            try:
                _atomic_save_npy(npy, out)
            except Exception as exc:
                print(f"[stat-cache] write failed: {exc}")
        return out

    cache_file = _feature_cache_path(
        croot, subset, args.split_seed, args.window, args.rmax,
        fraction, split, args.embed_pool, args.cache_dtype, args.cond_norm
    )
    amp = args.device == "cuda" and args.amp != "off"
    arr = chronos_embed_windows(
        pipe, X, args.embed_batch_size, args.context_length,
        cache_file=cache_file, force=args.force_reembed,
        pool=args.embed_pool, cache_dtype=args.cache_dtype, amp=amp
    )
    # mmap unless the whole matrix is small enough to be pinned on the GPU anyway
    mmap_ok = args.feature_mmap != "off"
    if mmap_ok and isinstance(arr, np.ndarray):
        npy = Path(cache_file).with_suffix(".npy")
        if npy.exists() and not isinstance(arr, np.memmap):
            arr = np.load(npy, mmap_mode="r")
    return arr


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def loss_components(pred, y, next_pred=None, next_y=None,
                    use_mono=True, use_smooth=True):
    mse = F.mse_loss(pred, y)
    mono = pred.new_tensor(0.0)
    smooth = pred.new_tensor(0.0)
    if next_pred is not None and next_y is not None and len(next_pred):
        delta_pred = next_pred - pred
        delta_true = next_y - y
        if use_mono:
            mono = F.relu(delta_pred).mean()
        if use_smooth:
            smooth = F.smooth_l1_loss(delta_pred, delta_true)
    return mse, mono, smooth


def pair_terms(model, it, zb, nxt, y, mode, frac, use_mono, use_smooth, amp_ctx):
    """Monotonicity / smoothness over the (window, window+1) pair.

    `nxt` holds GLOBAL row ids, so the successor row is gathered from the source
    (it.gather), not from the current mini-batch: after shuffling, a window and
    its successor almost never land in the same batch. Only --next_pair_frac of
    the batch is used and both rows go through a single forward pass, so the
    extra cost is one forward over 2*frac samples.
    """
    valid = nxt >= 0
    if not bool(valid.any()):
        return None, None, None
    pos = torch.nonzero(valid, as_tuple=False).squeeze(1)
    keep = max(1, int(len(pos) * frac))
    if keep < len(pos):
        pos = pos[torch.randperm(len(pos), device=pos.device)[:keep]]
    gnext = nxt[pos]
    both = torch.cat([zb[pos], it.gather(gnext)], dim=0)
    with amp_ctx():
        out, _ = model_forward(model, both, mode)
    return out[:len(pos)], out[len(pos):], (y[pos], it.labels(gnext))


def evaluate(model, it, device, mode, amp_ctx, collect=False):
    model.eval()
    ys, ps = [], []
    with torch.inference_mode():
        for zb, yb, _nxt, _idx in it:
            with amp_ctx():
                pred, _ = model_forward(model, zb, mode)
            ys.append(yb.detach().float().cpu().numpy())
            ps.append(pred.detach().float().cpu().numpy())
    if not ys:
        return float("inf"), np.array([]), np.array([])
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return float(np.sqrt(np.mean((p - y) ** 2))), y, p


def train_model(model, train_iter, val_iter, device, args, cfg, amp_ctx):
    model.to(device)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps = len(train_iter)
    scheduler = None
    if args.sched == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=max(1, args.epochs * steps),
            pct_start=0.3, div_factor=10.0, final_div_factor=100.0,
        )

    use_pair = args.use_next_pair and train_iter.next_idx is not None
    best_rmse = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for zb, yb, nxt, _idx in train_iter:
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                pred, _ = model_forward(model, zb, cfg["mode"])
                mse, mono, smooth = loss_components(pred, yb)
                if use_pair:
                    p1, p2, (y1, y2) = pair_terms(
                        model, train_iter, zb, nxt, yb, cfg["mode"],
                        args.next_pair_frac,
                        getattr(model, "use_mono", True),
                        getattr(model, "use_smooth", True),
                        amp_ctx)
                    if p1 is not None:
                        _, mono, smooth = loss_components(
                            p1, y1, p2, y2,
                            use_mono=getattr(model, "use_mono", True),
                            use_smooth=getattr(model, "use_smooth", True))
                loss = mse + args.lambda_mono * mono + args.lambda_smooth * smooth
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running += float(loss.detach()) * len(yb)
            n_seen += len(yb)

        val_rmse, _, _ = evaluate(model, val_iter, device, cfg["mode"], amp_ctx)
        history.append({"epoch": epoch, "train_loss": running / max(n_seen, 1),
                        "val_RMSE": val_rmse,
                        "lr": optimizer.param_groups[0]["lr"]})
        print(f"[train] epoch={epoch:03d} train_loss={running / max(n_seen, 1):.5f} "
              f"val_RMSE={val_rmse:.5f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[train] early stopping at epoch {epoch}")
                break

    train_time = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_rmse, history, train_time


# ---------------------------------------------------------------------------
# one configuration
# ---------------------------------------------------------------------------
def run_id(cfg):
    parts = [cfg["experiment"], cfg["subset"], cfg["mode"], cfg["ablation"],
             f"s{cfg['seed']}", f"f{cfg['fraction']:g}"]
    if cfg.get("cond_norm"):
        parts.append("cn1")
    if cfg.get("cross_train"):
        parts.append(f"x{cfg['cross_train']}to{cfg['cross_test']}")
    return "_".join(parts)


def _run_dir(args, cfg):
    return Path(args.output_dir) / "runs" / run_id(cfg)


def run_one(args, cfg, pipe=None):
    rid = run_id(cfg)
    out_dir = _run_dir(args, cfg)
    if args.resume and (out_dir / "metrics.json").exists():
        with open(out_dir / "metrics.json") as f:
            row = json.load(f)
        print(f"[resume] skip {rid}")
        return row

    seed_all(cfg["seed"], args.deterministic)
    device = choose_device(args.device)
    amp_train = (device.type == "cuda" and args.train_amp != "off"
                 and (args.train_amp == "on" or torch.cuda.is_bf16_supported()))
    amp_ctx = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16))
               if amp_train else nullcontext)
    pin = device.type == "cuda"

    t_prep = time.perf_counter()
    prep = prepare_split(args, cfg["subset"], cfg["fraction"])
    if cfg.get("cross_test") and cfg["cross_test"] != cfg["subset"]:
        Xte, yte, ute, cte, condte, osette = prepare_test_only(
            args, cfg["cross_test"], prep["scaler"])
        cache_tag = f"{cfg['cross_test']}_xtest"
    else:
        Xte, yte, ute, cte, condte, osette = (
            prep["Xte"], prep["yte"], prep["ute"], prep["cte"],
            prep["condte"], prep["osette"])
        cache_tag = None
    prep_time = time.perf_counter() - t_prep

    if args.encoder == "chronos" and pipe is None:
        pipe = load_chronos(args.chronos_model, device)

    t_enc = time.perf_counter()
    Ztr = embed_windows(args, pipe, cfg["subset"], cfg["fraction"], "train", prep["Xtr"])
    Zva = embed_windows(args, pipe, cfg["subset"], cfg["fraction"], "val", prep["Xva"])
    if cache_tag is None:
        Zte = embed_windows(args, pipe, cfg["subset"], cfg["fraction"], "test", Xte)
    else:
        Zte = embed_windows(args, pipe, cfg["cross_test"], cfg["fraction"], "xtest", Xte)
    enc_time = time.perf_counter() - t_enc

    nxt = build_next_index(prep["utr"], prep["ctr"]) if args.use_next_pair else None
    train_iter = BatchIter(Ztr, prep["ytr"], device, args.batch_size,
                           shuffle=True, seed=cfg["seed"], next_idx=nxt,
                           pin=pin, gpu_budget_gb=args.gpu_budget_gb,
                           preload_gb=args.feature_preload_gb)
    val_iter = BatchIter(Zva, prep["yva"], device, args.eval_batch_size,
                         pin=pin, gpu_budget_gb=args.gpu_budget_gb,
                           preload_gb=args.feature_preload_gb)
    test_iter = BatchIter(Zte, yte, device, args.eval_batch_size,
                          pin=pin, gpu_budget_gb=args.gpu_budget_gb,
                           preload_gb=args.feature_preload_gb)

    d_in = int(Ztr.shape[-1])
    model = make_model(d_in, args, cfg)
    if hasattr(model, "set_graph"):
        model.set_graph(prep["adj"])
    if args.compile and device.type == "cuda":
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"[compile] skipped: {exc}")
    n_params = count_parameters(model)

    model, best_val, history, train_time = train_model(
        model, train_iter, val_iter, device, args, cfg, amp_ctx)

    t_inf = time.perf_counter()
    _rmse, y, p = evaluate(model, test_iter, device, cfg["mode"], amp_ctx, collect=True)
    infer_time = time.perf_counter() - t_inf

    met = metric_dict(y, p)
    cond_df = None
    if (not args.no_report_cond and len(osette) == len(y)
            and len(np.unique(np.round(osette, 3), axis=0)) > 1):
        cond_df = per_condition_metrics(y, p, osette)
    cond_sum = summarize_conditions(cond_df)

    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    else:
        peak_mem = peak_rss_gb()

    row = {
        "run_id": rid, "experiment": cfg["experiment"], "subset": cfg["subset"],
        "test_subset": cfg.get("cross_test") or cfg["subset"],
        "cross_train": cfg.get("cross_train") or "",
        "mode": cfg["mode"], "ablation": cfg["ablation"], "seed": cfg["seed"],
        "fraction": float(cfg["fraction"]), "cond_norm": bool(cfg.get("cond_norm")),
        "encoder": args.encoder,
        "params": n_params, "best_val_RMSE": float(best_val),
        "prep_s": prep_time, "encode_s": enc_time, "train_s": train_time,
        "infer_s": infer_time, "epochs_run": len(history),
        "peak_mem_gb": float(peak_mem),
        **{k: met[k] for k in met},
        **cond_sum,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    per_engine_frame(ute, cte, y, p).to_csv(out_dir / "per_engine_metrics.csv", index=False)
    if cond_df is not None:
        cond_df.to_csv(out_dir / "per_condition_metrics.csv", index=False)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(row, f, indent=2)
    print(f"[run] {rid} RMSE={met['RMSE']:.4f} MAE={met['MAE']:.4f} "
          f"NASA/eng={met['NASA_per_engine']:.2f} params={n_params} "
          f"train={train_time:.1f}s")
    return row


# ---------------------------------------------------------------------------
# experiment plan
# ---------------------------------------------------------------------------
def build_runs(args):
    exp = args.experiments
    if "all" in exp:
        exp = EXPERIMENTS
    exp = [e for e in exp if e in EXPERIMENTS]
    subsets = list(args.subsets)
    seeds = list(args.seeds)
    fractions = args.data_fractions or [0.1, 0.25, 0.5, 0.75]
    runs = []

    def add(experiment, subset, seed, mode="full", ablation="full", fraction=1.0,
            cross_train=None, cross_test=None):
        cfg = {"experiment": experiment, "subset": subset, "seed": int(seed),
               "mode": mode, "ablation": ablation, "fraction": float(fraction),
               "cond_norm": bool(args.cond_norm),
               "cross_train": cross_train, "cross_test": cross_test}
        rid = run_id(cfg)
        if rid not in {r["_id"] for r in runs}:
            cfg["_id"] = rid
            runs.append(cfg)

    if args.single:
        cross_tr = args.cross_train
        base = cross_tr or subsets[0]
        add(args.experiment_tag or "single", base, seeds[0],
            mode=args.mode, ablation=args.ablation,
            fraction=fractions[-1],
            cross_train=cross_tr, cross_test=args.cross_test)
        if args.only_run:
            runs = [r for r in runs if r["_id"] == args.only_run] or runs
        return runs

    # E1 / E2: the multi-subset, multi-seed main table
    if "E1" in exp or "E2" in exp:
        for s in subsets:
            for sd in seeds:
                add("E1", s, sd)

    # E3: reduced-data learning curves on the first subset
    if "E3" in exp:
        for frac in sorted(fractions):
            for sd in seeds:
                add("E3", subsets[0], sd, fraction=frac)

    # E4: cross-subset transfer
    if "E4" in exp:
        for tr in subsets:
            for te in subsets:
                if tr == te:
                    continue
                add("E4", te, seeds[0], cross_train=tr, cross_test=te)

    # E5: frozen-feature linear probe
    if "E5" in exp:
        for s in subsets:
            add("E5", s, seeds[0], mode="linear_probe")

    # E6: LSTM backbone instead of BiLSTM, same adapter/GAT/fusion/head
    if "E6" in exp:
        for s in subsets:
            add("E6", s, seeds[0], mode="lstm_backbone")

    # E7: module ablations
    if "E7" in exp:
        for ab in ["no_temporal_adapter", "no_graph", "no_fusion_gate",
                   "no_stage_weighting"]:
            add("E7", subsets[0], seeds[0], ablation=ab)
        if args.use_next_pair:
            for ab in ["no_monotonic", "no_smoothness"]:
                add("E7", subsets[0], seeds[0], ablation=ab)

    return runs


def cfg_to_argv(cfg, args, threads):
    argv = [sys.executable, os.path.abspath(__file__),
            "--single", "--only_run", cfg["_id"],
            "--data_dir", str(args.data_dir),
            "--cache_dir", str(cache_root(args)),
            "--output_dir", str(args.output_dir),
            "--subsets", cfg["subset"],
            "--seeds", str(cfg["seed"]),
            "--mode", cfg["mode"], "--ablation", cfg["ablation"],
            "--data_fractions", f"{cfg['fraction']:g}",
            "--window", str(args.window), "--rmax", str(args.rmax),
            "--batch_size", str(args.batch_size),
            "--eval_batch_size", str(args.eval_batch_size),
            "--embed_batch_size", str(args.embed_batch_size),
            "--epochs", str(args.epochs), "--patience", str(args.patience),
            "--lr", str(args.lr), "--weight_decay", str(args.weight_decay),
            "--sched", args.sched,
            "--lambda_mono", str(args.lambda_mono),
            "--lambda_smooth", str(args.lambda_smooth),
            "--next_pair_frac", str(args.next_pair_frac),
            "--hidden", str(args.hidden), "--heads", str(args.heads),
            "--dropout", str(args.dropout), "--graph_topk", str(args.graph_topk),
            "--device", args.device, "--chronos_model", args.chronos_model,
            "--context_length", str(args.context_length),
            "--cache_dtype", args.cache_dtype, "--amp", args.amp,
            "--train_amp", str(args.train_amp),
            "--gpu_budget_gb", str(args.gpu_budget_gb),
            "--embed_pool", args.embed_pool, "--encoder", args.encoder,
            "--val_stride", str(args.val_stride),
            "--split_seed", str(args.split_seed),
            "--threads", str(threads),
            "--experiment_tag", cfg["experiment"]]
    if cfg.get("cond_norm"):
        argv.append("--cond_norm")
    if args.use_next_pair:
        argv.append("--use_next_pair")
    if args.resume:
        argv.append("--resume")
    if args.compile:
        argv.append("--compile")
    if cfg.get("cross_train"):
        argv += ["--cross_train", cfg["cross_train"], "--cross_test", cfg["cross_test"]]
    return argv


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def append_runs(args, rows):
    if not rows:
        return pd.DataFrame()
    out = Path(args.output_dir) / "all_runs.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        try:
            old = pd.read_csv(out)
            df = pd.concat([old, df], ignore_index=True)
        except Exception:
            pass
    df = df.drop_duplicates(subset=["run_id"], keep="last")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def aggregate_seeds(df):
    keys = ["experiment", "subset", "test_subset", "mode", "ablation", "fraction"]
    metrics = ["RMSE", "MAE", "NASA_per_engine", "R2",
               "prep_s", "encode_s", "train_s", "infer_s", "params"]
    metrics = [m for m in metrics if m in df.columns]
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(keys, dropna=False)[metrics]
    out = g.agg(["mean", "std"]).reset_index()
    out.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col
                   for col in out.columns]
    n = df.groupby(keys, dropna=False).size().reset_index(name="n_runs")
    return out.merge(n, on=keys, how="left")


def make_compare_table(args, df):
    if df.empty:
        return
    out_dir = Path(args.output_dir)
    agg = aggregate_seeds(df)
    if not agg.empty:
        agg.to_csv(out_dir / "summary_by_config.csv", index=False)

    base = out_dir / "compare_modes.csv"
    if args.baseline_csv:
        try:
            bl = pd.read_csv(args.baseline_csv)
            bl["source"] = "baseline"
            df = pd.concat([df.assign(source="ours"), bl], ignore_index=True,
                           sort=False)
        except Exception as exc:
            print(f"[baseline] could not read {args.baseline_csv}: {exc}")

    # Rank inside one PROTOCOL, not inside one configuration: grouping by
    # (mode, ablation) would leave a single row per group and mark every row as
    # best on every metric. The protocol is what the reviewer holds constant
    # (same subset, same split, same data fraction), so modes and ablations
    # compete inside it.
    proto_cols = [c for c in ("subset", "test_subset", "seed", "fraction", "cond_norm")
                  if c in df.columns]
    if "source" in df.columns:
        proto_cols = proto_cols + ["source"]
    metrics = [("RMSE", False), ("MAE", False), ("NASA_per_engine", False),
               ("R2", True), ("train_s", False), ("params", False)]
    rows = []
    for key, g in df.groupby(proto_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        proto = dict(zip(proto_cols, key))
        for metric, higher_better in metrics:
            if metric not in g.columns:
                continue
            vals = pd.to_numeric(g[metric], errors="coerce")
            if vals.notna().sum() == 0:
                continue
            best = vals.max() if higher_better else vals.min()
            for i, v in vals.items():
                rows.append({"mode": g.loc[i, "mode"], "ablation": g.loc[i, "ablation"],
                             "run_id": g.loc[i, "run_id"], "metric": metric, "value": v,
                             "is_best": bool(pd.notna(v) and v == best),
                             "higher_better": higher_better,
                             **proto})
    cmp = pd.DataFrame(rows)
    if cmp.empty:
        return
    best_on = (cmp[cmp.is_best].groupby("run_id")
               .metric.apply(lambda s: "|".join(sorted(s))).rename("best_on"))
    idx = ["mode", "ablation", "run_id"] + [c for c in proto_cols if c not in ("mode", "ablation")]
    wide = (cmp.set_index(idx + ["metric"])["value"]
            .unstack("metric").reset_index())
    wide = wide.merge(best_on, on="run_id", how="left")
    wide["best_on"] = wide["best_on"].fillna("")
    wide.to_csv(base, index=False)
    print(f"[table] wrote {base}")


def write_paper_table(args, df):
    if df.empty:
        return
    agg = aggregate_seeds(df)
    if agg.empty:
        return
    e1 = agg[agg.experiment == "E1"]
    if e1.empty:
        return
    lines = ["| subset | RMSE (mean±std) | MAE (mean±std) | NASA/engine | runs |",
             "|---|---|---|---|---|"]
    for _, r in e1.iterrows():
        # a single-seed run has std = NaN, which must print as +-0.000, not +-nan
        sd_r = 0.0 if pd.isna(r.get("RMSE_std", np.nan)) else r["RMSE_std"]
        sd_m = 0.0 if pd.isna(r.get("MAE_std", np.nan)) else r["MAE_std"]
        rmse = f"{r.get('RMSE_mean', float('nan')):.3f}±{sd_r:.3f}"
        mae = f"{r.get('MAE_mean', float('nan')):.3f}±{sd_m:.3f}"
        nasa = f"{r.get('NASA_per_engine_mean', float('nan')):.2f}"
        lines.append(f"| {r['subset']} | {rmse} | {mae} | {nasa} | {int(r['n_runs'])} |")
    path = Path(args.output_dir) / "paper_table_E1.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[table] wrote {path}")


def merge_runs(args):
    frames = []
    for d in args.merge_from:
        p = Path(d) / "all_runs.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            print(f"[merge] no all_runs.csv in {d}")
    if not frames:
        print("[merge] nothing to merge")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["run_id"], keep="last")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.output_dir) / "all_runs.csv", index=False)
    make_compare_table(args, df)
    write_paper_table(args, df)
    print(f"[merge] {len(df)} runs -> {args.output_dir}")
    return df


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def main(argv=None):
    args = args_parser(argv)
    device = choose_device(args.device)
    args.device = str(device)
    if args.merge_from:
        return merge_runs(args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    croot = cache_root(args)
    croot.mkdir(parents=True, exist_ok=True)

    runs = build_runs(args)
    if not runs:
        print("[plan] no runs selected")
        return pd.DataFrame()
    print(f"[plan] {len(runs)} configuration(s); cache={croot}; device={args.device}")

    nproc = os.cpu_count() or 1
    jobs = max(1, min(args.jobs, len(runs)))
    threads = args.threads if args.threads else max(1, nproc // jobs)
    if jobs > 1 and args.encoder == "chronos":
        print(f"[jobs] {jobs} workers will each load the Chronos weights; "
              f"use --jobs 1 on a single GPU unless memory allows.")

    rows = []
    if jobs > 1:
        cmds = [cfg_to_argv(c, args, threads) for c in runs]
        fails = []

        def _run(cmd):
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True)
            return p.returncode, p.stdout

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for c, (code, log) in zip(runs, pool.map(_run, cmds)):
                print(f"[job] {c['_id']} exit={code}")
                if code != 0:
                    fails.append((c["_id"], log[-1500:]))
        for rid, log in fails:
            print(f"[job] FAILED {rid}\n{log}")
        if (Path(args.output_dir) / "all_runs.csv").exists():
            rows = pd.read_csv(Path(args.output_dir) / "all_runs.csv").to_dict("records")
    else:
        configure_threads(threads)
        pipe = load_chronos(args.chronos_model, device) if (
            args.encoder == "chronos" and runs) else None
        for cfg in runs:
            rows.append(run_one(args, cfg, pipe))
        df = append_runs(args, rows)
        make_compare_table(args, df)
        write_paper_table(args, df)
        return df

    df = pd.read_csv(Path(args.output_dir) / "all_runs.csv") if rows else pd.DataFrame(rows)
    if not df.empty:
        make_compare_table(args, df)
        write_paper_table(args, df)
    return df


def run_config(epochs=60, **kwargs):
    """Notebook entry point: one configuration per call, caches and the loaded
    Chronos pipeline persist across calls in this process."""
    argv = ["--single"]
    for k, v in kwargs.items():
        flag = "--" + k
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif isinstance(v, (list, tuple)):
            argv += [flag] + [str(x) for x in v]
        else:
            argv += [flag, str(v)]
    argv += ["--epochs", str(epochs)]
    return main(argv)


if __name__ == "__main__":
    main()
