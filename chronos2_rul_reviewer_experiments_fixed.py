#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chronos-2 frozen-backbone RUL adaptation for NASA C-MAPSS.

Implements the reviewer-facing experiments:
  E1: FD001-FD004, repeated seeds
  E2: multi-seed mean/std
  E3: reduced-data learning curves
  E4: cross-subset transfer (train on one subset, evaluate another)
  E5: frozen Chronos-2 feature linear probe
  E6: LSTM-backbone replacement with the same adapter/GAT/fusion/head
  E7: module ablations
  E8: parameters, training/inference time, peak CUDA memory

Chronos-2 features are obtained from the official Chronos2Pipeline.embed()
API. The pretrained Chronos-2 encoder is frozen. The adapter, graph module,
fusion and regression head are trained downstream.

This script does NOT claim Chronos-2 is a native RUL regressor. It uses
Chronos-2 encoder representations as frozen features for an RUL head.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


COLS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
FEATURES = [f"s{i}" for i in range(1, 22)] + ["os1", "os2", "os3"]


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="chronos2_rul_revision")
    p.add_argument("--subsets", nargs="+", default=["FD001"],
                   choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    p.add_argument("--mode", choices=["full", "linear_probe", "lstm_backbone"], default="full")
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
    p.add_argument("--embed_batch_size", type=int, default=16)
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
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--deterministic", action="store_true")
    return p.parse_args()


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


def load_cmapss(root, subset):
    root = Path(root)
    tr = pd.read_csv(root / f"train_{subset}.txt", sep=r"\s+", header=None, names=COLS)
    te = pd.read_csv(root / f"test_{subset}.txt", sep=r"\s+", header=None, names=COLS)
    rul = pd.read_csv(root / f"RUL_{subset}.txt", sep=r"\s+", header=None, names=["RUL"])
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
    xs, ys, us, cs = [], [], [], []
    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        if len(g) < window:
            continue
        x = g[FEATURES].to_numpy(np.float32)
        y = g["RUL"].to_numpy(np.float32)
        c = g["cycle"].to_numpy(np.int64)
        for end in range(window - 1, len(g)):
            xs.append(x[end-window+1:end+1])
            ys.append(y[end])
            us.append(int(unit))
            cs.append(int(c[end]))
    return (np.asarray(xs, np.float32), np.asarray(ys, np.float32),
            np.asarray(us, np.int64), np.asarray(cs, np.int64))


def make_pseudotest(df, window, cap, fractions=(0.5, 0.6, 0.7, 0.8, 0.9)):
    xs, ys, ids, cycles = [], [], [], []
    sid = 0
    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        if len(g) <= window:
            continue
        x = g[FEATURES].to_numpy(np.float32)
        c = g["cycle"].to_numpy(np.int64)
        failure_cycle = int(c[-1])
        used = set()
        for frac in fractions:
            end = int(round((len(g)-1) * frac))
            end = max(window-1, min(end, len(g)-2))
            if end in used:
                continue
            used.add(end)
            xs.append(x[end-window+1:end+1])
            ys.append(float(np.clip(failure_cycle-int(c[end]), 0, cap)))
            ids.append(sid)
            cycles.append(int(c[end]))
            sid += 1
    return (np.asarray(xs, np.float32), np.asarray(ys, np.float32),
            np.asarray(ids, np.int64), np.asarray(cycles, np.int64))


class ArrayDataset(Dataset):
    def __init__(self, x, y, units, cycles):
        self.x = torch.from_numpy(np.asarray(x, np.float32))
        self.y = torch.from_numpy(np.asarray(y, np.float32))
        self.units = np.asarray(units)
        self.cycles = np.asarray(cycles)
        self.next_idx = np.full(len(y), -1, np.int64)
        for u in np.unique(self.units):
            ix = np.where(self.units == u)[0]
            ix = ix[np.argsort(self.cycles[ix])]
            for a, b in zip(ix[:-1], ix[1:]):
                if self.cycles[b] == self.cycles[a] + 1:
                    self.next_idx[a] = b

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], i


def final_indices(units, cycles):
    out = []
    for u in np.unique(units):
        ix = np.where(units == u)[0]
        out.append(ix[np.argmax(cycles[ix])])
    return np.asarray(out, np.int64)


def metric_dict(y, p):
    e = p-y
    score = np.where(e < 0, np.exp(-e/13)-1, np.exp(e/10)-1).sum()
    ss = np.sum((y-y.mean())**2)
    return {
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "NASA_Score": float(score),
        "R2": float(1-np.sum(e**2)/ss) if ss > 0 else float("nan"),
    }


def load_chronos(model_id, device):
    try:
        from chronos import Chronos2Pipeline
    except Exception as exc:
        raise ImportError(
            "Chronos-2 package missing. Install with: "
            "pip install \"chronos-forecasting>=2.0\""
        ) from exc
    pipe = Chronos2Pipeline.from_pretrained(
        model_id, device_map=str(device)
    )
    pipe.model.eval()
    for p in pipe.model.parameters():
        p.requires_grad_(False)
    return pipe


@torch.no_grad()
def chronos_embed_windows(pipe, X, batch_size, context_length, cache_file=None,
                          force=False):
    """
    X: [N, window, 24]. Chronos2Pipeline.embed expects each item to be
    [n_variates, history_length]. It returns one encoder embedding tensor
    per input item. Patch/token dimensions are mean-pooled, preserving
    the 24 variable representations.
    """
    if cache_file is not None and Path(cache_file).exists() and not force:
        z = np.load(cache_file)
        return z["features"].astype(np.float32)

    feats = []
    for start in range(0, len(X), batch_size):
        xb = X[start:start+batch_size]
        inputs = [torch.as_tensor(a.T, dtype=torch.float32) for a in xb]
        embeddings, _ = pipe.embed(
            inputs, batch_size=max(24, batch_size*24),
            context_length=context_length
        )
        for emb in embeddings:
            # Chronos2 embed returns per-item encoder states. Depending on
            # package version, shape is [variates, tokens, d_model] or
            # [tokens, variates, d_model]; normalize to [24, tokens, d].
            e = torch.as_tensor(emb).detach().cpu()
            if e.ndim == 2:
                if e.shape[0] == len(FEATURES):
                    e = e.unsqueeze(1)
                else:
                    raise RuntimeError(f"Unexpected Chronos embedding shape: {tuple(e.shape)}")
            elif e.ndim == 3:
                if e.shape[0] == len(FEATURES):
                    pass
                elif e.shape[1] == len(FEATURES):
                    e = e.transpose(0, 1)
                else:
                    raise RuntimeError(
                        f"Cannot identify 24 variable axis in Chronos embedding {tuple(e.shape)}"
                    )
            else:
                raise RuntimeError(f"Unexpected Chronos embedding rank: {e.ndim}")
            # Drop no tokens implicitly: mean pool encoder sequence positions.
            feats.append(e.mean(dim=1).numpy().astype(np.float32))
    out = np.stack(feats, axis=0)
    if cache_file is not None:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, features=out)
    return out


def build_graph(X_train, topk):
    """Train-split-only absolute Pearson correlation graph with self loops."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    corr = np.corrcoef(flat, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    n = corr.shape[0]
    adj = np.eye(n, dtype=np.float32)
    k = min(max(1, topk), n-1)
    for i in range(n):
        ids = np.argsort(corr[i])[-(k+1):-1]
        adj[i, ids] = 1.0
    adj = np.maximum(adj, adj.T)
    return torch.tensor(adj, dtype=torch.bool)


class DenseGraphAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden dimension must be divisible by attention heads")
        self.dim, self.heads, self.dh = dim, heads, dim//heads
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
    def __init__(self, d_in, hidden=128, heads=4, dropout=0.1,
                 use_temporal=True, use_graph=True, use_fusion=True):
        super().__init__()
        self.temporal_adapter = (
            nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                          nn.LayerNorm(hidden), nn.Dropout(dropout))
            if use_temporal else nn.Linear(d_in, hidden, bias=False)
        )
        self.graph = DenseGraphAttention(hidden, heads, dropout) if use_graph else None
        self.fusion_gate = (
            nn.Sequential(nn.Linear(hidden*2, hidden), nn.GELU(),
                          nn.Linear(hidden, hidden), nn.Sigmoid())
            if use_fusion and use_graph else None
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, z, adj):
        # z: [batch, variables, Chronos hidden dimension]
        temporal = self.temporal_adapter(z)
        if self.graph is not None:
            graph = self.graph(temporal, adj)
            if self.fusion_gate is not None:
                g = self.fusion_gate(torch.cat([temporal, graph], dim=-1))
                fused = g*temporal + (1-g)*graph
            else:
                fused = temporal + graph
        else:
            fused = temporal
        rep = fused.mean(dim=1)
        return self.regressor(rep).squeeze(-1)


class LSTMFeatureBackbone(nn.Module):
    """Trainable LSTM replacement used only for the backbone-swap experiment."""
    def __init__(self, n_features=24, hidden=128, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden//2, batch_first=True,
                            num_layers=2, bidirectional=True, dropout=dropout)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        # Produce one node embedding per variable using the same temporal
        # window, then pass through the identical graph/fusion/regression head.
        b, t, f = x.shape
        outs = []
        for j in range(f):
            seq = x[:, :, j:j+1]
            h, _ = self.lstm(seq)
            outs.append(self.proj(h.mean(dim=1)))
        return torch.stack(outs, dim=1)


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


def select_engine_split(df, seed, val_ratio=0.2):
    engines = np.sort(df.unit.unique())
    tr_ids, va_ids = train_test_split(engines, test_size=val_ratio, random_state=seed)
    return df[df.unit.isin(tr_ids)].copy(), df[df.unit.isin(va_ids)].copy()


def normalize_frames(train_df, other_frames):
    """
    Fit normalization statistics on the training split only, then apply the
    same statistics to training, validation, and test frames.
    Returns: normalized_train, normalized_validation, normalized_test, ...
    """
    mu = train_df[FEATURES].mean()
    sd = train_df[FEATURES].std().replace(0, 1.0).fillna(1.0)

    def transform(frame):
        z = frame.copy()
        vals = ((z[FEATURES].astype(np.float64) - mu) / sd).astype(np.float32)
        for col in FEATURES:
            z[col] = vals[col].to_numpy(np.float32)
        return z

    return [transform(train_df)] + [transform(frame) for frame in other_frames]


def make_cache_path(root, subset, split, window, seed):
    return Path(root) / f"chronos2_cache_{subset}_{split}_w{window}_s{seed}.npz"


def prepare_features(pipe, X, cache, args):
    return chronos_embed_windows(
        pipe, X, args.embed_batch_size, args.context_length,
        cache_file=cache, force=args.force_reembed
    )


def batch_pairs(ds, ids, device):
    ids = np.asarray(ids, np.int64)
    partner = ds.next_idx[ids]
    valid = partner >= 0
    ai, bi = ids[valid], partner[valid]
    all_ids = np.unique(np.concatenate([ids, bi]))
    lut = {int(v): i for i, v in enumerate(all_ids)}
    pi = torch.tensor([lut[int(v)] for v in ai], dtype=torch.long, device=device)
    pj = torch.tensor([lut[int(v)] for v in bi], dtype=torch.long, device=device)
    return all_ids, pi, pj


def evaluate_features(model, X, y, adj, device, batch_size, raw_x=None):
    model.eval()
    pred = []
    with torch.no_grad():
        for st in range(0, len(y), batch_size):
            if raw_x is None:
                xb = torch.from_numpy(X[st:st+batch_size]).to(device)
                out = model(xb, adj)
            else:
                xb = torch.from_numpy(raw_x[st:st+batch_size]).to(device)
                out = model(xb, adj)
            pred.append(out.cpu().numpy())
    return np.concatenate(pred)


def train_model(model, Xtr, ytr, utr, ctr, Xva, yva, adj, args,
                seed, use_mono=True, use_smooth=True, use_stage=True, data_tag=""):
    device = next(model.parameters()).device
    ds = ArrayDataset(Xtr, ytr, utr, ctr)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay
    )
    best, best_epoch, bad = float("inf"), 0, 0
    best_state = None
    history = []
    start_time = time.perf_counter()
    stage_edges = torch.tensor([args.rmax/3.0, 2.0*args.rmax/3.0], device=device)
    y_train_tensor = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    train_stage_id = torch.bucketize(y_train_tensor, stage_edges)
    stage_counts = torch.bincount(train_stage_id, minlength=3).float().clamp_min(1.0)
    fixed_stage_weights = (stage_counts.sum()/(3.0*stage_counts)).sqrt()
    fixed_stage_weights = fixed_stage_weights / fixed_stage_weights.mean()

    for ep in range(1, args.epochs+1):
        model.train()
        order = np.random.permutation(len(ds))
        losses, mses, monos, smooths = [], [], [], []
        for st in range(0, len(order), args.batch_size):
            anchor = order[st:st+args.batch_size]
            ix, pi, pj = batch_pairs(ds, anchor, device)
            yb = ds.y[ix].to(device)
            opt.zero_grad(set_to_none=True)
            if isinstance(model, RULModelLSTM):
                xb = ds.x[ix].to(device)
                pred = model(xb, adj)
            else:
                xb = torch.from_numpy(Xtr[ix]).to(device)
                pred = model(xb, adj)
            per_sample = (pred-yb).pow(2)
            if use_stage:
                # Three RUL stages are defined by target RUL, not by arbitrary
                # sensor ordering. Inverse-frequency weights are estimated
                # from the training split only.
                stage_id = torch.bucketize(yb.detach(), stage_edges)
                mse = (per_sample*fixed_stage_weights[stage_id]).mean()
            else:
                mse = per_sample.mean()
            mono = F.relu(pred[pj]-pred[pi]).mean() if use_mono and len(pi) else pred.new_tensor(0.)
            # Penalize departures from an approximately one-cycle RUL decrease
            # for truly consecutive windows from the same engine.
            if use_smooth and len(pi):
                uncapped = yb[pi] < (args.rmax - 1e-5)
                smooth = ((pred[pj[uncapped]]-pred[pi[uncapped]]) + 1.0).pow(2).mean() if uncapped.any() else pred.new_tensor(0.)
            else:
                smooth = pred.new_tensor(0.)
            loss = mse + args.lambda_mono*mono + args.lambda_smooth*smooth
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            mses.append(float(mse.detach().cpu()))
            monos.append(float(mono.detach().cpu()))
            smooths.append(float(smooth.detach().cpu()))

        # Validation predictions are made on pseudo-test snapshots.
        model.eval()
        vp = []
        with torch.no_grad():
            for st in range(0, len(yva), args.batch_size):
                if isinstance(model, RULModelLSTM):
                    raise RuntimeError("LSTM-backbone mode validation needs raw windows; use mode=lstm_backbone path.")
                xb = torch.from_numpy(Xva[st:st+args.batch_size]).to(device)
                vp.append(model(xb, adj).cpu().numpy())
        vm = metric_dict(yva, np.concatenate(vp))
        history.append({
            "epoch": ep, "train_loss": np.mean(losses), "train_mse": np.mean(mses),
            "mono_raw": np.mean(monos), "smooth_raw": np.mean(smooths),
            "val_pseudotest_RMSE": vm["RMSE"], "val_pseudotest_MAE": vm["MAE"]
        })
        if ep == 1 or ep % 5 == 0:
            print(f"{data_tag} epoch {ep:03d} | train MSE {np.mean(mses):.3f} "
                  f"| val pseudo-test RMSE {vm['RMSE']:.3f}")
        if vm["RMSE"] < best-1e-4:
            best, best_epoch, bad = vm["RMSE"], ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, time.perf_counter()-start_time, best, best_epoch


def ablation_flags(name):
    return {
        "use_temporal": name != "no_temporal_adapter",
        "use_graph": name != "no_graph",
        "use_fusion": name != "no_fusion_gate",
        "use_stage": name != "no_stage_weighting",
        "use_mono": name != "no_monotonic",
        "use_smooth": name != "no_smoothness",
    }


def run_one(args, subset, seed, ablation, pipe, device, fraction=1.0):
    seed_all(seed, args.deterministic)
    run_name = f"{subset}_{args.mode}_{ablation}_seed{seed}_frac{fraction:g}"
    out = Path(args.output_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)

    tr0, te0, rul = load_cmapss(args.data_dir, subset)
    tr0 = attach_train_rul(tr0, args.rmax)
    te0 = attach_test_rul(te0, rul, args.rmax)
    tr_df, va_df = select_engine_split(tr0, seed)
    tr_df, va_df, te_df = normalize_frames(tr_df, [va_df, te0])

    # Optional low-data experiment samples whole training engines only.
    if fraction < 1.0:
        eng = np.sort(tr_df.unit.unique())
        rng = np.random.default_rng(seed)
        keep_n = max(2, int(round(len(eng)*fraction)))
        keep = rng.choice(eng, size=min(keep_n, len(eng)), replace=False)
        tr_df = tr_df[tr_df.unit.isin(keep)].copy()

    Xtr_raw, ytr, utr, ctr = make_windows(tr_df, args.window)
    Xva_raw, yva, uva, cva = make_pseudotest(va_df, args.window, args.rmax)
    Xte_raw, yte_all, ute_all, cte_all = make_windows(te_df, args.window)
    te_ids = final_indices(ute_all, cte_all)
    Xte_final_raw = Xte_raw[te_ids]
    yte = yte_all[te_ids]
    ute, cte = ute_all[te_ids], cte_all[te_ids]

    # Graph topology is estimated exclusively from training engines.
    adj = build_graph(Xtr_raw, args.graph_topk).to(device)

    if args.mode == "lstm_backbone":
        flags = ablation_flags(ablation)
        model = RULModelLSTM(
            hidden=args.hidden, heads=args.heads, dropout=args.dropout,
            use_graph=flags["use_graph"], use_fusion=flags["use_fusion"]
        ).to(device)
        # This is a trainable LSTM replacement, not a frozen Chronos model.
        # Train/validation/test use raw windows and same graph/head structure.
        ds = ArrayDataset(Xtr_raw, ytr, utr, ctr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best, bad, best_state, best_epoch = float("inf"), 0, None, 0
        history = []
        t0 = time.perf_counter()
        for ep in range(1, args.epochs+1):
            model.train()
            order = np.random.permutation(len(ds))
            for st in range(0, len(order), args.batch_size):
                ix = order[st:st+args.batch_size]
                xb, yb = ds.x[ix].to(device), ds.y[ix].to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb, adj)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            vp = []
            with torch.no_grad():
                for st in range(0, len(Xva_raw), args.batch_size):
                    vp.append(model(torch.from_numpy(Xva_raw[st:st+args.batch_size]).to(device), adj).cpu().numpy())
            val = metric_dict(yva, np.concatenate(vp))["RMSE"]
            history.append({"epoch": ep, "val_pseudotest_RMSE": val})
            if val < best-1e-4:
                best, bad, best_epoch = val, 0, ep
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        train_sec = time.perf_counter()-t0
        if best_state:
            model.load_state_dict(best_state)
        test_pred = evaluate_features(model, None, yte, adj, device, args.batch_size, raw_x=Xte_final_raw)
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        infer_start = time.perf_counter()
        _ = evaluate_features(model, None, yte, adj, device, args.batch_size, raw_x=Xte_final_raw)
        infer_sec = time.perf_counter()-infer_start
    else:
        # Chronos-2 frozen embeddings. Cache includes raw normalized windows.
        croot = Path(args.output_dir) / "_feature_cache"
        Ztr = prepare_features(pipe, Xtr_raw, make_cache_path(croot, subset, f"train_f{fraction:g}", args.window, seed), args)
        Zva = prepare_features(pipe, Xva_raw, make_cache_path(croot, subset, f"val_f{fraction:g}", args.window, seed), args)
        Zte = prepare_features(pipe, Xte_final_raw, make_cache_path(croot, subset, f"test_f{fraction:g}", args.window, seed), args)

        flags = ablation_flags(ablation)
        if args.mode == "linear_probe":
            # Frozen Chronos feature mean-pooling + linear regression only.
            Ztr_p = Ztr.mean(axis=1)
            Zva_p = Zva.mean(axis=1)
            Zte_p = Zte.mean(axis=1)
            dim = Ztr_p.shape[1]
            model = nn.Linear(dim, 1).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best, bad, state, best_epoch = float("inf"), 0, None, 0
            history = []
            t0 = time.perf_counter()
            for ep in range(1, args.epochs+1):
                model.train()
                order = np.random.permutation(len(ytr))
                for st in range(0, len(order), args.batch_size):
                    ix = order[st:st+args.batch_size]
                    xb = torch.from_numpy(Ztr_p[ix]).to(device)
                    yb = torch.from_numpy(ytr[ix]).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.mse_loss(model(xb).squeeze(-1), yb)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    pv = model(torch.from_numpy(Zva_p).to(device)).squeeze(-1).cpu().numpy()
                val = metric_dict(yva, pv)["RMSE"]
                history.append({"epoch":ep,"val_pseudotest_RMSE":val})
                if val < best-1e-4:
                    best,bad,best_epoch=val,0,ep
                    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= args.patience: break
            train_sec = time.perf_counter()-t0
            if state: model.load_state_dict(state)
            with torch.no_grad():
                test_pred = model(torch.from_numpy(Zte_p).to(device)).squeeze(-1).cpu().numpy()
            n_total = sum(p.numel() for p in model.parameters())
            n_train = n_total
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(torch.from_numpy(Zte_p).to(device))
            infer_sec = time.perf_counter()-t0
        else:
            model = ChronosRULAdapter(
                d_in=Ztr.shape[-1], hidden=args.hidden, heads=args.heads,
                dropout=args.dropout, use_temporal=flags["use_temporal"],
                use_graph=flags["use_graph"], use_fusion=flags["use_fusion"]
            ).to(device)
            n_total = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in pipe.model.parameters())
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model, history, train_sec, best_val, best_epoch = train_model(
                model, Ztr, ytr, utr, ctr, Zva, yva, adj, args, seed,
                use_mono=flags["use_mono"], use_smooth=flags["use_smooth"],
                use_stage=flags["use_stage"], data_tag=run_name
            )
            t0 = time.perf_counter()
            test_pred = evaluate_features(model, Zte, yte, adj, device, args.batch_size)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_sec = time.perf_counter()-t0

    if args.mode == "lstm_backbone":
        best_val = best
    elif args.mode == "linear_probe":
        best_val = best

    metrics = metric_dict(yte, test_pred)
    result = {
        "subset": subset, "mode": args.mode, "ablation": ablation,
        "seed": seed, "data_fraction": fraction,
        "best_epoch": int(best_epoch), "best_val_pseudotest_RMSE": float(best_val),
        **metrics, "total_params_including_backbone": int(n_total),
        "trainable_params": int(n_train),
        "trainable_fraction_of_total": float(n_train/max(n_total,1)),
        "train_time_sec": float(train_sec),
        "inference_time_sec": float(infer_sec),
        "inference_ms_per_engine": float(infer_sec/max(len(yte),1)*1000),
        "peak_cuda_memory_MB": float(torch.cuda.max_memory_allocated(device)/(1024**2))
            if device.type == "cuda" else float("nan"),
        "device": str(device), "n_train_windows": len(ytr),
        "n_val_pseudotest": len(yva), "n_test_engines": len(yte),
        "chronos_model": args.chronos_model if args.mode != "lstm_backbone" else "LSTM",
        "window": args.window, "rul_cap": args.rmax,
    }
    pd.DataFrame(history).to_csv(out/"training_history.csv", index=False)
    pd.DataFrame({"unit":ute,"cycle":cte,"true_RUL":yte,"pred_RUL":test_pred,
                  "error":test_pred-yte}).to_csv(out/"test_predictions.csv",index=False)
    with open(out/"metrics.json","w",encoding="utf-8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
    print(f"RESULT {run_name}: RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f} "
          f"NASA={metrics['NASA_Score']:.2f}")
    return result



def run_cross_subset_transfer(args, source_subset, target_subset, seed, pipe, device):
    """Train on source-subset training engines; evaluate target-subset test engines."""
    seed_all(seed, args.deterministic)
    run_name = f"TRANSFER_{source_subset}_to_{target_subset}_seed{seed}"
    out = Path(args.output_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)

    src_train, _, _ = load_cmapss(args.data_dir, source_subset)
    _, tgt_test, tgt_rul = load_cmapss(args.data_dir, target_subset)
    src_train = attach_train_rul(src_train, args.rmax)
    src_tr, src_va = select_engine_split(src_train, seed)
    mu = src_tr[FEATURES].mean()
    sd = src_tr[FEATURES].std().replace(0, 1.0)

    def apply_source_scaler(frame):
        z = frame.copy()
        v = ((z[FEATURES].astype(np.float64)-mu)/sd).astype(np.float32)
        for c in FEATURES:
            z[c] = v[c].to_numpy(np.float32)
        return z

    src_tr = apply_source_scaler(src_tr)
    src_va = apply_source_scaler(src_va)
    tgt_test = apply_source_scaler(attach_test_rul(tgt_test, tgt_rul, args.rmax))

    Xtr, ytr, utr, ctr = make_windows(src_tr, args.window)
    Xva, yva, _, _ = make_pseudotest(src_va, args.window, args.rmax)
    Xte_all, yte_all, ute_all, cte_all = make_windows(tgt_test, args.window)
    ix = final_indices(ute_all, cte_all)
    Xte, yte = Xte_all[ix], yte_all[ix]
    ute, cte = ute_all[ix], cte_all[ix]

    adj = build_graph(Xtr, args.graph_topk).to(device)
    cache_root = Path(args.output_dir)/"_feature_cache"
    Ztr = prepare_features(pipe, Xtr, make_cache_path(cache_root, source_subset, f"xfer_train_{target_subset}", args.window, seed), args)
    Zva = prepare_features(pipe, Xva, make_cache_path(cache_root, source_subset, f"xfer_val_{target_subset}", args.window, seed), args)
    Zte = prepare_features(pipe, Xte, make_cache_path(cache_root, target_subset, f"xfer_test_from_{source_subset}", args.window, seed), args)

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
    result = {
        "source_subset": source_subset, "target_subset": target_subset, "seed": seed,
        "best_epoch": best_epoch, "best_val_RMSE": best_val, **met,
        "train_time_sec": train_sec, "n_source_train_windows": len(ytr),
        "n_target_test_engines": len(yte), "transfer_protocol": "source-only scaler/graph/training; target official test",
        "chronos_model": args.chronos_model
    }
    pd.DataFrame(hist).to_csv(out/"training_history.csv", index=False)
    pd.DataFrame({"unit":ute,"cycle":cte,"true_RUL":yte,"pred_RUL":pred,
                  "error":pred-yte}).to_csv(out/"target_predictions.csv",index=False)
    with open(out/"metrics.json","w",encoding="utf-8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
    return result


def main():
    args = args_parser()
    device = choose_device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    pipe = None
    if args.mode != "lstm_backbone":
        pipe = load_chronos(args.chronos_model, device)

    if args.cross_train and args.cross_test:
        rows = [run_cross_subset_transfer(args, args.cross_train, args.cross_test,
                                          seed, pipe, device) for seed in args.seeds]
        pd.DataFrame(rows).to_csv(Path(args.output_dir)/"cross_subset_transfer.csv", index=False)
        return

    if args.ablation == "all":
        ablations = ["full","no_temporal_adapter","no_graph","no_fusion_gate",
                     "no_monotonic","no_smoothness","no_stage_weighting"]
    else:
        ablations = [args.ablation]
    fractions = args.data_fractions or [1.0]
    results = []
    for subset in args.subsets:
        for frac in fractions:
            for ab in ablations:
                for seed in args.seeds:
                    results.append(run_one(args,subset,seed,ab,pipe,device,frac))

    raw = pd.DataFrame(results)
    raw.to_csv(Path(args.output_dir)/"all_runs.csv",index=False)
    cols = ["RMSE","MAE","NASA_Score","R2","total_params_including_backbone",
            "trainable_params","trainable_fraction_of_total","train_time_sec",
            "inference_ms_per_engine","peak_cuda_memory_MB"]
    summary = raw.groupby(["subset","mode","ablation","data_fraction"])[cols].agg(["mean","std"])
    summary.to_csv(Path(args.output_dir)/"mean_std_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
