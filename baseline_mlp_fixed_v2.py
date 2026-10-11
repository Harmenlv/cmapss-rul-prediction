"""
baseline_mlp.py
C-MAPSS / FD001 MLP baseline
D1 clean reproducible version

主要修正：
1. 按 engine 划分 train / validation，避免滑动窗口泄漏
2. train-only normalization，避免统计量泄漏
3. 正确构造 C-MAPSS test RUL
4. 使用 validation Early Stopping / best checkpoint
5. 测试集严格使用 official test set
6. 同时报告 RMSE / MAE / Score / Bias / over/under-estimation
7. 每台发动机只取最后一个窗口进行最终 RUL 评价
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# 配置
# ============================================================
DATA_DIR = r"F:\Users\jsj_s\Desktop\Code_testing\venv\project2026.np"

SUBSET = "FD001"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
WINDOW = 30
RMAX = 125

BATCH = 256
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-3

VAL_RATIO = 0.20
PATIENCE = 10
MIN_DELTA = 1e-4

CHECKPOINT = "best_mlp.pt"


# ============================================================
# C-MAPSS columns
# ============================================================
COLS = (
    ["unit", "cycle", "os1", "os2", "os3"]
    + [f"s{i}" for i in range(1, 22)]
)

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
OP_COLS = ["os1", "os2", "os3"]
FEATS = SENSOR_COLS + OP_COLS


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 保证尽可能可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 数据加载
# ============================================================
def load_subset(subset):
    train_path = os.path.join(DATA_DIR, f"train_{subset}.txt")
    test_path = os.path.join(DATA_DIR, f"test_{subset}.txt")
    rul_path = os.path.join(DATA_DIR, f"RUL_{subset}.txt")

    train = pd.read_csv(
        train_path,
        sep=r"\s+",
        header=None,
        names=COLS,
    )

    test = pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        names=COLS,
    )

    rul = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )

    return train, test, rul


# ============================================================
# Train RUL
# ============================================================
def add_train_rul(df, rmax=RMAX):
    """
    Official training-set RUL:
        RUL(t) = max_cycle(unit) - cycle(t)

    Then apply the commonly used upper cap:
        RUL = min(RUL, rmax)
    """
    out = df.copy()

    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = (max_cycle - out["cycle"]).clip(upper=rmax)

    return out


# ============================================================
# Test RUL
# ============================================================
def add_test_rul(test_df, rul_df, rmax=RMAX):
    """
    Correctly reconstruct test-set RUL.

    For each test engine u:
        RUL(last cycle) = official RUL[u]

    Therefore:
        RUL(t) = official_RUL[u] + last_cycle[u] - cycle(t)

    Finally apply the same upper cap.
    """
    out = test_df.copy().reset_index(drop=True)

    units = np.sort(out["unit"].unique())

    if len(units) != len(rul_df):
        raise ValueError(
            f"Test engine count ({len(units)}) != RUL file rows ({len(rul_df)})"
        )

    # official RUL corresponding to sorted engine IDs
    rul_map = {
        int(unit): float(rul)
        for unit, rul in zip(units, rul_df["RUL"].values)
    }

    last_cycle = out.groupby("unit")["cycle"].transform("max")

    out["RUL"] = (
        out["unit"].map(rul_map)
        + last_cycle
        - out["cycle"]
    )

    out["RUL"] = out["RUL"].clip(upper=rmax)

    return out


# ============================================================
# Engine-wise train / validation split
# ============================================================
def split_by_engine(df, val_ratio=VAL_RATIO, seed=SEED):
    """
    Split by engine ID, never by overlapping windows.

    This prevents almost-identical neighboring windows from
    appearing simultaneously in train and validation.
    """
    engines = np.sort(df["unit"].unique())

    tr_eng, va_eng = train_test_split(
        engines,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )

    tr_eng = np.sort(tr_eng)
    va_eng = np.sort(va_eng)

    tr_df = df[df["unit"].isin(tr_eng)].copy()
    va_df = df[df["unit"].isin(va_eng)].copy()

    return tr_df, va_df, tr_eng, va_eng


# ============================================================
# Train-only standardization
# ============================================================
def fit_scaler(df, features):
    """
    Fit normalization statistics using TRAIN ONLY.
    """
    mu = df[features].mean()
    sd = df[features].std()

    # Avoid division by zero
    sd = sd.replace(0, 1.0)

    return mu, sd


def apply_scaler(df, features, mu, sd):
    """
    Apply train-fitted standardization.

    Important:
    Some C-MAPSS columns are stored by pandas as integer dtype
    (especially sensors that happen to contain integer-valued data).
    The standardized values are floating-point, so assignment must
    explicitly convert the feature block to float32.
    """
    out = df.copy()

    scaled = (
        out[features].astype(np.float32)
        .sub(mu.astype(np.float32), axis="columns")
        .div(sd.astype(np.float32), axis="columns")
    )

    out[features] = scaled.astype(np.float32)

    return out


# ============================================================
# Sliding windows
# ============================================================
def make_windows(df, window=WINDOW):
    """
    For each engine:

        X_i = [x_i, ..., x_{i+window-1}]

    The label is the RUL at the last frame:

        y_i = RUL_{i+window-1}

    Output X is flattened to [window * feature_dim] for MLP.
    """
    Xs, ys, units = [], [], []

    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle").reset_index(drop=True)

        if len(g) < window:
            continue

        feats = g[FEATS].values.astype(np.float32)
        labels = g["RUL"].values.astype(np.float32)

        for i in range(len(g) - window + 1):
            Xs.append(
                feats[i:i + window].reshape(-1)
            )
            ys.append(
                labels[i + window - 1]
            )
            units.append(unit)

    if not Xs:
        raise ValueError(
            f"No windows generated. Check window={window} "
            f"and input engine lengths."
        )

    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(units),
    )


# ============================================================
# MLP
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# Metrics
# ============================================================
def cmapss_score(y_true, y_pred):
    """
    Official C-MAPSS asymmetric scoring function.

    d = prediction - truth

    Under-estimation and over-estimation receive different
    exponential penalties.
    """
    d = y_pred - y_true

    score = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1.0,
        np.exp(d / 10.0) - 1.0,
    )

    return float(score.sum())


def regression_metrics(y_true, y_pred):
    error = y_pred - y_true

    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))

    under_mask = error < 0
    over_mask = error > 0

    under_rate = float(np.mean(under_mask) * 100.0)
    over_rate = float(np.mean(over_mask) * 100.0)

    score = cmapss_score(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Score": score,
        "Bias": bias,
        "UnderRate": under_rate,
        "OverRate": over_rate,
    }


# ============================================================
# Validation
# ============================================================
def evaluate_model(model, loader, device):
    model.eval()

    total_sq_error = 0.0
    total_n = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)

            total_sq_error += (
                (pred - yb) ** 2
            ).sum().item()

            total_n += len(yb)

    return float(np.sqrt(total_sq_error / total_n))


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    set_seed(SEED)

    print("=" * 70)
    print("C-MAPSS MLP BASELINE")
    print("=" * 70)
    print(f"Device : {DEVICE}")
    print(f"Subset : {SUBSET}")
    print(f"Window : {WINDOW}")
    print(f"RMAX   : {RMAX}")
    print(f"Seed   : {SEED}")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load official C-MAPSS files
    # --------------------------------------------------------
    train_raw, test_raw, rul_raw = load_subset(SUBSET)

    print("\n[1] Raw data")
    print(f"train: {train_raw.shape}")
    print(f"test : {test_raw.shape}")
    print(f"RUL  : {rul_raw.shape}")

    # --------------------------------------------------------
    # 2. Construct labels
    # --------------------------------------------------------
    train_raw = add_train_rul(train_raw)
    test_raw = add_test_rul(test_raw, rul_raw)

    # --------------------------------------------------------
    # 3. Engine-wise train / validation split
    # --------------------------------------------------------
    tr_df, va_df, tr_eng, va_eng = split_by_engine(
        train_raw,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    print("\n[2] Engine-wise split")
    print(f"train engines: {len(tr_eng)}")
    print(f"val engines  : {len(va_eng)}")
    print(f"train IDs    : {tr_eng.tolist()}")
    print(f"val IDs      : {va_eng.tolist()}")

    # Explicit leakage check
    overlap = set(tr_eng).intersection(set(va_eng))
    if overlap:
        raise RuntimeError(
            f"Engine leakage detected: {sorted(overlap)}"
        )

    # --------------------------------------------------------
    # 4. Fit scaler on TRAIN ONLY
    # --------------------------------------------------------
    mu, sd = fit_scaler(tr_df, FEATS)

    tr_df = apply_scaler(tr_df, FEATS, mu, sd)
    va_df = apply_scaler(va_df, FEATS, mu, sd)
    test_scaled = apply_scaler(test_raw, FEATS, mu, sd)

    # Verify that all model features are floating-point after scaling.
    for name, frame in [
        ("train", tr_df),
        ("val", va_df),
        ("test", test_scaled),
    ]:
        if not all(
            np.issubdtype(frame[c].dtype, np.floating)
            for c in FEATS
        ):
            raise TypeError(
                f"{name} feature block is not floating-point after scaling."
            )

    print("\n[3] Normalization")
    print("Scaler fitted on training engines only.")
    print("All model features converted to float32 after scaling.")

    # --------------------------------------------------------
    # 5. Window construction
    # --------------------------------------------------------
    Xtr, ytr, tr_units = make_windows(tr_df)
    Xva, yva, va_units = make_windows(va_df)
    Xte, yte, te_units = make_windows(test_scaled)

    print("\n[4] Sliding windows")
    print(f"train: {Xtr.shape}")
    print(f"val  : {Xva.shape}")
    print(
        f"test : {Xte.shape} "
        f"({len(np.unique(te_units))} engines)"
    )

    print(f"Input dimension: {Xtr.shape[1]}")

    # --------------------------------------------------------
    # 6. DataLoaders
    # --------------------------------------------------------
    dl_tr = DataLoader(
        TensorDataset(
            torch.from_numpy(Xtr),
            torch.from_numpy(ytr),
        ),
        batch_size=BATCH,
        shuffle=True,
    )

    dl_va = DataLoader(
        TensorDataset(
            torch.from_numpy(Xva),
            torch.from_numpy(yva),
        ),
        batch_size=BATCH,
        shuffle=False,
    )

    # --------------------------------------------------------
    # 7. Model
    # --------------------------------------------------------
    model = MLP(in_dim=Xtr.shape[1]).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = nn.MSELoss()

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\n[5] Model parameters: {n_params:,}")

    # --------------------------------------------------------
    # 8. Training + Early Stopping
    # --------------------------------------------------------
    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0

    print("\n[6] Training")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):

        model.train()

        train_sq_error = 0.0
        train_n = 0

        for xb, yb in dl_tr:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            optimizer.step()

            train_sq_error += (
                (pred - yb) ** 2
            ).sum().item()

            train_n += len(yb)

        train_mse = train_sq_error / train_n

        val_rmse = evaluate_model(
            model,
            dl_va,
            DEVICE,
        )

        improved = val_rmse < (best_val - MIN_DELTA)

        if improved:
            best_val = val_rmse
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_rmse": best_val,
                    "seed": SEED,
                    "window": WINDOW,
                    "rmax": RMAX,
                    "features": FEATS,
                    "mu": mu,
                    "sd": sd,
                },
                CHECKPOINT,
            )

        else:
            patience_counter += 1

        if (
            epoch == 1
            or epoch % 10 == 0
            or improved
        ):
            print(
                f"Epoch {epoch:3d} | "
                f"train MSE {train_mse:9.3f} | "
                f"val RMSE {val_rmse:7.3f} | "
                f"best {best_val:7.3f}"
            )

        if patience_counter >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best epoch = {best_epoch}, "
                f"best val RMSE = {best_val:.3f}"
            )
            break

    # --------------------------------------------------------
    # 9. Restore best checkpoint
    # --------------------------------------------------------
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    best_epoch = checkpoint["epoch"]

    print("\n[7] Best checkpoint restored")
    print(f"Best epoch    : {best_epoch}")
    print(f"Best val RMSE : {checkpoint['val_rmse']:.4f}")

    # --------------------------------------------------------
    # 10. Official test inference
    # --------------------------------------------------------
    model.eval()

    with torch.no_grad():
        xt = torch.from_numpy(Xte).to(DEVICE)
        pred_all = model(xt).cpu().numpy()

    # --------------------------------------------------------
    # 11. C-MAPSS final protocol:
    #     one prediction per test engine,
    #     using the final available window
    # --------------------------------------------------------
    pred_last = []
    true_last = []

    for unit in np.unique(te_units):

        idx = np.where(te_units == unit)[0]

        # Last window = latest observed cycle
        last_idx = idx[-1]

        pred_last.append(pred_all[last_idx])
        true_last.append(yte[last_idx])

    pred_last = np.asarray(pred_last, dtype=np.float64)
    true_last = np.asarray(true_last, dtype=np.float64)

    # --------------------------------------------------------
    # 12. Metrics
    # --------------------------------------------------------
    metrics = regression_metrics(
        true_last,
        pred_last,
    )

    print("\n" + "=" * 70)
    print(f"{SUBSET} FINAL TEST RESULTS")
    print("=" * 70)
    print(f"Best epoch       : {best_epoch}")
    print(f"Val RMSE         : {best_val:.3f}")
    print(f"Test engines     : {len(true_last)}")
    print(f"RMSE             : {metrics['RMSE']:.3f}")
    print(f"MAE              : {metrics['MAE']:.3f}")
    print(f"Score            : {metrics['Score']:.1f}")
    print(f"Bias             : {metrics['Bias']:.3f}")
    print(f"Under-estimation : {metrics['UnderRate']:.1f}%")
    print(f"Over-estimation  : {metrics['OverRate']:.1f}%")
    print("=" * 70)

    print(
        "\nReference only: "
        "Wide&Deep FD001 RMSE reported as 9.97. "
        "This MLP is intended as a simple baseline, "
        "not as the proposed model."
    )

    # --------------------------------------------------------
    # 13. Save engine-level predictions
    # --------------------------------------------------------
    result_df = pd.DataFrame(
        {
            "unit": np.unique(te_units),
            "true_RUL": true_last,
            "pred_RUL": pred_last,
            "error": pred_last - true_last,
        }
    )

    result_path = f"{SUBSET}_mlp_predictions.csv"
    result_df.to_csv(
        result_path,
        index=False,
    )

    print(f"\nSaved predictions: {result_path}")
    print(f"Saved checkpoint : {CHECKPOINT}")
