"""
proposed_method.py
C-MAPSS FD001 Proposed Method

Proposed model:
Local-Global Temporal Fusion Network (LGTF-Net)

Core idea:
1. Local temporal branch: lightweight multi-scale 1D convolutions capture
   short-term degradation patterns.
2. Global temporal branch: LSTM captures long-range degradation evolution.
3. Adaptive gated fusion: learns sample-dependent weights to combine
   local and global temporal representations.
4. Regression head: maps the fused degradation representation to RUL.

Experimental protocol is kept identical to the validated baselines:
- FD001
- official train/test split
- engine-wise 80/20 train/validation split
- train-only standardization
- window = 30
- RUL cap = 125
- early stopping
- official 100-engine test set
- one final prediction per test engine
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


# =========================
# Configuration
# =========================
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

CHECKPOINT = "best_proposed.pt"
PREDICTION_FILE = "FD001_proposed_predictions.csv"
HISTORY_FILE = "proposed_training_history.csv"


COLS = (
    ["unit", "cycle", "os1", "os2", "os3"]
    + [f"s{i}" for i in range(1, 22)]
)
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
OP_COLS = ["os1", "os2", "os3"]
FEATS = SENSOR_COLS + OP_COLS


# =========================
# Reproducibility
# =========================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Data
# =========================
def load_subset(subset):
    train = pd.read_csv(
        os.path.join(DATA_DIR, f"train_{subset}.txt"),
        sep=r"\s+",
        header=None,
        names=COLS,
    )

    test = pd.read_csv(
        os.path.join(DATA_DIR, f"test_{subset}.txt"),
        sep=r"\s+",
        header=None,
        names=COLS,
    )

    rul = pd.read_csv(
        os.path.join(DATA_DIR, f"RUL_{subset}.txt"),
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )

    return train, test, rul


def add_train_rul(df, rmax=RMAX):
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = (max_cycle - out["cycle"]).clip(upper=rmax)
    return out


def add_test_rul(test_df, rul_df, rmax=RMAX):
    """
    Correct test-set RUL propagation:
    RUL(t) = official_RUL(unit) + last_cycle(unit) - cycle(t)
    """
    out = test_df.copy().reset_index(drop=True)

    units = np.sort(out["unit"].unique())
    if len(units) != len(rul_df):
        raise ValueError(
            f"Test engine count ({len(units)}) != RUL rows ({len(rul_df)})"
        )

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


def make_windows(df, window=WINDOW):
    """
    Each engine is converted into overlapping windows.
    The label is the RUL of the final cycle in each window.
    """
    Xs, ys, units = [], [], []

    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle").reset_index(drop=True)

        feats = g[FEATS].values.astype(np.float32)
        labels = g["RUL"].values.astype(np.float32)

        for i in range(len(g) - window + 1):
            Xs.append(feats[i:i + window])
            ys.append(labels[i + window - 1])
            units.append(unit)

    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(units),
    )


# =========================
# Proposed Model
# =========================
class LGTFNet(nn.Module):
    """
    Local-Global Temporal Fusion Network.

    Local branch:
        Conv1d(24 -> 64) -> Conv1d(64 -> 64)

    Global branch:
        2-layer LSTM, hidden size 64

    Fusion:
        adaptive gate z = sigmoid(W[h_local; h_global])
        h_fused = z*h_local + (1-z)*h_global

    Regression:
        64 -> 32 -> 1
    """

    def __init__(self, in_features=24, hidden=64, dropout=0.20):
        super().__init__()

        # ---- Local temporal branch ----
        self.local_branch = nn.Sequential(
            nn.Conv1d(in_features, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden),

            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
        )

        self.local_pool = nn.AdaptiveAvgPool1d(1)

        # ---- Global temporal branch ----
        self.global_branch = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # ---- Adaptive fusion ----
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )

        # ---- Regression head ----
        self.regressor = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: [B, T, F]

        # Local branch: [B, F, T] -> [B, H, T]
        xl = x.transpose(1, 2)
        hl = self.local_branch(xl)
        hl = self.local_pool(hl).squeeze(-1)  # [B, H]

        # Global branch
        hg_seq, _ = self.global_branch(x)
        hg = hg_seq[:, -1, :]                 # [B, H]

        # Adaptive gated fusion
        h_cat = torch.cat([hl, hg], dim=1)
        z = self.gate(h_cat)

        h_fused = z * hl + (1.0 - z) * hg

        return self.regressor(h_fused).squeeze(-1)


# =========================
# Metrics
# =========================
def cmapss_score(y_true, y_pred):
    d = y_pred - y_true

    # Official asymmetric C-MAPSS scoring convention:
    # early prediction (d < 0) is penalized with exp(-d/13),
    # late prediction (d > 0) with exp(d/10).
    s = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1.0,
        np.exp(d / 10.0) - 1.0,
    )

    return float(np.sum(s))


def regression_metrics(y_true, y_pred):
    err = y_pred - y_true

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    score = cmapss_score(y_true, y_pred)

    bias = float(np.mean(err))
    under = float(np.mean(err < 0) * 100.0)
    over = float(np.mean(err > 0) * 100.0)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    pearson = float(np.corrcoef(y_true, y_pred)[0, 1])

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Score": score,
        "Bias": bias,
        "Under_%": under,
        "Over_%": over,
        "Pearson_r": pearson,
        "R2": r2,
    }


# =========================
# Main
# =========================
if __name__ == "__main__":
    set_seed()

    print("=" * 72)
    print("C-MAPSS FD001 — Proposed LGTF-Net")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print(f"Subset: {SUBSET}")
    print(f"Window: {WINDOW}")
    print(f"RUL cap: {RMAX}")
    print()

    # 1. Load
    train_raw, test_raw, rul_raw = load_subset(SUBSET)

    train_raw = add_train_rul(train_raw)
    test_raw = add_test_rul(test_raw, rul_raw)

    # 2. Engine-wise train/validation split
    engines = train_raw["unit"].unique()

    tr_eng, va_eng = train_test_split(
        engines,
        test_size=VAL_RATIO,
        random_state=SEED,
    )

    tr_df = train_raw[train_raw["unit"].isin(tr_eng)].copy()
    va_df = train_raw[train_raw["unit"].isin(va_eng)].copy()

    # 3. Train-only standardization
    # Create independent float DataFrames to avoid pandas dtype conflicts.

    tr_features = tr_df[FEATS].astype(np.float32)
    va_features = va_df[FEATS].astype(np.float32)
    te_features = test_raw[FEATS].astype(np.float32)

    mu = tr_features.mean()
    sd = tr_features.std().replace(0, 1)

    tr_scaled = ((tr_features - mu) / sd).astype(np.float32)
    va_scaled = ((va_features - mu) / sd).astype(np.float32)
    te_scaled = ((te_features - mu) / sd).astype(np.float32)

    # Replace the feature blocks rather than assigning float values
    # into the original integer-typed columns.
    tr_df = pd.concat(
        [
            tr_df.drop(columns=FEATS),
            tr_scaled
        ],
        axis=1
    )

    va_df = pd.concat(
        [
            va_df.drop(columns=FEATS),
            va_scaled
        ],
        axis=1
    )

    test_raw = pd.concat(
        [
            test_raw.drop(columns=FEATS),
            te_scaled
        ],
        axis=1
    )

    # 4. Windows
    Xtr, ytr, _ = make_windows(tr_df)
    Xva, yva, _ = make_windows(va_df)
    Xte, yte, te_units = make_windows(test_raw)

    print(f"Train engines: {len(tr_eng)}")
    print(f"Val engines:   {len(va_eng)}")
    print(f"Train windows: {Xtr.shape}")
    print(f"Val windows:   {Xva.shape}")
    print(f"Test windows:  {Xte.shape}")
    print()

    # 5. Data loaders
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

    # 6. Model
    model = LGTFNet(
        in_features=len(FEATS),
        hidden=64,
        dropout=0.20,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = nn.MSELoss()

    # 7. Training
    best_val = float("inf")
    best_epoch = 0
    patience_count = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss = 0.0

        for xb, yb in dl_tr:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(yb)

        train_mse = train_loss / len(ytr)
        train_rmse = float(np.sqrt(train_mse))

        # Validation
        model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(yb)

        val_mse = val_loss / len(yva)
        val_rmse = float(np.sqrt(val_mse))

        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
            }
        )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"train MSE {train_mse:9.3f} | "
                f"train RMSE {train_rmse:7.3f} | "
                f"val RMSE {val_rmse:7.3f}"
            )

        if val_rmse < best_val - MIN_DELTA:
            best_val = val_rmse
            best_epoch = epoch
            patience_count = 0

            torch.save(
                model.state_dict(),
                CHECKPOINT,
            )
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best epoch = {best_epoch}, "
                f"best val RMSE = {best_val:.3f}"
            )
            break

    pd.DataFrame(history).to_csv(
        HISTORY_FILE,
        index=False,
    )

    # 8. Test
    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    model.eval()

    with torch.no_grad():
        pred_all = []

        for start in range(0, len(Xte), BATCH):
            xb = torch.from_numpy(
                Xte[start:start + BATCH]
            ).to(DEVICE)

            pred = model(xb)
            pred_all.append(pred.cpu().numpy())

        pred_all = np.concatenate(pred_all)

    # One final prediction per test engine
    pred_last = []
    true_last = []
    unit_last = []

    for u in np.unique(te_units):
        idx = np.where(te_units == u)[0]
        j = idx[-1]

        pred_last.append(pred_all[j])
        true_last.append(yte[j])
        unit_last.append(u)

    pred_last = np.asarray(pred_last)
    true_last = np.asarray(true_last)
    unit_last = np.asarray(unit_last)

    metrics = regression_metrics(
        true_last,
        pred_last,
    )

    # 9. Save predictions
    pred_df = pd.DataFrame(
        {
            "unit": unit_last,
            "true_RUL": true_last,
            "pred_RUL": pred_last,
            "error": pred_last - true_last,
        }
    )

    pred_df.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    # 10. Print results
    print()
    print("=" * 72)
    print("FD001 TEST RESULTS — Proposed LGTF-Net")
    print("=" * 72)

    print(f"RMSE       : {metrics['RMSE']:.3f}")
    print(f"MAE        : {metrics['MAE']:.3f}")
    print(f"C-MAPSS    : {metrics['Score']:.2f}")
    print(f"Bias       : {metrics['Bias']:.3f}")
    print(f"Under-est. : {metrics['Under_%']:.1f}%")
    print(f"Over-est.  : {metrics['Over_%']:.1f}%")
    print(f"Pearson r  : {metrics['Pearson_r']:.4f}")
    print(f"R^2        : {metrics['R2']:.4f}")
    print()
    print(f"Best epoch : {best_epoch}")
    print(f"Best val RMSE : {best_val:.3f}")
    print()
    print(f"Prediction file : {PREDICTION_FILE}")
    print(f"Training history: {HISTORY_FILE}")
    print(f"Checkpoint      : {CHECKPOINT}")
    print("=" * 72)
