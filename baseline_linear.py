"""
C-MAPSS / FD001 Linear Regression baseline.

Same protocol as the validated MLP and 1D-CNN baselines:
- official C-MAPSS train/test split
- engine-wise 80/20 train/validation split
- train-only standardization
- 30-cycle sliding window
- RUL cap = 125
- official 100-engine test set
- one final prediction per test engine

Model:
- Flatten 30 x 24 input features -> Linear(720, 1)

Outputs:
- validation RMSE
- test RMSE
- MAE
- C-MAPSS Score
- Bias
- under/over-estimation rate
- Pearson correlation
- R2
- FD001_linear_predictions.csv
"""

import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ============================================================
# Configuration
# ============================================================

DATA_DIR = r"F:\Users\jsj_s\Desktop\Code_testing\venv\project2026.np"

SUBSET = "FD001"

SEED = 42
WINDOW = 30
RMAX = 125
VAL_RATIO = 0.20

RESULT_PATH = f"{SUBSET}_linear_predictions.csv"


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


# ============================================================
# Data loading
# ============================================================

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


# ============================================================
# RUL labels
# ============================================================

def add_train_rul(df, rmax=RMAX):
    out = df.copy()

    max_cycle = out.groupby("unit")["cycle"].transform("max")

    out["RUL"] = (
        max_cycle - out["cycle"]
    ).clip(upper=rmax)

    return out


def add_test_rul(test_df, rul_df, rmax=RMAX):
    out = test_df.copy().reset_index(drop=True)

    units = np.sort(out["unit"].unique())

    if len(units) != len(rul_df):
        raise ValueError(
            f"Test engine count ({len(units)}) "
            f"!= RUL rows ({len(rul_df)})"
        )

    rul_map = {
        int(unit): float(rul)
        for unit, rul in zip(
            units,
            rul_df["RUL"].values
        )
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
# Engine-wise split
# ============================================================

def split_by_engine(df, val_ratio=VAL_RATIO, seed=SEED):
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
    mu = df[features].mean()
    sd = df[features].std().replace(0, 1.0)

    return mu, sd


def apply_scaler(df, features, mu, sd):
    out = df.copy()

    scaled = (
        out[features]
        .astype(np.float32)
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
    Return:
        X = [N, T, F]
        y = [N]
        units = engine ID for every window
    """

    Xs = []
    ys = []
    units = []

    for unit, g in df.groupby("unit"):

        g = (
            g.sort_values("cycle")
            .reset_index(drop=True)
        )

        if len(g) < window:
            continue

        feats = g[FEATS].values.astype(np.float32)
        labels = g["RUL"].values.astype(np.float32)

        for i in range(len(g) - window + 1):

            Xs.append(
                feats[i:i + window]
            )

            ys.append(
                labels[i + window - 1]
            )

            units.append(unit)

    if not Xs:
        raise ValueError(
            f"No windows generated for window={window}"
        )

    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(units),
    )


# ============================================================
# Metrics
# ============================================================

def cmapss_score(y_true, y_pred):

    d = y_pred - y_true

    score = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1.0,
        np.exp(d / 10.0) - 1.0,
    )

    return float(score.sum())


def regression_metrics(y_true, y_pred):

    error = y_pred - y_true

    rmse = float(
        np.sqrt(np.mean(error ** 2))
    )

    mae = float(
        np.mean(np.abs(error))
    )

    bias = float(
        np.mean(error)
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    r2 = float(
        1.0 - ss_res / ss_tot
    )

    corr = float(
        np.corrcoef(
            y_true,
            y_pred
        )[0, 1]
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Score": cmapss_score(
            y_true,
            y_pred
        ),
        "Bias": bias,
        "UnderRate": float(
            np.mean(error < 0) * 100.0
        ),
        "OverRate": float(
            np.mean(error > 0) * 100.0
        ),
        "R2": r2,
        "Correlation": corr,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    set_seed()

    print("=" * 70)
    print("C-MAPSS LINEAR REGRESSION BASELINE")
    print("=" * 70)

    print(f"Subset : {SUBSET}")
    print(f"Window : {WINDOW}")
    print(f"RMAX   : {RMAX}")
    print(f"Seed   : {SEED}")

    print("=" * 70)

    # --------------------------------------------------------
    # 1. Official data
    # --------------------------------------------------------

    train_raw, test_raw, rul_raw = load_subset(
        SUBSET
    )

    print("\n[1] Raw data")

    print(f"train: {train_raw.shape}")
    print(f"test : {test_raw.shape}")
    print(f"RUL  : {rul_raw.shape}")

    # --------------------------------------------------------
    # 2. Labels
    # --------------------------------------------------------

    train_raw = add_train_rul(
        train_raw
    )

    test_raw = add_test_rul(
        test_raw,
        rul_raw
    )

    # --------------------------------------------------------
    # 3. Engine-wise split
    # --------------------------------------------------------

    tr_df, va_df, tr_eng, va_eng = (
        split_by_engine(train_raw)
    )

    overlap = set(tr_eng).intersection(
        set(va_eng)
    )

    if overlap:
        raise RuntimeError(
            f"Engine leakage detected: {sorted(overlap)}"
        )

    print("\n[2] Engine-wise split")

    print(f"train engines: {len(tr_eng)}")
    print(f"val engines  : {len(va_eng)}")

    # --------------------------------------------------------
    # 4. Train-only normalization
    # --------------------------------------------------------

    mu, sd = fit_scaler(
        tr_df,
        FEATS
    )

    tr_df = apply_scaler(
        tr_df,
        FEATS,
        mu,
        sd
    )

    va_df = apply_scaler(
        va_df,
        FEATS,
        mu,
        sd
    )

    test_scaled = apply_scaler(
        test_raw,
        FEATS,
        mu,
        sd
    )

    print("\n[3] Normalization")
    print(
        "Scaler fitted on training engines only."
    )

    # --------------------------------------------------------
    # 5. Sliding windows
    # --------------------------------------------------------

    Xtr, ytr, _ = make_windows(
        tr_df
    )

    Xva, yva, _ = make_windows(
        va_df
    )

    Xte, yte, te_units = make_windows(
        test_scaled
    )

    print("\n[4] Sliding windows")

    print(f"train: {Xtr.shape}")
    print(f"val  : {Xva.shape}")
    print(
        f"test : {Xte.shape} "
        f"({len(np.unique(te_units))} engines)"
    )

    # --------------------------------------------------------
    # 6. Flatten windows
    # --------------------------------------------------------

    Xtr_flat = Xtr.reshape(
        Xtr.shape[0],
        -1
    )

    Xva_flat = Xva.reshape(
        Xva.shape[0],
        -1
    )

    Xte_flat = Xte.reshape(
        Xte.shape[0],
        -1
    )

    print("\n[5] Linear input")

    print(
        f"Input dimension: {Xtr_flat.shape[1]}"
    )

    # --------------------------------------------------------
    # 7. Linear Regression
    # --------------------------------------------------------

    model = LinearRegression()

    model.fit(
        Xtr_flat,
        ytr
    )

    n_params = (
        model.coef_.size
        + 1
    )

    print(
        f"Model parameters: {n_params:,}"
    )

    # --------------------------------------------------------
    # 8. Validation
    # --------------------------------------------------------

    pred_val = model.predict(
        Xva_flat
    )

    val_metrics = regression_metrics(
        yva,
        pred_val
    )

    print("\n[6] Validation")

    print(
        f"Val RMSE : "
        f"{val_metrics['RMSE']:.4f}"
    )

    # --------------------------------------------------------
    # 9. Official test inference
    # --------------------------------------------------------

    pred_all = model.predict(
        Xte_flat
    )

    # --------------------------------------------------------
    # 10. Final engine-level prediction
    # --------------------------------------------------------

    pred_last = []
    true_last = []
    unit_last = []

    for unit in np.unique(
        te_units
    ):

        idx = np.where(
            te_units == unit
        )[0]

        last_idx = idx[-1]

        unit_last.append(unit)

        pred_last.append(
            pred_all[last_idx]
        )

        true_last.append(
            yte[last_idx]
        )

    pred_last = np.asarray(
        pred_last,
        dtype=np.float64
    )

    true_last = np.asarray(
        true_last,
        dtype=np.float64
    )

    unit_last = np.asarray(
        unit_last
    )

    # --------------------------------------------------------
    # 11. Final metrics
    # --------------------------------------------------------

    metrics = regression_metrics(
        true_last,
        pred_last
    )

    print("\n" + "=" * 70)
    print(f"{SUBSET} FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Test engines     : "
        f"{len(true_last)}"
    )

    print(
        f"RMSE             : "
        f"{metrics['RMSE']:.3f}"
    )

    print(
        f"MAE              : "
        f"{metrics['MAE']:.3f}"
    )

    print(
        f"Score            : "
        f"{metrics['Score']:.1f}"
    )

    print(
        f"Bias             : "
        f"{metrics['Bias']:.3f}"
    )

    print(
        f"Under-estimation : "
        f"{metrics['UnderRate']:.1f}%"
    )

    print(
        f"Over-estimation  : "
        f"{metrics['OverRate']:.1f}%"
    )

    print(
        f"Correlation      : "
        f"{metrics['Correlation']:.4f}"
    )

    print(
        f"R2               : "
        f"{metrics['R2']:.4f}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 12. Save predictions
    # --------------------------------------------------------

    result_df = pd.DataFrame({
        "unit": unit_last,
        "true_RUL": true_last,
        "pred_RUL": pred_last,
        "error": pred_last - true_last,
    })

    result_df.to_csv(
        RESULT_PATH,
        index=False
    )

    print(
        f"\nSaved predictions: "
        f"{RESULT_PATH}"
    )
