# ============================================================
# FD001 Performance Comparison
# Publication-quality visualization
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ------------------------------------------------------------
# 1. Path configuration
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

print("Script directory:")
print(BASE_DIR)
print("Figures directory:")
print(FIGURES_DIR)

# ------------------------------------------------------------
# 2. Global plotting style
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# ------------------------------------------------------------
# 3. File configuration
# ------------------------------------------------------------
DATA_DIR = BASE_DIR   # CSV 数据和脚本同目录

files = {
    "CNN": "FD001_cnn_predictions.csv",
    "LGTF-Net": "FD001_LGTF-Net_predictions.csv",
    "Linear": "FD001_linear_predictions.csv",
    "LSTM": "FD001_lstm_predictions.csv",
    "MLP": "FD001_mlp_predictions.csv",
    "Proposed": "FD001_proposed_predictions.csv",
    "Transformer": "FD001_transformer_predictions.csv",
}

history_file = os.path.join(BASE_DIR, "proposed_training_history.csv")

# ------------------------------------------------------------
# 4. Load prediction results
# ------------------------------------------------------------
predictions = {}
for model, filename in files.items():
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    predictions[model] = pd.read_csv(path)

# ------------------------------------------------------------
# 5. Calculate performance metrics
# ------------------------------------------------------------
results = []
for model, df in predictions.items():
    error = df["pred_RUL"] - df["true_RUL"]
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    y_true = df["true_RUL"].values
    y_pred = df["pred_RUL"].values
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    results.append({"Model": model, "RMSE": rmse, "MAE": mae, "R2": r2})

metrics = pd.DataFrame(results)
print("\n================ PERFORMANCE =================")
print(metrics.round(4).to_string(index=False))

# ------------------------------------------------------------
# 6. Model order
# ------------------------------------------------------------
model_order = ["Linear", "MLP", "CNN", "LSTM", "Transformer", "LGTF-Net", "Proposed"]
metrics["Model"] = pd.Categorical(metrics["Model"], categories=model_order, ordered=True)
metrics = metrics.sort_values("Model")

# ============================================================
# FIGURE 1: RMSE + MAE comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
x = np.arange(len(metrics))
width = 0.68

ax = axes[0]
bars = ax.bar(x, metrics["RMSE"], width=width, edgecolor="black", linewidth=0.8)
ax.set_ylabel("RMSE")
ax.set_title("(a) RMSE Comparison", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics["Model"], rotation=35, ha="right")
ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
ax.set_axisbelow(True)
for bar, value in zip(bars, metrics["RMSE"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3,
            f"{value:.2f}", ha="center", va="bottom", fontsize=9)

ax = axes[1]
bars = ax.bar(x, metrics["MAE"], width=width, edgecolor="black", linewidth=0.8)
ax.set_ylabel("MAE")
ax.set_title("(b) MAE Comparison", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics["Model"], rotation=35, ha="right")
ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
ax.set_axisbelow(True)
for bar, value in zip(bars, metrics["MAE"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.2,
            f"{value:.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "Fig_performance_RMSE_MAE.png"))
plt.close()

# ============================================================
# FIGURE 2: Prediction vs Ground Truth
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for i, model in enumerate(model_order):
    ax = axes[i]
    df = predictions[model]
    y_true = df["true_RUL"].values
    y_pred = df["pred_RUL"].values

    ax.scatter(y_true, y_pred, s=30, alpha=0.65, edgecolors="none")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.2)

    row = metrics[metrics["Model"] == model].iloc[0]
    ax.text(0.05, 0.93,
            f"RMSE = {row['RMSE']:.2f}\nMAE = {row['MAE']:.2f}",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.set_title(model, fontweight="bold")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.35)

axes[-1].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "Fig_prediction_vs_true.png"))
plt.close()

# ============================================================
# FIGURE 3: Absolute error distribution
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5))
error_data = []
for model in model_order:
    df = predictions[model]
    abs_error = np.abs(df["pred_RUL"] - df["true_RUL"])
    error_data.append(abs_error)

bp = ax.boxplot(error_data, patch_artist=False, widths=0.62, showfliers=True)
ax.set_xticks(np.arange(1, len(model_order)+1))
ax.set_xticklabels(model_order, rotation=35, ha="right")
ax.set_ylabel("|Prediction Error|")
ax.set_title("Absolute Error Distribution", fontweight="bold")
ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "Fig_absolute_error_distribution.png"))
plt.close()

# ============================================================
# FIGURE 4: Proposed model training history
# ============================================================
if os.path.exists(history_file):
    history = pd.read_csv(history_file)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(history["epoch"], history["train_rmse"],
            marker="o", markersize=3.5, linewidth=1.8, label="Training RMSE")
    ax.plot(history["epoch"], history["val_rmse"],
            marker="s", markersize=3.5, linewidth=1.8, label="Validation RMSE")

    best_idx = history["val_rmse"].idxmin()
    best_epoch = history.loc[best_idx, "epoch"]
    best_rmse = history.loc[best_idx, "val_rmse"]

    ax.scatter(best_epoch, best_rmse, s=70, zorder=5)
    ax.annotate(f"Best = {best_rmse:.2f}\nEpoch {int(best_epoch)}",
                xy=(best_epoch, best_rmse),
                xytext=(best_epoch+1, best_rmse+5),
                arrowprops=dict(arrowstyle="->", linewidth=1.0), fontsize=9)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("Training Convergence of Proposed Model", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(linestyle="--", linewidth=0.7, alpha=0.35)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Fig_pro posed_training_curve.png"))
    plt.close()

# ============================================================
# Save numerical results
# ============================================================
metrics.to_csv(os.path.join(FIGURES_DIR, "FD001_performance_comparison.csv"), index=False)

print("\nSaved files:")
print(f"  {os.path.join(FIGURES_DIR, 'Fig_performance_RMSE_MAE.png')}")
print(f"  {os.path.join(FIGURES_DIR, 'Fig_prediction_vs_true.png')}")
print(f"  {os.path.join(FIGURES_DIR, 'Fig_absolute_error_distribution.png')}")
print(f"  {os.path.join(FIGURES_DIR, 'Fig_proposed_training_curve.png')}")
print(f"  {os.path.join(FIGURES_DIR, 'FD001_performance_comparison.csv')}")
