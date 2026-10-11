"""
C-MAPSS 可视化 —— 一次跑出 4 张论文级图
输出: figures/fig1_degradation.png, fig2_corr.png, fig3_regime.png, fig4_variance.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.cluster import KMeans

# ---------- 全局风格 ----------
sns.set_theme(style="whitegrid", context="talk", font="DejaVu Sans")
mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

DATA_DIR = r"E:\project2026.np"
FIG_DIR  = r"E:\project2026.np\figures"
os.makedirs(FIG_DIR, exist_ok=True)

COLS = ["unit","cycle","os1","os2","os3"] + [f"s{i}" for i in range(1,22)]
SENS = [f"s{i}" for i in range(1,22)]


def load(subset):
    df = pd.read_csv(os.path.join(DATA_DIR, f"train_{subset}.txt"),
                     sep=r"\s+", header=None, names=COLS)
    mx = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (mx - df["cycle"]).clip(upper=125)
    return df


# ============ 图1：单台发动机退化轨迹 + RUL ============
def fig1_degradation(df, unit_id=1):
    g = df[df["unit"] == unit_id].sort_values("cycle")
    # C-MAPSS 文献里常用的 6 个"有信息"传感器（FD001）
    show = ["s2", "s3", "s4", "s7", "s11", "s12", "s15", "s17", "s20", "s21"]

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(show)))
    for c, col in zip(cmap, show):
        # 归一化到 [0,1] 画在同一坐标
        v = g[col].values
        v = (v - v.min()) / (v.max() - v.min() + 1e-9)
        ax1.plot(g["cycle"], v, lw=1.4, color=c, label=col, alpha=0.85)
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("Normalized sensor reading", color="gray")
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(g["cycle"], g["RUL"], color="crimson", lw=2.6, ls="--",
             label="RUL (true)")
    ax2.set_ylabel("RUL (cycles)", color="crimson")
    ax2.tick_params(axis="y", colors="crimson")
    ax2.spines["top"].set_visible(False)

    # 合并图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, ncol=4, fontsize=9,
               loc="upper right", framealpha=0.9)

    ax1.set_title(f"Run-to-failure trajectory — Unit #{unit_id} (FD001)",
                  fontweight="bold", pad=12)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "fig1_degradation.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print("  saved:", p)


# ============ 图2：传感器相关矩阵热力图 ============
def fig2_correlation(df):
    corr = df[SENS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True,
                annot=False, cbar_kws={"shrink": 0.75, "label": "Pearson r"},
                ax=ax)
    ax.set_title("Sensor correlation matrix — FD001 (train)",
                 fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "fig2_correlation.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print("  saved:", p)


# ============ 图3：单台发动机的多工况切换轨迹 (FD004)，不用 KMeans ============
def fig3_regime(df, unit_id=1):
    g = df[df["unit"] == unit_id].sort_values("cycle").reset_index(drop=True)

    # 3 个操作设置归一化到 [0,1] 画在同一轴
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    colors = {"os1": "#1f77b4", "os2": "#2ca02c", "os3": "#9467bd"}
    labels = {"os1": "Op. setting 1", "os2": "Op. setting 2", "os3": "Op. setting 3"}
    for col in ["os1", "os2", "os3"]:
        v = g[col].values
        v = (v - v.min()) / (v.max() - v.min() + 1e-9)
        ax1.plot(g["cycle"], v, lw=1.6, color=colors[col], label=labels[col], alpha=0.9)
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("Normalized operating setting", color="gray")
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(g["cycle"], g["RUL"], color="crimson", lw=2.6, ls="--",
             label="RUL (true)")
    ax2.set_ylabel("RUL (cycles)", color="crimson")
    ax2.tick_params(axis="y", colors="crimson")
    ax2.spines["top"].set_visible(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=10,
               framealpha=0.9, ncol=2)

    ax1.set_title(f"Multi-regime operation over the whole life — Unit #{unit_id} (FD004)",
                  fontweight="bold", pad=12)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "fig3_regime.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print("  saved:", p)


# ============ 图4：传感器方差排序 ============
def fig4_variance(df):
    var = df[SENS].var().sort_values(ascending=True)
    colors = ["#d62728" if v < 1e-6 else "#1f77b4" for v in var.values]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(var.index, var.values, color=colors, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Variance (log scale)")
    ax.set_title("Sensor variance ranking — FD001\n"
                 "(red ≈ constant sensors, to be dropped)",
                 fontweight="bold", pad=10)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "fig4_variance.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print("  saved:", p)


# ============ 图5：3D 退化地形曲面 (FD001) ============
def fig5_3d_surface(df, unit_id=1):
    g = df[df["unit"] == unit_id].sort_values("cycle").reset_index(drop=True)
    # 选 10 个有信息的传感器
    sensors = ["s2", "s3", "s4", "s7", "s11", "s12", "s15", "s17", "s20", "s21"]

    # 每个传感器归一化到 [0,1]
    Z = g[sensors].values.astype(float)
    Z = (Z - Z.min(0)) / (Z.max(0) - Z.min(0) + 1e-9)

    # 采样 cycle 到 ~120 个点，曲面不至于太密
    step = max(1, len(g) // 120)
    Z = Z[::step]
    cycles = g["cycle"].values[::step]

    X, Y = np.meshgrid(np.arange(len(sensors)), cycles)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis",
                           edgecolor="none", alpha=0.92,
                           antialiased=True, rstride=1, cstride=1)
    ax.set_xticks(np.arange(len(sensors)))
    ax.set_xticklabels(sensors, fontsize=9)
    ax.set_xlabel("Sensor", labelpad=10)
    ax.set_ylabel("Cycle", labelpad=10)
    ax.set_zlabel("Normalized reading", labelpad=10)
    ax.view_init(elev=25, azim=-60)
    ax.set_title(f"Degradation landscape — Unit #{unit_id} (FD001)",
                 fontweight="bold", pad=15)
    fig.colorbar(surf, shrink=0.55, aspect=12, pad=0.1,
                 label="Normalized reading")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "fig5_3d_surface.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print("  saved:", p)


# ============ 主流程 ============
if __name__ == "__main__":
    print("加载 FD001 ...")
    df001 = load("FD001")
    print("加载 FD004 ...")
    df004 = load("FD004")

    print("\n开始画图：")
    fig1_degradation(df001, unit_id=1)
    fig2_correlation(df001)
    fig3_regime(df004, unit_id=1)
    fig4_variance(df001)
    fig5_3d_surface(df001, unit_id=1)

    print(f"\n[OK] 全部图已保存到: {FIG_DIR}")
