"""
baseline_mlp.py  ——  D1 最小可跑 baseline
纯 PyTorch，不依赖 Chronos，跑 FD001 出第一个 RMSE
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# ============ 配置 ============
DATA_DIR = r"F:\Users\jsj_s\Desktop\Code_testing\venv\project2026.np"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
WINDOW = 30
RMAX = 125
BATCH = 256
EPOCHS = 60
LR = 1e-3

COLS = ["unit","cycle","os1","os2","os3"] + [f"s{i}" for i in range(1,22)]
SENSOR_COLS = [f"s{i}" for i in range(1,22)]
FEATS = SENSOR_COLS + ["os1","os2","os3"]   # 24 维

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============ 数据加载 ============
def load_subset(subset):
    train = pd.read_csv(os.path.join(DATA_DIR, f"train_{subset}.txt"),
                        sep=r"\s+", header=None, names=COLS)
    test  = pd.read_csv(os.path.join(DATA_DIR, f"test_{subset}.txt"),
                        sep=r"\s+", header=None, names=COLS)
    rul   = pd.read_csv(os.path.join(DATA_DIR, f"RUL_{subset}.txt"),
                        sep=r"\s+", header=None, names=["RUL"])
    return train, test, rul


def make_windows(df, window=WINDOW, label_col="RUL"):
    """每台发动机切成窗口，标签取窗口最后一帧"""
    Xs, ys, units = [], [], []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle").reset_index(drop=True)
        feats = g[FEATS].values.astype(np.float32)
        labels = g[label_col].values.astype(np.float32)
        for i in range(len(g) - window + 1):
            Xs.append(feats[i:i+window].flatten())   # 拉平成 [window*24]
            ys.append(labels[i+window-1])
            units.append(unit)
    return np.array(Xs), np.array(ys), np.array(units)


# ============ 模型 ============
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 64),     nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============ C-MAPSS 官方 Score 函数 ============
def cmapss_score(y_true, y_pred):
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d/13) - 1, np.exp(d/10) - 1)
    return float(s.sum())


# ============ 主流程 ============
if __name__ == "__main__":
    subset = "FD001"
    print(f"设备: {DEVICE} | 子集: {subset}\n")

    # --- 1. 加载 ---
    train_raw, test_raw, rul_raw = load_subset(subset)

    # 训练集 RUL（piecewise linear cap=125）
    mx = train_raw.groupby("unit")["cycle"].transform("max")
    train_raw["RUL"] = (mx - train_raw["cycle"]).clip(upper=RMAX)

    # 测试集：把 RUL 接到每台发动机的最后一行
    last_idx = test_raw.groupby("unit").tail(1).index
    test_raw.loc[last_idx, "RUL"] = rul_raw["RUL"].values
    test_raw["RUL"] = test_raw["RUL"].ffill().clip(upper=RMAX)

    # --- 2. 按 engine 划分 train/val（防泄漏） ---
    engines = train_raw["unit"].unique()
    tr_eng, va_eng = train_test_split(engines, test_size=0.2, random_state=SEED)
    tr_df = train_raw[train_raw["unit"].isin(tr_eng)]
    va_df = train_raw[train_raw["unit"].isin(va_eng)]

    # --- 3. 归一化（用 train 的 mu/sd） ---
    mu = tr_df[FEATS].mean()
    sd = tr_df[FEATS].std().replace(0, 1)
    tr_df[FEATS] = (tr_df[FEATS] - mu) / sd
    va_df[FEATS] = (va_df[FEATS] - mu) / sd
    test_raw[FEATS] = (test_raw[FEATS] - mu) / sd

    # --- 4. 窗口化 ---
    Xtr, ytr, _ = make_windows(tr_df)
    Xva, yva, _ = make_windows(va_df)
    Xte, yte, te_units = make_windows(test_raw)

    print(f"train 窗口: {Xtr.shape}")
    print(f"val   窗口: {Xva.shape}")
    print(f"test  窗口: {Xte.shape}  ({len(np.unique(te_units))} 台发动机)\n")

    # --- 5. DataLoader ---
    dl_tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                       batch_size=BATCH, shuffle=True)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                       batch_size=BATCH)

    # --- 6. 模型 ---
    model = MLP(in_dim=Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}\n")

    # --- 7. 训练 ---
    best_val = 1e9
    for ep in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(ytr)

        model.eval()
        va_loss = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                va_loss += ((model(xb) - yb)**2).sum().item()
        rmse_va = np.sqrt(va_loss / len(yva))

        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep:3d} | train MSE {tr_loss:.2f} | val RMSE {rmse_va:.3f}")
        if rmse_va < best_val:
            best_val = rmse_va
            torch.save(model.state_dict(), "best_mlp.pt")

    # --- 8. 测试集评估 ---
    model.load_state_dict(torch.load("best_mlp.pt", map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(Xte).to(DEVICE)
        pred_all = model(xt).cpu().numpy()

    # 每台发动机取最后一个窗口的预测
    pred_last, true_last = [], []
    for u in np.unique(te_units):
        idx = np.where(te_units == u)[0]
        pred_last.append(pred_all[idx[-1]])
        true_last.append(yte[idx[-1]])
    pred_last = np.array(pred_last)
    true_last = np.array(true_last)

    rmse_test = float(np.sqrt(((pred_last - true_last)**2).mean()))
    score_test = cmapss_score(true_last, pred_last)

    print(f"\n{'='*45}")
    print(f"  FD001 测试集结果")
    print(f"  RMSE : {rmse_test:.3f}")
    print(f"  Score: {score_test:.1f}")
    print(f"{'='*45}")
    print("\n参考：Wide&Deep 那篇论文 FD001 RMSE=9.97；MLP baseline 应该在 12-14 之间")
#（注：内容由AI生成）
