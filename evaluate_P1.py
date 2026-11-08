# -*- coding: utf-8 -*-
"""
P1 评估管线：分组重复交叉验证 + 全指标同源 + 显著性检验
- 数据：239 条真实观测（不做插值，不做全库 IQR 清洗）
- 划分：按配合比分组（水泥类型 × 骨料），5 折 × R 次重复
- 标度器：仅在训练折内拟合（消除泄漏）
- 扩散模型：仅用训练折重训（消除泄漏），PCCDM 用先验采样
"""
import numpy as np, pandas as pd, re, torch, torch.nn as nn, torch.optim as optim, warnings, json, time
warnings.filterwarnings('ignore')
import os
from sklearn.model_selection import StratifiedGroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.neighbors import NearestNeighbors
from lightgbm import LGBMRegressor
from scipy import stats
import pccdm_model as M
import train as T

import os
SEED_BASE = 20261106
N_SPLITS = 5
N_REPEATS = int(os.environ.get('N_REPEATS', '2'))
N_GEN_MULT = 2.5
EPOCHS = 600
torch.set_num_threads(1)

# ---------------- 数据 ----------------
def firstnum(s):
    if pd.isna(s): return np.nan
    m = re.findall(r'[0-9]+\.?[0-9]*', str(s))
    return float(m[0]) if m else np.nan

df = pd.read_csv('data/processed/concrete_processed.csv')
df['agg_c'] = df['骨料水泥比'].map(firstnum)
FEATS = ['快中子辐照注量 n/cm2', '辐照温度', '水灰比', 'agg_c', 'concrete_type_encoded']
Y = df['相对抗压强度'].values.astype(np.float64).reshape(-1, 1)
Xr = df[FEATS].copy()
Xr['快中子辐照注量 n/cm2'] = np.log1p(Xr['快中子辐照注量 n/cm2'])
Xr = Xr.values.astype(np.float64)
groups = (df['水泥类型'].astype(str) + '|' + df['其它特殊成分'].astype(str)).values
print('样本 %d, 特征 %d, 分组 %d' % (len(Y), Xr.shape[1], len(set(groups))))

# ---------------- 下游预测器 ----------------
GBM = dict(n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
           min_child_samples=10, subsample=0.9, colsample_bytree=0.9, verbose=-1)

def fit_eval(Xtr, ytr, Xte, yte, sd):
    m = LGBMRegressor(random_state=sd, **GBM)
    m.fit(Xtr, ytr.ravel())
    p = m.predict(Xte)
    return dict(R2=r2_score(yte.ravel(), p),
                RMSE=float(np.sqrt(mean_squared_error(yte.ravel(), p))),
                MAE=mean_absolute_error(yte.ravel(), p),
                MAPE=mean_absolute_percentage_error(yte.ravel(), p) * 100)

# ---------------- 增广 ----------------
def aug_gaussian(X, y, n, sd):
    r = np.random.RandomState(sd)
    idx = r.randint(0, len(X), n)
    return X[idx] + r.normal(0, 0.1, (n, X.shape[1])), y[idx] + r.normal(0, 0.1, (n, 1))

def aug_over(X, y, n, sd):
    r = np.random.RandomState(sd)
    idx = r.randint(0, len(X), n)
    return X[idx].copy(), y[idx].copy()

def aug_smote(X, y, n, sd):
    r = np.random.RandomState(sd)
    k = min(5, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, ind = nn.kneighbors(X)
    out_x, out_y = np.zeros((n, X.shape[1])), np.zeros((n, 1))
    for i in range(n):
        a = r.randint(0, len(X)); b = r.choice(ind[a, 1:])
        lam = r.rand()
        out_x[i] = X[a] + lam * (X[b] - X[a]) + r.normal(0, 0.05, X.shape[1])
        out_y[i] = (y[a] + lam * (y[b] - y[a])).item()
    return out_x, out_y

_RAW = {'X': None, 'y': None}

def train_diffusion(X_raw, y_raw, sd):
    """X_raw / y_raw 保持物理量纲：物理边界 [0.1, 2.0] 定义在物理量纲上。
    早期版本传入标准化后的数据，使边界被施加在标准化空间，已修正。"""
    torch.manual_seed(sd); np.random.seed(sd)
    cfg = M.Config()
    LR, PHYS_W = 1e-3, 0.1
    model = M.PhysicsAwareResNetModel(cfg)
    opt = optim.Adam(model.parameters(), lr=LR)
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps)
    ac = torch.cumprod(1 - betas, dim=0)
    n = len(y_raw)
    ym, ys = y_raw.mean(0), y_raw.std(0) + 1e-8
    xm, xs = X_raw.mean(0), X_raw.std(0) + 1e-8
    ysc = torch.tensor((y_raw - ym) / ys, dtype=torch.float32)
    csc = torch.tensor((X_raw - xm) / xs, dtype=torch.float32)
    bs = min(128, n)
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n - bs + 1, bs):
            b = perm[i:i + bs]
            yb, cb = ysc[b], csc[b]
            t = torch.randint(0, cfg.timesteps, (len(b),)).long()
            noise = torch.randn_like(yb)
            at = ac[t].view(-1, 1)
            yn = torch.sqrt(at) * yb + torch.sqrt(1 - at) * noise
            pred = model(yn, t, cb)
            loss = nn.MSELoss()(pred, noise)
            x0 = (yn - torch.sqrt(1 - at) * pred) / torch.sqrt(at)
            loss = loss + PHYS_W * T.compute_physics_loss(x0, cb, ym, ys, xm, xs)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    cfg2 = M.Config()
    return model, M.DiffusionSampler(cfg2, torch.device('cpu')), (ym, ys, xm, xs)

def aug_pccdm(X, y, n, sd):
    # X, y 为标准化空间（用于采样与下游）；物理损失需原始量纲，取自 _RAW
    model, sampler, _ = train_diffusion(_RAW['X'], _RAW['y'], sd)
    r = np.random.RandomState(sd)
    idx = r.randint(0, len(X), n)
    Xc = X[idx]
    with torch.no_grad():
        Yg = sampler.generate(model, torch.tensor(Xc, dtype=torch.float32)).cpu().numpy()
    if Yg.ndim == 1: Yg = Yg.reshape(-1, 1)
    return Xc + r.normal(0, 0.001, Xc.shape), Yg.astype(np.float64)

METHODS = [('Raw (no aug)', None), ('Gaussian noise', aug_gaussian),
           ('Random oversampling', aug_over), ('SMOTE', aug_smote), ('PCCDM (prior sampling)', aug_pccdm)]

# ---------------- 主循环 ----------------
rows = {k: [] for k, _ in METHODS}
t0 = time.time()
for rep in range(N_REPEATS):
    sd = SEED_BASE + rep
    if os.environ.get('SPLIT', 'grouped') == 'random':
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=sd).split(Xr)
        print('  [SPLIT=random]')
    else:
        strat = (Y.ravel() > np.median(Y)).astype(int)
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                        random_state=sd).split(Xr, strat, groups)
    # 汇总折外预测：每次重复得到一组完整 OOF 预测，再统一计指标
    oof_pred = {k: np.zeros(len(Y)) for k, _ in METHODS}
    for fold, (tri, tei) in enumerate(splitter):
        med = np.nanmedian(Xr[tri], axis=0)
        Xraw_tr = np.where(np.isnan(Xr[tri]), med, Xr[tri])
        Xraw_te = np.where(np.isnan(Xr[tei]), med, Xr[tei])
        _RAW['X'], _RAW['y'] = Xraw_tr, Y[tri]
        sc_x, sc_y = StandardScaler(), StandardScaler()
        Xtr = sc_x.fit_transform(Xraw_tr); Xte = sc_x.transform(Xraw_te)
        ytr = sc_y.fit_transform(Y[tri]); yte = sc_y.transform(Y[tei])
        Xtr = np.nan_to_num(Xtr, nan=0.0); Xte = np.nan_to_num(Xte, nan=0.0)
        n_gen = int(len(Xtr) * N_GEN_MULT)
        for name, fn in METHODS:
            if fn is None:
                Xa, ya = Xtr, ytr
            else:
                Xs, ysg = fn(Xtr, ytr, n_gen, sd * 100 + fold)
                Xa = np.vstack([Xtr, Xs]); ya = np.vstack([ytr, ysg])
            m = LGBMRegressor(random_state=sd * 100 + fold, **GBM)
            m.fit(Xa, ya.ravel())
            oof_pred[name][tei] = sc_y.inverse_transform(
                m.predict(Xte).reshape(-1, 1)).ravel()
    for name, _ in METHODS:
        p, t = oof_pred[name], Y.ravel()
        rows[name].append(dict(R2=r2_score(t, p),
                               RMSE=float(np.sqrt(mean_squared_error(t, p))),
                               MAE=mean_absolute_error(t, p),
                               MAPE=mean_absolute_percentage_error(t, p) * 100,
                               r=float(np.corrcoef(t, p)[0, 1])))
    print('  repeat %d/%d done  (%.0fs)' % (rep + 1, N_REPEATS, time.time() - t0))

# ---------------- 汇总 ----------------
print('\n' + '=' * 96)
print('%-24s %16s %16s %16s' % ('Method', 'R2', 'RMSE', 'MAE'))
print('-' * 96)
summ = {}
for name, _ in METHODS:
    d = rows[name]
    summ[name] = {k: (np.mean([x[k] for x in d]), np.std([x[k] for x in d], ddof=1)) for k in d[0]}
    print('%-24s %8.4f±%-7.4f %8.4f±%-7.4f %8.4f±%-7.4f   r=%.3f' % (
        name, summ[name]['R2'][0], summ[name]['R2'][1],
        summ[name]['RMSE'][0], summ[name]['RMSE'][1],
        summ[name]['MAE'][0], summ[name]['MAE'][1], summ[name]['r'][0]))
print('=' * 96)
print('\n配对 Wilcoxon（对 Raw baseline，按折配对，n=%d）' % len(rows['Raw (no aug)']))
base = np.array([x['R2'] for x in rows['Raw (no aug)']])
for name, _ in METHODS[1:]:
    a = np.array([x['R2'] for x in rows[name]])
    d = a - base
    try: p = stats.wilcoxon(a, base).pvalue
    except Exception: p = float('nan')
    print('  %-24s ΔR2=%+.4f  正值 %2d/%2d  p=%.4f  %s' % (
        name, d.mean(), int((d > 0).sum()), len(d), p, '显著' if p < 0.05 else '不显著'))

OUT = '_p1_%s_r%d.json' % (os.environ.get('SPLIT', 'grouped'), N_REPEATS)
json.dump({k: rows[k] for k in rows}, open(OUT, 'w'), indent=1)
print('\nsaved %s   总耗时 %.0fs' % (OUT, time.time() - t0))
