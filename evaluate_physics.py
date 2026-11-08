# -*- coding: utf-8 -*-
"""
意见 #4 的物理指标管线
=========================
按新管线（真实 239 样本、配合比分组、训练折内拟合）重算：

1. 各增广方法的物理约束残差与 PVR
2. **真实观测的 PVR（T1）** —— 这是 PVR 的"地板"，旧稿用随机过采样间接近似，
   这里直接测量留出真实样本的违例率。
3. 各方法的 Wasserstein 距离（y 维，物理量纲）

关键方法学处理
--------------
PVR 的判据依赖样本量 N（window = N//10，比较第 i 与第 i+N//10 个样本），
所以生成样本（N≈525）与真实测试折（N≈48）的 PVR 不能直接比。
为此，生成样本的 PVR 采用**与测试折等量的子采样**（B 次重复取均值），
使两者在同一 N 下可比。
"""
import numpy as np, pandas as pd, re, torch, torch.nn as nn, torch.optim as optim
import warnings, json, time, os
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import wasserstein_distance
import pccdm_model as M
import train as T

torch.set_num_threads(1)

SEED_BASE = 20261106
N_SPLITS = 5
N_REPEATS = int(os.environ.get('N_REPEATS', '10'))
N_GEN_MULT = 2.5
EPOCHS = 600
LR = 1e-3
N_SUB = 30          # 子采样重复次数
PROG = '_physics_prog_r%d.jsonl' % N_REPEATS

def firstnum(s):
    if pd.isna(s):
        return np.nan
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
print('样本 %d | 分组 %d' % (len(Y), len(set(groups))), flush=True)

_RAW = {'X': None, 'y': None}

def train_diffusion(X_raw, y_raw, sd):
    torch.manual_seed(sd); np.random.seed(sd)
    cfg = M.Config()
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
            loss = loss + 0.1 * T.compute_physics_loss(x0, cb, ym, ys, xm, xs)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model

def sample_prior(model, Xc, seed):
    torch.manual_seed(seed)
    s = M.DiffusionSampler(M.Config(), torch.device('cpu'))
    with torch.no_grad():
        return s.generate(model, torch.tensor(Xc, dtype=torch.float32)).cpu().numpy()

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
    ox, oy = np.zeros((n, X.shape[1])), np.zeros((n, 1))
    for i in range(n):
        a = r.randint(0, len(X)); b = r.choice(ind[a, 1:])
        lam = r.rand()
        ox[i] = X[a] + lam * (X[b] - X[a]) + r.normal(0, 0.05, X.shape[1])
        oy[i] = (y[a] + lam * (y[b] - y[a])).item()
    return ox, oy

def aug_pccdm(X, y, n, sd):
    mdl = train_diffusion(_RAW['X'], _RAW['y'], sd)
    r = np.random.RandomState(sd)
    idx = r.randint(0, len(X), n)
    Xc = X[idx]
    Yg = sample_prior(mdl, Xc, sd).reshape(-1, 1)
    return Xc + r.normal(0, 0.001, Xc.shape), Yg.astype(np.float64)

METHODS = [('Raw (no aug)', None), ('Gaussian noise', aug_gaussian),
           ('Random oversampling', aug_over), ('SMOTE', aug_smote),
           ('PCCDM (prior sampling)', aug_pccdm)]
NAMES = [m[0] for m in METHODS]

def metrics_scaled(ev, Xs, ys, n_ref, rng):
    """在标准化空间上算 PVR / 残差；PVR 用子采样对齐到 n_ref 个样本。"""
    N = len(ys)
    if N > n_ref:
        idx = rng.choice(N, n_ref, replace=False)
        Xq, yq = Xs[idx], ys[idx]
    else:
        Xq, yq = Xs, ys
    pvr = ev.compute_physics_violation_rate(Xq, yq)
    res = ev.compute_physics_residual_error(Xq, yq)
    return pvr, res

results = {k: [] for k in NAMES}
truth = []
t0 = time.time()
done = 0
if os.path.exists(PROG):
    for line in open(PROG):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get('rep', -1) != done:
            continue
        for k in NAMES:
            results[k].append(rec['res'][k])
        truth.append(rec['truth'])
        done += 1
    print('断点续跑：已完成 %d/%d' % (done, N_REPEATS), flush=True)

for rep in range(done, N_REPEATS):
    sd = SEED_BASE + rep
    strat = (Y.ravel() > np.median(Y)).astype(int)
    sp = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=sd).split(Xr, strat, groups)
    acc = {k: [] for k in NAMES}
    tru = []
    for fold, (tri, tei) in enumerate(sp):
        fseed = sd * 100 + fold
        med = np.nanmedian(Xr[tri], axis=0)
        Xraw_tr = np.where(np.isnan(Xr[tri]), med, Xr[tri])
        Xraw_te = np.where(np.isnan(Xr[tei]), med, Xr[tei])
        sc_x, sc_y = StandardScaler(), StandardScaler()
        Xtr = sc_x.fit_transform(Xraw_tr); Xte = sc_x.transform(Xraw_te)
        ytr = sc_y.fit_transform(Y[tri]); yte = sc_y.transform(Y[tei])
        Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
        _RAW['X'], _RAW['y'] = Xraw_tr, Y[tri]
        ev = M.PhysicsResidualEvaluator(sc_y.mean_[0], sc_y.scale_[0],
                                        sc_x.mean_, sc_x.scale_)
        rng = np.random.RandomState(fseed)
        n_gen = int(len(Xtr) * N_GEN_MULT)
        n_ref = len(yte)

        # 真实留出样本（T1 基线）
        pvr_t, res_t = metrics_scaled(ev, Xte, yte, n_ref, rng)
        tru.append(dict(pvr=float(pvr_t), residual=float(res_t), n=int(n_ref),
                        wd_vs_train=float(wasserstein_distance(
                            sc_y.inverse_transform(yte).ravel(),
                            sc_y.inverse_transform(ytr).ravel()))))

        for name, fn in METHODS:
            if fn is None:
                Xs, ys = Xtr.copy(), ytr.copy()
                n_use = min(len(Xs), n_gen)
                ii = rng.choice(len(Xs), n_use, replace=False)
                Xs, ys = Xs[ii], ys[ii]
            else:
                Xs, ys = fn(Xtr, ytr, n_gen, fseed)
                ys = np.asarray(ys).reshape(-1, 1)
            pvr, res = metrics_scaled(ev, Xs, ys, n_ref, rng)
            wd = wasserstein_distance(sc_y.inverse_transform(ys).ravel(),
                                      sc_y.inverse_transform(ytr).ravel())
            acc[name].append(dict(pvr=float(pvr), residual=float(res), wd=float(wd)))

    for k in NAMES:
        results[k].append({f: float(np.mean([d[f] for d in acc[k]]))
                           for f in ('pvr', 'residual', 'wd')})
    truth.append({f: float(np.mean([d[f] for d in tru])) for f in ('pvr', 'residual', 'wd_vs_train')})
    with open(PROG, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'rep': rep,
                            'res': {k: results[k][-1] for k in NAMES},
                            'truth': truth[-1]}) + '\n')
    print('  repeat %d/%d  (%.0fs)' % (rep + 1, N_REPEATS, time.time() - t0), flush=True)

print('\n' + '=' * 92)
print('%-24s %10s %12s %12s' % ('Method', 'PVR(%)', 'Residual', 'WD'))
print('-' * 92)
tp = np.array([t['pvr'] for t in truth]); tr_ = np.array([t['residual'] for t in truth])
tw = np.array([t['wd_vs_train'] for t in truth])
print('%-24s %10.2f %12.5f %12.4f   <- 真实留出样本（T1 基线）' % (
    'Measured (held-out)', tp.mean(), tr_.mean(), tw.mean()))
for k in NAMES:
    p = np.array([x['pvr'] for x in results[k]])
    r_ = np.array([x['residual'] for x in results[k]])
    w = np.array([x['wd'] for x in results[k]])
    print('%-24s %10.2f %12.5f %12.4f' % (k, p.mean(), r_.mean(), w.mean()))
print('=' * 92)
print('\nPVR 是否优于真实基线（配对单侧检验）：')
from scipy import stats as st
for k in NAMES:
    p = np.array([x['pvr'] for x in results[k]])
    try:
        pv = st.wilcoxon(p, tp, alternative='less').pvalue
    except Exception:
        pv = float('nan')
    print('  %-24s ΔPVR=%+.2f  p(更优)=%.4f' % (k, p.mean() - tp.mean(), pv))

json.dump({'results': results, 'truth': truth},
          open('_physics_r%d.json' % N_REPEATS, 'w'), indent=1)
print('\nsaved _physics_r%d.json  %.0fs' % (N_REPEATS, time.time() - t0))
