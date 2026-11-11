# -*- coding: utf-8 -*-
"""
意见 #3 消融评估管线
=========================
回答三类问题，全部在同一批折实例上完成（配对可比）：

1) 组件消融：分别关闭 L_bound 与 L_mono，以及两者全关（lambda_phy = 0，
   即无约束条件扩散），与完整模型对照。
2) 采样机制消融：同一套权重下，先验采样 vs SDEdit。
   复用 A0 已训练模型，仅换采样入口，故不受训练随机性干扰。
3) 下游预测器鲁棒性：GBDT / GPR / 单调约束 GBDT 三种预测器，
   在完全相同的增广数据上比较，回答"增益是否只是 GBDT 的阶梯拟合效应"。

物理量纲说明（重要）：
    物理损失中的边界 [0.1, 2.0] 与单调阈值均定义在物理量纲上，
    因此训练函数接收未标准化的 X_raw / y_raw，在其内部计算统计量并标准化。
    （早期版本传入标准化后的统计量，导致边界被施加在标准化空间，已修正。）

统计口径：
    分组 5 折（配合比 = 水泥类型 x 骨料）x R 次重复
    标度器与扩散模型只在训练折内拟合
    汇总全部折外预测后统一计算 R2/RMSE/MAE
    95% CI 用 t 分布；配对 Wilcoxon + Holm 校正 + rank-biserial 效应量
"""
import numpy as np, pandas as pd, re, torch, torch.nn as nn, torch.optim as optim
import warnings, json, time, os
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from lightgbm import LGBMRegressor
from scipy import stats
import pccdm_model as M

torch.set_num_threads(1)

SEED_BASE = 20261106
N_SPLITS = 5
N_REPEATS = int(os.environ.get('N_REPEATS', '3'))
N_GEN_MULT = 2.5
EPOCHS = 600
LAMBDA_PHY = 0.1
LR = 1e-3
SDEDIT_START = 20

# ---------------- 数据 ----------------
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
print('样本 %d | 特征 %d | 配合比组 %d' % (len(Y), Xr.shape[1], len(set(groups))), flush=True)

# ---------------- 可开关的物理损失（物理量纲） ----------------
def phys_loss(x0_hat, cond, ym, ys, xm, xs, use_bound=True, use_mono=True):
    y_m = torch.tensor(np.asarray(ym).ravel(), dtype=torch.float32)
    y_s = torch.tensor(np.asarray(ys).ravel(), dtype=torch.float32)
    x_m = torch.tensor(np.asarray(xm).ravel(), dtype=torch.float32)
    x_s = torch.tensor(np.asarray(xs).ravel(), dtype=torch.float32)
    # 反标准化到物理量纲
    strength = (x0_hat * y_s + y_m).squeeze()
    wc = cond[:, 2] * x_s[2] + x_m[2]
    fl = cond[:, 0] * x_s[0] + x_m[0]

    total = torch.zeros((), dtype=torch.float32)
    if use_bound:
        b = torch.mean(torch.relu(0.1 - strength) ** 2 + torch.relu(strength - 2.0) ** 2)
        total = total + b
    if use_mono and strength.shape[0] > 1:
        idx = torch.randperm(strength.shape[0])
        dw = wc - wc[idx]
        ds = strength - strength[idx]
        df_ = fl - fl[idx]
        wv = torch.where((dw > 0.05) & (ds > 0), ds ** 2, torch.zeros_like(ds))
        fv = torch.where((df_ > 0.5) & (ds > 0), ds ** 2, torch.zeros_like(ds))
        total = total + torch.mean(wv) + torch.mean(fv)
    return total

# ---------------- 扩散模型训练（接受物理量纲输入） ----------------
def train_diffusion(X_raw, y_raw, sd, use_bound=True, use_mono=True):
    """X_raw: log1p 后未标准化；y_raw: 物理量纲。返回 model 与标度统计量。"""
    torch.manual_seed(sd); np.random.seed(sd)
    cfg = M.Config()
    model = M.PhysicsAwareResNetModel(cfg)
    opt = optim.Adam(model.parameters(), lr=LR)
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps)
    ac = torch.cumprod(1 - betas, dim=0)

    ym = y_raw.mean(0); ys = y_raw.std(0) + 1e-8
    xm = X_raw.mean(0); xs = X_raw.std(0) + 1e-8
    ysc = torch.tensor((y_raw - ym) / ys, dtype=torch.float32)
    csc = torch.tensor((X_raw - xm) / xs, dtype=torch.float32)
    n = len(ysc); bs = min(128, n)
    use_phys = use_bound or use_mono

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
            if use_phys:
                x0 = (yn - torch.sqrt(1 - at) * pred) / torch.sqrt(at)
                loss = loss + LAMBDA_PHY * phys_loss(x0, cb, ym, ys, xm, xs,
                                                     use_bound, use_mono)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, (ym, ys, xm, xs)

def sample_prior(model, Xc, seed):
    torch.manual_seed(seed)
    s = M.DiffusionSampler(M.Config(), torch.device('cpu'))
    with torch.no_grad():
        return s.generate(model, torch.tensor(Xc, dtype=torch.float32)).cpu().numpy()

def sample_sdedit(model, Xc, Yc, seed, start=SDEDIT_START):
    torch.manual_seed(seed)
    s = M.DiffusionSampler(M.Config(), torch.device('cpu'))
    with torch.no_grad():
        return s.generate_with_partial_noise(
            model, torch.tensor(Xc, dtype=torch.float32),
            torch.tensor(Yc, dtype=torch.float32).reshape(-1, 1), start).cpu().numpy()

# ---------------- 下游预测器 ----------------
GBM = dict(n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
           min_child_samples=10, subsample=0.9, colsample_bytree=0.9, verbose=-1)

def make_pred(kind, sd):
    if kind == 'gbdt':
        return LGBMRegressor(random_state=sd, **GBM)
    if kind == 'mono_gbdt':
        return LGBMRegressor(random_state=sd, monotone_constraints=[-1, 0, 0, 0, 0], **GBM)
    if kind == 'gpr':
        k = ConstantKernel(1.0, (1e-3, 1e3)) * \
            Matern(length_scale=np.ones(Xr.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=1.5) + \
            WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e0))
        return GaussianProcessRegressor(kernel=k, normalize_y=True, alpha=1e-8,
                                        n_restarts_optimizer=0, random_state=sd)
    raise ValueError(kind)

PREDICTORS = ['gbdt', 'mono_gbdt', 'gpr']
VARIANTS = [
    ('A0 full PCCDM', True, True, 'prior'),
    ('A1 no L_bound', False, True, 'prior'),
    ('A2 no L_mono', True, False, 'prior'),
    ('A3 no physics', False, False, 'prior'),
    ('A4 SDEdit', True, True, 'sdedit'),
]
NAMES = ['Raw (no aug)'] + [v[0] for v in VARIANTS]

results = {v: {p: [] for p in PREDICTORS} for v in NAMES}
phys_res = {v[0]: [] for v in VARIANTS}
t0 = time.time()

# ---- 断点续跑：每个 repeat 完成即落盘 ----
PROG = '_ablation_prog_r%d.jsonl' % N_REPEATS
done = 0
if os.path.exists(PROG):
    for line in open(PROG):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get('rep', -1) != done:
            continue
        for v in NAMES:
            for p in PREDICTORS:
                results[v][p].append(rec['res'][v][p])
        for nm in phys_res:
            phys_res[nm].extend(rec['phys'].get(nm, []))
        done += 1
    print('断点续跑：已完成 %d/%d 次重复' % (done, N_REPEATS), flush=True)

for rep in range(done, N_REPEATS):
    sd = SEED_BASE + rep
    strat = (Y.ravel() > np.median(Y)).astype(int)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                    random_state=sd).split(Xr, strat, groups)
    oof = {(v, p): np.zeros(len(Y)) for v in NAMES for p in PREDICTORS}
    phys_mark = {nm: len(phys_res[nm]) for nm in phys_res}

    for fold, (tri, tei) in enumerate(splitter):
        fseed = sd * 100 + fold
        # 缺失以训练折中位数填充（保持物理量纲；水灰比缺失 106/239）
        med = np.nanmedian(Xr[tri], axis=0)
        Xraw_tr = np.where(np.isnan(Xr[tri]), med, Xr[tri])
        Xraw_te = np.where(np.isnan(Xr[tei]), med, Xr[tei])
        yraw_tr, yraw_te = Y[tri], Y[tei]
        sc_x, sc_y = StandardScaler(), StandardScaler()
        Xtr = np.nan_to_num(sc_x.fit_transform(Xraw_tr))
        Xte = np.nan_to_num(sc_x.transform(Xraw_te))
        ytr = sc_y.fit_transform(yraw_tr)
        yte = sc_y.transform(yraw_te)
        n_gen = int(len(Xtr) * N_GEN_MULT)
        r = np.random.RandomState(fseed)
        idx = r.randint(0, len(Xtr), n_gen)
        Xc = Xtr[idx]
        Yc = ytr[idx].ravel()

        # A0 训练一次，A4 复用同一模型
        mdl0, stats0 = train_diffusion(Xraw_tr, yraw_tr, fseed, True, True)
        sets = {'A0 full PCCDM': sample_prior(mdl0, Xc, fseed),
                'A4 SDEdit': sample_sdedit(mdl0, Xc, Yc, fseed)}
        for nm, ub, um in [('A1 no L_bound', False, True), ('A2 no L_mono', True, False),
                           ('A3 no physics', False, False)]:
            m_, _ = train_diffusion(Xraw_tr, yraw_tr, fseed, ub, um)
            sets[nm] = sample_prior(m_, Xc, fseed)

        # 物理指标（标准化空间输入，evaluator 内部反变换）
        ev = M.PhysicsResidualEvaluator(stats0[0], stats0[1], stats0[2], stats0[3])
        for nm in sets:
            yy = np.asarray(sets[nm]).reshape(-1, 1)
            try:
                phys_res[nm].append(dict(
                    residual=float(ev.compute_physics_residual_error(Xc, yy)),
                    pvr=float(ev.compute_physics_violation_rate(Xc, yy))))
            except Exception:
                pass

        for vname in NAMES:
            for p in PREDICTORS:
                if vname == 'Raw (no aug)':
                    Xa, ya = Xtr, ytr.ravel()
                else:
                    Xa = np.vstack([Xtr, Xc])
                    ya = np.concatenate([ytr.ravel(), np.asarray(sets[vname]).ravel()])
                pm = make_pred(p, fseed)
                pm.fit(Xa, ya)
                oof[(vname, p)][tei] = sc_y.inverse_transform(
                    pm.predict(Xte).reshape(-1, 1)).ravel()

    for vname in NAMES:
        for p in PREDICTORS:
            pr, tr_ = oof[(vname, p)], Y.ravel()
            results[vname][p].append(dict(
                R2=r2_score(tr_, pr),
                RMSE=float(np.sqrt(mean_squared_error(tr_, pr))),
                MAE=mean_absolute_error(tr_, pr)))
    # 本 repeat 落盘
    with open(PROG, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'rep': rep,
                            'res': {v: {p: results[v][p][-1] for p in PREDICTORS} for v in NAMES},
                            'phys': {nm: phys_res[nm][phys_mark[nm]:] for nm in phys_res}}) + '\n')
    print('  repeat %d/%d 完成  (%.0fs)' % (rep + 1, N_REPEATS, time.time() - t0), flush=True)

# ---------------- 统计 ----------------
def ci95(vals):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    if n < 2:
        return v.mean(), np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return v.mean(), v.mean() - h, v.mean() + h

def rank_biserial(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    rk = stats.rankdata(np.abs(d))
    return (rk[d > 0].sum() - rk[d < 0].sum()) / rk.sum()

def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p); out = np.empty(m); run = 0.0
    for i, k in enumerate(order):
        v = min(1.0, (m - i) * p[k])
        run = max(run, v); out[k] = run
    return out

print('\n' + '=' * 104)
print('物理指标（各折均值；残差为训练目标，仅作诊断）')
print('-' * 104)
for nm, _, _, _ in VARIANTS:
    if phys_res[nm]:
        rr = np.mean([x['residual'] for x in phys_res[nm]])
        pp = np.mean([x['pvr'] for x in phys_res[nm]])
        print('  %-18s residual=%.5f   PVR=%.2f%%' % (nm, rr, pp))

for p in PREDICTORS:
    print('\n' + '=' * 104)
    print('下游预测器：%s' % p)
    print('-' * 104)
    print('%-20s %26s %12s %12s' % ('变体', 'R2 [95% CI]', 'RMSE', 'MAE'))
    for vname in NAMES:
        d = results[vname][p]
        m, lo, hi = ci95([x['R2'] for x in d])
        rm, _, _ = ci95([x['RMSE'] for x in d])
        ma, _, _ = ci95([x['MAE'] for x in d])
        print('%-20s  %+.4f [%+.4f, %+.4f]  %10.4f  %10.4f' % (vname, m, lo, hi, rm, ma))

print('\n' + '=' * 104)
print('配对 Wilcoxon（同重复配对，vs A0 完整模型；Holm 校正）')
print('-' * 104)
for p in PREDICTORS:
    a0 = [x['R2'] for x in results['A0 full PCCDM'][p]]
    rows = []
    for nm, _, _, _ in VARIANTS[1:]:
        v = [x['R2'] for x in results[nm][p]]
        d = np.array(v) - np.array(a0)
        try:
            pv = stats.wilcoxon(v, a0).pvalue
        except Exception:
            pv = np.nan
        rows.append((nm, d.mean(), rank_biserial(v, a0), pv))
    adj = holm([r[3] for r in rows])
    for (nm, dm, rb, pv), pa in zip(rows, adj):
        print('  [%-9s] %-16s dR2=%+.4f  rb=%+.2f  p=%.4f  p_holm=%.4f  %s' % (
            p, nm, dm, rb, pv, pa, '显著' if pa < 0.05 else '不显著'))

OUT = '_ablation_r%d.json' % N_REPEATS
json.dump({'results': results,
           'phys': {k: v for k, v in phys_res.items()}},
          open(OUT, 'w'), indent=1)
print('\nsaved %s   总耗时 %.0fs' % (OUT, time.time() - t0))
