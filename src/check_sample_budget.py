# -*- coding: utf-8 -*-
"""Sample budget of the grouped protocol, and the out-of-band rate of the generated values.

Section 4.7 of the manuscript reports three quantities that the other scripts do not expose:

  (a) the size of the training and held-out folds, which are unequal because the formulation
      groups range from 1 to 50 observations;
  (b) the number of generated values per fold, 2.5 n_train, and their total over a repetition;
  (c) how often a generated value leaves the admissible interval [0.1, 2.0], which is the
      empirical answer to "how are the constraints enforced during sampling": they are not,
      and the out-of-band rate is the consequence.

The training and sampling path below is a copy of the PCCDM branch of evaluate_P1.py, with the
first repetition seed, so the draws are the ones behind the reported numbers.
"""
import os, json, re, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import pccdm_model as M
import train as T

SEED = 20261106
N_SPLITS, EPOCHS, N_GEN_MULT, PHYS_W = 5, 600, 2.5, 0.1
torch.set_num_threads(1)


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

print('N=%d  n_groups=%d  y_obs=[%.4f, %.4f]' % (len(Y), len(set(groups)), Y.min(), Y.max()))

splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                random_state=SEED).split(Xr, (Y.ravel() > np.median(Y)).astype(int), groups)

out = {'N': int(len(Y)), 'n_groups': int(len(set(groups))),
       'y_min': float(Y.min()), 'y_max': float(Y.max()), 'folds': []}
all_gen = []
for fold, (tri, tei) in enumerate(splitter):
    med = np.nanmedian(Xr[tri], axis=0)
    Xraw_tr = np.where(np.isnan(Xr[tri]), med, Xr[tri])
    sc_x, sc_y = StandardScaler(), StandardScaler()
    Xtr = np.nan_to_num(sc_x.fit_transform(Xraw_tr), nan=0.0)
    ytr = sc_y.fit_transform(Y[tri])
    n_gen = int(len(Xtr) * N_GEN_MULT)

    # --- 与 evaluate_P1.py 完全一致的训练 ---
    torch.manual_seed(SEED * 100 + fold); np.random.seed(SEED * 100 + fold)
    cfg = M.Config()
    model = M.PhysicsAwareResNetModel(cfg)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps)
    ac = torch.cumprod(1 - betas, dim=0)
    ym, ys = Y[tri].mean(0), Y[tri].std(0) + 1e-8
    xm, xs = Xraw_tr.mean(0), Xraw_tr.std(0) + 1e-8
    ysc = torch.tensor((Y[tri] - ym) / ys, dtype=torch.float32)
    csc = torch.tensor((Xraw_tr - xm) / xs, dtype=torch.float32)
    bs = min(128, len(tri))
    for ep in range(EPOCHS):
        perm = torch.randperm(len(tri))
        for i in range(0, len(tri) - bs + 1, bs):
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
    sampler = M.DiffusionSampler(M.Config(), torch.device('cpu'))
    r = np.random.RandomState(SEED * 100 + fold)
    idx = r.randint(0, len(Xtr), n_gen)
    Xc = Xtr[idx]
    with torch.no_grad():
        Yg = sampler.generate(model, torch.tensor(Xc, dtype=torch.float32)).cpu().numpy().reshape(-1, 1)
    yphys = Yg * ys + ym                       # 反标准化回物理量纲

    all_gen.append(yphys)
    n_lo, n_hi = int((yphys < 0.1).sum()), int((yphys > 2.0).sum())
    rec = dict(fold=fold, n_train=int(len(tri)), n_test=int(len(tei)), n_gen=int(n_gen),
               n_groups_train=int(len(set(groups[tri]))), n_groups_test=int(len(set(groups[tei]))),
               gen_min=float(yphys.min()), gen_max=float(yphys.max()),
               gen_p01=float(np.percentile(yphys, 1)), gen_p99=float(np.percentile(yphys, 99)),
               below_lo=n_lo, above_hi=n_hi,
               frac_outside=float((n_lo + n_hi) / len(yphys)),
               train_min=float(Y[tri].min()), train_max=float(Y[tri].max()))
    out['folds'].append(rec)
    print('fold %d  n_tr=%3d n_te=%3d n_gen=%4d | gen phys [%.3f, %.3f]  out=%d (%.2f%%)'
          % (fold, rec['n_train'], rec['n_test'], rec['n_gen'],
             rec['gen_min'], rec['gen_max'], n_lo + n_hi, 100 * rec['frac_outside']))

G = np.vstack(all_gen)
out['gen_total'] = int(len(G))
out['gen_global_min'] = float(G.min()); out['gen_global_max'] = float(G.max())
out['gen_frac_outside'] = float(((G < 0.1) | (G > 2.0)).mean())
out['gen_mean'] = float(G.mean()); out['gen_sd'] = float(G.std())
out['obs_mean'] = float(Y.mean()); out['obs_sd'] = float(Y.std())
# 观测样本自身有多少落在 [0.1,2.0] 外
out['obs_frac_outside'] = float(((Y < 0.1) | (Y > 2.0)).mean())
print('\nTOTAL gen=%d  phys [%.4f, %.4f]  outside=%.4f%%  (obs outside=%.4f%%)'
      % (len(G), G.min(), G.max(), 100 * out['gen_frac_outside'], 100 * out['obs_frac_outside']))
print('gen mean/sd = %.4f/%.4f   obs mean/sd = %.4f/%.4f'
      % (G.mean(), G.std(), Y.mean(), Y.std()))
json.dump(out, open('sample_budget.json', 'w'), indent=1)
print('saved sample_budget.json')
