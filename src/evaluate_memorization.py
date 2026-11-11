# -*- coding: utf-8 -*-
"""
Comment 7 -- memorisation and diversity of the synthetic retention values.

The reviewer's concern is that SDEdit, which starts from a measured sample, may
only reproduce the training data. That is a testable statement, so it is tested
here on the quantity the generator actually produces: the strength retention.

For each augmentation strategy, over the same grouped folds used everywhere else
in the paper, four quantities are reported:

  copy rate   fraction of synthetic values lying within EPS of some training value
  NNDR        median nearest-neighbour distance from a synthetic value to the
              training set, divided by the median nearest-neighbour distance
              within the training set.  Values well below one mean the synthetic
              points are closer to the training data than the training data are
              to each other, which is the signature of replication.
  coverage    fraction of synthetic values inside the observed range
  KS          Kolmogorov-Smirnov statistic between the synthetic and training
              distributions, with its p-value

Prior sampling and SDEdit are compared directly, because the reviewer's
objection names SDEdit specifically and the manuscript reports the prior.
"""
import os
import re
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats

import pccdm_model as M
import train as T

warnings.filterwarnings('ignore')

N_SPLITS = 5
N_REPEATS = int(os.environ.get('N_REPEATS', '5'))
SEED_BASE = 20261107
EPOCHS = 600
PHYS_W = 0.1
LR = 1e-3
N_GEN_MULT = 2.5
SDEDIT_START = 20
EPS = 0.01          # retention units; 1% of the initial strength

def firstnum(s):
    if pd.isna(s):
        return np.nan
    m = re.findall(r'[0-9]+\.?[0-9]*', str(s))
    return float(m[0]) if m else np.nan

df = pd.read_csv('data/processed/concrete_processed.csv')
df['agg_c'] = df['骨料水泥比'].map(firstnum)
FEATS = ['快中子辐照注量 n/cm2', '辐照温度', '水灰比', 'agg_c', 'concrete_type_encoded']

Xr = df[FEATS].copy()
Xr['快中子辐照注量 n/cm2'] = np.log1p(Xr['快中子辐照注量 n/cm2'])
Xr = Xr.values.astype(float)
Y = df['相对抗压强度'].values.reshape(-1, 1)
groups = (df['水泥类型'].astype(str) + '|' + df['其它特殊成分'].astype(str)).values
strat = (Y.ravel() > np.median(Y)).astype(int)

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
            loss = loss + PHYS_W * T.compute_physics_loss(x0, cb, ym, ys, xm, xs)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, M.DiffusionSampler(M.Config(), torch.device('cpu')), (ym, ys, xm, xs)

def metrics(gen, train):
    """All quantities computed on the retention scalar, in physical units."""
    gen = np.asarray(gen).ravel()
    train = np.asarray(train).ravel()

    # distance from each synthetic value to the closest training value
    d = np.abs(gen[:, None] - train[None, :]).min(axis=1)

    # spacing of the training set itself, which sets the scale of "close"
    T2 = train[:, None]
    d_tt = np.abs(T2 - T2.T)
    np.fill_diagonal(d_tt, np.inf)
    nn_self = d_tt.min(axis=1)
    positive = nn_self[nn_self > 0]
    nnd_train = float(np.median(positive)) if len(positive) else np.nan
    eps_adaptive = 0.1 * nnd_train if np.isfinite(nnd_train) else EPS

    exact_copy = float((d < 1e-6).mean())
    near_copy = float((d < eps_adaptive).mean())
    nndr = float(np.median(d) / nnd_train) if np.isfinite(nnd_train) and nnd_train > 0 else np.nan

    lo, hi = train.min(), train.max()
    coverage = float(((gen >= lo) & (gen <= hi)).mean())

    ks = stats.ks_2samp(gen, train)

    return dict(exact_copy=exact_copy, near_copy=near_copy, nndr=nndr,
                nnd=float(np.median(d)), nnd_train=nnd_train, coverage=coverage,
                ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
                gen_std=float(gen.std()), train_std=float(train.std()),
                uniq_ratio=float(len(np.unique(np.round(gen, 4))) / len(gen)))

def aug_gaussian(X, y, n, sd):
    r = np.random.RandomState(sd)
    i = r.randint(0, len(X), n)
    return X[i] + r.normal(0, 0.001, (n, X.shape[1])), y[i] + r.normal(0, 0.15, (n, 1))

def aug_over(X, y, n, sd):
    r = np.random.RandomState(sd)
    i = r.randint(0, len(X), n)
    return X[i], y[i]

def aug_smote(X, y, n, sd):
    r = np.random.RandomState(sd)
    i = r.randint(0, len(X), n)
    j = r.randint(0, len(X), n)
    lam = r.uniform(0, 1, (n, 1))
    return X[i] + lam * (X[j] - X[i]), y[i] + lam * (y[j] - y[i])

NAMES = ['Gaussian noise', 'Random oversampling', 'SMOTE',
         'PCCDM (prior sampling)', 'PCCDM (SDEdit)']
res = {k: [] for k in NAMES}

t0 = time.time()
for rep in range(N_REPEATS):
    sd = SEED_BASE + rep
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=sd)
    for fold, (tri, tei) in enumerate(sgkf.split(Xr, strat, groups)):
        fseed = sd * 100 + fold
        med = np.nanmedian(Xr[tri], axis=0)
        Xraw_tr = np.where(np.isnan(Xr[tri]), med, Xr[tri])
        sc_x, sc_y = StandardScaler(), StandardScaler()
        Xtr = np.nan_to_num(sc_x.fit_transform(Xraw_tr))
        ytr = sc_y.fit_transform(Y[tri])
        n_gen = int(len(Xtr) * N_GEN_MULT)

        # non-diffusion baselines operate in standardised space, then inverse-transform
        for name, fn in [('Gaussian noise', aug_gaussian), ('Random oversampling', aug_over),
                         ('SMOTE', aug_smote)]:
            Xs, ys = fn(Xtr, ytr, n_gen, fseed)
            gen = sc_y.inverse_transform(ys.reshape(-1, 1)).ravel()
            res[name].append(metrics(gen, Y[tri].ravel()))

        # diffusion model, sampled two ways
        model, sampler, _ = train_diffusion(Xraw_tr, Y[tri], fseed)
        r = np.random.RandomState(fseed)
        idx = r.randint(0, len(Xtr), n_gen)
        Xc = Xtr[idx]
        with torch.no_grad():
            yg = sampler.generate(model, torch.tensor(Xc, dtype=torch.float32)).cpu().numpy()
            yg = yg.reshape(-1, 1)
            ys2 = sampler.generate_with_partial_noise(
                model, torch.tensor(Xc, dtype=torch.float32),
                torch.tensor(ytr[idx], dtype=torch.float32).reshape(-1, 1),
                SDEDIT_START).cpu().numpy().reshape(-1, 1)
        res['PCCDM (prior sampling)'].append(
            metrics(sc_y.inverse_transform(yg).ravel(), Y[tri].ravel()))
        res['PCCDM (SDEdit)'].append(
            metrics(sc_y.inverse_transform(ys2).ravel(), Y[tri].ravel()))
    print('  repeat %d/%d  (%.0fs)' % (rep + 1, N_REPEATS, time.time() - t0), flush=True)

print()
print('%-24s %8s %8s %7s %8s %8s %7s' % ('method', 'exact', 'near', 'NNDR', 'cover', 'KS', 'uniq'))
print('-' * 76)
summary = {}
for k in NAMES:
    a = {f: np.array([d[f] for d in res[k]]) for f in res[k][0]}
    summary[k] = {f: [float(v.mean()), float(v.std(ddof=1))] for f, v in a.items()}
    print('%-24s %7.2f%% %7.1f%% %7.3f %7.1f%% %8.3f %7.3f' % (
        k, 100 * a['exact_copy'].mean(), 100 * a['near_copy'].mean(), a['nndr'].mean(),
        100 * a['coverage'].mean(), a['ks_stat'].mean(), a['uniq_ratio'].mean()))

print()
print('Paired comparison, SDEdit vs prior sampling (Wilcoxon, n=%d folds):' % len(res[NAMES[3]]))
for f in ['exact_copy', 'near_copy', 'nndr', 'ks_stat']:
    x = np.array([d[f] for d in res['PCCDM (SDEdit)']])
    y = np.array([d[f] for d in res['PCCDM (prior sampling)']])
    p = stats.wilcoxon(x, y).pvalue
    print('  %-12s SDEdit = %.4f   prior = %.4f   p = %.2e' % (f, x.mean(), y.mean(), p))

json.dump(summary, open('_c7_memorization.json', 'w'), indent=1)
print('\nsaved _c7_memorization.json  (%.0fs)' % (time.time() - t0))
