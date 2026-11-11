# -*- coding: utf-8 -*-
"""
Comment 6 -- rebuild of the Section 6 life-extension assessment.

One representative mix design throughout (the dataset median), so that every
number in the table and in Figure 7 refers to the same profile.

What is reported:
  (a) the fluence integral, evaluated for three flux assumptions;
  (b) a monotone-constrained prediction with a bootstrap 95% interval;
  (c) the spread of measured retention in the same fluence neighbourhood, which
      bounds how much information the model can have about that range;
  (d) the fluence at which the lower 95% bound crosses a range of thresholds, so
      that the answer can be read as a function of the threshold rather than of
      one chosen value.
"""
import json
import re
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0
TEMPERATURE = 60.0
N_BOOT = 200
SEED = 20261107
SCREENING = 1.0e19

FLUXES = {
    'phi = 1.5e9 (original submission)': 1.5e9,
    'phi = 5.0e9 (intermediate)': 5.0e9,
    'phi = 1.2e10 (published shield level)': 1.2e10,
}
YEARS = [40, 60, 80]

GBM = dict(n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
           min_child_samples=5, subsample=0.9, colsample_bytree=0.9,
           verbose=-1, n_jobs=1)
MONO = [-1, -1, 0, 0, 0]

def firstnum(s):
    if pd.isna(s):
        return np.nan
    m = re.findall(r'[0-9]+\.?[0-9]*', str(s))
    return float(m[0]) if m else np.nan

df = pd.read_csv('data/processed/concrete_processed.csv')
df['agg_c'] = df['骨料水泥比'].map(firstnum)

FEATS = ['快中子辐照注量 n/cm2', '辐照温度', '水灰比', 'agg_c', 'concrete_type_encoded']
Y = df['相对抗压强度'].values
FL = df['快中子辐照注量 n/cm2'].values
TT = df['辐照温度'].values

def design(rows):
    X = np.array(rows, dtype=float)
    X[:, 0] = np.log1p(X[:, 0])
    return np.nan_to_num(X)

X_all = design(df[FEATS].values)
med = np.nanmedian(df[FEATS].values, axis=0)

def fit(X, y, seed):
    return LGBMRegressor(random_state=seed, monotone_constraints=MONO, **GBM).fit(X, y)

base = fit(X_all, Y, SEED)
rng = np.random.RandomState(SEED)
boots = []
for b in range(N_BOOT):
    idx = rng.randint(0, len(Y), len(Y))
    boots.append(fit(X_all[idx], Y[idx], SEED + b))

def query(fluence):
    row = med.copy()
    row[0] = fluence
    row[1] = TEMPERATURE
    Xq = design([row])
    p = float(base.predict(Xq)[0])
    bs = np.array([float(m.predict(Xq)[0]) for m in boots])
    return p, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def local_spread(fluence, factor=2.0):
    """Measured retention within a factor `factor` of the given fluence, at any
    temperature, so wide that it is not a model statement but a data statement."""
    m = (FL > fluence / factor) & (FL < fluence * factor)
    v = Y[m]
    if len(v) < 5:
        return None
    return len(v), float(np.percentile(v, 5)), float(np.percentile(v, 95)), float(v.min()), float(v.max())

print('=' * 78)
print('Representative mix design (dataset median): w/c = %.2f, a/c = %.1f, cement type = %.0f'
      % (med[2], med[3], med[4]))
print('Temperature held at %.0f degC (%d observations at this temperature)' % (TEMPERATURE, (TT == TEMPERATURE).sum()))
print('=' * 78)

print('\n(a) Fluence integral  Phi = phi * t * 365.25*24*3600   (E > 0.1 MeV)\n')
print('%-40s %12s %12s %12s' % ('flux assumption', '40 yr', '60 yr', '80 yr'))
for name, phi in FLUXES.items():
    vals = [phi * yr * SECONDS_PER_YEAR for yr in YEARS]
    print('%-40s ' % name + ' '.join('%12.3e' % v for v in vals))
print('\nNUREG-2192 (SRP-SLR) 3.5.2.2.2.6 screening level: %.1e n/cm2' % SCREENING)
for name, phi in FLUXES.items():
    print('  %-40s 80-yr = %.2f x screening level' % (name, phi * 80 * SECONDS_PER_YEAR / SCREENING))

print('\n(b) Point prediction and 95% bootstrap interval\n')
print('%-40s %-22s %-22s %-22s' % ('flux assumption', '40 yr', '60 yr', '80 yr'))
rows_out = {}
for name, phi in FLUXES.items():
    rows_out[name] = {}
    cells = []
    for yr in YEARS:
        F = phi * yr * SECONDS_PER_YEAR
        p, lo, hi = query(F)
        rows_out[name][yr] = dict(fluence=F, point=p, lo=lo, hi=hi)
        cells.append('%.3f [%.3f,%.3f]' % (p, lo, hi))
    print('%-40s ' % name + ' '.join('%-22s' % c for c in cells))

print('\n(c) Measured retention within a factor of two of the same fluence\n')
print('%-40s %-22s %-22s %-22s' % ('flux assumption', '40 yr', '60 yr', '80 yr'))
spread = {}
for name, phi in FLUXES.items():
    spread[name] = {}
    cells = []
    for yr in YEARS:
        F = phi * yr * SECONDS_PER_YEAR
        s = local_spread(F)
        spread[name][yr] = s
        cells.append('n=%d p5=%.2f p95=%.2f' % (s[0], s[1], s[2]) if s else 'n<5')
    print('%-40s ' % name + ' '.join('%-22s' % c for c in cells))

print('\n(d) Threshold sensitivity: fluence at which the lower 95%% bound crosses tau\n')
grid = np.logspace(np.log10(1e17), np.log10(1e20), 160)
G = np.tile(med, (len(grid), 1))
G[:, 0] = grid
G[:, 1] = TEMPERATURE
Xg = design(G)
pg = base.predict(Xg)
bg = np.array([m.predict(Xg) for m in boots])
lg = np.percentile(bg, 2.5, axis=0)

cross = {}
for tau in [0.95, 0.90, 0.85, 0.80, 0.75]:
    below = np.where(lg < tau)[0]
    cross[tau] = float(grid[below[0]]) if len(below) else None
    print('  tau = %.2f   lower bound crosses at Phi = %s' %
          (tau, ('%.2e n/cm2' % cross[tau]) if cross[tau] else 'not reached below 1e20'))

# curve for the figure
curve = dict(fluence=grid.tolist(), point=pg.tolist(), low=lg.tolist(),
             high=np.percentile(bg, 97.5, axis=0).tolist())
json.dump(dict(fluxes=FLUXES, years=YEARS, table=rows_out, spread=spread,
               cross={str(k): v for k, v in cross.items()}, curve=curve,
               temperature=TEMPERATURE, median_profile=med.tolist(),
               screening=SCREENING, n_boot=N_BOOT),
          open('_c6_life.json', 'w'), indent=1)
print('\nsaved _c6_life.json')
