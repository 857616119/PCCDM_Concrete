# -*- coding: utf-8 -*-
"""
Comment 5 -- ablation of the derived physics features.

The reviewer asks for an ablation of "each loss term derived feature".  The three
derived factors of Section 4.2 (f_Arr, f_coup, D_th) are not part of the condition
vector used by the generator, so the relevant question is whether they *should* be.
This script tests that directly: each factor is added, one at a time and jointly,
to the same five measured variables, under the identical grouped protocol used in
Section 5.2.1 (StratifiedGroupKFold, five folds, ten repetitions, metrics pooled
over out-of-fold predictions).

It also reports collinearity diagnostics, because a factor that is a deterministic
function of the other inputs cannot add information regardless of its physical
meaning.
"""
import os
import re
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

N_SPLITS = 5
N_REPEATS = int(os.environ.get('N_REPEATS', '10'))
SEED_BASE = 20261107

GBM = dict(n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
           min_child_samples=10, subsample=0.9, colsample_bytree=0.9,
           verbose=-1, n_jobs=1)

BASE = 'data/processed/concrete_processed.csv'
PHYS = 'data/processed/concrete_processed_with_physics_features.csv'

def firstnum(s):
    if pd.isna(s):
        return np.nan
    m = re.findall(r'[0-9]+\.?[0-9]*', str(s))
    return float(m[0]) if m else np.nan

df = pd.read_csv(BASE)
df['agg_c'] = df['骨料水泥比'].map(firstnum)

# the derived factors, taken from the file that ships with the repository
pf = pd.read_csv(PHYS)
assert len(pf) == len(df), 'row mismatch between the two processed files'
for col in ['arrhenius_term', 'fluence_temp_normalized', 'thermal_damage_factor',
            'defect_density', 'amorphous_fraction', 'crack_density',
            'temperature_compensated_dose', 'damage_saturation']:
    df[col] = pf[col].values

MEASURED = ['快中子辐照注量 n/cm2', '辐照温度', '水灰比', 'agg_c', 'concrete_type_encoded']
F_ARR = 'arrhenius_term'
F_COUP = 'fluence_temp_normalized'
D_TH = 'thermal_damage_factor'
ALL9 = ['arrhenius_term', 'fluence_temp_normalized', 'thermal_damage_factor',
        'defect_density', 'amorphous_fraction', 'crack_density',
        'temperature_compensated_dose', 'damage_saturation']

CONFIGS = [
    ('measured only (used in the paper)', MEASURED),
    ('+ f_Arr', MEASURED + [F_ARR]),
    ('+ f_coup', MEASURED + [F_COUP]),
    ('+ D_th', MEASURED + [D_TH]),
    ('+ all three derived factors', MEASURED + [F_ARR, F_COUP, D_TH]),
    ('+ all eight derived columns', MEASURED + ALL9),
    ('derived factors only', [F_ARR, F_COUP, D_TH]),
    # controls: the same low dimensionality reached with measured variables only,
    # so that a gain from "derived factors only" can be attributed to content
    # rather than to the smaller number of inputs.
    ('control: 3 measured (Phi, T, W/C)', ['快中子辐照注量 n/cm2', '辐照温度', '水灰比']),
    ('control: 2 measured (Phi, T)', ['快中子辐照注量 n/cm2', '辐照温度']),
    ('control: 4 measured (drop W/C)', ['快中子辐照注量 n/cm2', '辐照温度', 'agg_c',
                                        'concrete_type_encoded']),
]

Y = df['相对抗压强度'].values.reshape(-1, 1)
groups = (df['水泥类型'].astype(str) + '|' + df['其它特殊成分'].astype(str)).values
strat = (Y.ravel() > np.median(Y)).astype(int)

def build(cols):
    X = df[cols].copy()
    if '快中子辐照注量 n/cm2' in X.columns:
        X['快中子辐照注量 n/cm2'] = np.log1p(X['快中子辐照注量 n/cm2'])
    return X.values.astype(float)

print('=' * 78)
print('Derived-feature ablation  (%d folds x %d repetitions, grouped)' % (N_SPLITS, N_REPEATS))
print('=' * 78)

results = {name: [] for name, _ in CONFIGS}
t0 = pd.Timestamp.now()

for rep in range(N_REPEATS):
    sd = SEED_BASE + rep
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                    random_state=sd).split(df, strat, groups)
    folds = list(splitter)
    for name, cols in CONFIGS:
        X = build(cols)
        pred = np.zeros(len(Y))
        for tri, tei in folds:
            med = np.nanmedian(X[tri], axis=0)
            Xtr = np.where(np.isnan(X[tri]), med, X[tri])
            Xte = np.where(np.isnan(X[tei]), med, X[tei])
            sc = StandardScaler()
            Xtr = np.nan_to_num(sc.fit_transform(Xtr))
            Xte = np.nan_to_num(sc.transform(Xte))
            m = LGBMRegressor(random_state=sd, **GBM).fit(Xtr, Y[tri].ravel())
            pred[tei] = m.predict(Xte)
        t, p = Y.ravel(), pred
        results[name].append(dict(R2=r2_score(t, p),
                                  RMSE=float(np.sqrt(mean_squared_error(t, p))),
                                  MAE=mean_absolute_error(t, p),
                                  r=float(np.corrcoef(t, p)[0, 1])))
    print('  repeat %d/%d  ok' % (rep + 1, N_REPEATS), flush=True)

print()
print('%-38s %18s %10s %10s' % ('configuration', 'R2', 'RMSE', 'r'))
print('-' * 78)
base = np.array([d['R2'] for d in results[CONFIGS[0][0]]])
for name, _ in CONFIGS:
    a = np.array([d['R2'] for d in results[name]])
    rm = np.mean([d['RMSE'] for d in results[name]])
    rr = np.mean([d['r'] for d in results[name]])
    print('%-38s %+.4f +/- %.4f  %8.4f %9.3f' % (name, a.mean(), a.std(ddof=1), rm, rr))

print()
print('Paired difference against "measured only" (Wilcoxon signed-rank, n=%d):' % N_REPEATS)
for name, _ in CONFIGS[1:]:
    a = np.array([d['R2'] for d in results[name]])
    diff = a - base
    try:
        p = stats.wilcoxon(a, base).pvalue
    except ValueError:
        p = float('nan')
    print('  %-36s dR2 = %+.4f   p = %.3f' % (name, diff.mean(), p))

# collinearity: R2 of regressing each derived factor on the five measured variables
print()
print('Collinearity diagnostic: R2 of the derived factor regressed on the five measured inputs')
Xm = build(MEASURED)
for col in [F_ARR, F_COUP, D_TH]:
    v = df[col].values
    ok = np.isfinite(v)
    m = LGBMRegressor(random_state=0, **GBM).fit(np.nan_to_num(Xm[ok]), v[ok])
    r2 = r2_score(v[ok], m.predict(np.nan_to_num(Xm[ok])))
    sp = stats.spearmanr(v[ok], Y.ravel()[ok]).statistic
    print('  %-26s R2(measured) = %.4f   Spearman vs y = %+.3f' % (col, r2, sp))

json.dump({k: v for k, v in results.items()}, open('_c5_derived.json', 'w'), indent=1)
print('\nsaved _c5_derived.json   (%.0fs)' % (pd.Timestamp.now() - t0).total_seconds())
