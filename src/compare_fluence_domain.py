# -*- coding: utf-8 -*-
"""Comment 10: comparison with RIVE-based models, made in the fluence domain.

RIVE models predict a swelling law eps*(Phi, T) for the aggregate, not a strength. Converting swelling to
strength needs a damage relation on top, and both that relation and the aggregate mineralogy are absent from
this dataset. The two descriptions therefore share only one quantity: the fluence at which degradation
begins. This script measures where the measured retention begins to fall, first on the pooled data and then
with the temperature confound controlled.

Reported here, and quoted in Section 6.3 and Table 11 of the manuscript:
  (a) retention binned by fluence, all 239 records, with bootstrap 95% intervals and the mean temperature of
      each bin, because the bins are not temperature-matched;
  (b) the same split at 1e19 for the 55--85 C window in which the case study is evaluated;
  (c) the within-formulation comparison, which removes the cement and aggregate dependence.
"""
import json
import numpy as np
import pandas as pd

RNG = np.random.RandomState(20261108)
d = pd.read_csv('data/processed/concrete_processed.csv')
f = d['快中子辐照注量 n/cm2'].astype(float).values
y = d['相对抗压强度'].astype(float).values
T = d['辐照温度'].astype(float).values
x = np.log10(f)
grp = (d['水泥类型'].astype(str) + '|' + d['其它特殊成分'].astype(str)).values


def boot_ci(v, n=4000):
    if len(v) < 2:
        return float('nan'), float('nan')
    m = [RNG.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


out = {'n': int(len(y)), 'corr_logphi_T': float(np.corrcoef(x, T)[0, 1])}
print('n = %d   corr(log10 Phi, T) = %.3f' % (len(y), out['corr_logphi_T']))

# (a) pooled bins
EDGES = [0, 17, 18, 19, 20, 21, 22]
LAB = ['<1e17', '1e17-1e18', '1e18-1e19', '1e19-1e20', '1e20-1e21', '>1e21']
print('\n(a) pooled')
out['bins'] = []
for lo, hi, lab in zip(EDGES[:-1], EDGES[1:], LAB):
    m = (x >= lo) & (x < hi)
    if m.sum() == 0:
        continue
    l95, h95 = boot_ci(y[m])
    rec = dict(bin=lab, n=int(m.sum()), mean=float(y[m].mean()), median=float(np.median(y[m])),
               lo=l95, hi=h95, T=float(T[m].mean()), T_min=float(T[m].min()), T_max=float(T[m].max()))
    out['bins'].append(rec)
    print('  %-10s n=%3d  mean=%.3f [%.3f, %.3f]  median=%.3f  T=%5.1f (%3.0f-%3.0f)'
          % (lab, rec['n'], rec['mean'], l95, h95, rec['median'], rec['T'], rec['T_min'], rec['T_max']))

# (b) temperature-matched split
print('\n(b) 55-85 C window')
win = (T >= 55) & (T <= 85)
lo_m, hi_m = win & (x < 19), win & (x >= 19)
out['window'] = dict(
    n_lo=int(lo_m.sum()), n_hi=int(hi_m.sum()),
    mean_lo=float(y[lo_m].mean()), mean_hi=float(y[hi_m].mean()),
    lo_ci=list(boot_ci(y[lo_m])), hi_ci=list(boot_ci(y[hi_m])),
    T_lo=float(T[lo_m].mean()), T_hi=float(T[hi_m].mean()),
    corr_within=float(np.corrcoef(x[win], y[win])[0, 1]),
    phi_min=float(f[win].min()), phi_max=float(f[win].max()))
w = out['window']
print('  below 1e19: n=%d mean=%.3f [%.3f, %.3f] T=%.0f' % (w['n_lo'], w['mean_lo'], *w['lo_ci'], w['T_lo']))
print('  above 1e19: n=%d mean=%.3f [%.3f, %.3f] T=%.0f' % (w['n_hi'], w['mean_hi'], *w['hi_ci'], w['T_hi']))
print('  corr(log10 Phi, retention) within window = %+.3f' % w['corr_within'])

# (c) within-formulation
print('\n(c) within formulation, 55-85 C, groups with >=3 records on each side')
rows = []
for g in np.unique(grp[win]):
    m = win & (grp == g)
    a, b = m & (x < 19), m & (x >= 19)
    if a.sum() >= 3 and b.sum() >= 3:
        rows.append(dict(group=g.split('|')[1], n_lo=int(a.sum()), n_hi=int(b.sum()),
                         y_lo=float(y[a].mean()), y_hi=float(y[b].mean()), diff=float(y[b].mean() - y[a].mean())))
        print('  %-24s n=%d/%d  %.3f -> %.3f  diff %+.3f'
              % (rows[-1]['group'], rows[-1]['n_lo'], rows[-1]['n_hi'], rows[-1]['y_lo'], rows[-1]['y_hi'], rows[-1]['diff']))
out['within_group'] = rows
out['within_group_mean_diff'] = float(np.mean([r['diff'] for r in rows]))
print('  mean within-group difference: %+.3f' % out['within_group_mean_diff'])

json.dump(out, open('fluence_domain.json', 'w'), indent=1)
print('\nsaved fluence_domain.json')
