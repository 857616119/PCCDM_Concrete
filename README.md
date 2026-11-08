# PCCDM: a physics-constrained conditional diffusion model for irradiated concrete

Code and data for the revised manuscript

> Wang, Z., Li, Z., Liu, X. *A physics-constrained diffusion data augmentation approach for strength
> degradation prediction of nuclear biological shield concrete under limited irradiation data.*
> Annals of Nuclear Energy, manuscript ANUCENE-D-26-01360 (revised version R1).

The archive reproduces every table and figure of Sections 5 and 6 of the revised manuscript on a single
CPU core in about two hours. This file maps each script to the table or figure it produces;
`data/README.md` documents the analysis table.

---

## What the method is, and what it is not

PCCDM is a conditional diffusion model that generates **one scalar**, the relative compressive strength
retention of a concrete specimen, conditioned on five measured variables: fast neutron fluence,
irradiation temperature, water–cement ratio, aggregate–cement ratio and cement type. Two physical penalty
terms are added to the training objective: an admissible interval on the retention, and a conditional
monotonicity constraint in fluence and water–cement ratio. The constraints act on training only.
Generation is an unmodified reverse diffusion from a standard-normal prior, with no projection, clipping
or rejection of the generated values.

Under the formulation-grouped evaluation used throughout the paper, **no augmentation strategy improves
downstream accuracy**: every strategy lowers R² relative to training on the measured observations alone,
and PCCDM lowers it least. This negative result is the main empirical finding of the revision, and the
code reproduces it rather than a favourable one.

The numbers in the original submission (R² ≈ 0.80) came from a random split at the level of individual
observations and from a baseline trained on an interpolated pool. Both are described under
*What changed relative to the original submission* below, and neither is reproduced here.

---

## Repository layout

| Path | Contents |
| --- | --- |
| `pccdm_model.py` | conditional residual denoiser, diffusion sampler, physics evaluator |
| `train.py` | training loop, physics loss terms, dataset wrapper |
| `augment.py` | data augmentation entry point |
| `evaluate_P1.py` | downstream comparison of five augmentation strategies, grouped and random splits, 10 repetitions |
| `evaluate_ablation.py` | component ablation and the three-predictor comparison |
| `evaluate_physics.py` | constraint residual, violation rate, Wasserstein distance, measured-data reference |
| `evaluate_derived.py` | ablation of the derived physics features |
| `evaluate_memorization.py` | exact and near copy rates, nearest-neighbour distance ratio, coverage, KS statistic |
| `evaluate_life.py` | Section 6 fluence integral, monotone prediction, bootstrap interval, threshold sweep |
| `compare_fluence_domain.py` | measured retention by fluence interval, the temperature confound, and the within-formulation comparison (Section 6.3, Table 11) |
| `check_sample_budget.py` | sample counts per fold and the out-of-band rate of the generated values (Section 4.7) |
| `data/processed/concrete_processed.csv` | the 239 measured observations (analysis table) |
| `data/processed/concrete_processed_with_physics_features.csv` | same rows with the derived indicators, used only by `evaluate_derived.py` |
| `results/r1/*.json` | outputs of the runs reported in the manuscript |

---

## Data

The 239 observations come from the database *Dataset of concrete material properties under neutron
irradiation and high temperature environments*, National Basic Science Data Center (NBSDC) of China,
permanent identifier **CSTR:16666.11.nbsdc.zMXRoDX8**, version 1.0 (2024-11-29).

The NBSDC workbook is distributed under NBSDC's own access terms and is **not** redistributed here.
`data/processed/concrete_processed.csv` is the analysis table derived from it: 12 columns, 239 rows,
documented column by column in `data/README.md`. The two files under `data/processed/` are row-aligned, and
the second adds the eight derived indicators used in the ablation of Table 6.

The dataset contains **no interpolated or synthetic rows**. A file of 800 interpolated rows shipped with
the original submission (`concrete_augmented.csv`) has been removed; it is not used anywhere in the
revised results.

---

## Environment

Python 3.12 with the versions pinned in `requirements.txt` (PyTorch 2.8 CPU build, LightGBM 4.6,
scikit-learn 1.5). No GPU is required; every experiment runs on one core with `OMP_NUM_THREADS=1`
and `torch.set_num_threads(1)`.

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproducing the results

All randomness is seeded from `SEED_BASE = 20261106` with the repetition index added, so the runs below
are deterministic. Run from the repository root.

| Command | Produces | Section | Runtime |
| --- | --- | --- | --- |
| `N_REPEATS=10 SPLIT=grouped python evaluate_P1.py` | Table 4, Figures 5–6 | 5.2 | 9 min |
| `N_REPEATS=10 SPLIT=random python evaluate_P1.py` | random-split rows of Table 4 | 5.2 | 9 min |
| `N_REPEATS=10 python evaluate_ablation.py` | Tables 5 and 7 | 5.6 | 78 min |
| `N_REPEATS=10 python evaluate_physics.py` | Table 8 | 5.4 | 2 min |
| `python evaluate_derived.py` | Table 6 | 5.6 | 1 min |
| `N_REPEATS=10 python evaluate_memorization.py` | Table 9 | 5.7 | 10 min |
| `python evaluate_life.py` | Tables 10 and 12, Figure 7 | 6 | 1 min |
| `python compare_fluence_domain.py` | Table 11, numbers in Section 6.3 | 6.3 | 1 min |
| `python check_sample_budget.py` | numbers quoted in Section 4.7 | 4.7 | 2 min |

Each script writes a JSON file to the working directory; the copies in `results/r1/` are the outputs
behind the published numbers. `evaluate_ablation.py` writes a progress file after every repetition and can
be interrupted and resumed.

The ablation dominates the wall time. Two scripts were re-run from a clean checkout of the tagged version
after the release tree was assembled, and both reproduce the archived JSON exactly: `evaluate_derived.py`
(`derived_features.json`) and `check_sample_budget.py` (`sample_budget.json`).

---

## What changed relative to the original submission

The code and the results of the first submission were superseded in four respects. Each is documented in
the response to the reviewers, and each is visible in this repository.

1. **Partition.** The original pipeline split at the level of individual observations, which places
   repeated measurements of the same mix design on both sides of the boundary. Grouped five-fold
   cross-validation with the pair (cement system, aggregate type) as the grouping key replaces it
   (`evaluate_P1.py`, `SPLIT=grouped`).

2. **Metrics.** R² had been overwritten by the training-fold cross-validation score while RMSE and MAE
   were test-set quantities, so the three columns of the previous table were not computed on the same
   data. Metrics are now computed on pooled out-of-fold predictions; `metrics['R2_trainCV']` is retained in
   the code for reference and is not reported.

3. **Sampling.** The original pipeline started the reverse process from a perturbed measured observation
   (SDEdit). The reported results now sample from the prior. The comparison is in
   `evaluate_ablation.py` (`A4`): SDEdit lowers downstream R² by 0.35 (p = 0.002).

4. **Objective.** The feature-consistency term of the original Equation 11 has no counterpart in the
   implementation, because the generator emits a scalar rather than a feature vector. It has been removed
   from both the manuscript and the code; the objective is the boundary term plus the monotonicity term.

---

## License

Code released under the MIT license (`LICENSE`). The analysis table is derived from NBSDC data and is
provided for verification of this manuscript; the terms of the NBSDC database take precedence for any
reuse of the underlying records.
