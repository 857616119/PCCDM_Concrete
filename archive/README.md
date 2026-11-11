# Superseded pipeline

Files kept only so that the original submission remains traceable. **Nothing in this directory is used by
the revised manuscript**, and none of the scripts under `src/` imports anything from here.

| Path | What it is | Why it is not used |
| --- | --- | --- |
| `train.py` | training loop of the original submission | superseded by the training path inside `src/evaluate_*.py`, which fits the scaler and the generator inside the training folds only |
| `augment.py` | augmentation entry point of the original submission | it draws on `concrete_augmented.csv`, the 800-row interpolated pool that the revision removed |
| `results/evaluation/`, `results/training_history/` | metrics and logs of the original pipeline | computed with R² taken from the training-fold CV score rather than the held-out fold (see `README.md`, *What changed relative to the original submission*) |
| `results/figures/` | figures of the original pipeline | plotted from those metrics; the figures of the revised manuscript are built from `results/r1/` |

Model checkpoints (`results/models/`, 171 MB) were removed from the repository. They belong to the
superseded training run and are not needed to reproduce anything in the paper; the archive tree is small
enough to inspect without cloning a checkpoint directory.
