# Analysis table

`processed/concrete_processed.csv` — 239 rows × 12 columns. This is the table every script in the
repository reads; no script uses any other data file except `concrete_processed_with_physics_features.csv`,
which repeats the same 239 rows in the same order and appends the derived indicators.

Column names are those of the source database and are kept in Chinese so that the table can be traced back
to it without a lookup. The dictionary below is the lookup.

## Provenance

Source: *Dataset of concrete material properties under neutron irradiation and high temperature
environments*, National Basic Science Data Center (NBSDC) of China, permanent identifier
**CSTR:16666.11.nbsdc.zMXRoDX8**, version 1.0, 2024-11-29. The workbook is not redistributed here; it is
available from NBSDC under that accession.

The workbook holds seven sheets. Of these, `构筑物材料基本信息` (material information), `辐照条件`
(irradiation conditions), `混凝土劣化前性能` and `混凝土劣化后性能` (performance before and after
degradation) supply the fields used here; the remaining sheets describe mixing proportions, the
high-temperature test programme and the database itself and do not enter the model.

## Columns

| # | Column | Meaning | Range / values | Notes |
| --- | --- | --- | --- | --- |
| 1 | `数据ID` | record identifier | 1–239 | row order of the source table |
| 2 | `水泥类型` | cement system | 6 values | see *Cement system* below |
| 3 | `试样尺寸m×m×m` | specimen dimensions | 10 strings | descriptive; not used as a feature |
| 4 | `水灰比` | water–cement ratio | 0.35–1.61 | 106 of 239 missing; see *Water–cement ratio* |
| 5 | `骨料水泥比` | aggregate–cement ratio | 3–22 (text) | some records list two values, e.g. `3.0,6.0`; the code takes the first |
| 6 | `其它特殊成分` | aggregate type | 24 values | e.g. `Serpentine`, `Heston Sand`, `magnetite` |
| 7 | `试样` | specimen shape | `立方体`, `/`, `形状` | shape, **not** an identifier: the database carries no per-specimen identifier |
| 8 | `快中子辐照注量 n/cm2` | fast neutron fluence (n cm⁻²) | 4.9×10¹³ – 2.19×10²¹ | spans 7.6 decades; enters the model as ln(1+Φ) |
| 9 | `辐照温度` | irradiation temperature (°C) | 20–375 | |
| 10 | `相对抗压强度` | relative compressive strength retention | 0.303–2.162 | the target; 2 of 239 exceed 2.0 |
| 11 | `水灰比_missing` | missing-value flag | 0/1 | 1 for the 106 records of column 4 |
| 12 | `concrete_type_encoded` | ordinal code of column 2 | 0–5 | 0 = combined label, 1 = 硅酸盐水泥, 2 = 拉法基铝水泥, 3 = 硅酸盐水泥/水玻璃, 4 = 氧化镁水泥, 5 = 普通硅酸盐水泥 |

Standardisation is **not** stored in the table. Every script fits its scaler inside the training folds of
the current split, so no column of this file carries a statistic of the full dataset.

## Cement system

Column 2 takes six values, of which one is a combined label:

| Value | Records | Meaning |
| --- | --- | --- |
| `普通硅酸盐水泥，高铝水泥，低热水泥，` | 115 | the source reports this test campaign as covering three cement systems jointly; it is not resolved further |
| `硅酸盐水泥` | 60 | ordinary Portland cement |
| `拉法基铝水泥` | 56 | Lafarge alumina cement |
| `氧化镁水泥` | 5 | magnesium oxide cement |
| `硅酸盐水泥、水玻璃` | 2 | Portland cement with water glass |
| `普通硅酸盐水泥` | 1 | ordinary Portland cement |

The grouping key of the cross-validation is the pair (column 2, column 6). For the 115 records carrying the
combined label the key resolves, in effect, to the aggregate type. This merges groups rather than splitting
them: the 115 records fall into 13 aggregate types, and the largest single group of the table
(`拉法基铝水泥` | `Serpentine`, 50 records) is unaffected. Merging makes the held-out fold *more* disjoint
from the training folds, so the partition is, if anything, stricter than a finer key would produce.

## Water–cement ratio

Two properties of this column limit what can be concluded from it, and both are stated in Section 4.2 of
the manuscript.

- **106 of 239 values are missing.** They are imputed with the median of the training folds, not of the
  full table. For those records the ratio carries no information, so the water–cement branch of the
  monotonicity penalty is inactive.
- **8 values exceed unity**: three at 1.01 and five at 1.61. A water–cement ratio above 1 is not physically
  admissible for a hardened concrete and is most likely a transcription artefact in the source compilation.
  The records are retained rather than deleted on our own judgement, and the anomaly is reported in the
  manuscript because it limits the weight that the water–cement branch can bear.

## Files not shipped

`concrete_augmented.csv` (800 rows, identifiers 1–800, the first 239 of which are the measured records)
belonged to the original submission. It was produced by nearest-neighbour interpolation and was used to
train the baseline of that version. It is not read by any script here and has been removed from the
repository, because the revised manuscript trains every method on measured observations only.
