# Known issues and version boundaries

## Blocking issues in the original model code

- `01_core_model/train.py` imports `Wake_rankloss.py`, which was not present in the supplied source directories.
- `train.py` references an undefined variable named `d_sleep` in the constraint-loss calculation.
- The Stage-1 CNN architecture is present, but a complete Stage-1 training entry point matching the manuscript cohorts was not found.
- Required model checkpoints are external and are not included.

## Configuration requirements

- Several scripts retain author-machine absolute paths and must be changed before execution.
- Clinical cohort files are not distributed in this repository.
- BZ-CPAP source data are not distributed in this repository.

## Analysis version boundaries

- The primary phenotype analysis fits separate 16-feature GMMs in CFS, MrOS, and SHHS.
- Eight-feature GMM, pooled CFS+MrOS consensus GMM, and SHHS-only reclustering implementations are historical or sensitivity versions.
- The manuscript SHAP figure uses 14 inputs: SDI-6 plus eight handcrafted SDI-3D biomarkers. It does not include the eight learned embeddings.

## Terminology compatibility

Some source column names are retained for compatibility with existing result tables:

- `rem_logit_rem` and `rem_prob_rem` correspond to the manuscript's Brain sleep-depth output.
- `BRAI` corresponds to the displayed name BBAI.
- `Disturbed Sleep` corresponds to the displayed name Fragmented Sleep.
