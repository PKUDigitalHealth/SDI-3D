# SDI-3D experiment code

This directory combines the original model implementation with the analysis scripts used across the SDI-3D manuscript. Historical and sensitivity implementations are retained under `legacy/` and should not be mixed with the current primary workflow.

## 01_core_model

- `net1d.py`: Stage-1 one-dimensional CNN architecture.
- `model_transformer.py`: temporal Transformer and task heads.
- `dataset_sliding.py`: overnight window dataset and Wake, REM, and depth-ranking targets.
- `split_data.py`: file-level dataset splitting.
- `train.py`: Stage-2 multitask training.
- `test.py`: internal/external evaluation, bootstrap intervals, and epoch-level exports.
- `run_wenxin_sdi3d.py`: adapter from ambulatory ECG input to SDI-3D outputs.

## 02_trajectory_features

- `extract_body_brain_dual_features.py`: current handcrafted Body--Brain trajectory biomarkers.
- `augment_tos_tci_sri.py`: adds TOS, TCI, and SRI to an existing feature table.
- `sleep_autoencoder_8d.py`: outcome-agnostic eight-dimensional night embedding.
- `trajectory_ml/`: sequence loading, encoding, feature, evaluation, and pipeline modules.
- `legacy/`: original full-layer feature implementation and early case plot.

## 03_coordinate_validation

Participant-level and stage-level PCA, sleep-stage distribution analysis, and PSG-derived EEG/EOG/EMG probes used to assess coordinate validity and complementarity.

## 04_prediction_and_shap

- `compare_sdi_feature_sets.py`: within-cohort five-fold out-of-fold comparison and frozen CFS+MrOS-to-SHHS transfer for SDI-6, handcrafted-8, embedding-8, fused-16, and combined-22 representations.
- `shhs_xgboost_14_features.py`: SHAP analysis using SDI-6 plus eight handcrafted SDI-3D biomarkers.
- `analysis_specification.json`: classifier, bootstrap, seed, and multiplicity settings.
- `legacy/`: earlier ECG-SDI plus five-feature SDI-3D analyses.

## 05_phenotype_survival

The primary chain is:

1. `cluster_three_cohorts_sdi8_embedding8.py`
2. `analyze_sdi_cluster_outcomes.py`
3. `plot_combined_three_cohort_forest.py`
4. `shhs_frozen_cluster_survival_analysis.py`

It uses eight handcrafted biomarkers plus eight learned embeddings, with a separate two-component GMM fitted within each cohort. Alternative eight-feature, consensus-boundary, and SHHS-only implementations are stored under `legacy/`.

## 06_cpap

- `analyze_pre_post_features.py`: paired Wilcoxon tests, paired rank-biserial effects, and Benjamini--Hochberg correction for 500 matched participants.
- `plot_publication_pre_post_boxplots.py`: publication CPAP boxplots.

## 07_case_figures

Scripts for the four SHHS cases, two-coordinate time series, three-dimensional trajectories, and exploratory wearable illustrations.

## 08_optional_and_utilities

Quadrant analyses and vector-PDF terminology updates. These scripts are not part of the primary manuscript evidence chain.

## Recommended execution order

```text
ECG preprocessing and SDI-3D inference
  -> coordinate validation
  -> handcrafted-8 and embedding-8 extraction
  -> SDI-6/SDI-3D comparison and frozen external transfer
  -> cohort-specific 16-feature GMM
  -> clinical association and SHHS survival analyses
  -> paired BZ-CPAP analysis
  -> manuscript figures
```

Install the Python dependencies listed in `requirements.txt`. Data paths and checkpoint paths must be configured locally. See `KNOWN_ISSUES.md` before attempting model training.
