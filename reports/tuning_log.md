# Model Tuning Log

- Generated at: `2026-04-17T16:04:38`
- Objective: maximize validation `macro_f1`, use `accuracy` as tie-breaker
- Environment: `conda activate /blue/ruogu.fang/from_red/hanwen/3d-gen/conda/envs/flux_new`
- Features: `features/mpnet_*_embeddings.npy` for `xgboost` and `mlp`, raw `text` for `bert`
- Model downloads: Hugging Face models were downloaded into `hf_cache/` before training and then loaded from local snapshot paths

## Best Per Model

### bert

- Best run: `bert_ep6_lr2e5_bs32_len192`
- Validation macro_f1: `0.8495`
- Validation accuracy: `0.8740`
- Train seconds: `135.57`
- Canonical outputs: `experiments/bert/outputs`
- Config: `["--device", "cuda", "--model-name", "hf_cache/models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594", "--hf-cache-dir", "hf_cache", "--epochs", "6", "--batch-size", "32", "--max-length", "192", "--learning-rate", "2e-05", "--weight-decay", "0.01", "--warmup-ratio", "0.1", "--logging-steps", "100", "--seed", "42"]`

### mlp

- Best run: `mlp_h768_do01_lr15e3_bs512`
- Validation macro_f1: `0.7895`
- Validation accuracy: `0.8151`
- Train seconds: `14.22`
- Canonical outputs: `experiments/mlp/outputs`
- Config: `["--device", "cuda", "--epochs", "24", "--batch-size", "512", "--hidden-dim", "768", "--dropout", "0.1", "--learning-rate", "0.0015", "--weight-decay", "0.00005", "--patience", "6"]`

### xgboost

- Best run: `xgb_lr004_d10_est1200_ss085_cs085_l15`
- Validation macro_f1: `0.7990`
- Validation accuracy: `0.8386`
- Train seconds: `13.42`
- Canonical outputs: `experiments/xgboost/outputs`
- Config: `["--device", "cuda", "--n-estimators", "1200", "--max-depth", "10", "--learning-rate", "0.04", "--subsample", "0.85", "--colsample-bytree", "0.85", "--reg-lambda", "1.5", "--early-stopping-rounds", "50", "--n-jobs", "8"]`

## Overall Best

- Model: `bert`
- Run: `bert_ep6_lr2e5_bs32_len192`
- Validation macro_f1: `0.8495`
- Validation accuracy: `0.8740`
- Canonical outputs: `experiments/bert/outputs`

## All Runs

| Model | Run | Macro F1 | Accuracy | Train Seconds |
| --- | --- | ---: | ---: | ---: |
| bert | `bert_ep6_lr2e5_bs32_len192` | 0.8495 | 0.8740 | 135.57 |
| bert | `bert_ep8_lr2e5_bs32_len192` | 0.8468 | 0.8718 | 180.19 |
| bert | `bert_ep7_lr2e5_bs32_len192` | 0.8449 | 0.8699 | 157.67 |
| bert | `bert_ep5_lr2e5_bs32_len192` | 0.8438 | 0.8700 | 111.87 |
| bert | `bert_ep4_lr2e5_bs32_len192` | 0.8360 | 0.8639 | 89.87 |
| bert | `bert_ep3_lr2e5_bs32_len192` | 0.8299 | 0.8584 | 67.58 |
| bert | `bert_ep2_lr3e5_bs32_len192` | 0.8258 | 0.8568 | 45.71 |
| bert | `bert_ep2_lr2e5_bs32_len192` | 0.8166 | 0.8497 | 45.73 |
| bert | `bert_ep2_lr15e5_bs32_len256` | 0.8027 | 0.8392 | 45.80 |
| mlp | `mlp_h768_do01_lr15e3_bs512` | 0.7895 | 0.8151 | 14.22 |
| mlp | `mlp_h1024_do03_lr8e4_bs1024` | 0.7795 | 0.8102 | 9.55 |
| mlp | `mlp_h1024_do02_lr1e3_bs1024` | 0.7692 | 0.7982 | 8.06 |
| mlp | `mlp_h512_do02_lr2e3_bs1024` | 0.7673 | 0.7932 | 8.13 |
| mlp | `mlp_h768_do02_lr1e3_bs1024` | 0.7664 | 0.8046 | 8.14 |
| xgboost | `xgb_lr004_d10_est1200_ss085_cs085_l15` | 0.7990 | 0.8386 | 13.42 |
| xgboost | `xgb_lr007_d8_est600_ss08_cs09_l1` | 0.7973 | 0.8366 | 7.88 |
| xgboost | `xgb_lr005_d8_est500_ss09_cs09_l1` | 0.7970 | 0.8358 | 10.11 |
| xgboost | `xgb_lr003_d10_est900_ss09_cs09_l1` | 0.7961 | 0.8362 | 18.21 |
| xgboost | `xgb_lr005_d12_est700_ss09_cs08_l2` | 0.7931 | 0.8339 | 14.82 |

## Files Written

- `reports/tuning_summary.json`
- `reports/tuning_results.csv`
- `experiments/<model>/outputs/` updated to the best run for each model

## Notes

- `xgboost` uses validation-set early stopping during tuning.
- `mlp` keeps the checkpoint with the best validation `macro_f1` inside each run.
- `bert` keeps improving through the later epoch sweep until `epoch=6`, then falls off at `epoch=7` and `epoch=8`.
