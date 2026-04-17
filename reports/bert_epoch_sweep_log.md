# BERT Epoch Sweep Log

- Generated at: `2026-04-17`
- Starting best: `bert_ep3_lr2e5_bs32_len192`
- Starting validation macro_f1: `0.8299`
- Stopping rule: stop after `2` consecutive larger epochs do not beat the current best `macro_f1`

## Sweep Results

| Run | Epoch | Macro F1 | Accuracy | Train Seconds |
| --- | ---: | ---: | ---: | ---: |
| `bert_ep4_lr2e5_bs32_len192` | 4 | 0.8360 | 0.8639 | 89.87 |
| `bert_ep5_lr2e5_bs32_len192` | 5 | 0.8438 | 0.8700 | 111.87 |
| `bert_ep6_lr2e5_bs32_len192` | 6 | 0.8495 | 0.8740 | 135.57 |
| `bert_ep7_lr2e5_bs32_len192` | 7 | 0.8449 | 0.8699 | 157.67 |
| `bert_ep8_lr2e5_bs32_len192` | 8 | 0.8468 | 0.8718 | 180.19 |

## Best Result

- Best run: `bert_ep6_lr2e5_bs32_len192`
- Validation macro_f1: `0.8495`
- Validation accuracy: `0.8740`
- Canonical outputs updated to: `experiments/bert/outputs`

## Notes

- `epoch=7` and `epoch=8` both failed to beat the `epoch=6` best, so the sweep stopped.
- Detailed machine-readable results live in `reports/bert_epoch_sweep_summary.json`.
