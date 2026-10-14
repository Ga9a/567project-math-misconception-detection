# Logistic regression baseline

A four-class baseline using sparse word TF-IDF, character TF-IDF, and MPNet embeddings. Fit feature extraction on the training partition before running this model.

## Run

From the repository root:

```sh
python experiments/logistic_regression/train_logreg.py --C 1.0 --max-iter 1000 --solver saga
```

Inputs: `features/X_train.npz`, `features/X_val.npz`, `features/y_train.npy`, and `features/y_val.npy`. The recorded feature dimension was 50,233, including 768 MPNet dimensions; future dimensions depend on feature settings and data.

Outputs under `experiments/logistic_regression/outputs/` include `metrics.json`, classification reports, confusion matrix CSVs and plots, validation predictions, feature summaries, and `logreg_model.joblib`. Compare macro-F1 and per-class metrics as well as accuracy.

## Recorded metrics

The following values are copied from the committed metrics artifact, rather than recomputed during documentation maintenance:

```json
{
  "accuracy": 0.8219346049046321,
  "macro_f1": 0.7996120686934094,
  "weighted_f1": 0.8245652980974194
}
```

See `outputs/classification_report.txt` and `outputs/confusion_matrix.csv` for the full recorded report. Category imbalance and the validation protocol affect comparisons. A question-grouped holdout is needed to evaluate unseen question contexts.
