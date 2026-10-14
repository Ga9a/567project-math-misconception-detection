# Math Misconception Detection

A four-class student math misconception classification project comparing logistic regression, Linear SVM, XGBoost, an MLP, and BERT. The source data have six categories; preprocessing maps them to the project-specific `Category_4` target.

## Task and labels

| Original category | Four-class target |
| --- | --- |
| `True_Correct`, `False_Correct` | `True_Correct` |
| `False_Misconception`, `True_Misconception` | `False_Misconception` |
| `False_Neither` | `False_Neither` |
| `True_Neither` | `True_Neither` |

This target differs from the original six-class task. Scores and outputs must be interpreted using the four-class definition; they are not directly interchangeable with a six-class competition metric.

## Setup

Use Python 3.11 or 3.12 and run commands from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-models.txt
```

On Windows, activate `.venv\Scripts\Activate.ps1`. `requirements.txt` contains the classical ML and reporting dependencies; `requirements-models.txt` adds model training and embedding dependencies. Use a PyTorch build appropriate for your accelerator when training on CUDA. No machine-specific conda path is required.

## Data and preprocessing

Place source files at `data/train.csv` and `data/test.csv`. Required text columns are `QuestionText`, `MC_Answer`, and `StudentExplanation`; training also requires `Category`. Row and question identifiers are retained. Obtain data from its original provider and follow its distribution terms.

```sh
python preprocess.py
```

Outputs are `data/processed_train.csv` and `data/processed_test.csv`. Text combines the question, selected answer, and student explanation. Outputs are written atomically, so invalid labels do not replace an existing processed file. Input and output paths must differ.

## Features and validation split

```sh
python build_features.py --device cpu --hf-cache-dir hf_cache
```

The script makes a stratified 80/20 train/validation split using seed 42, then fits word and character TF-IDF vectorizers only on training text. It downloads MPNet to the configured cache and generates normalized embeddings. Use `--device cuda` for a supported GPU.

| Model | Input |
| --- | --- |
| Logistic regression, Linear SVM | Sparse TF-IDF + MPNet fusion (`features/X_*.npz`) |
| XGBoost, MLP | Dense MPNet embeddings (`features/mpnet_*_embeddings.npy`) |
| BERT | Raw text from `features/train_split.csv` and `features/val_split.csv` |

Labels are saved as `features/y_train.npy` and `features/y_val.npy`; metadata and vectorizers are also saved under `features/`. Features, caches, and checkpoints are generated artifacts. Their sizes depend on the dataset and configured vocabulary limits.

## Train models

```sh
python experiments/logistic_regression/train_logreg.py
python experiments/svm/train_svm.py --C 1.0 --max-iter 5000
python experiments/xgboost/train_xgboost.py --n-estimators 400 --max-depth 8 --n-jobs 8
python experiments/mlp/train_mlp.py --epochs 20 --batch-size 256 --device cpu
python experiments/bert/train_bert.py --model-name google-bert/bert-base-uncased --hf-cache-dir hf_cache --epochs 1 --batch-size 8 --max-length 256 --device cpu
```

Run each command with `--help` for supported options. BERT downloads its model snapshot before loading it locally. `experiments/tune_models.py` and `experiments/bert/sweep_epochs.py` run multi-configuration experiments and can require substantial compute.

## Outputs and interpretation

Each model writes to `experiments/<model>/outputs/`: metrics, classification reports, confusion matrices, validation predictions, plots, and model artifacts. Focus on macro-F1 and per-class precision/recall alongside accuracy and weighted F1. See the baseline READMEs and `reports/` for recorded experiments. Historical metrics are retained results, not claims of a fresh training run.

The random row split may share question contexts between training and validation. For claims about generalization to unseen questions, add a grouped split by `QuestionId` and report that protocol separately. Hyperparameter selection also requires a held-out evaluation beyond the repeatedly used validation split.

## Verify

```sh
python -m unittest discover -s tests -v
python -m compileall -q preprocess.py build_features.py experiments
python scripts/smoke_classical.py
python test.py --feature-dir features
```

Preprocessing tests use small synthetic CSVs and need only the Python standard library. They check label mapping, required columns, atomic failure behavior, and protection against overwriting raw input. Full feature generation and neural training additionally require the model dependencies, data, downloaded weights, and suitable hardware.

The classical smoke command uses small synthetic fixtures to verify training and report generation; its scores are not dataset benchmark results. `test.py` checks generated feature dimensions, finite values, and label alignment without training a model.
