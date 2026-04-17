# Math Misconception Detection

本项目用于数学误解检测课程项目。当前任务被简化为 **4 分类**，并支持以下训练路线：

- `Logistic Regression`
- `Linear SVM`
- `XGBoost`
- `Simple MLP`
- `BERT`

当前仓库面向 **Linux + conda** 使用。你已经有现成环境：

`/blue/ruogu.fang/from_red/hanwen/3d-gen/conda/envs/flux_new`

因此这里不再使用 `uv`。

## 任务定义

原始 `Category` 有 6 类，这里合并为 4 类：

| 原始标签 | 4 分类标签 |
| --- | --- |
| `True_Correct` | `True_Correct` |
| `False_Correct` | `True_Correct` |
| `False_Misconception` | `False_Misconception` |
| `True_Misconception` | `False_Misconception` |
| `False_Neither` | `False_Neither` |
| `True_Neither` | `True_Neither` |

新的 4 分类标签保存在 `Category_4` 字段中。

## 项目结构

```text
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── processed_train.csv
│   └── processed_test.csv
├── features/                            # build_features.py 生成
├── hf_cache/                            # Hugging Face 模型缓存
├── preprocess.py
├── build_features.py
├── experiments/
│   ├── common.py
│   ├── logistic_regression/
│   ├── svm/
│   ├── xgboost/
│   ├── mlp/
│   └── bert/
└── README.md
```

## 环境配置

先激活现有 conda 环境：

```bash
source /apps/conda/25.7.0/etc/profile.d/conda.sh
conda activate /blue/ruogu.fang/from_red/hanwen/3d-gen/conda/envs/flux_new
```

如果缺依赖，直接在这个环境里补：

```bash
pip install scikit-learn sentence-transformers xgboost matplotlib
```

如果你要训练 BERT，通常还需要这些：

```bash
pip install transformers datasets accelerate huggingface_hub
```

检查 PyTorch / CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 执行流程

### 1. 数据预处理

```bash
python preprocess.py
```

输入：

```text
data/train.csv
data/test.csv
```

输出：

```text
data/processed_train.csv
data/processed_test.csv
```

预处理阶段会将以下字段拼成统一文本：

```text
Question: {QuestionText}
Answer: {MC_Answer}
Student explanation: {StudentExplanation}
```

### 2. 特征生成

```bash
python build_features.py \
  --device cpu \
  --hf-cache-dir hf_cache
```

说明：

- 该脚本会先做 `train/val` 划分，再生成特征
- MPNet 模型会 **先下载到本地缓存**，再从本地路径加载
- 如果当前节点有可用 GPU，可以把 `--device cpu` 改为 `--device cuda`

主要输出：

```text
features/train_split.csv
features/val_split.csv
features/test_processed.csv

features/y_train.npy
features/y_val.npy

features/mpnet_train_embeddings.npy
features/mpnet_val_embeddings.npy
features/mpnet_test_embeddings.npy

features/X_train.npz
features/X_val.npz
features/X_test.npz
features/feature_metadata.json
```

## 训练集和验证集划分

由于当前 `test.csv` 样本很少，评估使用从训练集中切出的验证集。

默认划分方式：

```text
train: 80%
validation: 20%
random_state: 42
stratify: Category_4
```

## 模型输入约定

| 模型 | 输入 | 文件 |
| --- | --- | --- |
| Logistic Regression | TF-IDF + MPNet 融合特征 | `features/X_train.npz`, `features/X_val.npz` |
| Linear SVM | TF-IDF + MPNet 融合特征 | `features/X_train.npz`, `features/X_val.npz` |
| XGBoost | MPNet embedding | `features/mpnet_train_embeddings.npy`, `features/mpnet_val_embeddings.npy` |
| Simple MLP | MPNet embedding | `features/mpnet_train_embeddings.npy`, `features/mpnet_val_embeddings.npy` |
| BERT | 原始文本 | `features/train_split.csv`, `features/val_split.csv` |

简要规则：

```text
线性模型 -> 融合稀疏特征
树模型 / MLP -> MPNet dense embedding
Transformer -> 原始 text
```

## 训练命令

### Logistic Regression

```bash
python experiments/logistic_regression/train_logreg.py
```

### Linear SVM

```bash
python experiments/svm/train_svm.py
```

### XGBoost

```bash
python experiments/xgboost/train_xgboost.py \
  --n-estimators 400 \
  --max-depth 8 \
  --n-jobs 8
```

### Simple MLP

```bash
python experiments/mlp/train_mlp.py \
  --epochs 20 \
  --batch-size 256 \
  --device cpu
```

如果当前节点有 GPU，可改为：

```bash
python experiments/mlp/train_mlp.py \
  --epochs 20 \
  --batch-size 256 \
  --device cuda
```

### BERT

`BERT` 训练脚本会先下载模型到 `hf_cache/`，然后再从本地 snapshot 加载。

```bash
python experiments/bert/train_bert.py \
  --model-name google-bert/bert-base-uncased \
  --hf-cache-dir hf_cache \
  --epochs 1 \
  --batch-size 8 \
  --max-length 256 \
  --device cpu
```

如果当前节点有 GPU，可改为：

```bash
python experiments/bert/train_bert.py \
  --model-name google-bert/bert-base-uncased \
  --hf-cache-dir hf_cache \
  --epochs 1 \
  --batch-size 16 \
  --max-length 256 \
  --device cuda
```

## 输出结果

每个训练脚本都会在各自目录下生成：

```text
experiments/<model>/outputs/
├── metrics.json
├── classification_report.txt
├── classification_report.csv
├── confusion_matrix.csv
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── per_class_metrics.png
├── val_predictions.csv
├── top_features_by_class.csv
└── model artifact
```

其中：

- `metrics.json` 包含 accuracy、macro F1、weighted F1、训练时间等
- `val_predictions.csv` 便于后续 error analysis
- `top_features_by_class.csv` 提供每个模型的可解释性摘要
- BERT / MPNet 相关模型会把 Hugging Face snapshot 路径写入指标文件，便于复现

## 推荐执行顺序

为了尽快拿到可比较结果，建议按下面顺序运行：

1. `python preprocess.py`
2. `python build_features.py --device cpu --hf-cache-dir hf_cache`
3. `python experiments/xgboost/train_xgboost.py`
4. `python experiments/mlp/train_mlp.py --device cpu`
5. `python experiments/bert/train_bert.py --hf-cache-dir hf_cache --epochs 1 --device cpu`

如果你在 GPU 节点上，优先把 `build_features.py`、`train_mlp.py`、`train_bert.py` 的 `--device` 改成 `cuda`。
