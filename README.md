# Math Misconception Detection

本项目用于数学误解检测课程项目。当前任务被简化为 **4 分类**，并已经完成数据预处理与特征向量生成流程。

## 任务定义

原始 `Category` 有 6 类，本项目合并为 4 类：

| 原始标签 | 4 分类标签 |
| --- | --- |
| `True_Correct` | `True_Correct` |
| `False_Correct` | `True_Correct` |
| `False_Misconception` | `False_Misconception` |
| `True_Misconception` | `False_Misconception` |
| `False_Neither` | `False_Neither` |
| `True_Neither` | `True_Neither` |

4 分类标签保存在 `Category_4` 字段中。

## 项目结构

```text
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── processed_train.csv      # preprocess.py 生成
│   └── processed_test.csv       # preprocess.py 生成
├── features/                    # build_features.py 生成，不提交 Git
├── preprocess.py
├── build_features.py
├── .gitignore
└── README.md
```

`features/`、虚拟环境和缓存目录都已在 `.gitignore` 中忽略。

## 环境配置

本项目使用 `uv` 管理环境，推荐 Python 3.11。

```powershell
uv venv --python 3.11
.venv\Scripts\activate
```

安装 CUDA 12.8 版本 PyTorch：

```powershell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

安装其余依赖：

```powershell
uv pip install pandas numpy scipy scikit-learn sentence-transformers tqdm joblib matplotlib
```

开发环境验证结果：

```text
torch: 2.11.0+cu128
torch CUDA: 12.8
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
sentence-transformers: 5.4.1
scikit-learn: 1.8.0
pandas: 3.0.2
matplotlib: 3.10.8
```

检查 GPU：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 执行流程

### 1. 数据预处理

```powershell
.\.venv\Scripts\python.exe preprocess.py
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

预处理会把以下字段拼成统一文本：

```text
Question: {QuestionText}
Answer: {MC_Answer}
Student explanation: {StudentExplanation}
```

### 2. 特征生成

```powershell
.\.venv\Scripts\python.exe build_features.py
```

输入：

```text
data/processed_train.csv
data/processed_test.csv
```

输出目录：

```text
features/
```

## 训练集和验证集划分

由于当前 `test.csv` 样本很少，模型评估使用从训练集中划分出的验证集。

划分方式：

```text
train: 80%
validation: 20%
random_state: 42
stratify: Category_4
```

当前样本数量：

```text
train: 29356
validation: 7340
test: 3
```

当前类别分布：

| 类别 | Train | Validation |
| --- | ---: | ---: |
| `True_Correct` | 12023 | 3006 |
| `False_Misconception` | 7888 | 1972 |
| `False_Neither` | 5233 | 1309 |
| `True_Neither` | 4212 | 1053 |

划分后的文本文件：

```text
features/train_split.csv
features/val_split.csv
features/test_processed.csv
```

## 特征向量说明

当前特征提取阶段会同时生成两种向量，分别用于不同模型。

### 1. 768 维 MPNet Embedding

模型：

```text
sentence-transformers/all-mpnet-base-v2
```

输出文件：

```text
features/mpnet_train_embeddings.npy
features/mpnet_val_embeddings.npy
features/mpnet_test_embeddings.npy
```

矩阵形状：

```text
train: (29356, 768)
val:   (7340, 768)
test:  (3, 768)
```

用途：

```text
XGBoost / LightGBM
Simple MLP
其他适合 dense vector 的模型
```

### 2. 50233 维融合特征

融合特征由以下部分拼接：

```text
word-level TF-IDF
char-level TF-IDF
768 维 MPNet embedding
```

当前维度：

```text
TF-IDF: 49465
MPNet embedding: 768
Total: 50233
```

输出文件：

```text
features/X_train.npz
features/X_val.npz
features/X_test.npz
```

矩阵形状：

```text
X_train: (29356, 50233)
X_val:   (7340, 50233)
X_test:  (3, 50233)
```

用途：

```text
Logistic Regression
Linear SVM
其他适合高维稀疏文本特征的线性模型
```

## 模型输入安排

后续实验请按下表选择输入特征。这个安排是当前项目的默认约定。

| 模型 | 推荐输入 | 文件 | 维度 | 备注 |
| --- | --- | --- | ---: | --- |
| Logistic Regression | TF-IDF + MPNet 融合特征 | `features/X_train.npz`, `features/X_val.npz` | 50233 | 推荐首个 baseline |
| SVM | TF-IDF + MPNet 融合特征 | `features/X_train.npz`, `features/X_val.npz` | 50233 | 使用 `LinearSVC`，不建议 RBF SVM |
| XGBoost / LightGBM | MPNet embedding | `features/mpnet_train_embeddings.npy`, `features/mpnet_val_embeddings.npy` | 768 | 不建议直接使用 50233 维稀疏特征 |
| Simple MLP | MPNet embedding | `features/mpnet_train_embeddings.npy`, `features/mpnet_val_embeddings.npy` | 768 | dense vector 更适合 MLP |
| BERT / DistilBERT | 原始 `text` | `features/train_split.csv`, `features/val_split.csv` | 不适用 | 直接 fine-tune 文本，不使用预生成向量 |

简要规则：

```text
线性模型 -> 50233 维融合特征
树模型 / MLP -> 768 维 MPNet embedding
Transformer -> 原始 text
```

## 标签编码

标签文件：

```text
features/y_train.npy
features/y_val.npy
```

标签映射：

```text
0 -> True_Correct
1 -> False_Misconception
2 -> False_Neither
3 -> True_Neither
```

完整元信息保存在：

```text
features/feature_metadata.json
```

## 读取特征示例

### 读取 50233 维融合特征

```python
import numpy as np
from scipy import sparse

X_train = sparse.load_npz("features/X_train.npz")
X_val = sparse.load_npz("features/X_val.npz")
y_train = np.load("features/y_train.npy")
y_val = np.load("features/y_val.npy")
```

### 读取 768 维 MPNet Embedding

```python
import numpy as np

X_train = np.load("features/mpnet_train_embeddings.npy")
X_val = np.load("features/mpnet_val_embeddings.npy")
y_train = np.load("features/y_train.npy")
y_val = np.load("features/y_val.npy")
```

### Logistic Regression 示例

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="saga",
    n_jobs=-1,
)

clf.fit(X_train, y_train)
pred = clf.predict(X_val)

print(classification_report(y_val, pred))
print("macro_f1:", f1_score(y_val, pred, average="macro"))
```

推荐评估指标：

```text
accuracy
macro-F1
classification report
confusion matrix
```

其中 `macro-F1` 比 accuracy 更重要，因为类别分布不完全均衡。

## 已搭建实验

### Logistic Regression Baseline

实验目录：

```text
experiments/logistic_regression/
```

运行：

```powershell
.\.venv\Scripts\python.exe experiments\logistic_regression\train_logreg.py
```

该实验使用：

```text
features/X_train.npz
features/X_val.npz
```

也就是 50233 维 TF-IDF + MPNet 融合特征。

默认输出：

```text
experiments/logistic_regression/outputs/
```

主要结果包括：

```text
metrics.json
classification_report.txt
classification_report.csv
confusion_matrix.png
confusion_matrix_normalized.png
per_class_metrics.png
val_predictions.csv
logreg_model.joblib
top_features_by_class.csv
```

### Support Vector Machine Baseline

实验目录：

```text
experiments/svm/
```

运行：

```powershell
.\.venv\Scripts\python.exe experiments\svm\train_svm.py
```

该实验使用：

```text
features/X_train.npz
features/X_val.npz
```

也就是 50233 维 TF-IDF + MPNet 融合特征。当前默认模型为 `LinearSVC`，不建议在该高维稀疏特征上使用 RBF SVM。

默认输出：

```text
experiments/svm/outputs/
```

## Git 提交说明

推荐提交：

```text
README.md
.gitignore
preprocess.py
build_features.py
```

不建议提交：

```text
.venv/
.uv-cache/
.uv-python/
.hf-cache/
features/
```

`data/processed_train.csv` 和 `data/processed_test.csv` 是可复现中间文件，一般不需要提交，除非课程要求提交处理后的数据。

## 完整复现命令

```powershell
uv venv --python 3.11
.venv\Scripts\activate

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install pandas numpy scipy scikit-learn sentence-transformers tqdm joblib matplotlib

python preprocess.py
python build_features.py
```

预期主要输出：

```text
features/mpnet_train_embeddings.npy  # 768 维 embedding
features/X_train.npz                 # 50233 维融合特征
features/y_train.npy
features/y_val.npy
```
