# Math Misconception Detection

本仓库包含数学误解检测课程项目的数据预处理和特征生成流程。

当前项目目标被简化为 **4 分类任务**。处理流程会把题目文本、选择题答案和学生解释拼接成统一文本，然后生成后续传统机器学习模型可以直接使用的特征向量。

后续可以使用的传统机器学习算法包括：

```text
Logistic Regression
Linear SVM
Random Forest
XGBoost / LightGBM
其他 sklearn 风格分类器
```

## 项目目标

原始训练数据中 `Category` 一共有 6 个类别：

```text
True_Correct
False_Correct
False_Misconception
True_Misconception
False_Neither
True_Neither
```

为了降低长尾类别带来的难度，本项目将任务简化为 4 分类：

```text
True_Correct
False_Misconception
False_Neither
True_Neither
```

6 类到 4 类的映射规则如下：

| 原始 Category | 简化后的 Category |
| --- | --- |
| `True_Correct` | `True_Correct` |
| `False_Correct` | `True_Correct` |
| `False_Misconception` | `False_Misconception` |
| `True_Misconception` | `False_Misconception` |
| `False_Neither` | `False_Neither` |
| `True_Neither` | `True_Neither` |

简化后的 4 分类标签会保存到 `Category_4` 字段中。

## 仓库结构

```text
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── processed_train.csv        # 由 preprocess.py 生成
│   └── processed_test.csv         # 由 preprocess.py 生成
├── features/                      # 由 build_features.py 生成，已被 Git 忽略
├── preprocess.py
├── build_features.py
├── .gitignore
└── README.md
```

其中 `features/` 目录存放生成后的特征矩阵、embedding、TF-IDF 向量器和标签文件。该目录文件体积较大，并且可以重新生成，所以不会提交到 Git。

## 环境配置

本项目使用 `uv` 管理 Python 虚拟环境。

推荐 Python 版本：

```text
Python 3.11
```

选择 Python 3.11 的原因是它与当前使用的 PyTorch、sentence-transformers、scikit-learn、pandas 等依赖兼容性较好。

### 1. 创建虚拟环境

在项目根目录运行：

```powershell
uv venv --python 3.11
```

Windows PowerShell 中激活虚拟环境：

```powershell
.venv\Scripts\activate
```

如果不想激活环境，也可以直接使用虚拟环境里的 Python 执行脚本：

```powershell
.\.venv\Scripts\python.exe script_name.py
```

### 2. 安装 PyTorch

本项目开发时使用的 GPU 是：

```text
NVIDIA GeForce RTX 5060 Laptop GPU
```

当前安装的 PyTorch 版本是：

```text
torch 2.11.0+cu128
torchvision 0.26.0+cu128
torchaudio 2.11.0+cu128
PyTorch 使用的 CUDA runtime: 12.8
```

安装 CUDA 12.8 版本的 PyTorch：

```powershell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

开发机器上的 `nvidia-smi` 显示驱动支持到 CUDA 13.1，因此可以运行 CUDA 12.8 的 PyTorch wheel。一般来说，只要 NVIDIA 驱动足够新，就可以向后兼容运行较低版本 CUDA runtime 的 PyTorch 包。

如果在没有 NVIDIA GPU 的机器上运行，可以安装 CPU 版本 PyTorch，或者根据当前机器配置参考 PyTorch 官方安装命令。

### 3. 安装其他依赖

安装数据处理、embedding 和传统机器学习需要的包：

```powershell
uv pip install pandas numpy scipy scikit-learn sentence-transformers tqdm joblib
```

当前流程主要依赖：

```text
pandas
numpy
scipy
scikit-learn
sentence-transformers
torch
torchvision
torchaudio
tqdm
joblib
```

## 验证 GPU 环境

安装完成后，可以用下面的命令检查 PyTorch 是否能访问 GPU：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

在本项目开发机器上的预期输出为：

```text
2.11.0+cu128
True
12.8
NVIDIA GeForce RTX 5060 Laptop GPU
```

如果 `torch.cuda.is_available()` 输出为 `False`，说明当前 PyTorch 没有正确使用 GPU，需要检查 PyTorch 版本、CUDA wheel、NVIDIA 驱动或虚拟环境是否正确。

## 第一步：数据预处理

运行：

```powershell
.\.venv\Scripts\python.exe preprocess.py
```

该脚本读取：

```text
data/train.csv
data/test.csv
```

并生成：

```text
data/processed_train.csv
data/processed_test.csv
```

### 预处理做了什么

对于每一条样本，脚本会把三个文本字段拼接成一个统一的 `text` 字段：

```text
Question: {QuestionText}
Answer: {MC_Answer}
Student explanation: {StudentExplanation}
```

这样做的原因是：学生解释本身并不足以判断是否正确或是否存在误解，还需要结合题目和选择的答案一起判断。

训练集处理后保留：

```text
row_id
QuestionId
text
Category
Category_4
```

其中：

```text
Category   原始 6 分类标签
Category_4 简化后的 4 分类标签
```

测试集没有标签，因此 `processed_test.csv` 只包含：

```text
row_id
QuestionId
text
```

## 第二步：生成特征向量

运行：

```powershell
.\.venv\Scripts\python.exe build_features.py
```

该脚本读取：

```text
data/processed_train.csv
data/processed_test.csv
```

并在下面目录生成特征文件：

```text
features/
```

后续传统机器学习模型最主要使用的输入文件是：

```text
features/X_train.npz
features/X_val.npz
features/X_test.npz
features/y_train.npy
features/y_val.npy
```

其中：

```text
X_train.npz  训练集特征矩阵
X_val.npz    验证集特征矩阵
X_test.npz   测试集特征矩阵
y_train.npy  训练集标签
y_val.npy    验证集标签
```

## 训练集和验证集划分

由于当前 `test.csv` 样本较少，因此模型开发和评估主要依赖从训练集中划分出的验证集。

当前划分方式：

```text
训练集: 80%
验证集: 20%
random_state: 42
stratify 字段: Category_4
```

这里使用 `stratify=Category_4`，保证训练集和验证集中的 4 个类别比例尽量一致。

当前划分后的样本数量：

```text
train: 29356
validation: 7340
test: 3
```

当前训练集类别分布：

```text
True_Correct           12023
False_Misconception     7888
False_Neither           5233
True_Neither            4212
```

当前验证集类别分布：

```text
True_Correct            3006
False_Misconception     1972
False_Neither           1309
True_Neither            1053
```

划分后的 CSV 也会保存：

```text
features/train_split.csv
features/val_split.csv
features/test_processed.csv
```

## 特征设计

最终特征向量由三部分拼接而成：

```text
word-level TF-IDF 特征
char-level TF-IDF 特征
dense sentence embedding 特征
```

也就是说，每条样本最终的向量可以理解为：

```text
X = [TF-IDF word/char n-gram features, MPNet embedding]
```

这样设计的原因是：

```text
embedding 负责捕捉整体语义
word TF-IDF 负责捕捉关键词和短语
char TF-IDF 负责捕捉拼写错误、数学符号和局部字符串模式
```

### 1. Dense Embedding

当前使用的 embedding 模型是：

```text
sentence-transformers/all-mpnet-base-v2
```

该模型生成的 embedding 维度是：

```text
768
```

生成 embedding 时使用了归一化：

```python
normalize_embeddings=True
```

保存的 embedding 文件：

```text
features/mpnet_train_embeddings.npy
features/mpnet_val_embeddings.npy
features/mpnet_test_embeddings.npy
```

如果后续实验只想使用 embedding，不使用 TF-IDF，可以直接读取这些 `.npy` 文件。

### 2. Word-Level TF-IDF

word-level TF-IDF 配置如下：

```text
analyzer: word
ngram_range: (1, 2)
min_df: 2
max_features: 50000
sublinear_tf: True
```

它可以捕捉词和短语，例如：

```text
not shaded
same denominator
one third
because
```

### 3. Char-Level TF-IDF

char-level TF-IDF 配置如下：

```text
analyzer: char_wb
ngram_range: (3, 5)
min_df: 2
max_features: 50000
sublinear_tf: True
```

它可以捕捉局部字符模式，例如：

```text
1/3
6/9
frac
nin
deno
```

char-level TF-IDF 对这个任务比较有用，因为学生解释中可能存在：

```text
拼写错误
非正式表达
数学符号
分数写法
短文本答案
```

### 4. 最终拼接特征

当前生成的特征维度如下：

```text
TF-IDF 维度: 49465
MPNet embedding 维度: 768
最终拼接维度: 50233
```

最终矩阵形状：

```text
X_train: (29356, 50233)
X_val:   (7340, 50233)
X_test:  (3, 50233)
```

## 标签编码

机器学习模型训练时需要整数标签，因此 4 个类别被编码为：

```text
0 -> True_Correct
1 -> False_Misconception
2 -> False_Neither
3 -> True_Neither
```

该映射保存于：

```text
features/feature_metadata.json
```

## 后续机器学习如何读取特征

示例代码：

```python
import numpy as np
from scipy import sparse

X_train = sparse.load_npz("features/X_train.npz")
X_val = sparse.load_npz("features/X_val.npz")
X_test = sparse.load_npz("features/X_test.npz")

y_train = np.load("features/y_train.npy")
y_val = np.load("features/y_val.npy")
```

这些 `X_*` 文件是 scipy sparse matrix，可以直接输入给大多数 sklearn 模型。

## Logistic Regression 示例

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1,
)

clf.fit(X_train, y_train)
pred = clf.predict(X_val)

print(classification_report(y_val, pred))
print("macro_f1:", f1_score(y_val, pred, average="macro"))
```

这里使用 `class_weight="balanced"` 是因为类别分布并不完全均衡。

推荐评估指标：

```text
accuracy
macro-F1
classification report
confusion matrix
```

其中 `macro-F1` 比 accuracy 更重要，因为它能更好反映每个类别的平均表现，而不是被样本最多的类别主导。

## 生成文件说明

运行 `preprocess.py` 后会生成：

```text
data/processed_train.csv
data/processed_test.csv
```

运行 `build_features.py` 后会生成：

```text
features/train_split.csv
features/val_split.csv
features/test_processed.csv
features/tfidf_train.npz
features/tfidf_val.npz
features/tfidf_test.npz
features/mpnet_train_embeddings.npy
features/mpnet_val_embeddings.npy
features/mpnet_test_embeddings.npy
features/X_train.npz
features/X_val.npz
features/X_test.npz
features/y_train.npy
features/y_val.npy
features/feature_metadata.json
features/tfidf_word_vectorizer.joblib
features/tfidf_char_vectorizer.joblib
```

其中最重要的是：

```text
features/X_train.npz
features/X_val.npz
features/X_test.npz
features/y_train.npy
features/y_val.npy
```

这些就是后续传统机器学习算法的入口。

## Git 和提交说明

不要提交虚拟环境、缓存文件或大型特征矩阵。

当前 `.gitignore` 已忽略：

```text
.venv/
.uv-cache/
.uv-python/
.hf-cache/
features/
__pycache__/
*.pyc
```

推荐提交的文件：

```text
README.md
.gitignore
preprocess.py
build_features.py
```

是否提交 `data/` 下的原始数据文件，需要根据课程要求和数据集许可决定。

`processed_train.csv` 和 `processed_test.csv` 是可复现的中间文件，可以通过下面命令重新生成：

```powershell
.\.venv\Scripts\python.exe preprocess.py
```

因此一般不需要提交这两个 processed CSV，除非课程提交要求包含中间处理结果。

`features/` 目录不建议提交，因为文件较大，并且可以通过下面命令重新生成：

```powershell
.\.venv\Scripts\python.exe build_features.py
```

## 完整复现流程

从零开始复现当前数据处理和特征生成结果：

```powershell
uv venv --python 3.11
.venv\Scripts\activate

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install pandas numpy scipy scikit-learn sentence-transformers tqdm joblib

python preprocess.py
python build_features.py
```

预期最终特征矩阵形状：

```text
X_train: (29356, 50233)
X_val:   (7340, 50233)
X_test:  (3, 50233)
```

