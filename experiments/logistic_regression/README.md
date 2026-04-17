# Logistic Regression Baseline

本实验是 4 分类任务的第一个基础 baseline。

## 输入特征

本实验使用 50233 维融合特征：

```text
features/X_train.npz
features/X_val.npz
features/y_train.npy
features/y_val.npy
```

该特征由以下部分拼接：

```text
word-level TF-IDF
char-level TF-IDF
768 维 MPNet embedding
```

## 运行方式

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe experiments\logistic_regression\train_logreg.py
```

也可以调整参数：

```powershell
.\.venv\Scripts\python.exe experiments\logistic_regression\train_logreg.py --C 1.0 --max-iter 1000 --solver saga
```

## 输出结果

默认输出到：

```text
experiments/logistic_regression/outputs/
```

主要文件：

```text
metrics.json
classification_report.txt
classification_report.csv
confusion_matrix.csv
confusion_matrix.png
confusion_matrix_normalized.png
per_class_metrics.png
val_predictions.csv
logreg_model.joblib
top_features_by_class.csv
```

## 评价指标

重点关注：

```text
accuracy
macro-F1
weighted-F1
per-class precision / recall / F1
confusion matrix
```

由于类别分布不完全均衡，`macro-F1` 比 accuracy 更适合作为主要比较指标。

## 当前 Baseline 结果

当前使用默认参数运行：

```text
solver: saga
C: 1.0
max_iter: 1000
class_weight: balanced
input: 50233 维 TF-IDF + MPNet 融合特征
```

验证集结果：

```text
accuracy:    0.8219
macro-F1:    0.7996
weighted-F1: 0.8246
```

分类报告：

```text
                     precision    recall  f1-score   support

       True_Correct       0.92      0.85      0.88      3006
False_Misconception       0.84      0.83      0.83      1972
      False_Neither       0.77      0.77      0.77      1309
       True_Neither       0.65      0.80      0.71      1053

           accuracy                           0.82      7340
          macro avg       0.79      0.81      0.80      7340
       weighted avg       0.83      0.82      0.82      7340
```

混淆矩阵：

```text
                      Pred True_Correct  Pred False_Misconception  Pred False_Neither  Pred True_Neither
True_Correct                      2557                         25                  10                414
False_Misconception                 24                       1632                 283                 33
False_Neither                       16                        279                1006                  8
True_Neither                       195                         15                   5                838
```

主要观察：

```text
1. True_Correct 整体表现最好，F1 = 0.88。
2. False_Misconception 表现稳定，F1 = 0.83。
3. True_Neither 是当前最弱类别，precision = 0.65，说明模型较容易把其他类别预测成 True_Neither。
4. False_Misconception 和 False_Neither 之间仍有明显混淆。
```
