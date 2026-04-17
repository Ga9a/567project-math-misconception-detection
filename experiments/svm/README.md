# Support Vector Machine Baseline

本实验是 4 分类任务的 SVM baseline。

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

## 模型选择

本实验默认使用：

```text
LinearSVC
```

不建议在当前特征上使用 RBF kernel SVM，因为当前输入是 29356 条样本、50233 维高维稀疏特征，RBF SVM 训练会非常慢且内存压力较大。

## 运行方式

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe experiments\svm\train_svm.py
```

也可以调整参数：

```powershell
.\.venv\Scripts\python.exe experiments\svm\train_svm.py --C 1.0 --max-iter 5000
```

## 输出结果

默认输出到：

```text
experiments/svm/outputs/
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
linear_svm_model.joblib
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
model: LinearSVC
C: 1.0
max_iter: 5000
class_weight: balanced
input: 50233 维 TF-IDF + MPNet 融合特征
```

验证集结果：

```text
accuracy:    0.8403
macro-F1:    0.8130
weighted-F1: 0.8399
```

分类报告：

```text
                     precision    recall  f1-score   support

       True_Correct       0.90      0.90      0.90      3006
False_Misconception       0.84      0.86      0.85      1972
      False_Neither       0.80      0.76      0.78      1309
       True_Neither       0.72      0.72      0.72      1053

           accuracy                           0.84      7340
          macro avg       0.82      0.81      0.81      7340
       weighted avg       0.84      0.84      0.84      7340
```

混淆矩阵：

```text
                      Pred True_Correct  Pred False_Misconception  Pred False_Neither  Pred True_Neither
True_Correct                      2715                         12                  15                264
False_Misconception                 25                       1700                 221                 26
False_Neither                       12                        296                 995                  6
True_Neither                       272                         16                   7                758
```

主要观察：

```text
1. SVM 当前整体优于 Logistic Regression baseline。
2. True_Correct 和 False_Misconception 表现最好，F1 分别为 0.90 和 0.85。
3. False_Misconception 与 False_Neither 仍有明显混淆。
4. True_Neither 的 F1 约为 0.72，仍是较难分类的类别之一。
```
