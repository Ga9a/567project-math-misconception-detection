# Project Midterm Check-in Slides

---

## Slide 1 - Recap & Scope Change

### Content

- Goal: classify student math responses into a simplified 4-class misconception taxonomy.
- Scope changed from the original 6 `Category` labels to `Category_4` so evaluation is more stable for a midterm model comparison.
- Final labels: `True_Correct`, `False_Misconception`, `False_Neither`, and `True_Neither`.
- Data split: 80% train / 20% validation, stratified by `Category_4`; validation size is 7,340 examples.
- Class balance still matters: `True_Correct` is largest, while `True_Neither` is smallest and harder to separate.

### Image

Path: `experiments/logistic_regression/outputs/per_class_metrics.png`

Description:
Per-class baseline metrics for Logistic Regression, useful as the starting point for the project scope and class difficulty discussion.

### Speaker Notes

We reframed the task into four labels because the original label space was too fine-grained for the amount of signal we can reliably model in a midterm cycle. The new taxonomy keeps the important educational distinction: correct reasoning, misconception, neither, and whether the answer itself is true or false. The class distribution is not fully balanced, so macro F1 is the main metric because it gives the smaller classes enough weight.

---

## Slide 2 - Algorithms & Baselines

### Content

- Baseline models: Logistic Regression and Linear SVM on 50,233-dimensional TF-IDF + MPNet features.
- Dense-vector models: XGBoost and Simple MLP on 768-dimensional MPNet embeddings.
- Transformer model: BERT fine-tuned directly on raw text with question, selected answer, and student explanation.
- Why this mix: linear models give fast, interpretable baselines; MPNet models test semantic embeddings; BERT tests end-to-end contextual learning.
- Current baseline comparison: Logistic Regression macro F1 0.7996; Linear SVM macro F1 0.8130.

### Image

Path: `experiments/svm/outputs/per_class_metrics.png`

Description:
Per-class precision, recall, and F1 for the Linear SVM baseline, which is the strongest non-tuned linear baseline in the repository outputs.

### Speaker Notes

The model set is intentionally staged from simple to richer representations. Logistic Regression and SVM are strong sanity checks because they train quickly and expose feature-level behavior. XGBoost and MLP isolate how much MPNet embeddings help without full transformer fine-tuning. BERT is the most expensive route, but it can jointly model the question, selected answer, and explanation instead of relying on frozen embeddings.

---

## Slide 3 - Results & Analysis

### Content

- Best result in `reports/tuning_summary.json`: BERT epoch 6, learning rate 2e-5, batch size 32, max length 192.
- Best tuned BERT validation macro F1: 0.8495; validation accuracy: 0.8740.
- Best tuned XGBoost macro F1: 0.7990; best tuned MLP macro F1: 0.7895.
- SVM remains a strong fast baseline at macro F1 0.8130, outperforming tuned XGBoost and MLP on current validation metrics.
- BERT improves most because it can preserve word order and local reasoning context in the student explanation.

### Image

Path: `experiments/bert/outputs/confusion_matrix_normalized.png`

Description:
Normalized BERT confusion matrix showing where the strongest current model separates the four labels and where remaining confusions cluster.

### Speaker Notes

The main result is that contextual fine-tuning helps beyond frozen embeddings. The tuning log shows BERT improving through epoch 6, then slightly declining at epochs 7 and 8, so the current best point is not simply "train longer." Linear SVM is still important because it is cheap, robust, and competitive. The BERT image should be presented together with the tuning summary number, since the current canonical BERT output file reports a slightly lower macro F1 than the sweep summary.

---

## Slide 4 - Challenges

### Content

- The hardest labels are `True_Neither` and the boundary between `False_Misconception` and `False_Neither`.
- Some student explanations are short or ambiguous, so the model must infer intent from limited evidence.
- The class distribution pushes models toward the larger `True_Correct` class unless macro F1 and class-balanced training are emphasized.
- Frozen MPNet embeddings do not fully capture the interaction between question text, answer choice, and explanation.
- Output alignment needs cleanup: tuning summary and canonical BERT metrics currently differ slightly and should be reconciled before final reporting.

### Image

Path: `experiments/xgboost/outputs/confusion_matrix_normalized.png`

Description:
Normalized XGBoost confusion matrix showing class confusions when using frozen MPNet embeddings instead of end-to-end transformer fine-tuning.

### Speaker Notes

The remaining errors are not just model capacity issues. Some categories are semantically close: a wrong explanation without a clear misconception can look similar to a misconception, and `True_Neither` is relatively underrepresented. The XGBoost confusion matrix is useful because it shows what happens when we rely on dense semantic embeddings without letting the classifier adapt token-level interactions to this task.

---

## Slide 5 - Next Steps

### Content

- Reconcile `reports/tuning_summary.json` with `experiments/bert/outputs/metrics.json` so the final report has one authoritative metric source.
- Add error analysis from `experiments/bert/outputs/val_predictions.csv`, especially high-confidence mistakes.
- Create a model comparison chart from `reports/tuning_results.csv` for macro F1, accuracy, and training time.
- Try targeted improvements: class-weighted loss, focal loss, or oversampling for weaker classes.
- Prepare final deliverables: reproducible commands, cleaned result tables, and qualitative examples of misconception errors.

### Image

Path: `experiments/bert/outputs/per_class_metrics.png`

Description:
BERT per-class metrics chart to guide the next iteration toward the weakest classes rather than only improving overall accuracy.

### Speaker Notes

The next phase should focus less on adding many new models and more on making the strongest path reliable and explainable. The first cleanup item is metric alignment, because the team needs one number for the report. After that, qualitative error analysis will tell us whether the model is missing mathematical concepts, struggling with ambiguous student language, or mostly confusing nearby labels.

---

## Assets to Prepare

- Pipeline diagram showing preprocessing, feature generation, model training, and validation outputs.
- Model comparison bar chart from `reports/tuning_results.csv`.
- Error-analysis table with representative rows from `experiments/bert/outputs/val_predictions.csv`.
- Optional slide-ready class distribution chart from `data/processed_train.csv`.

---

## Collaboration Notes

- Slide 1 needs a better pipeline or class-distribution visual; the current real repo image is a useful baseline chart but not a perfect scope diagram.
- Slide 3 should be checked by the teammate responsible for experiments because the tuning summary and canonical BERT metrics differ slightly.
- Slide 4 can be strengthened by adding two or three concrete validation examples from `val_predictions.csv`.
- Slide 5 should be updated after metric reconciliation so the final next-step list matches the team plan.
