# Experiment 4: Binary Classification using Linear and Kernel-Based Models

## Aim
To classify emails as spam or ham using Logistic Regression and Support Vector Machine (SVM) classifiers and to analyze the effect of hyperparameter tuning on classification performance, on the Spambase dataset.

## Dataset
- Dataset: Spambase (Kaggle: somesh24/spambase)
- Train Shape: (3680, 57), Test Shape: (921, 57)
- Feature Count: 57

## Required Tables

### Hyperparameter Tuning Results
| Model | Search Method | Best Parameters | Best CV Accuracy |
|---|---|---|---|
| Logistic Regression | GridSearchCV | {'C': 100, 'max_iter': 5000, 'penalty': 'l1', 'solver': 'liblinear'} | 0.9261 |
| SVM | GridSearchCV | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'} | 0.9356 |

### Logistic Regression Performance
| Metric | Value |
|---|---|
| Accuracy | 0.9262 |
| Precision | 0.9202 |
| Recall | 0.8898 |
| F1 Score | 0.9048 |
| Training Time (s) | 0.2729 |

*(Tuned model shown; untuned baseline: Accuracy = 0.9294, Precision = 0.9209, Recall = 0.8981, F1 = 0.9093, Training Time = 0.0226s.)*

### Logistic Penalty Comparison
| Penalty | Best C | Best Solver | Best CV Accuracy |
|---|---|---|---|
| L1 | 100 | liblinear | 0.9261 |
| L2 | 100 | liblinear | 0.9258 |

### SVM Kernel-wise Performance
| Kernel | Accuracy | F1 Score | Training Time (s) |
|---|---|---|---|
| Linear | 0.9294 | 0.9093 | 0.1428 |
| Polynomial | 0.7796 | 0.6220 | 0.1265 |
| RBF | 0.9273 | 0.9055 | 0.0605 |
| Sigmoid | 0.8849 | 0.8528 | 0.0602 |

### K-Fold Cross-Validation Results (K = 5)
| Fold | Logistic Regression | SVM |
|---|---|---|
| Fold 1 | 0.9416 | 0.9443 |
| Fold 2 | 0.9253 | 0.9416 |
| Fold 3 | 0.9348 | 0.9348 |
| Fold 4 | 0.9117 | 0.9307 |
| Fold 5 | 0.9171 | 0.9266 |
| **Average** | **0.9261** | **0.9356** |

### Comparative Analysis (Tuned Logistic Regression vs Tuned SVM)
| Criterion | Logistic Regression | SVM |
|---|---|---|
| Accuracy | 0.9262 | 0.9207 |
| ROC-AUC | 0.9679 | 0.9702 |
| Model Complexity | Low | High |
| Training Time (s) | 0.2729 | 0.0576 |
| Prediction Time (s) | 0.000626 | 0.022344 |
| Interpretability | High | Low |

## Observations
- **Best-performing classifier:** Logistic Regression (Tuned) had the best held-out test performance overall — Accuracy = 0.9262, F1 = 0.9048, ROC-AUC = 0.9679 — narrowly ahead of the tuned SVM (Accuracy = 0.9207, F1 = 0.8976), though SVM's ROC-AUC (0.9702) was marginally higher, meaning SVM ranks positive/negative examples slightly better even though its default-threshold accuracy trails.
- **Impact of regularization:** The best cross-validated Logistic Regression configuration used penalty = L1 with C = 100 (CV accuracy 0.9261). Among tested C values, C=100 gave the best CV accuracy (0.9261) while C=0.01 gave the weakest (0.9090) — confirming that too much regularization (small C) underfits this 57-feature dataset.
- **L1 vs L2:** L1 (0.9261) marginally beat L2 (0.9258) at the same C=100, suggesting a small amount of sparsity helps generalization slightly on Spambase's feature set.
- **Kernel behavior in SVM:** Linear was the strongest baseline kernel (Accuracy = 0.9294, F1 = 0.9093), while Polynomial was by far the weakest (Accuracy = 0.7796, F1 = 0.6220) — Spambase's word-frequency features appear close to linearly separable, so the added polynomial curvature does not help and likely overfits/underfits depending on degree. After tuning, RBF (C=10, gamma=scale) was selected as the best kernel, reaching CV accuracy 0.9356.
- **Bias-variance trade-off:** The tuned Logistic Regression had a train-test CV accuracy gap of 0.0067, while the tuned SVM had a larger gap of 0.0344 — the SVM is fitting the training folds more aggressively, i.e. carries more variance than the logistic model here.
- **Computational trade-offs:** SVM (Tuned) trained fastest (0.0576s) but predicted slowest (0.0223s) due to support-vector-based scoring; Logistic Regression (Tuned) predicted fastest (0.000626s) but took longer to train (0.2729s) during grid search.

## Learning Outcomes
- Understood the sigmoid-based probabilistic formulation of Logistic Regression and the margin-maximization principle behind SVM.
- Learned how L1 vs L2 regularization and the inverse-strength parameter C shape the bias-variance behavior of Logistic Regression.
- Compared four SVM kernels (Linear, Polynomial, RBF, Sigmoid) and connected kernel choice to the underlying geometry of the data (near-linear separability of Spambase).
- Practiced GridSearchCV-based hyperparameter tuning for both a linear model and a kernel-based model.
- Evaluated models with a complete metric suite (Accuracy, Precision, Recall, F1, ROC-AUC) plus 5-fold cross-validation, and used the CV-vs-test gap as a concrete bias-variance diagnostic.
- Recognized the practical trade-off between training cost, prediction cost, and interpretability when choosing between a linear and a kernel-based classifier.
