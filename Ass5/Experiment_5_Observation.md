# Experiment 5: Decision Tree and Random Forest Classification

## Aim
To implement and compare a Decision Tree classifier and a Random Forest ensemble on the Wisconsin Diagnostic Breast Cancer dataset using 5-fold cross-validation and hyperparameter tuning.

## Dataset
- Dataset: Wisconsin Diagnostic Breast Cancer (UCI)
- Train Shape: (455, 30), Test Shape: (114, 30)
- Feature Count: 30, Classes: B (negative), M (positive)

## Required Tables

### Decision Tree Performance
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Training Time (s) | Prediction Time (s) |
|---|---|---|---|---|---|---|---|
| Decision Tree | 0.9298 | 0.9048 | 0.9048 | 0.9048 | 0.9246 | 0.003441 | 0.000241 |
| Decision Tree (Tuned) | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9744 | 0.002554 | 0.000264 |

### Decision Tree Hyperparameter Evaluation (5-Fold CV, top 10)
| Criterion | Max Depth | Avg CV Accuracy (%) | Avg CV F1 |
|---|---|---|---|
| entropy | 5 | 93.63 | 0.9121 |
| entropy | 7 | 93.41 | 0.9094 |
| entropy | 10 | 93.41 | 0.9094 |
| entropy | None | 93.41 | 0.9094 |
| gini | 7 | 92.97 | 0.9039 |
| gini | 10 | 92.97 | 0.9039 |
| gini | None | 92.97 | 0.9039 |
| gini | 3 | 92.75 | 0.8990 |
| entropy | 3 | 92.53 | 0.8941 |
| gini | 5 | 92.53 | 0.8971 |

### Random Forest Performance
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Training Time (s) | Prediction Time (s) |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9929 | 0.050279 | 0.002366 |
| Random Forest (Tuned) | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9937 | 0.104470 | 0.004145 |

### Random Forest Hyperparameter Evaluation (5-Fold CV, top rows)
| n_estimators | Max Depth | Max Features | Avg CV Accuracy (%) | Avg CV F1 |
|---|---|---|---|---|
| 200 | None | sqrt | 97.14 | 0.9615 |
| 200 | 5 | sqrt | 97.14 | 0.9608 |
| 50 | 10 | sqrt | 97.14 | 0.9617 |
| 200 | 10 | sqrt | 97.14 | 0.9615 |
| 50 | 10 | log2 | 96.92 | 0.9584 |
| 50 | None | log2 | 96.92 | 0.9584 |
| 50 | None | sqrt | 96.92 | 0.9589 |
| 100 | 5 | sqrt | 96.92 | 0.9580 |
| 100 | 10 | log2 | 96.92 | 0.9585 |
| 100 | 5 | log2 | 96.70 | 0.9550 |

### Hyperparameter Tuning Summary
| Model | Search Method | Best Parameters | Best CV Accuracy | Execution Time (s) |
|---|---|---|---|---|
| Decision Tree | GridSearchCV | {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 10} | 0.9363 | 1.8366 |
| Random Forest | GridSearchCV | {'bootstrap': False, 'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 200} | 0.9714 | 1.2757 |

### Cross Validation Results
| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Average |
|---|---|---|---|---|---|---|
| Decision Tree | 0.9011 | 0.9560 | 0.9121 | 0.9560 | 0.9560 | 0.9363 |
| Random Forest | 0.9560 | 1.0000 | 0.9451 | 0.9670 | 0.9890 | 0.9714 |

### Decision Tree vs Random Forest Comparison (Tuned, Test Set)
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Training Time (s) | Prediction Time (s) |
|---|---|---|---|---|---|---|---|
| Decision Tree (Tuned) | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9744 | 0.002554 | 0.000264 |
| Random Forest (Tuned) | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9937 | 0.104470 | 0.004145 |

## Observation Questions

**How does tree depth affect overfitting in Decision Trees?**
The depth sweep shows training accuracy climbing to 1.0000 as depth grows, while test accuracy peaks at 0.9386 around max_depth=7. Once the tree is left unbounded (max_depth=None) the train-test gap widens to 0.0702, so deeper trees increasingly memorize the training folds instead of generalizing further on this dataset.

**Which hyperparameter had the greatest impact on performance?**
For the Decision Tree, max_depth moved the best CV accuracy by 0.88 percentage points; for the Random Forest, bootstrap moved it by 0.44 percentage points. Both were the largest swings within their respective search spaces.

**How does Random Forest improve generalization?**
The Random Forest's CV train-test gap (0.0286) was narrower than the Decision Tree's gap (0.0467), and its 5-fold CV accuracy averaged 0.9714 against 0.9363 for the single tree — consistent with bagging and random feature selection averaging out the variance of individual trees.

**Did ensemble learning always improve performance? Why or why not?**
On the held-out test set, the Decision Tree matched the Random Forest (both F1 = 0.9500), so ensembling did not add test-set benefit here. Ensembling is not guaranteed to help whenever a single well-tuned tree already captures the decision boundary well, which can happen on a comparatively small, well-separated dataset like this one — though the Random Forest still had a clearly higher ROC-AUC (0.9937 vs 0.9744) and better fold-to-fold stability.

## Learning Outcomes
- Understood how Decision Tree splitting criteria (gini vs entropy) and depth/leaf/split constraints control model complexity and overfitting.
- Observed directly, via a depth-sweep experiment, how unconstrained tree depth drives training accuracy to 1.0 while test accuracy plateaus and then degrades — a concrete illustration of overfitting.
- Learned how Random Forest's bagging and random feature subsampling reduce variance relative to a single Decision Tree, evidenced by a narrower CV train-test gap.
- Practiced multi-parameter GridSearchCV tuning for both a single tree and an ensemble model, and interpreted `cv_results_` to isolate the impact of individual hyperparameters.
- Learned to use feature importance outputs to compare which features a Decision Tree vs a Random Forest rely on most.
- Understood that ensembling does not automatically outperform a single well-tuned model on every metric, and that ROC-AUC and stability across folds can favor an ensemble even when test-set point accuracy ties.
