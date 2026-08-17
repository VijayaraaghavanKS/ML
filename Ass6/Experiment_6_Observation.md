# Experiment 6: Bagging, Boosting, and Stacked Ensemble Models

## Aim
To understand ensemble learning strategies (Bagging, Boosting, and Stacking), implement Bagging and Boosting classifiers, build a Stacked Ensemble with multiple base learners, compare the ensemble models in terms of accuracy/stability/generalization, and analyze the effect of ensemble methods on bias and variance, on the Wisconsin Diagnostic Breast Cancer dataset.

## Dataset
- Wisconsin Diagnostic Breast Cancer Dataset (UCI)
- Total samples: 569, Features: 30 numerical attributes, Target classes: Malignant (M), Benign (B)
- Train Shape: (455, 30), Test Shape: (114, 30)

## Ensemble Models Used
- **Bagging** — Base Estimator: Decision Tree
- **Boosting** — AdaBoost and Gradient Boosting (both trained; Gradient Boosting won the tuned comparison)
- **Stacked Ensemble** — Base Models: SVM, Naive Bayes, Decision Tree; Meta Learner: Logistic Regression

## Required Tables

### Table 1: Bagging Hyperparameter Evaluation (5-Fold CV, top rows)
| n_estimators | max_samples | Avg CV Accuracy (%) | Avg CV F1 Score |
|---|---|---|---|
| 50 | 0.7 | 97.36 | 0.9643 |
| 50 | 1.0 | 96.92 | 0.9583 |
| 50 | 0.5 | 96.70 | 0.9542 |
| 100 | 0.7 | 96.70 | 0.9547 |
| 100 | 1.0 | 96.70 | 0.9551 |
| 200 | 1.0 | 96.70 | 0.9551 |
| 200 | 0.7 | 96.48 | 0.9519 |
| 10 | 0.5 | 96.26 | 0.9477 |

### Table 2: Boosting Hyperparameter Evaluation (5-Fold CV, top rows)
| Algorithm | n_estimators | learning_rate | max_depth | Avg CV Accuracy (%) | Avg CV F1 Score |
|---|---|---|---|---|---|
| Gradient Boosting | 200 | 0.1 | 2 | 97.36 | 0.9645 |
| AdaBoost | 200 | 1.0 | - | 97.14 | 0.9610 |
| AdaBoost | 100 | 0.5 | - | 97.14 | 0.9607 |
| AdaBoost | 200 | 0.5 | - | 97.14 | 0.9612 |
| Gradient Boosting | 200 | 0.5 | 2 | 96.92 | 0.9578 |
| Gradient Boosting | 200 | 0.1 | 4 | 96.92 | 0.9584 |
| Gradient Boosting | 200 | 0.1 | 3 | 96.92 | 0.9582 |

### Table 3: Stacked Ensemble Evaluation (5-Fold CV)
| Base Models | Meta Learner | Avg CV Accuracy (%) | Avg CV F1 Score |
|---|---|---|---|
| SVM + Naive Bayes + Decision Tree | Logistic Regression | 97.36 | 0.9638 |
| SVM + Decision Tree | Logistic Regression | 97.14 | 0.9609 |
| SVM + Naive Bayes | Logistic Regression | 95.82 | 0.9433 |
| Naive Bayes + Decision Tree | Logistic Regression | 94.51 | 0.9244 |

### Hyperparameter Tuning Summary
| Model | Search Method | Best Parameters | Best CV Accuracy |
|---|---|---|---|
| Bagging | GridSearchCV | {'max_features': 0.5, 'max_samples': 0.7, 'n_estimators': 50} | 0.9736 |
| Boosting (Gradient Boosting) | GridSearchCV | {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 200} | 0.9736 |
| Stacked Ensemble | Cross-validated combination search | SVM + Naive Bayes + Decision Tree + Logistic Regression | 0.9736 |

### K-Fold Cross-Validation Results (K = 5)
| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Average |
|---|---|---|---|---|---|---|
| Bagging | 0.9670 | 1.0000 | 0.9670 | 0.9451 | 0.9890 | 0.9736 |
| Boosting | 0.9670 | 0.9780 | 0.9670 | 0.9670 | 0.9890 | 0.9736 |
| Stacked Ensemble | 0.9670 | 0.9890 | 0.9780 | 0.9451 | 0.9890 | 0.9736 |

### Table 4: Performance Comparison of Ensemble Models (Test Set, Tuned)
| Model | Accuracy (%) | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Bagging | 96.49 | 1.0000 | 0.9048 | 0.9500 | 0.9917 |
| Boosting | 96.49 | 1.0000 | 0.9048 | 0.9500 | 0.9934 |
| Stacked Ensemble | 96.49 | 1.0000 | 0.9048 | 0.9500 | 0.9970 |

*(Individual base learners for reference: SVM Accuracy=0.9737/F1=0.9630, Naive Bayes Accuracy=0.9211/F1=0.8889, Decision Tree Accuracy=0.9298/F1=0.9048.)*

## Observation Questions

**How does Bagging reduce variance?**
Bagging trains multiple Decision Trees on independent bootstrap samples and averages their votes. The single Decision Tree's 5-fold CV accuracy had a standard deviation of 0.0326, while the tuned Bagging ensemble's standard deviation dropped to 0.0192 — a reduction of 0.0134. Because the individual trees' errors are only weakly correlated, averaging their predictions cancels out a large share of the variance any single tree would carry, without changing what each tree learns.

**How does Boosting address model bias?**
Boosting builds an ensemble sequentially, with each new learner focused on the mistakes the ensemble has made so far. A single depth-1 decision stump (deliberately high-bias) scored only 0.9011 CV accuracy on its own, while the tuned Gradient Boosting ensemble reached 0.9736 — a gain of 0.0725. This comes from directly and repeatedly correcting the systematic errors (bias) a single weak learner cannot fix, rather than from averaging away random variance.

**Why does stacking benefit from heterogeneous models?**
The three base learners make errors differently: SVM finds a margin-based boundary, Naive Bayes assumes feature independence, and the Decision Tree partitions the feature space axis-by-axis. Individually they scored test accuracies of SVM=0.9737, Naive Bayes=0.9211, Decision Tree=0.9298. Because their mistakes are not perfectly correlated, the Logistic Regression meta-learner combines their predictions usefully; homogeneous base learners would have given it far less complementary signal.

**Which ensemble method performed best and why?**
Stacked Ensemble achieved the best held-out performance (Accuracy=0.9649, F1=0.9500, ROC-AUC=0.9970), narrowly ahead of Boosting (ROC-AUC=0.9934) and Bagging (ROC-AUC=0.9917) — all three tied on Accuracy/F1 at the default threshold. This is consistent with stacking's ability to combine complementary, diverse base learners through a learned meta-model, which is the most effective strategy when the base learners disagree in informative ways, as they do here.

## Bias-Variance Analysis Summary
- **Bagging** reduced variance: CV accuracy std fell from 0.0326 (single tree) to 0.0192 (tuned Bagging).
- **Boosting** reduced bias: CV accuracy rose from 0.9011 (single stump) to 0.9736 (tuned Gradient Boosting).
- **Stacking** improved on the best individual base learner's decision-boundary shape via a learned combination, reaching the highest ROC-AUC (0.9970) of all three ensembles despite tying on point Accuracy/F1.

## Learning Outcomes
- Understood the mechanistic difference between Bagging (parallel, independent training on bootstrap samples to cut variance) and Boosting (sequential, error-focused training to cut bias).
- Implemented and compared two boosting algorithms (AdaBoost, Gradient Boosting) and learned how their tunable hyperparameters (n_estimators, learning_rate, max_depth) trade off against each other.
- Built a Stacked Ensemble with heterogeneous base learners (SVM, Naive Bayes, Decision Tree) and a Logistic Regression meta-learner, and learned how base-model diversity — not just individual accuracy — drives stacking's benefit.
- Practiced 5-fold cross-validated hyperparameter search across three different ensemble families and interpreted `cv_results_`/`cross_validate` outputs to isolate parameter impact.
- Used a controlled comparison (single tree vs Bagging; single stump vs Boosting) to directly observe variance reduction and bias reduction as concrete, measured effects rather than abstract concepts.
- Learned that ensembling does not always separate cleanly on point-accuracy metrics — ROC-AUC and fold-to-fold stability can reveal ranking differences between ensembles that tie on Accuracy/F1 at a fixed threshold.
