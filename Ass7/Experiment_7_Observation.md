# Experiment 7: Dimensionality Reduction and Model Evaluation (With and Without PCA)

## Aim
To study the effect of Principal Component Analysis (PCA) on classifier performance by training and hyperparameter-tuning ten models (SVM, Naive Bayes, KNN, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, Stacking) using 5-fold cross-validation, both on the original feature space and on a PCA-reduced feature space, on the Wisconsin Diagnostic Breast Cancer dataset.

## Dataset
- Wisconsin Diagnostic Breast Cancer Dataset (UCI), same dataset used in Experiments 5 and 6
- 569 samples, 30 numerical features, target: Malignant (M) / Benign (B)
- Train Shape: (455, 30), Test Shape: (114, 30)

## Table 1: PCA Variance Explained
| Setting | Chosen Components / Variance Target | Explained Variance (%) | Justification |
|---|---|---|---|
| With-PCA | 10 components (target 95%) | 95.21 | Smallest number of components whose cumulative explained variance reaches the 95% target, chosen from the scree plot to compress 30 original features while retaining most information. |

## Table 2: Hyperparameter Tuning Results (Best Parameters, No-PCA vs With-PCA)
| Model | Kernel / Params Tried | Best Parameters | Performance (No-PCA) | Performance (With-PCA) |
|---|---|---|---|---|
| SVM | kernel: linear/rbf, C: 0.1/1/10, gamma: scale/auto | C=10, gamma=scale, kernel=rbf | 0.9737 | 0.9649 |
| Naive Bayes | var_smoothing: 1e-9 to 1e-6 | var_smoothing=1e-09 | 0.9211 | 0.8947 |
| KNN | k: 3-11, weights: uniform/distance, metric: euclidean/manhattan | k=3, weights=uniform | 0.9649 | 0.9474 |
| Logistic Regression | C: 0.01-10, penalty: l2 | C=1 | 0.9649 | 0.9737 |
| Decision Tree | max_depth: 3/5/7/None, min_samples_split: 2/5/10 | max_depth=7 (No-PCA), 5 (With-PCA) | 0.9386 | 0.9386 |
| Random Forest | n_estimators: 100/200, max_depth: 5/10/None | n_estimators=100, max_depth=5 | 0.9737 | 0.9474 |
| AdaBoost | n_estimators: 50/100/200, learning_rate: 0.1/0.5/1.0 | n_estimators=100, learning_rate=0.5 | 0.9737 | 0.9474 |
| Gradient Boosting | n_estimators: 100/200, learning_rate: 0.05/0.1, max_depth: 2/3 | n_estimators=200, learning_rate=0.1, max_depth=2 | 0.9649 | 0.9561 |
| XGBoost | n_estimators: 100/200, learning_rate: 0.05/0.1, max_depth: 2/3 | n_estimators=200, learning_rate=0.05-0.1 | 0.9737 | 0.9737 |
| Stacking | base: SVM + Random Forest + Gradient Boosting, meta: Logistic Regression | final_estimator C=10 | 0.9737 | 0.9825 |

## Table 3: 5-Fold Cross-Validation Results (No-PCA vs With-PCA)
| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg (No-PCA) | Avg (With-PCA) |
|---|---|---|---|---|---|---|---|
| SVM | 0.967 | 0.989 | 0.967 | 0.978 | 0.978 | 0.9758 | 0.9714 |
| Naive Bayes | 0.956 | 0.978 | 0.9121 | 0.9121 | 0.9341 | 0.9385 | 0.9121 |
| KNN | 0.967 | 1.000 | 0.967 | 0.956 | 0.989 | 0.9758 | 0.9714 |
| Logistic Regression | 0.956 | 0.967 | 1.000 | 0.978 | 0.967 | 0.9736 | 0.9758 |
| Decision Tree | 0.8901 | 0.967 | 0.8901 | 0.9121 | 0.956 | 0.9231 | 0.9385 |
| Random Forest | 0.967 | 1.000 | 0.9451 | 0.9451 | 0.978 | 0.9670 | 0.9582 |
| AdaBoost | 0.956 | 0.967 | 0.967 | 0.978 | 0.989 | 0.9714 | 0.9714 |
| Gradient Boosting | 0.967 | 0.978 | 0.967 | 0.967 | 0.989 | 0.9736 | 0.9582 |
| XGBoost | 0.967 | 0.978 | 0.967 | 0.967 | 0.989 | 0.9736 | 0.9736 |
| Stacking | 0.956 | 0.989 | 0.978 | 0.978 | 0.989 | 0.9780 | 0.9714 |

## Observation Questions

**Which models improved most with PCA? Which did not? Why?**
Logistic Regression and Stacking improved the most with PCA (accuracy delta +0.0088 each), while Naive Bayes dropped the most (-0.0264). Linear/margin-based models (Logistic Regression, SVM) tend to benefit from PCA removing correlated, redundant directions, while tree-based ensembles (XGBoost, Gradient Boosting, Random Forest, AdaBoost) already perform implicit feature selection at each split and lose useful raw-feature signal when compressed into PCA components.

**Did PCA reduce variance across folds (more stable results)?**
Slightly, on average, but not uniformly. Only 2 of the 10 models showed a positive test-accuracy delta under PCA, though the average fold-to-fold CV standard deviation dropped a little overall.

**For high-dimensional data, was PCA beneficial in reducing overfitting?**
Only modestly. With 30 features on 455 training samples, the data is not severely high-dimensional, so overfitting was already limited before PCA. The clearest reduction showed up in the Decision Tree, whose CV fold standard deviation dropped from 0.0326 to 0.0164 under PCA.

**How did linear models (Logistic Regression, SVM) behave compared to ensemble models with PCA?**
Linear/margin-based models moved by close to 0.0000 average accuracy delta, versus -0.0153 for the boosted/bagged ensembles, consistent with linear models being more sensitive to the geometry PCA reshapes while tree ensembles are largely indifferent to (or slightly hurt by) it.

**Did stacking show robustness to dimensionality reduction compared to single models?**
Yes. Stacking's accuracy delta under PCA was +0.0088, better than the average absolute delta of its individual base models, and it achieved the best overall test accuracy (0.9825) under PCA, since its Logistic Regression meta-learner can compensate for a base learner's degraded feature space.

## Conclusion
The best No-PCA model was SVM (test accuracy 0.9737). The best With-PCA model was Stacking (test accuracy 0.9825). PCA helped distance- and margin-based models (Logistic Regression, Stacking) more than tree-based ensembles, which already perform their own feature selection and mostly lost a little accuracy under PCA. PCA is worth applying ahead of linear/margin/distance models on this dataset, but it is not necessary for tree-based ensembles.

## Learning Outcomes
- Learned to pick the number of PCA components using a cumulative explained-variance target instead of an arbitrary fixed number.
- Saw that PCA's benefit is model-dependent: it helps linear/distance-based models more than tree-based ensembles.
- Practiced comparing No-PCA vs With-PCA settings using both cross-validation and test-set accuracy, not just one number.
- Learned that a stacked ensemble's meta-learner can offset a base learner's accuracy loss from dimensionality reduction.
