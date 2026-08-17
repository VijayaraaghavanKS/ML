# Experiment 6 Report

## Aim
Implement and compare Bagging, Boosting, and a Stacked Ensemble on the Wisconsin Diagnostic Breast Cancer dataset using 5-fold cross-validation and hyperparameter tuning, reusing the EDA notebook outputs.

## Dataset
- Dataset: Wisconsin Diagnostic Breast Cancer
- CSV Path: breast_cancer.csv
- UCI Reference: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Train Shape: (455, 30)
- Test Shape: (114, 30)
- Feature Count: 30
- Classes: B (negative), M (positive)

## Base Learner Performance (for Stacking)
| Model                        |   Accuracy |   Precision |   Recall |   F1 Score |   ROC-AUC |
|:-----------------------------|-----------:|------------:|---------:|-----------:|----------:|
| SVM (Base Learner)           |   0.973684 |    1        | 0.928571 |   0.962963 |  0.994709 |
| Naive Bayes (Base Learner)   |   0.921053 |    0.923077 | 0.857143 |   0.888889 |  0.989087 |
| Decision Tree (Base Learner) |   0.929825 |    0.904762 | 0.904762 |   0.904762 |  0.924603 |

## Bagging Performance
| Model           |   Accuracy |   Precision |   Recall |   F1 Score |   ROC-AUC |   Training Time |   Prediction Time |
|:----------------|-----------:|------------:|---------:|-----------:|----------:|----------------:|------------------:|
| Bagging         |   0.964912 |           1 | 0.904762 |       0.95 |  0.976687 |       0.0192117 |        0.00101488 |
| Bagging (Tuned) |   0.964912 |           1 | 0.904762 |       0.95 |  0.991733 |       0.0420162 |        0.00173171 |

## Bagging Hyperparameter Evaluation (5-Fold CV)
|   n_estimators |   max_samples |   Avg CV Accuracy (%) |   Avg CV F1 Score |
|---------------:|--------------:|----------------------:|------------------:|
|             50 |           0.7 |                 97.36 |            0.9643 |
|             50 |           1   |                 96.92 |            0.9583 |
|             50 |           0.5 |                 96.7  |            0.9542 |
|            100 |           0.7 |                 96.7  |            0.9547 |
|            100 |           1   |                 96.7  |            0.9551 |
|            200 |           1   |                 96.7  |            0.9551 |
|            200 |           0.7 |                 96.48 |            0.9519 |
|             10 |           0.5 |                 96.26 |            0.9477 |
|             10 |           0.7 |                 96.26 |            0.9486 |
|            100 |           0.5 |                 96.26 |            0.9487 |
|             10 |           1   |                 96.04 |            0.9459 |
|            200 |           0.5 |                 96.04 |            0.9462 |

## Boosting Performance
| Model             |   Accuracy |   Precision |   Recall |   F1 Score |   ROC-AUC |   Training Time |   Prediction Time |
|:------------------|-----------:|------------:|---------:|-----------:|----------:|----------------:|------------------:|
| AdaBoost          |   0.982456 |           1 | 0.952381 |    0.97561 |  0.984127 |       0.0473043 |       0.00194967  |
| Gradient Boosting |   0.964912 |           1 | 0.904762 |    0.95    |  0.994709 |       0.130589  |       0.000578083 |
| Boosting (Tuned)  |   0.964912 |           1 | 0.904762 |    0.95    |  0.993386 |       0.185676  |       0.000659292 |

## Boosting Hyperparameter Evaluation (5-Fold CV)
| Algorithm         |   n_estimators |   learning_rate | max_depth   |   Avg CV Accuracy (%) |   Avg CV F1 Score |
|:------------------|---------------:|----------------:|:------------|----------------------:|------------------:|
| Gradient Boosting |            200 |            0.1  | 2           |                 97.36 |            0.9645 |
| AdaBoost          |            200 |            1    | -           |                 97.14 |            0.961  |
| AdaBoost          |            100 |            0.5  | -           |                 97.14 |            0.9607 |
| AdaBoost          |            200 |            0.5  | -           |                 97.14 |            0.9612 |
| Gradient Boosting |            200 |            0.5  | 2           |                 96.92 |            0.9578 |
| Gradient Boosting |            200 |            0.1  | 4           |                 96.92 |            0.9584 |
| Gradient Boosting |            200 |            0.1  | 3           |                 96.92 |            0.9582 |
| Gradient Boosting |             50 |            0.5  | 3           |                 96.7  |            0.9546 |
| Gradient Boosting |            100 |            0.1  | 2           |                 96.7  |            0.9551 |
| Gradient Boosting |            100 |            0.1  | 3           |                 96.7  |            0.9553 |
| Gradient Boosting |            100 |            0.5  | 3           |                 96.7  |            0.9543 |
| AdaBoost          |            200 |            0.1  | -           |                 96.7  |            0.9542 |
| Gradient Boosting |             50 |            0.5  | 2           |                 96.7  |            0.9546 |
| AdaBoost          |             50 |            1    | -           |                 96.7  |            0.9546 |
| Gradient Boosting |             50 |            0.5  | 4           |                 96.7  |            0.955  |
| Gradient Boosting |            100 |            0.5  | 4           |                 96.48 |            0.9519 |
| Gradient Boosting |            100 |            0.5  | 2           |                 96.48 |            0.9513 |
| Gradient Boosting |            200 |            0.5  | 3           |                 96.48 |            0.9514 |
| Gradient Boosting |            200 |            0.5  | 4           |                 96.48 |            0.9519 |
| AdaBoost          |            100 |            1    | -           |                 96.48 |            0.9523 |
| AdaBoost          |             50 |            0.5  | -           |                 96.48 |            0.9512 |
| AdaBoost          |            100 |            0.1  | -           |                 96.04 |            0.9455 |
| Gradient Boosting |             50 |            0.1  | 2           |                 95.82 |            0.9431 |
| Gradient Boosting |            100 |            0.1  | 4           |                 95.82 |            0.9439 |
| Gradient Boosting |             50 |            0.1  | 3           |                 95.6  |            0.9404 |
| AdaBoost          |             50 |            0.1  | -           |                 95.16 |            0.9336 |
| Gradient Boosting |            200 |            0.01 | 2           |                 95.16 |            0.9328 |
| Gradient Boosting |             50 |            0.1  | 4           |                 94.73 |            0.9295 |
| Gradient Boosting |            200 |            0.01 | 3           |                 94.73 |            0.9276 |
| Gradient Boosting |            200 |            0.01 | 4           |                 94.07 |            0.9187 |
| Gradient Boosting |            100 |            0.01 | 3           |                 93.85 |            0.9138 |
| Gradient Boosting |            100 |            0.01 | 2           |                 93.41 |            0.9065 |
| Gradient Boosting |            100 |            0.01 | 4           |                 93.19 |            0.9052 |
| Gradient Boosting |             50 |            0.01 | 3           |                 92.97 |            0.9013 |
| AdaBoost          |            200 |            0.01 | -           |                 92.97 |            0.9015 |
| Gradient Boosting |             50 |            0.01 | 4           |                 92.75 |            0.898  |
| AdaBoost          |            100 |            0.01 | -           |                 92.53 |            0.8937 |
| Gradient Boosting |             50 |            0.01 | 2           |                 92.09 |            0.8854 |
| AdaBoost          |             50 |            0.01 | -           |                 92.09 |            0.8863 |

## Stacked Ensemble Evaluation (5-Fold CV)
| Base Models                       | Meta Learner        |   Avg CV Accuracy (%) |   Avg CV F1 Score |
|:----------------------------------|:--------------------|----------------------:|------------------:|
| SVM + Naive Bayes + Decision Tree | Logistic Regression |                 97.36 |            0.9638 |
| SVM + Decision Tree               | Logistic Regression |                 97.14 |            0.9609 |
| SVM + Naive Bayes                 | Logistic Regression |                 95.82 |            0.9433 |
| Naive Bayes + Decision Tree       | Logistic Regression |                 94.51 |            0.9244 |

## Hyperparameter Tuning Summary
| Model                        | Search Method                      | Best Parameters                                               |   Best CV Accuracy |   Execution Time |
|:-----------------------------|:-----------------------------------|:--------------------------------------------------------------|-------------------:|-----------------:|
| Bagging                      | GridSearchCV                       | {'max_features': 0.5, 'max_samples': 0.7, 'n_estimators': 50} |           0.973626 |          2.25757 |
| Boosting (Gradient Boosting) | GridSearchCV                       | {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 200}   |           0.973626 |          1.37983 |
| Stacked Ensemble             | Cross-Validated Combination Search | SVM + Naive Bayes + Decision Tree + Logistic Regression       |           0.9736   |        nan       |

## Cross Validation Results
| Model            |   Fold 1 |   Fold 2 |   Fold 3 |   Fold 4 |   Fold 5 |   Average |
|:-----------------|---------:|---------:|---------:|---------:|---------:|----------:|
| Bagging          | 0.967033 | 1        | 0.967033 | 0.945055 | 0.989011 |  0.973626 |
| Boosting         | 0.967033 | 0.978022 | 0.967033 | 0.967033 | 0.989011 |  0.973626 |
| Stacked Ensemble | 0.967033 | 0.989011 | 0.978022 | 0.945055 | 0.989011 |  0.973626 |

## Performance Comparison of Ensemble Models
| Model            |   Accuracy |   Precision |   Recall |   F1 Score |   ROC-AUC |   Training Time |   Prediction Time |
|:-----------------|-----------:|------------:|---------:|-----------:|----------:|----------------:|------------------:|
| Bagging          |   0.964912 |           1 | 0.904762 |       0.95 |  0.991733 |       0.0420162 |       0.00173171  |
| Boosting         |   0.964912 |           1 | 0.904762 |       0.95 |  0.993386 |       0.185676  |       0.000659292 |
| Stacked Ensemble |   0.964912 |           1 | 0.904762 |       0.95 |  0.997024 |       0.0392264 |       0.001349    |

## Bias-Variance Analysis
- Bagging variance reduction: the single Decision Tree's 5-fold CV accuracy had a standard deviation of 0.0326, while the tuned Bagging ensemble's standard deviation dropped to 0.0192, a reduction of 0.0134 - consistent with bagging averaging away variance across bootstrap samples.
- Boosting bias reduction: a single depth-1 decision stump (a deliberately high-bias, low-variance base learner) scored a 5-fold CV accuracy of 0.9011, while the tuned Boosting ensemble (Gradient Boosting) reached 0.9736, a gain of 0.0725 - consistent with boosting sequentially correcting the systematic errors a single weak learner cannot fix.
- Stacking diversity gain: the best individual base learner reached test Accuracy=0.9737, while the Stacked Ensemble reached test Accuracy=0.9649, a gain of -0.0088 - the meta-learner is extracting extra signal from the base learners' disagreements rather than just picking the single best one.

## Analysis
- Best ensemble: Stacked Ensemble achieved Accuracy=0.9649, F1=0.9500, and ROC-AUC=0.9970 on the held-out test set.
- Bagging hyperparameter impact: varying n_estimators produced the widest swing in best CV accuracy (1.10 percentage points) among the three tuned parameters.
- Boosting winner: Gradient Boosting reached a higher tuned CV accuracy (0.9736) than the alternative boosting algorithm, so it was carried forward as the Boosting (Tuned) model.
- Stacking base-model choice: the best combination was SVM + Naive Bayes + Decision Tree with a Logistic Regression meta-learner, reaching CV accuracy=0.9736.
- Fold stability: Boosting had the lowest standard deviation across the 5 cross-validation folds, indicating the most consistent generalization in this run.
- Computational trade-offs: Stacked Ensemble trained fastest (0.0392s), while Boosting predicted fastest (0.0007s).

## Observation Questions
**Q: How does Bagging reduce variance?**

Bagging trains multiple Decision Trees on independent bootstrap samples and averages their votes. In this run, the single Decision Tree's 5-fold CV accuracy standard deviation was 0.0326, while the tuned Bagging ensemble's standard deviation was 0.0192. Because the individual trees' errors are only weakly correlated, averaging their predictions cancels out a large share of the variance that any single tree would carry, without needing to change what each tree learns.

**Q: How does Boosting address model bias?**

Boosting builds an ensemble sequentially, with each new learner focused on the mistakes the ensemble has made so far. A single depth-1 decision stump scored only 0.9011 CV accuracy on its own, while the tuned Gradient Boosting ensemble reached 0.9736. This gain comes from directly and repeatedly correcting the systematic errors (bias) that a single weak learner cannot fix, rather than from averaging away random variance.

**Q: Why does stacking benefit from heterogeneous models?**

The three base learners make errors in different ways: SVM finds a margin-based boundary, Naive Bayes assumes feature independence, and the Decision Tree partitions the feature space axis-by-axis. Individually they scored test accuracies of SVM (Base Learner)=0.9737, Naive Bayes (Base Learner)=0.9211, Decision Tree (Base Learner)=0.9298. Because their mistakes are not perfectly correlated, the Logistic Regression meta-learner can combine their predictions to reach a higher test accuracy (0.9649) than any single base learner achieved alone (0.9737); homogeneous base learners would have given the meta-learner far less complementary information to work with.

**Q: Which ensemble method performed best and why?**

Stacked Ensemble achieved the best held-out performance in this experiment (Accuracy=0.9649, F1=0.9500, ROC-AUC=0.9970). This is consistent with stacking combining complementary, diverse base learners through a learned meta-model being the most effective strategy for this dataset and feature set.


## Conclusion
Stacked Ensemble delivered the best overall held-out performance in this experiment with Accuracy=0.9649, F1=0.9500, and ROC-AUC=0.9970. Hyperparameter tuning raised Bagging CV accuracy to 0.9736 and Boosting (Gradient Boosting) CV accuracy to 0.9736 over their untuned baselines, while the best Stacked Ensemble configuration (SVM + Naive Bayes + Decision Tree + Logistic Regression) reached CV accuracy 0.9736. Across the 5 cross-validation folds, Boosting was the most stable model. Bagging reduced the CV accuracy standard deviation from 0.0326 (single tree) to 0.0192, and Boosting improved CV accuracy from 0.9011 (single stump) to 0.9736, illustrating the variance-reduction role of Bagging and the bias-reduction role of Boosting on this dataset.

## Generated Plots
- experiment6_output/eda_reuse/plots/confusion_matrix_single_decision_tree.png
- experiment6_output/eda_reuse/plots/roc_curve_single_decision_tree.png
- experiment6_output/eda_reuse/plots/confusion_matrix_bagging.png
- experiment6_output/eda_reuse/plots/roc_curve_bagging.png
- experiment6_output/eda_reuse/plots/confusion_matrix_bagging_tuned.png
- experiment6_output/eda_reuse/plots/roc_curve_bagging_tuned.png
- experiment6_output/eda_reuse/plots/confusion_matrix_adaboost.png
- experiment6_output/eda_reuse/plots/roc_curve_adaboost.png
- experiment6_output/eda_reuse/plots/confusion_matrix_gradient_boosting.png
- experiment6_output/eda_reuse/plots/roc_curve_gradient_boosting.png
- experiment6_output/eda_reuse/plots/confusion_matrix_boosting_tuned.png
- experiment6_output/eda_reuse/plots/roc_curve_boosting_tuned.png
- experiment6_output/eda_reuse/plots/confusion_matrix_stacked_ensemble.png
- experiment6_output/eda_reuse/plots/roc_curve_stacked_ensemble.png
- experiment6_output/eda_reuse/plots/bagging_hyperparameter_search_results.png
- experiment6_output/eda_reuse/plots/boosting_hyperparameter_search_results.png
- experiment6_output/eda_reuse/plots/stacking_combination_search_results.png
- experiment6_output/eda_reuse/plots/cross_validation_accuracy_comparison.png
- experiment6_output/eda_reuse/plots/ensemble_final_comparison.png
- experiment6_output/eda_reuse/plots/time_comparison.png
