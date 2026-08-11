# Experiment 5 Report

## Aim
Implement and compare a Decision Tree classifier and a Random Forest ensemble on the Wisconsin Diagnostic Breast Cancer dataset using 5-fold cross-validation and hyperparameter tuning, reusing the EDA notebook outputs.

## Dataset
- Dataset: Wisconsin Diagnostic Breast Cancer
- CSV Path: breast_cancer.csv
- UCI Reference: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Train Shape: (455, 30)
- Test Shape: (114, 30)
- Feature Count: 30
- Classes: B (negative), M (positive)

## Decision Tree Performance
| Model                 |   Accuracy |   Precision |   Recall |   F1 Score |   ROC-AUC |   Training Time |   Prediction Time |
|:----------------------|-----------:|------------:|---------:|-----------:|----------:|----------------:|------------------:|
| Decision Tree         |   0.929825 |    0.904762 | 0.904762 |   0.904762 |  0.924603 |      0.00344063 |       0.000240958 |
| Decision Tree (Tuned) |   0.964912 |    1        | 0.904762 |   0.95     |  0.974372 |      0.00255408 |       0.000264458 |

## Decision Tree Hyperparameter Evaluation (5-Fold CV)
| Criterion   | Max Depth   |   Avg CV Accuracy (%) |   Avg CV F1 Score |
|:------------|:------------|----------------------:|------------------:|
| entropy     | 5           |                 93.63 |            0.9121 |
| entropy     | 7           |                 93.41 |            0.9094 |
| entropy     | 10          |                 93.41 |            0.9094 |
| entropy     | None        |                 93.41 |            0.9094 |
| gini        | 7           |                 92.97 |            0.9039 |
| gini        | 10          |                 92.97 |            0.9039 |
| gini        | None        |                 92.97 |            0.9039 |
| gini        | 3           |                 92.75 |            0.899  |
| entropy     | 3           |                 92.53 |            0.8941 |
| gini        | 5           |                 92.53 |            0.8971 |

## Random Forest Performance
| Model                 |   Accuracy |   Precision |   Recall |   F1 Score |   ROC-AUC |   Training Time |   Prediction Time |
|:----------------------|-----------:|------------:|---------:|-----------:|----------:|----------------:|------------------:|
| Random Forest         |   0.973684 |           1 | 0.928571 |   0.962963 |  0.99289  |       0.0502788 |        0.00236583 |
| Random Forest (Tuned) |   0.964912 |           1 | 0.904762 |   0.95     |  0.993717 |       0.10447   |        0.00414454 |

## Random Forest Hyperparameter Evaluation (5-Fold CV)
|   n_estimators | Max Depth   | Max Features   |   Avg CV Accuracy (%) |   Avg CV F1 Score |
|---------------:|:------------|:---------------|----------------------:|------------------:|
|            200 | None        | sqrt           |                 97.14 |            0.9615 |
|            200 | 5           | sqrt           |                 97.14 |            0.9608 |
|             50 | 10          | sqrt           |                 97.14 |            0.9617 |
|            200 | 10          | sqrt           |                 97.14 |            0.9615 |
|             50 | 10          | log2           |                 96.92 |            0.9584 |
|             50 | None        | log2           |                 96.92 |            0.9584 |
|             50 | None        | sqrt           |                 96.92 |            0.9589 |
|            100 | 5           | sqrt           |                 96.92 |            0.958  |
|            100 | 10          | log2           |                 96.92 |            0.9585 |
|            100 | 5           | log2           |                 96.7  |            0.955  |
|            100 | None        | log2           |                 96.7  |            0.9556 |
|            200 | 10          | log2           |                 96.7  |            0.9557 |
|            200 | None        | log2           |                 96.48 |            0.9524 |
|            100 | 10          | sqrt           |                 96.48 |            0.9519 |
|            100 | None        | sqrt           |                 96.48 |            0.9519 |
|             50 | 5           | sqrt           |                 96.48 |            0.952  |
|            200 | 5           | log2           |                 96.26 |            0.949  |
|             50 | 5           | log2           |                 96.26 |            0.9485 |

## Hyperparameter Tuning Summary
| Model         | Search Method   | Best Parameters                                                                          |   Best CV Accuracy |   Execution Time |
|:--------------|:----------------|:-----------------------------------------------------------------------------------------|-------------------:|-----------------:|
| Decision Tree | GridSearchCV    | {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 10} |           0.936264 |          1.83661 |
| Random Forest | GridSearchCV    | {'bootstrap': False, 'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 200}        |           0.971429 |          1.27574 |

## Cross Validation Results
| Model         |   Fold 1 |   Fold 2 |   Fold 3 |   Fold 4 |   Fold 5 |   Average |
|:--------------|---------:|---------:|---------:|---------:|---------:|----------:|
| Decision Tree | 0.901099 | 0.956044 | 0.912088 | 0.956044 | 0.956044 |  0.936264 |
| Random Forest | 0.956044 | 1        | 0.945055 | 0.967033 | 0.989011 |  0.971429 |

## Decision Tree vs Random Forest Comparison
| Model                 |   Accuracy |   Precision |   Recall |   F1 Score |   ROC-AUC |   Training Time |   Prediction Time |
|:----------------------|-----------:|------------:|---------:|-----------:|----------:|----------------:|------------------:|
| Decision Tree (Tuned) |   0.964912 |           1 | 0.904762 |       0.95 |  0.974372 |      0.00255408 |       0.000264458 |
| Random Forest (Tuned) |   0.964912 |           1 | 0.904762 |       0.95 |  0.993717 |      0.10447    |       0.00414454  |

## Analysis
- Best classifier: Random Forest (Tuned) achieved Accuracy=0.9649, F1=0.9500, and ROC-AUC=0.9937 on the held-out test set.
- Decision Tree hyperparameter impact: varying max_depth produced the widest swing in best CV accuracy (0.88 percentage points), more than the other three tuned parameters.
- Random Forest hyperparameter impact: varying bootstrap produced the widest swing in best CV accuracy (0.44 percentage points) among the four tuned parameters.
- Bias-variance trade-off: the tuned Decision Tree had a train-test CV accuracy gap of 0.0467, while the tuned Random Forest had a gap of 0.0286, showing the ensemble reduced overfitting.
- Depth sweep: at max_depth=1 the tree scored Train=0.9297/Test=0.8772, while an unbounded tree (max_depth=None) scored Train=1.0000/Test=0.9298, a train-test gap of 0.0702.
- Computational trade-offs: Decision Tree (Tuned) trained fastest (0.0026s), while Decision Tree (Tuned) predicted fastest (0.0003s).

## Observation Questions
**Q: How does tree depth affect overfitting in Decision Trees?**

The depth sweep shows training accuracy climbing to 1.0000 as depth grows, while test accuracy peaks at 0.9386 around max_depth=7. Once the tree is left unbounded (max_depth=None) the train-test gap widens to 0.0702, so deeper trees increasingly memorize the training folds instead of generalizing further on this dataset.

**Q: Which hyperparameter had the greatest impact on performance?**

For the Decision Tree, max_depth moved the best CV accuracy by 0.88 percentage points, and for the Random Forest, bootstrap moved it by 0.44 percentage points. Both were the largest swings within their respective search spaces, so max_depth and bootstrap mattered more than the other tuned parameters in this run.

**Q: How does Random Forest improve generalization?**

The Random Forest's CV train-test gap (0.0286) was narrower than the Decision Tree's gap (0.0467), and its 5-fold CV accuracy averaged 0.9714 against 0.9363 for the tree. This is consistent with bagging and random feature selection averaging out the variance carried by individual trees.

**Q: Did ensemble learning always improve performance? Why or why not?**

On the held-out test set, the Decision Tree matched or outperformed the Random Forest (RF F1=0.9500 vs DT F1=0.9500), so ensembling did not help here. Ensembling is not guaranteed to help whenever a single well-tuned tree already captures the decision boundary well or when the additional trees mostly agree with each other, which can happen on a comparatively small, well-separated dataset like this one.


## Conclusion
Random Forest (Tuned) delivered the best overall held-out performance in this experiment with Accuracy=0.9649, F1=0.9500, and ROC-AUC=0.9937. Hyperparameter tuning raised Decision Tree CV accuracy to 0.9363 and Random Forest CV accuracy to 0.9714 over their untuned baselines. Across the 5 cross-validation folds, Random Forest was more stable (fold std: RF=0.0204, DT=0.0245), and ensembling reduced the train-test CV accuracy gap compared to the single tree (0.0286 vs 0.0467).

## Generated Plots
- experiment5_output/eda_reuse/plots/confusion_matrix_decision_tree.png
- experiment5_output/eda_reuse/plots/roc_curve_decision_tree.png
- experiment5_output/eda_reuse/plots/feature_importance_decision_tree.png
- experiment5_output/eda_reuse/plots/confusion_matrix_decision_tree_tuned.png
- experiment5_output/eda_reuse/plots/roc_curve_decision_tree_tuned.png
- experiment5_output/eda_reuse/plots/feature_importance_decision_tree_tuned.png
- experiment5_output/eda_reuse/plots/confusion_matrix_random_forest.png
- experiment5_output/eda_reuse/plots/roc_curve_random_forest.png
- experiment5_output/eda_reuse/plots/feature_importance_random_forest.png
- experiment5_output/eda_reuse/plots/confusion_matrix_random_forest_tuned.png
- experiment5_output/eda_reuse/plots/roc_curve_random_forest_tuned.png
- experiment5_output/eda_reuse/plots/feature_importance_random_forest_tuned.png
- experiment5_output/eda_reuse/plots/decision_tree_hyperparameter_search_results.png
- experiment5_output/eda_reuse/plots/decision_tree_overfitting_curve.png
- experiment5_output/eda_reuse/plots/random_forest_hyperparameter_search_results.png
- experiment5_output/eda_reuse/plots/cross_validation_accuracy_comparison.png
- experiment5_output/eda_reuse/plots/decision_tree_vs_random_forest_comparison.png
- experiment5_output/eda_reuse/plots/time_comparison.png
