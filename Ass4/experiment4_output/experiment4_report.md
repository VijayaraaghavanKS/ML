# Experiment 4 Report

## Aim
Classify the Spambase dataset using Logistic Regression and Support Vector Machine models while reusing the EDA notebook outputs.

## Dataset
- Dataset: Spambase
- CSV Path: spambase.csv
- Kaggle Reference: https://www.kaggle.com/datasets/somesh24/spambase
- Train Shape: (3680, 57)
- Test Shape: (921, 57)
- Feature Count: 57

## Logistic Regression Performance
| Model                       |   Accuracy |   Precision |   Recall |   F1 Score |   Training Time |
|:----------------------------|-----------:|------------:|---------:|-----------:|----------------:|
| Logistic Regression         |   0.929425 |    0.920904 | 0.898072 |   0.909344 |       0.0226314 |
| Logistic Regression (Tuned) |   0.926167 |    0.920228 | 0.889807 |   0.904762 |       0.272882  |

## Logistic Penalty Comparison
| Penalty   |   Best C | Best Solver   |   Best CV Accuracy |
|:----------|---------:|:--------------|-------------------:|
| L1        |      100 | liblinear     |           0.926087 |
| L2        |      100 | liblinear     |           0.925815 |

## SVM Kernel-wise Performance
| Kernel     |   Accuracy |   F1 Score |   Training Time |
|:-----------|-----------:|-----------:|----------------:|
| Linear     |   0.929425 |   0.909344 |       0.14282   |
| Polynomial |   0.779587 |   0.621974 |       0.126526  |
| RBF        |   0.927253 |   0.905501 |       0.0604745 |
| Sigmoid    |   0.884908 |   0.852778 |       0.0601579 |

## Hyperparameter Tuning Summary
| Model                  | Search Method   | Best Parameters                                                      |   Best CV Accuracy |   Execution Time |
|:-----------------------|:----------------|:---------------------------------------------------------------------|-------------------:|-----------------:|
| Logistic Regression    | GridSearchCV    | {'C': 100, 'max_iter': 5000, 'penalty': 'l1', 'solver': 'liblinear'} |           0.926087 |          8.37734 |
| Support Vector Machine | GridSearchCV    | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}                         |           0.935598 |          8.26375 |

## Cross Validation Results
| Fold    |   Best Logistic Regression |   Best SVM |
|:--------|---------------------------:|-----------:|
| 1       |                   0.941576 |   0.944293 |
| 2       |                   0.925272 |   0.941576 |
| 3       |                   0.934783 |   0.934783 |
| 4       |                   0.911685 |   0.930707 |
| 5       |                   0.91712  |   0.92663  |
| Average |                   0.926087 |   0.935598 |

## Logistic Regression vs SVM Comparison
| Model                       |   Accuracy |   Precision |   Recall |   F1 Score |   ROC-AUC |   Training Time |   Prediction Time |
|:----------------------------|-----------:|------------:|---------:|-----------:|----------:|----------------:|------------------:|
| Logistic Regression (Tuned) |   0.926167 |    0.920228 | 0.889807 |   0.904762 |  0.96786  |       0.272882  |         0.000626  |
| SVM (Tuned Best)            |   0.920738 |    0.914286 | 0.881543 |   0.897616 |  0.970181 |       0.0576145 |         0.0223442 |

## Analysis
- Best classifier: Logistic Regression (Tuned) achieved Accuracy=0.9262, F1=0.9048, and ROC-AUC=0.9679 on the held-out test set.
- Effect of regularization: the best cross-validated Logistic Regression setting used penalty=l1 with C=100, reaching CV accuracy=0.9261. This indicates that that regularization choice generalized better than the other explored settings for this dataset.
- L1 vs L2 comparison: the best L1 configuration reached CV accuracy=0.9261, while the best L2 configuration reached CV accuracy=0.9258. The stronger performer was L1 in this experiment.
- Effect of C: among the tested values, C=100 produced the best maximum CV accuracy (0.9261), while C=0.01 gave the weakest maximum CV accuracy (0.9090).
- Behaviour of SVM kernels: Linear was the strongest baseline kernel with Accuracy=0.9294 and F1=0.9093, whereas Polynomial was the weakest with Accuracy=0.7796 and F1=0.6220.
- Bias-variance trade-off: the tuned Logistic Regression configuration had a train-test CV gap of 0.0067, while the best tuned SVM configuration had a gap of 0.0344. The larger gap suggests the model is fitting the training folds more aggressively.
- Why one kernel performed better: the tuned SVM selected kernel=rbf with C=10 and gamma=scale, so the data benefited from that decision boundary more than the alternative kernels tested.
- Computational trade-offs: SVM (Tuned Best) trained fastest (0.0576s), while Logistic Regression (Tuned) predicted fastest (0.0006s). The final selection should balance metric gains against these time costs.

## Conclusion
Logistic Regression (Tuned) delivered the best overall held-out performance in this experiment with Accuracy=0.9262, F1=0.9048, and ROC-AUC=0.9679, while SVM (Tuned Best) offered the lower training cost at 0.0576s.

## Generated Plots
- experiment4_output/eda_reuse/plots/confusion_matrix_logistic_regression.png
- experiment4_output/eda_reuse/plots/roc_curve_logistic_regression.png
- experiment4_output/eda_reuse/plots/precision_recall_logistic_regression.png
- experiment4_output/eda_reuse/plots/confusion_matrix_logistic_regression_tuned.png
- experiment4_output/eda_reuse/plots/roc_curve_logistic_regression_tuned.png
- experiment4_output/eda_reuse/plots/precision_recall_logistic_regression_tuned.png
- experiment4_output/eda_reuse/plots/confusion_matrix_svm_linear.png
- experiment4_output/eda_reuse/plots/roc_curve_svm_linear.png
- experiment4_output/eda_reuse/plots/precision_recall_svm_linear.png
- experiment4_output/eda_reuse/plots/confusion_matrix_svm_polynomial.png
- experiment4_output/eda_reuse/plots/roc_curve_svm_polynomial.png
- experiment4_output/eda_reuse/plots/precision_recall_svm_polynomial.png
- experiment4_output/eda_reuse/plots/confusion_matrix_svm_rbf.png
- experiment4_output/eda_reuse/plots/roc_curve_svm_rbf.png
- experiment4_output/eda_reuse/plots/precision_recall_svm_rbf.png
- experiment4_output/eda_reuse/plots/confusion_matrix_svm_sigmoid.png
- experiment4_output/eda_reuse/plots/roc_curve_svm_sigmoid.png
- experiment4_output/eda_reuse/plots/precision_recall_svm_sigmoid.png
- experiment4_output/eda_reuse/plots/confusion_matrix_svm_tuned_best.png
- experiment4_output/eda_reuse/plots/roc_curve_svm_tuned_best.png
- experiment4_output/eda_reuse/plots/precision_recall_svm_tuned_best.png
- experiment4_output/eda_reuse/plots/logistic_hyperparameter_search_results.png
- experiment4_output/eda_reuse/plots/svm_hyperparameter_search_results.png
- experiment4_output/eda_reuse/plots/kernel_comparison.png
- experiment4_output/eda_reuse/plots/cross_validation_accuracy.png
- experiment4_output/eda_reuse/plots/final_model_comparison.png
- experiment4_output/eda_reuse/plots/time_comparison.png
