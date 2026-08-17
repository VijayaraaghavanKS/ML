# Classification Report

## Aim
Classify Spambase samples using Naive Bayes and KNN.

## Naive Bayes Comparison
|           |   Gaussian |   Multinomial |   Bernoulli |
|:----------|-----------:|--------------:|------------:|
| Accuracy  |   0.83279  |      0.895765 |    0.901194 |
| Precision |   0.866565 |      0.89933  |    0.901245 |
| Recall    |   0.83279  |      0.895765 |    0.901194 |
| F1        |   0.834536 |      0.8939   |    0.900515 |
| ROC-AUC   |   0.937612 |      0.961191 |    0.960393 |

## KNN Comparison
|   k |   Accuracy |   Precision |   Recall |       F1 |
|----:|-----------:|------------:|---------:|---------:|
|   1 |   0.899023 |    0.898777 | 0.899023 | 0.898846 |
|   3 |   0.897937 |    0.897846 | 0.897937 | 0.897887 |
|   5 |   0.907709 |    0.907594 | 0.907709 | 0.907641 |
|   7 |   0.908795 |    0.908571 | 0.908795 | 0.908611 |
|   9 |   0.908795 |    0.908571 | 0.908795 | 0.908611 |
|  11 |   0.909881 |    0.909656 | 0.909881 | 0.909578 |

## Grid Search vs Randomized Search
| Parameter      | GridSearchCV       | RandomizedSearchCV   |
|:---------------|:-------------------|:---------------------|
| Best k         | 9                  | 9                    |
| Metric         | manhattan          | manhattan            |
| Weights        | distance           | distance             |
| Algorithm      | auto               | kd_tree              |
| CV Accuracy    | 0.9252717391304348 | 0.9252717391304348   |
| Execution Time | 2.854243250039872  | 0.33956341695738956  |

## KDTree vs BallTree
| Metric          |     KDTree |   BallTree |
|:----------------|-----------:|-----------:|
| Accuracy        | 0.909881   |  0.909881  |
| Training Time   | 0.00219929 |  0.001984  |
| Prediction Time | 0.0543885  |  0.0522719 |

## Cross Validation
| Fold    |   Naive Bayes |   Best KNN |
|:--------|--------------:|-----------:|
| 1       |      0.91712  |   0.938859 |
| 2       |      0.907609 |   0.927989 |
| 3       |      0.889946 |   0.925272 |
| 4       |      0.898098 |   0.92663  |
| 5       |      0.899457 |   0.907609 |
| Average |      0.902446 |   0.925272 |

## Experimental Time Analysis
| Model          |   Training Time |   Prediction Time |
|:---------------|----------------:|------------------:|
| Gaussian NB    |     0.00144408  |       0.000428041 |
| Multinomial NB |     0.00209517  |       0.0004195   |
| Bernoulli NB   |     0.00218358  |       0.0006095   |
| KNN k=11       |     0.000658708 |       0.00359633  |

## Theoretical Complexity
| Algorithm   | Training   | Prediction   |
|:------------|:-----------|:-------------|
| Naive Bayes | O(nd)      | O(d)         |
| KNN (Brute) | O(1)       | O(nd)        |
| KDTree      | O(nlog n)  | O(log n) avg |
| BallTree    | O(nlog n)  | O(log n) avg |

## Analysis
- Best Naive Bayes variant: Bernoulli NB with F1=0.9005.
- Optimal k: 11 with F1=0.9096 and Accuracy=0.9099.
- GridSearchCV performed better by CV accuracy; Grid=0.9253, Randomized=0.9253.
- BallTree had faster prediction time in the KDTree vs BallTree comparison.
- Practical timing follows the theory: Naive Bayes trains and predicts quickly, while KNN has low training cost but prediction depends on neighbor search.
- Preferred large-dataset classifier from these results: Bernoulli NB.

## Conclusion
For Spambase, Bernoulli NB was the best Naive Bayes model, while KNN performed best at k=11. The final choice should balance F1 score, prediction time, and dataset size.
