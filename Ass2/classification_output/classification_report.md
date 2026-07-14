# Classification Report

## Aim
Classify Iris samples using Naive Bayes and KNN.

## Naive Bayes Comparison
|           |   Gaussian |   Multinomial |   Bernoulli |
|:----------|-----------:|--------------:|------------:|
| Accuracy  |   0.966667 |      0.833333 |    0.733333 |
| Precision |   0.969697 |      0.835017 |    0.819444 |
| Recall    |   0.966667 |      0.833333 |    0.733333 |
| F1        |   0.966583 |      0.832916 |    0.670552 |
| ROC-AUC   |   0.99     |      0.836667 |    0.85     |

## KNN Comparison
|   k |   Accuracy |   Precision |   Recall |       F1 |
|----:|-----------:|------------:|---------:|---------:|
|   1 |   0.966667 |    0.969697 | 0.966667 | 0.966583 |
|   3 |   0.933333 |    0.944444 | 0.933333 | 0.93266  |
|   5 |   0.933333 |    0.944444 | 0.933333 | 0.93266  |
|   7 |   0.966667 |    0.969697 | 0.966667 | 0.966583 |
|   9 |   0.966667 |    0.969697 | 0.966667 | 0.966583 |
|  11 |   0.966667 |    0.969697 | 0.966667 | 0.966583 |

## Grid Search vs Randomized Search
| Parameter      | GridSearchCV       | RandomizedSearchCV   |
|:---------------|:-------------------|:---------------------|
| Best k         | 3                  | 3                    |
| Metric         | euclidean          | euclidean            |
| Weights        | uniform            | uniform              |
| Algorithm      | auto               | brute                |
| CV Accuracy    | 0.9666666666666668 | 0.9666666666666668   |
| Execution Time | 1.6185255930004132 | 0.05976007099980052  |

## KDTree vs BallTree
| Metric          |      KDTree |    BallTree |
|:----------------|------------:|------------:|
| Accuracy        | 0.966667    | 0.966667    |
| Training Time   | 0.000849299 | 0.000652276 |
| Prediction Time | 0.00219343  | 0.00142053  |

## Cross Validation
| Fold    |   Naive Bayes |   Best KNN |
|:--------|--------------:|-----------:|
| 1       |      0.958333 |   0.958333 |
| 2       |      0.958333 |   1        |
| 3       |      0.958333 |   0.958333 |
| 4       |      0.958333 |   0.958333 |
| 5       |      0.916667 |   0.958333 |
| Average |      0.95     |   0.966667 |

## Experimental Time Analysis
| Model          |   Training Time |   Prediction Time |
|:---------------|----------------:|------------------:|
| Gaussian NB    |     0.00109344  |       0.000456839 |
| Multinomial NB |     0.00218828  |       0.000573045 |
| Bernoulli NB   |     0.0013615   |       0.000580518 |
| KNN k=1        |     0.000948522 |       0.00178676  |

## Theoretical Complexity
| Algorithm   | Training   | Prediction   |
|:------------|:-----------|:-------------|
| Naive Bayes | O(nd)      | O(d)         |
| KNN (Brute) | O(1)       | O(nd)        |
| KDTree      | O(nlog n)  | O(log n) avg |
| BallTree    | O(nlog n)  | O(log n) avg |

## Analysis
- Best Naive Bayes variant: Gaussian NB with F1=0.9666.
- Optimal k: 1 with F1=0.9666 and Accuracy=0.9667.
- GridSearchCV performed better by CV accuracy; Grid=0.9667, Randomized=0.9667.
- BallTree had faster prediction time in the KDTree vs BallTree comparison.
- Practical timing follows the theory: Naive Bayes trains and predicts quickly, while KNN has low training cost but prediction depends on neighbor search.
- Preferred large-dataset classifier from these results: Gaussian NB.

## Conclusion
For Iris, Gaussian NB was the best Naive Bayes model, while KNN performed best at k=1. The final choice should balance F1 score, prediction time, and dataset size.
