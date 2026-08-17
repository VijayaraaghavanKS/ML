# Experiment 2: Email Spam/Ham Classification using Naive Bayes and KNN

## Aim
To build spam classifiers using Naive Bayes (Gaussian, Multinomial, Bernoulli) and KNN on the Spambase dataset, compare KDTree vs BallTree, optimize KNN using GridSearchCV and RandomizedSearchCV, evaluate computational complexity and execution time, and validate using 5-fold cross validation.

## Required Tables

### Naive Bayes Comparison
| Metric | Gaussian | Multinomial | Bernoulli |
|---|---|---|---|
| Accuracy | 0.8328 | 0.8958 | 0.9012 |
| Precision | 0.8666 | 0.8993 | 0.9012 |
| Recall | 0.8328 | 0.8958 | 0.9012 |
| F1 | 0.8345 | 0.8939 | 0.9005 |
| ROC-AUC | 0.9376 | 0.9612 | 0.9604 |

### KNN Comparison
| k | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 1 | 0.8990 | 0.8988 | 0.8990 | 0.8988 |
| 3 | 0.8979 | 0.8978 | 0.8979 | 0.8979 |
| 5 | 0.9077 | 0.9076 | 0.9077 | 0.9076 |
| 7 | 0.9088 | 0.9086 | 0.9088 | 0.9086 |
| 9 | 0.9088 | 0.9086 | 0.9088 | 0.9086 |
| 11 | 0.9099 | 0.9097 | 0.9099 | 0.9096 |

### Grid Search vs Randomized Search
| Parameter | GridSearchCV | RandomizedSearchCV |
|---|---|---|
| Best k | 9 | 9 |
| Metric | manhattan | manhattan |
| Weights | distance | distance |
| Algorithm | auto | kd_tree |
| CV Accuracy | 0.9253 | 0.9253 |
| Execution Time (s) | 2.8542 | 0.3396 |

### KDTree vs BallTree
| Metric | KDTree | BallTree |
|---|---|---|
| Accuracy | 0.9099 | 0.9099 |
| Training Time (s) | 0.002199 | 0.001984 |
| Prediction Time (s) | 0.054389 | 0.052272 |

### Cross Validation
| Fold | Naive Bayes | Best KNN |
|---|---|---|
| 1 | 0.9171 | 0.9389 |
| 2 | 0.9076 | 0.9280 |
| 3 | 0.8899 | 0.9253 |
| 4 | 0.8981 | 0.9266 |
| 5 | 0.8995 | 0.9076 |
| **Average** | **0.9024** | **0.9253** |

### Theoretical Time Complexity
| Algorithm | Training | Prediction |
|---|---|---|
| Naive Bayes | O(nd) | O(d) |
| KNN (Brute) | O(1) | O(nd) |
| KDTree | O(n log n) | O(log n) avg |
| BallTree | O(n log n) | O(log n) avg |

### Experimental Time Analysis
| Algorithm | Training (s) | Prediction (s) |
|---|---|---|
| Gaussian NB | 0.001444 | 0.000428 |
| Multinomial NB | 0.002095 | 0.000420 |
| Bernoulli NB | 0.002184 | 0.000610 |
| Best KNN (k=11) | 0.000659 | 0.003596 |

## Analysis Questions
1. **Which Naive Bayes variant performed best?** Bernoulli NB, with F1 = 0.9005 and Accuracy = 0.9012 — it edges out Multinomial NB (F1 = 0.8939) because Spambase's word-presence features behave more like binary indicators than raw counts or continuous values, which is exactly what Bernoulli NB models.
2. **What is the optimal value of k?** k = 11, giving the highest Accuracy (0.9099) and F1 (0.9096) among the tested values {1,3,5,7,9,11}.
3. **Compare GridSearchCV and RandomizedSearchCV.** Both converged to the same best configuration (k=9, manhattan, distance weights, CV accuracy 0.9253), but RandomizedSearchCV reached it about 8.4x faster (0.3396s vs 2.8542s) since it samples a subset of the grid instead of evaluating every combination.
4. **Compare KDTree and BallTree.** Both achieved identical accuracy (0.9099); BallTree had marginally faster training (0.001984s vs 0.002199s) and prediction (0.052272s vs 0.054389s) on this dataset.
5. **Compare theoretical and practical complexity.** The practical timings track the theory: Naive Bayes trains and predicts in roughly constant, small time regardless of query, while KNN has near-zero training time (O(1), just storing the data) but its prediction step is the costliest of all models tested (0.003596s per the reported figure), consistent with its O(nd) brute-force or O(log n) average tree-based search cost.
6. **Which classifier is preferred for large datasets?** Bernoulli NB — it has the best accuracy/F1 among the three Naive Bayes variants while retaining Naive Bayes's O(nd) training and O(d) prediction complexity, both cheaper than KNN's per-query search cost at scale.

## Learning Outcomes
- Understood how Naive Bayes variants (Gaussian, Multinomial, Bernoulli) make different distributional assumptions and how that affects which one suits a given feature type.
- Learned that KNN is a lazy learner with negligible training cost but non-trivial prediction cost that grows with dataset size.
- Gained hands-on experience comparing exhaustive (GridSearchCV) vs sampled (RandomizedSearchCV) hyperparameter search strategies and their time/accuracy trade-offs.
- Understood how spatial index structures (KDTree, BallTree) accelerate nearest-neighbor search compared to brute-force search.
- Practiced evaluating classifiers using a full metric suite (Accuracy, Precision, Recall, F1, ROC-AUC) and 5-fold cross-validation rather than a single train/test split.
- Connected theoretical time-complexity notation to measured wall-clock training/prediction times.
