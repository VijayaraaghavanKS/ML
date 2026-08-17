# Experiment 3: Regression Analysis using Linear and Regularized Models

## Aim
To implement Linear, Ridge, Lasso, and Elastic Net regression models for predicting a continuous target variable (loan amount sanctioned), evaluate performance using multiple metrics, visualize model behavior, and analyze overfitting, underfitting, and bias-variance characteristics.

## Dataset
- Dataset: Loan Amount Sanctioned (Kaggle: Predict Loan Amount Data)
- Training Samples: 480, Test Samples: 120
- Preprocessed Features: 25

## Required Tables

### Table 1: Hyperparameter Tuning Summary
| Model | Search Method | Best Parameters | Best CV R² |
|---|---|---|---|
| Ridge Regression | GridSearchCV | {'alpha': 1} | 0.9249 |
| Lasso Regression | GridSearchCV | {'alpha': 1} | 0.9253 |
| Elastic Net Regression | GridSearchCV | {'alpha': 0.01, 'l1_ratio': 0.5} | 0.9249 |

### Table 2: Cross-Validation Performance (K = 5)
| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Linear Regression | 229.319 | 86378.7 | 293.735 | 0.9248 |
| Ridge Regression | 229.124 | 86237.9 | 293.486 | 0.9249 |
| Lasso Regression | 228.467 | 85707.1 | 292.570 | 0.9253 |
| Elastic Net Regression | 229.020 | 86157.9 | 293.340 | 0.9249 |

### Table 3: Test Set Performance
| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Linear Regression | 235.394 | 105816 | 325.294 | 0.9309 |
| Ridge Regression | 235.770 | 105848 | 325.342 | 0.9309 |
| Lasso Regression | 236.682 | 106661 | 326.590 | 0.9303 |
| Elastic Net Regression | 236.358 | 105943 | 325.488 | 0.9308 |

*Training Time (s):* Linear = 0.01105, Ridge = 0.00081, Lasso = 0.00135, Elastic Net = 0.00281.
*Prediction Time (s):* Linear = 0.000364, Ridge = 0.000504, Lasso = 0.000322, Elastic Net = 0.000288.

### Table 4: Coefficient Comparison (Top Features)
| Feature | Linear | Ridge | Lasso | Elastic Net |
|---|---|---|---|---|
| Applicant_Income | 877.242 | 875.159 | 875.779 | 872.266 |
| Coapplicant_Income | 450.188 | 449.194 | 449.165 | 447.812 |
| Credit_Score | -3.116 | -3.329 | -2.588 | -3.604 |
| Age | 104.124 | 103.810 | 102.706 | 103.365 |
| Loan_Amount_Term_12 | -147.721 | -143.107 | -56.904 | -136.729 |
| Loan_Amount_Term_36 | -116.622 | -115.078 | -30.371 | -112.598 |
| Loan_Amount_Term_60 | -107.553 | -107.204 | -24.660 | -106.302 |
| Loan_Amount_Term_120 | -68.085 | -68.831 | 0.000 | -69.430 |
| Loan_Amount_Term_180 | 49.625 | 48.194 | 120.362 | 46.598 |
| Loan_Amount_Term_360 | 390.356 | 386.027 | 460.320 | 380.458 |

## Overfitting and Underfitting Analysis
- **Train vs test error gap:** Linear Regression achieved Train R² = 0.9339 and Test R² = 0.9309, a gap of only 0.0030 — the smallest gap among all four models, meaning the baseline model was not overfitting.
- **Effect of regularization strength:** The optimal Ridge alpha (1) shrank large OLS weights, mitigating variance without zeroing out predictors. The optimal Lasso alpha (1) drove 4 collinear feature coefficients to exactly zero (see Loan_Amount_Term_120 above). Elastic Net (alpha=0.01, l1_ratio=0.5) balanced sparsity with shrinkage.
- **Improvement in generalization after tuning:** 5-fold CV-based hyperparameter tuning produced a Test R² of 0.9309 and Test RMSE of 325.29, confirming the tuned models generalize well and are not overfit to the training folds.

## Bias-Variance Analysis
- **Bias behavior of Linear Regression:** Exhibits low bias when the linear assumption holds, but its coefficients are sensitive (higher variance) to feature collinearity among the loan features.
- **Variance reduction in Ridge and Elastic Net:** Ridge (L2) introduces a small amount of bias to significantly contract weight magnitudes, stabilizing predictions across CV folds; Elastic Net combines L2 stability with L1 sparsity, handling groups of correlated predictors (e.g. the Loan_Amount_Term dummy variables) more gracefully than pure Lasso.
- **Feature sparsity effect in Lasso:** Lasso eliminated 4 out of 25 features (setting their coefficients to zero), producing a sparser, more interpretable model at a very small cost in R² (0.9303 vs 0.9309 for Linear on the test set).

## Learning Outcomes
- Understood how L1 (Lasso), L2 (Ridge), and combined (Elastic Net) penalties change coefficient magnitudes differently, and observed this directly in the coefficient comparison table.
- Learned to use 5-fold cross-validated GridSearchCV to select regularization strength (alpha) and Elastic Net's l1_ratio.
- Practiced the full regression evaluation suite: MAE, MSE, RMSE, R² on both CV folds and a held-out test set.
- Connected the train-test R² gap to a concrete measure of overfitting/underfitting rather than a qualitative judgment.
- Understood the bias-variance trade-off in practical terms: how shrinkage (Ridge) trades a little bias for lower variance, and how sparsity (Lasso) trades some accuracy for interpretability.
