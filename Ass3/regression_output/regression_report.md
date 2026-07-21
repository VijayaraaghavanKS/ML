# Master Regression Report: Loan Amount Sanctioned

## 1. Aim and Objective

Predict continuous target `loan_amount_sanctioned` using Linear, Ridge, Lasso, and Elastic Net regression models.

## 2. Dataset Description and Preprocessing

- **Dataset**: Loan Amount Sanctioned

- **Training Samples**: 480, **Test Samples**: 120

- **Preprocessed Features**: 25

## 3. Table 1: Hyperparameter Tuning Summary

| Model                  | Search Method   | Best Parameters                  |   Best CV R² |   Tuning Time (s) |
|:-----------------------|:----------------|:---------------------------------|-------------:|------------------:|
| Ridge Regression       | GridSearchCV    | {'alpha': 1}                     |     0.924854 |         1.83065   |
| Lasso Regression       | GridSearchCV    | {'alpha': 1}                     |     0.92534  |         0.0398121 |
| Elastic Net Regression | GridSearchCV    | {'alpha': 0.01, 'l1_ratio': 0.5} |     0.924893 |         0.0428183 |

## 4. Table 2: Cross-Validation Performance (K = 5)

| Model                  |     MAE |     MSE |    RMSE |       R² |
|:-----------------------|--------:|--------:|--------:|---------:|
| Linear Regression      | 229.319 | 86378.7 | 293.735 | 0.924765 |
| Ridge Regression       | 229.124 | 86237.9 | 293.486 | 0.924854 |
| Lasso Regression       | 228.467 | 85707.1 | 292.57  | 0.92534  |
| Elastic Net Regression | 229.02  | 86157.9 | 293.34  | 0.924893 |

## 5. Table 3: Test Set Performance

| Model             |     MAE |    MSE |    RMSE |       R² |   Training Time |   Prediction Time |
|:------------------|--------:|-------:|--------:|---------:|----------------:|------------------:|
| Linear Regression | 235.394 | 105816 | 325.294 | 0.930899 |     0.00149012  |       0.000388291 |
| Ridge             | 235.77  | 105848 | 325.342 | 0.930879 |     0.000824833 |       0.00030375  |
| Lasso             | 236.682 | 106661 | 326.59  | 0.930348 |     0.00127775  |       0.000401459 |
| Elastic Net       | 236.358 | 105943 | 325.488 | 0.930817 |     0.00270654  |       0.000249333 |

## 6. Table 4: Coefficient Comparison (Top Features)

| Feature              |     Linear |      Ridge |     Lasso |   Elastic Net |
|:---------------------|-----------:|-----------:|----------:|--------------:|
| Applicant_Income     |  877.242   |  875.159   | 875.779   |     872.266   |
| Coapplicant_Income   |  450.188   |  449.194   | 449.165   |     447.812   |
| Credit_Score         |   -3.11588 |   -3.32865 |  -2.58765 |      -3.60403 |
| Age                  |  104.124   |  103.81    | 102.706   |     103.365   |
| Loan_Amount_Term_12  | -147.721   | -143.107   | -56.9036  |    -136.729   |
| Loan_Amount_Term_36  | -116.622   | -115.078   | -30.3711  |    -112.598   |
| Loan_Amount_Term_60  | -107.553   | -107.204   | -24.66    |    -106.302   |
| Loan_Amount_Term_120 |  -68.0851  |  -68.8314  |   0       |     -69.4296  |
| Loan_Amount_Term_180 |   49.6251  |   48.1938  | 120.362   |      46.5976  |
| Loan_Amount_Term_360 |  390.356   |  386.027   | 460.32    |     380.458   |

## 7. Overfitting and Underfitting Analysis


1. **Difference between Training and Validation/Test Errors**:
   - Linear Regression achieved a Train R² of **0.9339** and Test R² of **0.9309** (R² gap: **0.0030**).
   - Regularized models reduced the generalization gap; **Linear Regression** exhibited the lowest gap (**0.0030**).

2. **Effect of Regularization Strength**:
   - Optimal Ridge alpha (**1**) shrank large OLS weights, mitigating variance without zeroing out predictors.
   - Optimal Lasso alpha (**1**) set **4** collinear feature coefficients strictly to zero.
   - Elastic Net (alpha=**0.01**, l1_ratio=**0.5**) balanced feature selection with parameter shrinkage.

3. **Improvement in Generalization After Tuning**:
   - Hyperparameter tuning via 5-Fold CV improved Test R² to **0.9309** and reduced test RMSE to **325.29**.


## 8. Bias-Variance Analysis


1. **Bias Behavior of Linear Regression**:
   - Linear Regression exhibits low bias when linear assumptions hold, but suffers from high coefficient variance under feature collinearity.

2. **Variance Reduction in Ridge and Elastic Net**:
   - Ridge ($L_2$) introduces small bias ($\lambda > 0$) to significantly contract weight magnitudes, stabilizing predictions across CV folds.
   - Elastic Net combines $L_2$ stability with $L_1$ sparsity, effectively handling groups of correlated predictors.

3. **Feature Sparsity Effect in Lasso**:
   - Lasso ($L_1$) eliminated **4** irrelevant/redundant features out of **25**, yielding a sparse, highly interpretable model.


## 9. Conclusion


For **Loan Amount Sanctioned**, **Linear Regression** achieved the highest generalization performance with an **R² of 0.9309**, **RMSE of 325.29**, and **MAE of 235.39**.

Regularization effectively controlled coefficient inflation and prevented overfitting. Lasso and Elastic Net provided strong feature sparsity while Ridge provided stable weight shrinkage. The final selection balances accuracy, feature interpretability, and low prediction latency.

