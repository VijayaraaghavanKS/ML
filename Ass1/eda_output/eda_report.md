# EDA Report

## Dataset
- Name: Loan Amount Sanctioned
- Shape: (600, 11)
- Target: loan_amount_sanctioned
- Task: regression

## Integrity

- Duplicate rows: 0
- Missing-value columns: 0
- Constant columns: None

## Feature Types

- Numerical features: 4
- Categorical features: 6

## Outliers

| feature            |   iqr_lower |   iqr_upper |   iqr_outliers |   z_score_outliers |
|:-------------------|------------:|------------:|---------------:|-------------------:|
| Applicant_Income   |   -1385.5   |    11569.8  |             25 |                  8 |
| Coapplicant_Income |   -3846.68  |     6495.27 |              0 |                  0 |
| Credit_Score       |      21.375 |     1140.38 |              0 |                  0 |
| Age                |       1.5   |       85.5  |              0 |                  0 |

## Feature Selection

| feature              |   select_k_best_score |      p_value |   mutual_information |
|:---------------------|----------------------:|-------------:|---------------------:|
| Applicant_Income     |           853.046     | 2.28666e-108 |           0.441558   |
| Coapplicant_Income   |           109.144     | 3.77363e-23  |           0.0839843  |
| Age                  |             1.67258   | 0.196538     |           0.0306342  |
| Loan_Amount_Term_360 |            16.7239    | 5.07234e-05  |           0.0301741  |
| Dependents_3+        |             0.0603393 | 0.806066     |           0.0287796  |
| Loan_Amount_Term_60  |             3.50854   | 0.0616632    |           0.0271911  |
| Existing_Loans_1     |             0.0766682 | 0.781984     |           0.0193057  |
| Dependents_1         |             0.348495  | 0.555245     |           0.0152112  |
| Credit_Score         |             0.0983137 | 0.753999     |           0.0135171  |
| Property_Area_Rural  |             0.0236637 | 0.877809     |           0.00772021 |

## Inference Summary

- Dataset contains 600 samples and 11 columns.
- Detected 4 numerical features and 6 categorical features.
- Target column `loan_amount_sanctioned` is treated as Numerical; task type is regression.
- Duplicate rows: 0.
- Columns with missing values: 0.
- Highly correlated feature pairs: 0.
- Constant features: None.
- Suggested models: Linear Regression, Ridge, Lasso, Decision Tree Regressor, Random Forest Regressor, SVR.