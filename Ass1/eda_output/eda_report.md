# EDA Report

## Dataset
- Name: Iris
- Shape: (150, 5)
- Target: species
- Task: classification

## Integrity

- Duplicate rows: 1
- Missing-value columns: 0
- Constant columns: None

## Feature Types

- Numerical features: 4
- Categorical features: 0

## Outliers

| feature           |   iqr_lower |   iqr_upper |   iqr_outliers |   z_score_outliers |
|:------------------|------------:|------------:|---------------:|-------------------:|
| sepal length (cm) |        3.15 |        8.35 |              0 |                  0 |
| sepal width (cm)  |        2.05 |        4.05 |              4 |                  1 |
| petal length (cm) |       -3.65 |       10.35 |              0 |                  0 |
| petal width (cm)  |       -1.95 |        4.05 |              0 |                  0 |

## Feature Selection

| feature           |   select_k_best_score |     p_value |   mutual_information |
|:------------------|----------------------:|------------:|---------------------:|
| petal width (cm)  |              803.214  | 4.56953e-69 |             1.00385  |
| petal length (cm) |              948.89   | 4.91526e-73 |             0.985075 |
| sepal length (cm) |              100.966  | 3.33078e-26 |             0.585014 |
| sepal width (cm)  |               36.0306 | 6.42152e-13 |             0.204446 |

## Inference Summary

- Dataset contains 150 samples and 5 columns.
- Detected 4 numerical features and 0 categorical features.
- Target column `species` is treated as Categorical; task type is classification.
- Duplicate rows: 1.
- Columns with missing values: 0.
- Highly correlated feature pairs: 3.
- Constant features: None.
- Suggested models: Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, SVM.