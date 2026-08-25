# EDA Report

## Dataset
- Name: Wisconsin Diagnostic Breast Cancer
- Shape: (569, 31)
- Target: diagnosis
- Task: classification

## Integrity

- Duplicate rows: 0
- Missing-value columns: 0
- Constant columns: None

## Feature Types

- Numerical features: 30
- Categorical features: 0

## Outliers

| feature                 |   iqr_lower |   iqr_upper |   iqr_outliers |   z_score_outliers |
|:------------------------|------------:|------------:|---------------:|-------------------:|
| mean radius             |      5.58   |     21.9    |             14 |                  5 |
| mean texture            |      7.725  |     30.245  |              7 |                  4 |
| mean perimeter          |     31.775  |    147.495  |             13 |                  7 |
| mean area               |   -123.3    |   1326.3    |             25 |                  8 |
| mean smoothness         |      0.058  |      0.1337 |              6 |                  5 |
| mean compactness        |     -0.0333 |      0.2286 |             16 |                  9 |
| mean concavity          |     -0.1222 |      0.2824 |             18 |                  9 |
| mean concave points     |     -0.0602 |      0.1545 |             10 |                  6 |
| mean symmetry           |      0.1112 |      0.2464 |             15 |                  5 |
| mean fractal dimension  |      0.0451 |      0.0787 |             15 |                  7 |
| radius error            |     -0.1374 |      0.8486 |             38 |                  7 |
| texture error           |     -0.1263 |      2.4341 |             20 |                  9 |
| perimeter error         |     -1.0205 |      5.9835 |             38 |                  8 |
| area error              |    -23.16   |     86.2    |             65 |                  6 |
| smoothness error        |      0.0007 |      0.0126 |             30 |                  7 |
| compactness error       |     -0.016  |      0.0615 |             28 |                 12 |
| concavity error         |     -0.0253 |      0.0825 |             22 |                  6 |
| concave points error    |     -0.003  |      0.0253 |             19 |                  6 |
| symmetry error          |      0.0027 |      0.036  |             27 |                 11 |
| fractal dimension error |     -0.0012 |      0.008  |             28 |                 10 |
| worst radius            |      4.34   |     27.46   |             17 |                  6 |
| worst texture           |      8.12   |     42.68   |              5 |                  4 |
| worst perimeter         |     22.175  |    187.335  |             15 |                  6 |
| worst area              |   -337.75   |   1937.05   |             35 |                 10 |
| worst smoothness        |      0.0725 |      0.1901 |              7 |                  3 |
| worst compactness       |     -0.1407 |      0.627  |             16 |                 10 |
| worst concavity         |     -0.2881 |      0.7855 |             12 |                  7 |
| worst concave points    |     -0.0798 |      0.3061 |              0 |                  0 |
| worst symmetry          |      0.1492 |      0.4192 |             23 |                  9 |
| worst fractal dimension |      0.0405 |      0.123  |             24 |                  9 |

## Feature Selection

| feature              |   select_k_best_score |     p_value |   mutual_information |
|:---------------------|----------------------:|------------:|---------------------:|
| worst radius         |               692.861 | 2.48369e-93 |             0.461221 |
| worst perimeter      |               717.246 | 2.09305e-95 |             0.454559 |
| worst area           |               522.189 | 1.93239e-77 |             0.453342 |
| mean concave points  |               684.527 | 1.30087e-92 |             0.445956 |
| worst concave points |               733.725 | 8.77859e-97 |             0.4329   |
| mean perimeter       |               548.413 | 4.68752e-80 |             0.409301 |
| mean area            |               444.858 | 2.69116e-69 |             0.374696 |
| mean radius          |               511.275 | 2.48505e-76 |             0.373251 |
| mean concavity       |               397.592 | 5.78193e-64 |             0.354085 |
| area error           |               180.561 | 6.99837e-35 |             0.350009 |

## Inference Summary

- Dataset contains 569 samples and 31 columns.
- Detected 30 numerical features and 0 categorical features.
- Target column `diagnosis` is treated as Categorical; task type is classification.
- Duplicate rows: 0.
- Columns with missing values: 0.
- Highly correlated feature pairs: 44.
- Constant features: None.
- Suggested models: Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, SVM.