# EDA Report

## Dataset
- Name: Spambase
- Shape: (4601, 58)
- Target: class
- Task: classification

## Integrity

- Duplicate rows: 391
- Missing-value columns: 0
- Constant columns: None

## Feature Types

- Numerical features: 57
- Categorical features: 0

## Outliers

| feature                    |   iqr_lower |   iqr_upper |   iqr_outliers |   z_score_outliers |
|:---------------------------|------------:|------------:|---------------:|-------------------:|
| word_freq_make             |      0      |      0      |           1053 |                 90 |
| word_freq_address          |      0      |      0      |            898 |                 43 |
| word_freq_all              |     -0.63   |      1.05   |            338 |                 94 |
| word_freq_3d               |      0      |      0      |             47 |                 13 |
| word_freq_our              |     -0.57   |      0.95   |            501 |                 81 |
| word_freq_over             |      0      |      0      |            999 |                104 |
| word_freq_remove           |      0      |      0      |            807 |                 99 |
| word_freq_internet         |      0      |      0      |            824 |                 77 |
| word_freq_order            |      0      |      0      |            773 |                113 |
| word_freq_mail             |     -0.24   |      0.4    |            852 |                 74 |
| word_freq_receive          |      0      |      0      |            709 |                100 |
| word_freq_will             |     -1.2    |      2      |            270 |                102 |
| word_freq_people           |      0      |      0      |            852 |                 89 |
| word_freq_report           |      0      |      0      |            357 |                106 |
| word_freq_addresses        |      0      |      0      |            336 |                 99 |
| word_freq_free             |     -0.15   |      0.25   |            957 |                 69 |
| word_freq_business         |      0      |      0      |            963 |                 97 |
| word_freq_email            |      0      |      0      |           1038 |                106 |
| word_freq_you              |     -3.96   |      6.6    |             75 |                 60 |
| word_freq_credit           |      0      |      0      |            424 |                 76 |
| word_freq_your             |     -1.905  |      3.175  |            229 |                 87 |
| word_freq_font             |      0      |      0      |            117 |                 57 |
| word_freq_000              |      0      |      0      |            679 |                107 |
| word_freq_money            |      0      |      0      |            735 |                 32 |
| word_freq_hp               |      0      |      0      |           1090 |                 86 |
| word_freq_hpl              |      0      |      0      |            811 |                105 |
| word_freq_george           |      0      |      0      |            780 |                123 |
| word_freq_650              |      0      |      0      |            463 |                107 |
| word_freq_lab              |      0      |      0      |            372 |                 64 |
| word_freq_labs             |      0      |      0      |            469 |                 90 |
| word_freq_telnet           |      0      |      0      |            293 |                 61 |
| word_freq_857              |      0      |      0      |            205 |                 51 |
| word_freq_data             |      0      |      0      |            405 |                 77 |
| word_freq_415              |      0      |      0      |            215 |                 52 |
| word_freq_85               |      0      |      0      |            485 |                 82 |
| word_freq_technology       |      0      |      0      |            599 |                 77 |
| word_freq_1999             |      0      |      0      |            829 |                105 |
| word_freq_parts            |      0      |      0      |             83 |                 21 |
| word_freq_pm               |      0      |      0      |            384 |                 69 |
| word_freq_direct           |      0      |      0      |            453 |                 61 |
| word_freq_cs               |      0      |      0      |            148 |                 62 |
| word_freq_meeting          |      0      |      0      |            341 |                 78 |
| word_freq_original         |      0      |      0      |            375 |                107 |
| word_freq_project          |      0      |      0      |            327 |                 51 |
| word_freq_re               |     -0.165  |      0.275  |           1001 |                 64 |
| word_freq_edu              |      0      |      0      |            517 |                 74 |
| word_freq_table            |      0      |      0      |             63 |                 28 |
| word_freq_conference       |      0      |      0      |            203 |                 49 |
| char_freq_semicolon        |      0      |      0      |            790 |                 31 |
| char_freq_open_paren       |     -0.282  |      0.47   |            296 |                 49 |
| char_freq_open_bracket     |      0      |      0      |            529 |                 30 |
| char_freq_exclamation      |     -0.4725 |      0.7875 |            411 |                 44 |
| char_freq_dollar           |     -0.078  |      0.13   |            811 |                 62 |
| char_freq_hash             |      0      |      0      |            750 |                 24 |
| capital_run_length_average |     -1.589  |      6.883  |            363 |                 21 |
| capital_run_length_longest |    -49.5    |     98.5    |            463 |                 68 |
| capital_run_length_total   |   -311.5    |    612.5    |            550 |                 86 |

## Feature Selection

| feature                    |   select_k_best_score |      p_value |   mutual_information |
|:---------------------------|----------------------:|-------------:|---------------------:|
| char_freq_dollar           |              407.75   | 4.46363e-86  |             0.204981 |
| char_freq_exclamation      |              190.643  | 2.57071e-42  |             0.199113 |
| capital_run_length_longest |              162.693  | 1.71306e-36  |             0.192971 |
| word_freq_your             |              688.059  | 3.58901e-139 |             0.17645  |
| capital_run_length_average |               45.7523 | 1.55478e-11  |             0.169185 |
| word_freq_free             |              240.144  | 1.6171e-52   |             0.148177 |
| word_freq_remove           |              442.755  | 6.60858e-93  |             0.141634 |
| capital_run_length_total   |              242.725  | 4.79345e-53  |             0.130185 |
| word_freq_hp               |              260.346  | 1.21636e-56  |             0.123209 |
| word_freq_money            |              189.129  | 5.30155e-42  |             0.121599 |

## Inference Summary

- Dataset contains 4601 samples and 58 columns.
- Detected 57 numerical features and 0 categorical features.
- Target column `class` is treated as Categorical; task type is classification.
- Duplicate rows: 391.
- Columns with missing values: 0.
- Highly correlated feature pairs: 3.
- Constant features: None.
- Suggested models: Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, SVM.