# Experiment 9: Perceptron vs MLP

## Dataset
- English Handwritten Characters (Kaggle)
- Total images: 3410, Classes: 62
- Image size: 32x32 grayscale, flattened, normalized

## MLP Tuning Table

| hidden_layer_sizes   | activation   | solver   |   learning_rate_init |   batch_size |   Val Accuracy |   Train Time (s) |   Iterations |
|:---------------------|:-------------|:---------|---------------------:|-------------:|---------------:|-----------------:|-------------:|
| (256, 128)           | relu         | adam     |                0.001 |           64 |         0.4469 |             8.17 |          300 |
| (128,)               | tanh         | adam     |                0.001 |           64 |         0.3059 |             4.03 |          300 |
| (128, 64)            | relu         | sgd      |                0.01  |           64 |         0.3059 |             2.65 |          284 |
| (128, 64)            | relu         | adam     |                0.001 |           64 |         0.2894 |             4.51 |          300 |
| (128, 64)            | relu         | adam     |                0.001 |          128 |         0.13   |             2.7  |          300 |
| (128, 64)            | relu         | adam     |                0.01  |           64 |         0.0165 |             0.83 |           56 |
| (128,)               | relu         | adam     |                0.001 |           64 |         0.0147 |             2.04 |          148 |
| (128, 64, 32)        | relu         | adam     |                0.001 |           64 |         0.0147 |             2.86 |          181 |

## Final Comparison

| Model             |   Train Accuracy |   Test Accuracy |   Precision (macro) |   Recall (macro) |   F1 (macro) |   ROC-AUC (micro) |   ROC-AUC (macro) |   Train Time (s) |
|:------------------|-----------------:|----------------:|--------------------:|-----------------:|-------------:|------------------:|------------------:|-----------------:|
| PLA (one-vs-rest) |           0.312  |          0.1804 |              0.3169 |           0.1804 |       0.1537 |            0.7947 |            0.8505 |             7.39 |
| MLP (256, 128)    |           0.9652 |          0.4355 |              0.4598 |           0.4355 |       0.4355 |            0.9382 |            0.9385 |            16.74 |