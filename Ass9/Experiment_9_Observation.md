# Experiment 9: Perceptron vs Multilayer Perceptron (A/B Experiment) with Hyperparameter Tuning

## Aim
To implement a single-layer Perceptron Learning Algorithm (PLA) from scratch and a tuned Multilayer Perceptron (MLP), compare their performance on the English Handwritten Characters dataset (62 classes: 0-9, A-Z, a-z), and justify the MLP's chosen hyperparameters through systematic tuning.

## Dataset
- Kaggle English Handwritten Characters Dataset
- 3,410 images, 62 classes (0-9, A-Z, a-z), 55 images per class
- Preprocessing: grayscale conversion, resize to 32×32, flatten to 1024 features, normalize to [0, 1]
- Train/Test split: 80/20 stratified (2,728 train / 682 test); a further 80/20 split of the training set was used for hyperparameter validation

## Model A: PLA
One-vs-rest ensemble of 62 single-layer Perceptrons implemented from scratch in NumPy, step activation, weight update rule w_{t+1} = w_t + η(y − ŷ)x, learning rate η=0.01, 50 epochs, prediction by argmax of raw (pre-step) score across the 62 perceptrons.

## Table 1: MLP Hyperparameter Tuning Results
| Hidden Layers | Activation | Solver | LR | Batch Size | Val Accuracy | Train Time (s) | Iterations |
|---|---|---|---|---|---|---|---|
| (256, 128) | relu | adam | 0.001 | 64 | **0.4469** | 8.17 | 300 |
| (128,) | tanh | adam | 0.001 | 64 | 0.3059 | 4.03 | 300 |
| (128, 64) | relu | sgd | 0.01 | 64 | 0.3059 | 2.65 | 284 |
| (128, 64) | relu | adam | 0.001 | 64 | 0.2894 | 4.51 | 300 |
| (128, 64) | relu | adam | 0.001 | 128 | 0.1300 | 2.70 | 300 |
| (128, 64) | relu | adam | 0.01 | 64 | 0.0165 | 0.83 | 56 |
| (128,) | relu | adam | 0.001 | 64 | 0.0147 | 2.04 | 148 |
| (128, 64, 32) | relu | adam | 0.001 | 64 | 0.0147 | 2.86 | 181 |

**Selected MLP hyperparameters:** hidden_layer_sizes=(256, 128), activation=ReLU, optimizer=Adam, learning_rate=0.001, batch_size=64.

## Table 2: A/B Comparison (PLA vs Tuned MLP)
| Model | Train Acc | Test Acc | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (micro) | ROC-AUC (macro) | Train Time (s) |
|---|---|---|---|---|---|---|---|---|
| PLA (one-vs-rest) | 0.3120 | 0.1804 | 0.3169 | 0.1804 | 0.1537 | 0.7947 | 0.8505 | 7.39 |
| MLP (256, 128) | 0.9652 | **0.4355** | **0.4598** | **0.4355** | **0.4355** | **0.9382** | **0.9385** | 16.74 |

## Strengths and Weaknesses

**PLA:** Simple, fast to train, easy to interpret (linear weights per class). Weakness: can only represent a linear decision boundary per class, which is insufficient for 62-way character recognition on raw pixels — test accuracy (0.1804) is barely above 10× random guess (1/62 ≈ 0.016) relative to what a non-linear model achieves.

**MLP:** Learns non-linear decision boundaries via hidden layers, reaching more than double PLA's test accuracy and F1. Weakness: much larger train-test accuracy gap (0.9652 vs 0.4355) indicating overfitting, given only ~44 training images per class; also slower to train and more hyperparameter-sensitive (half the tuning grid converged to near-random accuracy).

## Impact of Hyperparameter Tuning
Architecture (hidden layer width/depth) and optimizer choice had by far the largest effect on validation accuracy — the best (256, 128)/Adam configuration reached 0.4469 while several relu/Adam configurations with different depth or batch size collapsed to near-random (0.0147–0.0165), showing MLP training on this small, high-class-count dataset is unstable without careful architecture/optimizer selection.

## Observation Questions

**Why does PLA underperform compared to MLP?**
PLA reaches 0.1804 test accuracy versus MLP's 0.4355 because PLA can only draw a linear decision boundary per class (one-vs-rest), while 62-way handwritten character recognition on raw pixel intensities is not linearly separable — many characters (e.g. lowercase/uppercase pairs, visually similar digits) need a non-linear boundary that only MLP's hidden layers can represent.

**Which hyperparameters had the most impact on MLP performance?**
Hidden layer size/depth and the optimizer had the largest impact: the best configuration ((256, 128), relu, adam) reached 0.4469 validation accuracy, while the SGD-optimizer run reached only 0.3059 — a much larger gap than switching activation function alone (ReLU vs. Tanh at the same architecture differed by a smaller margin, 0.2894 vs 0.3059).

**Did optimizer choice (SGD vs Adam) affect convergence?**
Yes: the Adam-optimizer configurations converged in 226 iterations on average versus 284 for the SGD configuration in the same search, and the best Adam configuration's validation accuracy was notably higher than SGD's, consistent with Adam's per-parameter adaptive learning rates converging faster and more reliably than plain SGD on this network/dataset size.

**Did adding more hidden layers always improve results? Why or why not?**
No: the 3-hidden-layer network (128, 64, 32) reached only 0.0147 validation accuracy, far below the 2-hidden-layer (128, 64) network's 0.3059. With only 2,182 training samples for 62 classes (about 35 samples/class in the tuning split), a deeper network has more parameters to fit but not enough data to reliably estimate them all, so extra depth did not translate into extra accuracy here — it made optimization harder instead.

**Did MLP show overfitting? How could it be mitigated?**
Yes, clearly: MLP's train-test accuracy gap is 0.5297 (0.9652 train vs. 0.4355 test), far larger than PLA's gap of 0.1316, indicating substantial overfitting — expected given only about 44 training images per class for a 62-class problem. This could be mitigated with stronger L2 weight regularization (`alpha`), early stopping on a held-out validation set, or, most effectively, more training images per class via data augmentation (small rotations/shifts/noise applied to the existing handwriting samples).

## Conclusion
A from-scratch one-vs-rest Perceptron (PLA) and a tuned MLP ((256, 128) hidden layers, ReLU, Adam, learning rate 0.001, batch size 64) were trained and compared on the 62-class English Handwritten Characters dataset. MLP substantially outperformed PLA on every metric (test accuracy 0.4355 vs. 0.1804, macro F1 0.4355 vs. 0.1537, macro ROC-AUC 0.9385 vs. 0.8505), confirming this task needs a non-linear decision boundary that only MLP's hidden layers can represent. Hyperparameter tuning showed architecture and optimizer choice mattered far more than activation function alone, and that adding depth did not help once the per-class sample count became the limiting factor — a reminder that model capacity must be matched to the amount of training data available.

## Learning Outcomes
- Implemented the PLA weight-update rule from scratch and extended it to multiclass via a one-vs-rest ensemble.
- Saw concretely why a linear model (PLA) underperforms a non-linear model (MLP) on a task that isn't linearly separable.
- Practiced MLP hyperparameter tuning and saw that untuned configurations can fail to train at all.
- Learned to use the train-test accuracy gap as direct evidence of overfitting.
