# ML Algorithms Laboratory (ICS1512)

Lab work for the Machine Learning Algorithms Laboratory course, SSN College of Engineering (M.Tech Integrated CSE, Semester V).

Nine experiments, each in its own folder: a Jupyter notebook that actually runs end to end, an observation file I copy into the physical lab notebook, and a LaTeX report compiled to PDF. A few folders also carry the raw dataset, since two of the later experiments need real downloads (UCI's Human Activity Recognition set and Kaggle's handwritten characters set) rather than something `sklearn` ships with.

## Why it's set up this way

Experiments 2 through 9 mostly reuse the same EDA pipeline from Experiment 1 (`Ass1/EDA_Pipeline.ipynb`), invoked with `%run` and a small config cell at the top of each notebook. It handles missing values, outlier checks, correlation, feature selection, and the train/test split, so each later notebook only has to set a dataset path and target column, then get on with the actual modeling. That's also why a couple of the earlier reports share an "EDA pipeline output refresh" commit in the history: running Assignment 4-6's notebooks regenerated some of the shared output the earlier assignments also read from.

## The experiments

| # | Topic | Dataset |
|---|---|---|
| 1 | Working with Python packages (NumPy, SciPy, scikit-learn, Matplotlib) + EDA pipeline | Iris |
| 2 | Naive Bayes and KNN for spam classification | Spambase (UCI) |
| 3 | Linear and regularized regression | Loan Amount |
| 4 | Logistic Regression and SVM | Spambase (UCI) |
| 5 | Decision Tree and Random Forest | Wisconsin Diagnostic Breast Cancer |
| 6 | Bagging, Boosting, and a stacked ensemble | Wisconsin Diagnostic Breast Cancer |
| 7 | PCA vs. no-PCA across 10 classifiers | Wisconsin Diagnostic Breast Cancer |
| 8 | K-Means, DBSCAN, and hierarchical clustering | Human Activity Recognition (UCI) |
| 9 | Perceptron (from scratch) vs. a tuned MLP | English Handwritten Characters (Kaggle) |

Assignments 5-7 stick with the same breast cancer dataset on purpose. It made it easy to compare how Random Forest, boosting, and PCA each behaved on the exact same data instead of reasoning about it in the abstract.

## Running any of these

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib seaborn scipy xgboost jupyter nbconvert
```

Then, from inside an assignment folder:

```bash
jupyter nbconvert --to notebook --execute --inplace Experiment_N_*.ipynb
```

Assignment 8 needs the UCI HAR dataset unzipped into `Ass8/data/`, and Assignment 9 needs the Kaggle English Handwritten Characters dataset unzipped into `Ass9/data/` (both are already committed here, so you shouldn't need to fetch them again unless you're starting from a fresh clone of just the notebook).

## Compiling a report

Each `Experiment_N_Report.tex` is self-contained and expects to sit next to its assignment's output folder for the image paths to resolve:

```bash
cd Ass7
tectonic Experiment_7_Report.tex
```

Any LaTeX engine works; I used `tectonic` because it doesn't need a system-wide TeX install.

## A note on the git history

The commit dates on this repo match each assignment's actual submission deadline, not the moment I happened to run `git commit`. My professor asked for this specifically, so seniors reviewing the repo later see it in deadline order. If you're cloning this for reference, don't read commit timestamps as a record of when the work was literally typed.
