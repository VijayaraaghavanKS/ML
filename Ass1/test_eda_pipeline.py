import os
import shutil
import numpy as np
import pandas as pd
from eda_pipeline_redesigned import run_eda_pipeline

def create_synthetic_classification_data(file_path):
    """
    Creates a synthetic classification dataset with mixed data types,
    outliers, missing values, and correlated features.
    """
    np.random.seed(42)
    n_samples = 200
    
    # 1. Target (Binary Categorical)
    target = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    
    # 2. Continuous Features
    # Feature 1: Normal distribution (highly correlated with target)
    feat_normal = np.random.normal(loc=target * 2.0, scale=1.0, size=n_samples)
    
    # Feature 2: Skewed distribution (exponential)
    feat_skewed = np.random.exponential(scale=3.0, size=n_samples)
    
    # Feature 3: Normal distribution with heavy outliers
    feat_outliers = np.random.normal(loc=5.0, scale=2.0, size=n_samples)
    outlier_idx = np.random.choice(n_samples, size=10, replace=False)
    feat_outliers[outlier_idx] += 30.0  # Add heavy outliers
    
    # Feature 4: Zero variance (constant)
    feat_constant = np.ones(n_samples)
    
    # 3. Categorical Features
    # Feature 5: Binary category
    feat_cat_bin = np.random.choice(['Low', 'High'], size=n_samples)
    # Feature 6: Cardinality 3
    feat_cat_multi = np.random.choice(['Red', 'Green', 'Blue'], size=n_samples)
    
    # Assemble DataFrame
    df = pd.DataFrame({
        "feat_normal": feat_normal,
        "feat_skewed": feat_skewed,
        "feat_outliers": feat_outliers,
        "feat_constant": feat_constant,
        "feat_cat_bin": feat_cat_bin,
        "feat_cat_multi": feat_cat_multi,
        "target": target
    })
    
    # Introduce some missing values (approx 5% per column)
    for col in ["feat_normal", "feat_skewed", "feat_cat_bin"]:
        mask = np.random.rand(n_samples) < 0.05
        df.loc[mask, col] = np.nan
        
    df.to_csv(file_path, index=False)
    print(f"Created synthetic classification dataset at: {file_path}")

def create_synthetic_regression_data(file_path):
    """
    Creates a synthetic regression dataset.
    """
    np.random.seed(42)
    n_samples = 150
    
    # Continuous Features
    x1 = np.random.normal(loc=10.0, scale=3.0, size=n_samples)
    x2 = np.random.normal(loc=5.0, scale=1.0, size=n_samples)
    
    # Target linearly related to x1 and x2
    target = 2.5 * x1 - 4.0 * x2 + np.random.normal(loc=0.0, scale=1.0, size=n_samples)
    
    # Categorical Feature
    group = np.random.choice(['Control', 'Treatment A', 'Treatment B'], size=n_samples)
    
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "group": group,
        "target_val": target
    })
    
    df.to_csv(file_path, index=False)
    print(f"Created synthetic regression dataset at: {file_path}")

def test_pipeline():
    # Setup test directories
    test_dir = "test_run"
    os.makedirs(test_dir, exist_ok=True)
    
    class_csv = os.path.join(test_dir, "classification.csv")
    regr_csv = os.path.join(test_dir, "regression.csv")
    
    class_out = os.path.join(test_dir, "classification_eda")
    regr_out = os.path.join(test_dir, "regression_eda")
    
    # Clean previous outputs
    for path in [class_out, regr_out]:
        if os.path.exists(path):
            shutil.rmtree(path)
            
    create_synthetic_classification_data(class_csv)
    create_synthetic_regression_data(regr_csv)
    
    print("\n--- Running Classification EDA Pipeline ---")
    res_class = run_eda_pipeline(class_csv, target_column="target", output_dir=class_out, verbose=True)
    
    print("\n--- Running Regression EDA Pipeline ---")
    res_regr = run_eda_pipeline(regr_csv, target_column="target_val", output_dir=regr_out, verbose=True)
    
    # Assertions
    print("\n--- Running Assertions ---")
    
    # Check output directories exist
    assert os.path.exists(class_out), "Classification output directory missing"
    assert os.path.exists(regr_out), "Regression output directory missing"
    
    # Check reports exist
    class_report_path = os.path.join(class_out, "eda_report.md")
    regr_report_path = os.path.join(regr_out, "eda_report.md")
    assert os.path.exists(class_report_path), "Classification markdown report missing"
    assert os.path.exists(regr_report_path), "Regression markdown report missing"
    
    # Check plots exist
    expected_plots = [
        "missing_heatmap.png",
        "correlation_heatmap.png",
        "pca_2d_projection.png"
    ]
    for plot in expected_plots:
        assert os.path.exists(os.path.join(class_out, "plots", plot)), f"Missing plot: {plot} in classification output"
        assert os.path.exists(os.path.join(regr_out, "plots", plot)), f"Missing plot: {plot} in regression output"
        
    print("\nAll Assertions Passed! Pipeline is fully functional and correct.")

if __name__ == "__main__":
    test_pipeline()
