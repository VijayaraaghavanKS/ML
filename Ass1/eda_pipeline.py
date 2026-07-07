import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Suppress GUI window popup
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2, mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Set seaborn theme for beautiful plots
sns.set_theme(style="whitegrid", palette="muted")

def check_data_integrity(df):
    """
    Checks data integrity including dimensions, missing values, duplicates, dtypes,
    unique value counts, and constant/zero-variance features.
    """
    num_rows, num_cols = df.shape
    dtypes = df.dtypes.to_dict()
    missing_counts = df.isnull().sum().to_dict()
    missing_rates = (df.isnull().mean() * 100).to_dict()
    duplicate_rows = int(df.duplicated().sum())
    unique_counts = df.nunique().to_dict()
    
    # Identify constant or near-constant features (zero/near-zero variance)
    constant_features = []
    for col in df.columns:
        if unique_counts[col] <= 1:
            constant_features.append(col)
            
    # Classify columns into Numerical and Categorical
    num_cols_list = []
    cat_cols_list = []
    
    for col in df.columns:
        # Heuristic: if numeric dtype and unique values > 10, consider numerical.
        # Otherwise, if it has low cardinality or object/category dtype, consider categorical.
        if pd.api.types.is_numeric_dtype(df[col]) and unique_counts[col] > 10:
            num_cols_list.append(col)
        else:
            cat_cols_list.append(col)
            
    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "dtypes": dtypes,
        "missing_counts": missing_counts,
        "missing_rates": missing_rates,
        "duplicate_rows": duplicate_rows,
        "unique_counts": unique_counts,
        "constant_features": constant_features,
        "num_cols_list": num_cols_list,
        "cat_cols_list": cat_cols_list
    }

def detect_outliers(df, num_cols):
    """
    Detects outliers using IQR, Z-score, and multivariate Mahalanobis Distance.
    """
    outliers_report = {}
    if not num_cols:
        return outliers_report

    # 1. IQR & Z-score (univariate)
    for col in num_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        # IQR method
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        
        # Z-score method
        mean = col_data.mean()
        std = col_data.std()
        if std > 0:
            z_scores = (col_data - mean) / std
            z_outliers = col_data[np.abs(z_scores) > 3]
        else:
            z_outliers = pd.Series(dtype=float)
            
        outliers_report[col] = {
            "iqr_bounds": (lower_bound, upper_bound),
            "iqr_count": len(iqr_outliers),
            "z_count": len(z_outliers)
        }

    # 2. Mahalanobis Distance (multivariate)
    # Only calculate if we have at least 2 numerical columns and enough rows
    if len(num_cols) >= 2 and len(df) > len(num_cols):
        # Drop rows with missing numerical values for Mahalanobis
        df_num = df[num_cols].dropna()
        if len(df_num) > len(num_cols):
            X = df_num.values
            mean = np.mean(X, axis=0)
            try:
                cov = np.cov(X.T)
                # Pseudo-inverse in case of multicollinearity
                inv_cov = np.linalg.pinv(cov)
                diff = X - mean
                md_squared = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
                md = np.sqrt(md_squared)
                
                # Critical value from Chi-square distribution at significance 0.001
                threshold = np.sqrt(stats.chi2.ppf(0.999, df=len(num_cols)))
                multivariate_outliers_count = int(np.sum(md > threshold))
                
                outliers_report["_multivariate_"] = {
                    "method": "Mahalanobis Distance",
                    "threshold": threshold,
                    "outliers_count": multivariate_outliers_count,
                    "total_valid_rows": len(df_num)
                }
            except Exception as e:
                outliers_report["_multivariate_"] = {"error": str(e)}

    return outliers_report

def run_statistical_tests(df, num_cols, cat_cols, target_column=None):
    """
    Performs normality tests, correlation analysis, and target-specific hypothesis tests.
    """
    tests_report = {}
    
    # 1. Normality Tests (Shapiro-Wilk)
    normality_tests = {}
    for col in num_cols:
        col_data = df[col].dropna()
        if len(col_data) < 3:
            continue
        # Shapiro-Wilk is limited to N <= 5000, so we sample if larger
        if len(col_data) > 5000:
            sample_data = col_data.sample(5000, random_state=42)
        else:
            sample_data = col_data
            
        stat, p_val = stats.shapiro(sample_data)
        normality_tests[col] = {
            "statistic": float(stat),
            "p_value": float(p_val),
            "is_normal": bool(p_val > 0.05)
        }
    tests_report["normality"] = normality_tests

    # 2. Correlation Matrices (Pearson and Spearman)
    if len(num_cols) >= 2:
        df_num = df[num_cols].dropna()
        if len(df_num) >= 3:
            pearson_corr = df_num.corr(method='pearson')
            spearman_corr = df_num.corr(method='spearman')
            tests_report["pearson_corr"] = pearson_corr
            tests_report["spearman_corr"] = spearman_corr

    # 3. Target-Specific Tests
    if target_column and target_column in df.columns:
        target_data = df[target_column]
        target_unique = target_data.nunique()
        
        # Heuristic: Is target categorical or numerical?
        is_target_cat = (target_column in cat_cols) or (pd.api.types.is_categorical_dtype(target_data)) or (target_unique <= 10)
        tests_report["target_info"] = {
            "name": target_column,
            "type": "Categorical" if is_target_cat else "Numerical",
            "unique_values": target_unique
        }

        target_tests = {}
        
        if is_target_cat:
            # Categorical Target
            # ANOVA or Kruskal-Wallis for Numerical Features
            target_groups = target_data.dropna().unique()
            if len(target_groups) >= 2:
                for col in num_cols:
                    if col == target_column:
                        continue
                    groups = []
                    for g in target_groups:
                        g_data = df[df[target_column] == g][col].dropna()
                        if len(g_data) >= 3:
                            groups.append(g_data)
                    
                    if len(groups) >= 2:
                        # ANOVA
                        try:
                            f_stat, anova_p = stats.f_oneway(*groups)
                            # Kruskal-Wallis
                            h_stat, kruskal_p = stats.kruskal(*groups)
                            target_tests[col] = {
                                "test_type": "ANOVA / Kruskal-Wallis",
                                "anova_f_stat": float(f_stat) if not np.isnan(f_stat) else None,
                                "anova_p_val": float(anova_p) if not np.isnan(anova_p) else None,
                                "kruskal_h_stat": float(h_stat) if not np.isnan(h_stat) else None,
                                "kruskal_p_val": float(kruskal_p) if not np.isnan(kruskal_p) else None,
                                "is_significant_anova": bool(anova_p < 0.05) if not np.isnan(anova_p) else False
                            }
                        except Exception as e:
                            target_tests[col] = {"error": str(e)}
            
            # Chi-square test of independence for Categorical Features
            for col in cat_cols:
                if col == target_column:
                    continue
                # Create contingency table
                contingency_table = pd.crosstab(df[col], df[target_column])
                if contingency_table.size > 0:
                    try:
                        chi2_stat, p, dof, expected = stats.chi2_contingency(contingency_table)
                        target_tests[col] = {
                            "test_type": "Chi-Square",
                            "chi2_statistic": float(chi2_stat),
                            "p_value": float(p),
                            "dof": int(dof),
                            "is_significant": bool(p < 0.05)
                        }
                    except Exception as e:
                        target_tests[col] = {"error": str(e)}
        else:
            # Numerical Target
            # F-test for regression & Correlation for Numerical Features
            for col in num_cols:
                if col == target_column:
                    continue
                df_pair = df[[col, target_column]].dropna()
                if len(df_pair) >= 5:
                    x = df_pair[col].values
                    y = df_pair[target_column].values
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        # F-test for regression significance
                        # Null hypothesis: slope = 0
                        # F = t^2
                        t_stat = slope / std_err if std_err > 0 else 0
                        f_stat = t_stat ** 2
                        target_tests[col] = {
                            "test_type": "Linear Regression Significance (F-test)",
                            "slope": float(slope),
                            "r_squared": float(r_value ** 2),
                            "f_statistic": float(f_stat),
                            "p_value": float(p_value),
                            "is_significant": bool(p_value < 0.05)
                        }
                    except Exception as e:
                        target_tests[col] = {"error": str(e)}
                        
            # ANOVA or Kruskal-Wallis for Categorical Features vs Numerical Target
            for col in cat_cols:
                if col == target_column:
                    continue
                cat_groups = df[col].dropna().unique()
                if len(cat_groups) >= 2:
                    groups = []
                    for g in cat_groups:
                        g_data = df[df[col] == g][target_column].dropna()
                        if len(g_data) >= 3:
                            groups.append(g_data)
                    if len(groups) >= 2:
                        try:
                            f_stat, anova_p = stats.f_oneway(*groups)
                            target_tests[col] = {
                                "test_type": "ANOVA (Categorical vs Numerical Target)",
                                "f_statistic": float(f_stat) if not np.isnan(f_stat) else None,
                                "p_value": float(anova_p) if not np.isnan(anova_p) else None,
                                "is_significant": bool(anova_p < 0.05) if not np.isnan(anova_p) else False
                            }
                        except Exception as e:
                            target_tests[col] = {"error": str(e)}

        tests_report["target_tests"] = target_tests

    return tests_report

def preprocess_data(df, num_cols, cat_cols, target_column=None, scaling_method="standard", imputation_method="median"):
    """
    Fits and runs a preprocessing pipeline on a Train-Test split of the dataset to avoid data leakage.
    Returns processed training/testing splits and objects.
    """
    # Create a copy
    df_clean = df.copy()
    
    # Separate features and target
    if target_column and target_column in df_clean.columns:
        y = df_clean[target_column]
        X = df_clean.drop(columns=[target_column])
    else:
        y = None
        X = df_clean

    # Update columns lists for X
    x_num_cols = [c for c in num_cols if c != target_column]
    x_cat_cols = [c for c in cat_cols if c != target_column]

    # Handle target missing values (drop them as supervised models need targets)
    if y is not None:
        valid_idx = y.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
    # Split into Train (80%) and Test (20%)
    if y is not None:
        # Determine stratify based on target class distribution
        stratify = y if (target_column in cat_cols and y.value_counts().min() >= 2) else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    else:
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        y_train, y_test = None, None

    # Preprocessing pipelines fitted on Train, transformed on Test
    
    # 1. Imputation
    num_imputer = None
    cat_imputer = None
    
    if x_num_cols:
        if imputation_method == "mean":
            num_imputer = SimpleImputer(strategy="mean")
        elif imputation_method == "median":
            num_imputer = SimpleImputer(strategy="median")
        elif imputation_method == "knn":
            num_imputer = KNNImputer(n_neighbors=5)
        else:
            num_imputer = SimpleImputer(strategy="median")
            
        X_train[x_num_cols] = num_imputer.fit_transform(X_train[x_num_cols])
        X_test[x_num_cols] = num_imputer.transform(X_test[x_num_cols])
        
    if x_cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train[x_cat_cols] = cat_imputer.fit_transform(X_train[x_cat_cols])
        X_test[x_cat_cols] = cat_imputer.transform(X_test[x_cat_cols])

    # 2. Scaling
    scaler = None
    if x_num_cols and scaling_method:
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
            
        if scaler:
            X_train[x_num_cols] = scaler.fit_transform(X_train[x_num_cols])
            X_test[x_num_cols] = scaler.transform(X_test[x_num_cols])

    # 3. Categorical Encoding (One-Hot Encoding for Nominal)
    encoder = None
    encoded_features = []
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    if x_cat_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        # Fit on training categorical variables
        train_encoded_arr = encoder.fit_transform(X_train[x_cat_cols])
        test_encoded_arr = encoder.transform(X_test[x_cat_cols])
        
        encoded_feature_names = encoder.get_feature_names_out(x_cat_cols)
        encoded_features = list(encoded_feature_names)
        
        # Create DataFrames with encoded features
        df_train_enc = pd.DataFrame(train_encoded_arr, columns=encoded_feature_names, index=X_train.index)
        df_test_enc = pd.DataFrame(test_encoded_arr, columns=encoded_feature_names, index=X_test.index)
        
        # Drop original categoricals and concatenate encoded ones
        X_train_encoded = pd.concat([X_train.drop(columns=x_cat_cols), df_train_enc], axis=1)
        X_test_encoded = pd.concat([X_test.drop(columns=x_cat_cols), df_test_enc], axis=1)

    return {
        "X_train": X_train_encoded,
        "X_test": X_test_encoded,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "encoder": encoder,
        "encoded_features": encoded_features,
        "num_cols": x_num_cols,
        "cat_cols": x_cat_cols
    }

def perform_feature_selection(preprocess_results, target_column, target_type):
    """
    Performs SelectKBest and Mutual Information to identify top features.
    """
    X_train = preprocess_results["X_train"]
    y_train = preprocess_results["y_train"]
    
    if y_train is None:
        return {}

    feature_selection_report = {}
    features = X_train.columns.tolist()
    
    # 1. Mutual Information (Non-linear relevance)
    try:
        if target_type == "Categorical":
            mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        else:
            mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
            
        mi_series = pd.Series(mi_scores, index=features).sort_values(ascending=False)
        feature_selection_report["mutual_info"] = mi_series.to_dict()
    except Exception as e:
        feature_selection_report["mutual_info_error"] = str(e)

    # 2. SelectKBest Statistical scores
    try:
        if target_type == "Categorical":
            # Use ANOVA F-classif (handles continuous input features)
            # For purely positive features we could use chi2, but f_classif is generally safe for scaled inputs
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(X_train, y_train)
            scores = selector.scores_
            p_values = selector.pvalues_
            
            sel_df = pd.DataFrame({
                "score_anova": scores,
                "p_value": p_values
            }, index=features).sort_values(by="score_anova", ascending=False)
            feature_selection_report["kbest_stats"] = sel_df.to_dict(orient="index")
        else:
            # Regression F-test
            selector = SelectKBest(score_func=f_regression, k='all')
            selector.fit(X_train, y_train)
            scores = selector.scores_
            p_values = selector.pvalues_
            
            sel_df = pd.DataFrame({
                "score_f_regression": scores,
                "p_value": p_values
            }, index=features).sort_values(by="score_f_regression", ascending=False)
            feature_selection_report["kbest_stats"] = sel_df.to_dict(orient="index")
    except Exception as e:
        feature_selection_report["kbest_error"] = str(e)

    return feature_selection_report

def generate_plots(df, num_cols, cat_cols, target_column, output_dir):
    """
    Generates and saves the complete set of required EDA plots.
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Missingness Visualizations
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("Missing Value Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "missing_heatmap.png"))
    plt.close()
    
    missing_rates = df.isnull().mean() * 100
    if missing_rates.sum() > 0:
        plt.figure(figsize=(10, 6))
        missing_rates[missing_rates > 0].sort_values().plot(kind='barh', color='coral')
        plt.title("Percentage of Missing Values per Column")
        plt.xlabel("Percentage (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "missing_bar.png"))
        plt.close()

    # 2. Univariate Numerical plots: Histograms with normal curve & KDE, Boxplots, Violinplots, QQ plots
    for col in num_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Distribution & Normality Analysis: {col}", fontsize=16, fontweight='bold')
        
        # Hist + KDE + normal overlay
        sns.histplot(col_data, kde=True, ax=axes[0, 0], color="skyblue", stat="density")
        # Overlay Normal curve
        mu, std = stats.norm.fit(col_data)
        xmin, xmax = axes[0, 0].get_xlim()
        x_axis = np.linspace(xmin, xmax, 100)
        axes[0, 0].plot(x_axis, stats.norm.pdf(x_axis, mu, std), color="crimson", linestyle="--", linewidth=2, label="Normal Dist")
        axes[0, 0].set_title("Histogram with Normal Overlay")
        axes[0, 0].legend()
        
        # Boxplot
        sns.boxplot(y=col_data, ax=axes[0, 1], color="lightgreen")
        axes[0, 1].set_title("Boxplot (Outlier Detection)")
        
        # Violinplot
        sns.violinplot(y=col_data, ax=axes[1, 0], color="orange")
        axes[1, 0].set_title("Violin Plot")
        
        # QQ-Plot
        stats.probplot(col_data, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot (Normality check)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"dist_{col}.png"))
        plt.close()

    # 3. Categorical plots: Bar counts
    for col in cat_cols:
        if col == target_column and len(cat_cols) > 5:
            # Skip excess categorical plots to save time
            continue
        unique_count = df[col].nunique()
        if unique_count > 30:  # Skip high-cardinality bar plots
            continue
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, hue=col, legend=False)
        plt.title(f"Frequency Distribution of {col}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"count_{col}.png"))
        plt.close()

    # 4. Correlation Heatmap
    if len(num_cols) >= 2:
        df_num = df[num_cols].dropna()
        if len(df_num) >= 3:
            corr = df_num.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, linewidths=.5)
            plt.title("Correlation Matrix Heatmap")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "correlation_heatmap.png"))
            plt.close()
            
            # Cluster map
            try:
                cg = sns.clustermap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
                cg.figure.suptitle("Hierarchical Clustering of Correlation Matrix", y=1.02, fontsize=14)
                cg.savefig(os.path.join(plots_dir, "correlation_clustermap.png"))
                plt.close()
            except Exception:
                pass

    # 5. Variance plot
    if num_cols:
        variances = df[num_cols].var()
        plt.figure(figsize=(10, 5))
        variances.sort_values().plot(kind='bar', color='teal')
        plt.title("Feature Variance Comparison")
        plt.ylabel("Variance")
        plt.yscale("log")  # Log scale since variances can range widely
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "variance_plot.png"))
        plt.close()

    # 6. Target-Specific Plots
    if target_column and target_column in df.columns:
        target_data = df[target_column]
        is_target_cat = (target_column in cat_cols) or (target_data.nunique() <= 10)
        
        if is_target_cat:
            # Categorical Target Grouped Analysis
            # Grouped Boxplot / Violinplot for top 5 numerical columns
            num_features = [c for c in num_cols if c != target_column][:5]
            for num_col in num_features:
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df, x=target_column, y=num_col, hue=target_column, palette="Set2", legend=False)
                plt.title(f"Distribution of {num_col} by {target_column}")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"target_grouped_box_{num_col}.png"))
                plt.close()
        else:
            # Numerical Target Analysis
            # Scatter plots for top 5 numerical features vs target
            num_features = [c for c in num_cols if c != target_column][:5]
            for num_col in num_features:
                plt.figure(figsize=(8, 6))
                sns.regplot(data=df, x=num_col, y=target_column, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
                plt.title(f"{target_column} vs {num_col} (with linear regression line)")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"target_scatter_{num_col}.png"))
                plt.close()

            # Residual plot for target vs a simple regression model on the best numerical feature
            if num_features:
                best_num = num_features[0]
                df_pair = df[[best_num, target_column]].dropna()
                if len(df_pair) > 5:
                    X_reg = df_pair[[best_num]]
                    y_reg = df_pair[target_column]
                    reg_model = LinearRegression()
                    reg_model.fit(X_reg, y_reg)
                    preds = reg_model.predict(X_reg)
                    residuals = y_reg - preds
                    
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(x=preds, y=residuals, alpha=0.7)
                    plt.axhline(0, color='red', linestyle='--')
                    plt.title(f"Residual Plot (Linear Fit: {target_column} ~ {best_num})")
                    plt.xlabel("Fitted Values")
                    plt.ylabel("Residuals")
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, "residual_plot.png"))
                    plt.close()

    # 7. PCA Plot (Dimensionality Reduction)
    # Perform PCA on imputed/scaled continuous features
    num_feat_for_pca = [c for c in num_cols if c != target_column]
    if len(num_feat_for_pca) >= 2:
        df_pca = df[num_feat_for_pca].dropna()
        if len(df_pca) > 5:
            # Scale first
            scaler = StandardScaler()
            scaled_pca_data = scaler.fit_transform(df_pca)
            
            pca = PCA(n_components=2)
            pca_res = pca.fit_transform(scaled_pca_data)
            
            plt.figure(figsize=(8, 6))
            if target_column and target_column in df.columns:
                # Align targets
                aligned_target = df.loc[df_pca.index, target_column]
                sns.scatterplot(x=pca_res[:, 0], y=pca_res[:, 1], hue=aligned_target, palette="viridis", alpha=0.8)
            else:
                sns.scatterplot(x=pca_res[:, 0], y=pca_res[:, 1], alpha=0.8)
                
            plt.title(f"2D PCA Projection (Explained Var: {pca.explained_variance_ratio_.sum()*100:.1f}%)")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "pca_2d_projection.png"))
            plt.close()

def generate_markdown_report(integrity, outliers, tests, preprocess, feat_sel, output_dir, target_column):
    """
    Compiles all EDA statistics, test outputs, and model suggestions into an eda_report.md.
    """
    report_path = os.path.join(output_dir, "eda_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Exploratory Data Analysis & Statistical Report\n\n")
        
        # 1. Executive Summary
        f.write("## 1. Executive Summary\n")
        f.write(f"- **Total Rows:** {integrity['num_rows']}\n")
        f.write(f"- **Total Columns:** {integrity['num_cols']}\n")
        f.write(f"- **Duplicate Rows:** {integrity['duplicate_rows']}\n")
        f.write(f"- **Numerical Columns Identified ({len(integrity['num_cols_list'])}):** `{', '.join(integrity['num_cols_list'])}`\n")
        f.write(f"- **Categorical Columns Identified ({len(integrity['cat_cols_list'])}):** `{', '.join(integrity['cat_cols_list'])}`\n")
        if target_column:
            f.write(f"- **Target Column:** `{target_column}` (Type: {tests.get('target_info', {}).get('type', 'Unknown')})\n")
        f.write("\n")
        
        # 2. Data Integrity Checklist
        f.write("## 2. Data Integrity\n")
        f.write("| Column Name | Data Type | Null Count | Null Rate (%) | Unique Count | Constant? |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for col in integrity['dtypes']:
            is_const = "Yes" if col in integrity['constant_features'] else "No"
            f.write(f"| `{col}` | {integrity['dtypes'][col]} | {integrity['missing_counts'][col]} | {integrity['missing_rates'][col]:.2f}% | {integrity['unique_counts'][col]} | {is_const} |\n")
        f.write("\n")
        
        # 3. Outlier Analysis
        f.write("## 3. Outlier Detection\n")
        if outliers:
            f.write("### Univariate Outliers (IQR & Z-score)\n")
            f.write("| Column Name | IQR Lower Bound | IQR Upper Bound | IQR Outliers Count | Z-Score Outliers Count (|Z|>3) |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            for col, details in outliers.items():
                if col.startswith("_"):
                    continue
                f.write(f"| `{col}` | {details['iqr_bounds'][0]:.4f} | {details['iqr_bounds'][1]:.4f} | {details['iqr_count']} | {details['z_count']} |\n")
            f.write("\n")
            
            if "_multivariate_" in outliers:
                mv = outliers["_multivariate_"]
                f.write("### Multivariate Outliers (Mahalanobis Distance)\n")
                f.write(f"- **Method:** Mahalanobis Distance (using all numerical columns)\n")
                f.write(f"- **Critical Chi-Squared Value threshold:** {mv['threshold']:.4f}\n")
                f.write(f"- **Outliers Count:** {mv['outliers_count']} (out of {mv['total_valid_rows']} complete rows)\n\n")
        else:
            f.write("No numerical columns found to analyze outliers.\n\n")

        # 4. Statistical & Normality Tests
        f.write("## 4. Statistical Testing & Normality Check\n")
        
        f.write("### Normality Checks (Shapiro-Wilk Test)\n")
        norm = tests.get("normality", {})
        if norm:
            f.write("| Column Name | W-Statistic | p-value | Is Normally Distributed (p > 0.05)? |\n")
            f.write("| --- | --- | --- | --- |\n")
            for col, res in norm.items():
                is_norm = "**Yes**" if res['is_normal'] else "No"
                f.write(f"| `{col}` | {res['statistic']:.4f} | {res['p_value']:.4e} | {is_norm} |\n")
            f.write("\n")
        else:
            f.write("No numerical features available for normality testing.\n\n")

        # Target specific hypothesis tests
        if "target_tests" in tests and tests["target_tests"]:
            f.write("### Hypothesis Testing against Target\n")
            f.write("| Feature Name | Test Performed | Metric / Score | p-value | Statistically Significant (p < 0.05)? |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            for col, res in tests["target_tests"].items():
                if "error" in res:
                    f.write(f"| `{col}` | Error | N/A | N/A | Error: {res['error']} |\n")
                    continue
                
                test_type = res["test_type"]
                sig = "**Yes**" if res.get("is_significant", res.get("is_significant_anova", False)) else "No"
                
                if "chi2_statistic" in res:
                    metric = f"Chi2 = {res['chi2_statistic']:.4f}"
                    pval = f"{res['p_value']:.4e}"
                elif "anova_f_stat" in res:
                    metric = f"F-ANOVA = {res['anova_f_stat']:.4f}"
                    pval = f"{res['anova_p_val']:.4e}"
                elif "f_statistic" in res:
                    metric = f"F-Stat = {res['f_statistic']:.4f}"
                    pval = f"{res['p_value']:.4e}"
                elif "slope" in res:
                    metric = f"Slope = {res['slope']:.4f}, R² = {res['r_squared']:.4f}"
                    pval = f"{res['p_value']:.4e}"
                else:
                    metric = "N/A"
                    pval = "N/A"
                    
                f.write(f"| `{col}` | {test_type} | {metric} | {pval} | {sig} |\n")
            f.write("\n")

        # 5. Feature Selection
        if feat_sel:
            f.write("## 5. Feature Selection Insights\n")
            if "kbest_stats" in feat_sel:
                f.write("### Supervised SelectKBest Ranking\n")
                f.write("| Feature Name | Statistical Score | p-value | Significant? |\n")
                f.write("| --- | --- | --- | --- |\n")
                for feat, val in feat_sel["kbest_stats"].items():
                    # Pick key name
                    score_name = "score_anova" if "score_anova" in val else "score_f_regression"
                    pval = val.get("p_value")
                    pval_str = f"{pval:.4e}" if pval is not None else "N/A"
                    sig = "**Yes**" if pval is not None and pval < 0.05 else "No"
                    f.write(f"| `{feat}` | {val[score_name]:.4f} | {pval_str} | {sig} |\n")
                f.write("\n")
                
            if "mutual_info" in feat_sel:
                f.write("### Mutual Information Scores (Non-Linear Relevance)\n")
                f.write("| Feature Name | MI Score (Relevance) |\n")
                f.write("| --- | --- |\n")
                for feat, score in feat_sel["mutual_info"].items():
                    f.write(f"| `{feat}` | {score:.4f} |\n")
                f.write("\n")

        # 6. Machine Learning Model Selection Recommendations
        f.write("## 6. Machine Learning Recommendations\n")
        
        target_info = tests.get("target_info", {})
        if target_info:
            target_type = target_info["type"]
            f.write(f"### Target Type: **{target_type}**\n")
            if target_type == "Categorical":
                f.write("- **Task:** Classification\n")
                f.write("- **Suggested Models:**\n")
                # Recommendations based on statistical properties
                all_normal = all(res['is_normal'] for res in norm.values()) if norm else False
                if all_normal:
                    f.write("  - **Logistic Regression or LDA:** Suitable because continuous variables are largely normally distributed.\n")
                else:
                    f.write("  - **Random Forest or XGBoost:** Strongly recommended due to non-normal distributions, categorical variables, or potential non-linear relationships.\n")
                f.write("  - **K-Nearest Neighbors (KNN) / SVM (RBF):** Bounded and scaled features are ready for distance-based models.\n")
                f.write("  - **Naive Bayes:** Useful for quick classification baselines, especially if categorical features are dominant.\n")
            else:
                f.write("- **Task:** Regression\n")
                f.write("- **Suggested Models:**\n")
                
                # Check linearity
                linear_features = 0
                if "target_tests" in tests:
                    for val in tests["target_tests"].values():
                        if val.get("test_type", "").startswith("Linear Regression") and val.get("is_significant", False):
                            linear_features += 1
                            
                if linear_features > 0:
                    f.write("  - **Ridge / Lasso / Linear Regression:** Suitable since linear relationships and significant regression F-tests were detected for several features.\n")
                else:
                    f.write("  - **Decision Trees / Random Forest / Gradient Boosting:** Better suited due to lack of strong linear correlations or presence of non-linear trends.\n")
                f.write("  - **SVR or KNN Regressor:** Since variables are standardized/normalized, distance-sensitive regressors can be safely trained.\n")
        else:
            f.write("- **Task:** Unsupervised Learning (Clustering / Dimensionality Reduction)\n")
            f.write("- **Suggested Models:**\n")
            f.write("  - **K-Means / DBSCAN / Hierarchical Clustering:** The pipeline pre-processes (scales) features appropriately for these distance-based models.\n")
            f.write("  - **Principal Component Analysis (PCA):** 2D projection has been computed and plotted under `plots/pca_2d_projection.png`.\n")
            
        f.write("\n---\n*Report compiled automatically by the Automated EDA Pipeline.*")

def run_eda_pipeline(csv_path, target_column=None, output_dir="eda_output", verbose=True):
    """
    Full-length automated EDA pipeline execution wrapper.
    """
    if verbose:
        print(f"=== Starting Automated EDA Pipeline for: {csv_path} ===")
        print(f"Target column: {target_column if target_column else 'None (Unsupervised Mode)'}")
        print(f"Output directory: {output_dir}")
        
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin1")
        
    if verbose:
        print(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 2. Check Data Integrity
    integrity = check_data_integrity(df)
    num_cols = integrity["num_cols_list"]
    cat_cols = integrity["cat_cols_list"]

    # 3. Detect Outliers
    outliers = detect_outliers(df, num_cols)

    # 4. Statistical Testing
    tests = run_statistical_tests(df, num_cols, cat_cols, target_column)

    # 5. Preprocessing & Splitting
    # Decide target class based on heuristics
    target_type = None
    if target_column and target_column in df.columns:
        t_info = tests.get("target_info", {})
        target_type = t_info.get("type")
        
    preprocess = preprocess_data(df, num_cols, cat_cols, target_column, 
                                 scaling_method="standard", imputation_method="median")

    # 6. Feature Selection
    feat_sel = {}
    if target_column:
        feat_sel = perform_feature_selection(preprocess, target_column, target_type)

    # 7. Generate Visualizations
    if verbose:
        print("Generating and saving plots...")
    generate_plots(df, num_cols, cat_cols, target_column, output_dir)

    # 8. Generate Markdown Report
    if verbose:
        print("Compiling EDA Markdown report...")
    generate_markdown_report(integrity, outliers, tests, preprocess, feat_sel, output_dir, target_column)
    
    if verbose:
        print(f"=== EDA Pipeline finished successfully! ===")
        print(f"Report location: {os.path.join(output_dir, 'eda_report.md')}")
        print(f"Plots folder: {os.path.join(output_dir, 'plots/')}\n")

    return {
        "integrity": integrity,
        "outliers": outliers,
        "tests": tests,
        "preprocess": preprocess,
        "feature_selection": feat_sel
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full length automated EDA pipeline.")
    parser.add_argument("--csv", required=True, help="Path to the CSV dataset")
    parser.add_argument("--target", default=None, help="Name of the target column (optional)")
    parser.add_argument("--output", default="eda_output", help="Output directory (default: eda_output)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)
        
    run_eda_pipeline(args.csv, args.target, args.output)
