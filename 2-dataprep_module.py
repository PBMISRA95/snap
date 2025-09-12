# dataprep_module.py (updated version)
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_and_prepare_data():
    """Load and prepare the fraud detection dataset (no splitting)"""
    df = pd.read_csv("dataset.csv")
    
    # Handle missing values in target
    target_col = "Label"
    x = df.drop(columns=[target_col])    # feature matrix
    y = df[target_col]                   # target variable
    
    # Remove rows with missing target values
    mask = ~y.isna()
    x_clean = x[mask]   # feature matrix
    y_clean = y[mask]
    
    # Categorising feature columns into numerical & categorical feature
    num_col = selector(dtype_include=["int64", "float64"])(x_clean)
    cat_col = selector(dtype_include=["object","category",'bool'])(x_clean)
    
    # Building Numeric feature processing pipeline (Imputing and scaling)
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Building Categorical feature processing pipeline (Imputing and scaling)
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Applying above preprocessing pipeline to respective columns
    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_col),
            ("cat", cat_pipe, cat_col)
        ],
        remainder="drop"
    )
    
    # Return cleaned data WITHOUT splitting
    return x_clean, y_clean, preprocess