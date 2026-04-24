
# =============================================================================
# preprocessing.py
# Builds the full sklearn preprocessing pipeline.
# Handles imputation, encoding, scaling, and train/test splitting.
# The pipeline is designed to prevent data leakage at every step.
# =============================================================================

import logging
import numpy as np
import pandas as pd
import os
import sys

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from feature_engineering import FeatureEngineeringTransformer

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


# =============================================================================
# COLUMN DEFINITIONS
# These define exactly which columns go into which pipeline
# Must match the output of FeatureEngineeringTransformer exactly
# =============================================================================

# Numerical columns — will be imputed with median then scaled
NUMERICAL_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "TotalServices",
    "SpendPerService",
    "ChargesRatio",
    "ContractRiskScore",
    "TenureContractInteraction"
]

# Binary categorical columns — will be imputed then ordinally encoded
BINARY_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "HasPremiumServices",
    "IsAutomatedPayment"
]

# SeniorCitizen is int but categorical — treat as binary
# Already 0/1 so OrdinalEncoder passes it through unchanged
SENIOR_CITIZEN_COLUMN = ["SeniorCitizen"]

# Multi-class categorical columns — will be one-hot encoded
MULTICLASS_COLUMNS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "TenureGroup"
]


def split_features_target(df: pd.DataFrame):
    """
    Separate the dataframe into features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned and engineered dataframe including target column.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix — all columns except target.
    y : pd.Series
        Target vector — Churn column only.
    """
    logger.info(f"Splitting features and target — target column: {config.TARGET_COLUMN}")

    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Target distribution — 0: {(y==0).sum()}, 1: {(y==1).sum()}")

    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series):
    """
    Split data into training and test sets.

    Uses stratified splitting to ensure both sets have the same
    class distribution as the full dataset. This is critical for
    imbalanced datasets like churn where random splitting could
    produce an unrepresentative test set.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.

    Returns
    -------
    X_train : pd.DataFrame
    X_test  : pd.DataFrame
    y_train : pd.Series
    y_test  : pd.Series
    """
    logger.info(
        f"Splitting data — "
        f"test size: {config.TEST_SIZE}, "
        f"random seed: {config.RANDOM_SEED}, "
        f"stratified: True"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y        # Ensures same churn rate in both splits
    )

    logger.info(f"Training set: {X_train.shape[0]} rows")
    logger.info(f"Test set    : {X_test.shape[0]} rows")
    logger.info(
        f"Training churn rate: {y_train.mean():.2%} | "
        f"Test churn rate: {y_test.mean():.2%}"
    )

    return X_train, X_test, y_train, y_test


def build_numerical_pipeline() -> Pipeline:
    """
    Build pipeline for numerical features.

    Steps:
        1. SimpleImputer  — fills missing values with column median
                            Median is robust to outliers unlike mean
        2. StandardScaler — scales to mean=0, std=1
                            Required for logistic regression to converge properly

    Returns
    -------
    Pipeline
        Fitted-ready sklearn pipeline for numerical columns.
    """
    logger.info("Building numerical pipeline")

    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


def build_binary_pipeline() -> Pipeline:
    """
    Build pipeline for binary categorical features.

    Steps:
        1. SimpleImputer   — fills missing values with most frequent value
        2. OrdinalEncoder  — encodes two-value categories as 0 and 1
                             Safe for binary columns — no dummy variable trap

    Returns
    -------
    Pipeline
        Fitted-ready sklearn pipeline for binary categorical columns.
    """
    logger.info("Building binary categorical pipeline")

    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1    # Unknown categories get -1 in production
        ))
    ])


def build_multiclass_pipeline() -> Pipeline:
    """
    Build pipeline for multi-class categorical features.

    Steps:
        1. SimpleImputer  — fills missing values with most frequent value
        2. OneHotEncoder  — creates binary column per category
                            drop=first prevents dummy variable trap
                            handle_unknown=ignore is safe for production

    Returns
    -------
    Pipeline
        Fitted-ready sklearn pipeline for multi-class categorical columns.
    """
    logger.info("Building multi-class categorical pipeline")

    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            drop="first",               # Prevents dummy variable trap
            handle_unknown="ignore",    # Unknown categories become all zeros
            sparse_output=False         # Return dense array not sparse matrix
        ))
    ])


def build_preprocessor() -> ColumnTransformer:
    """
    Build the full preprocessing ColumnTransformer.

    Combines all three sub-pipelines and applies each to its
    designated set of columns. ColumnTransformer handles the
    column routing and concatenates the results automatically.

    Returns
    -------
    ColumnTransformer
        Full preprocessing transformer ready to be fitted.
    """
    logger.info("Building full ColumnTransformer preprocessor")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                build_numerical_pipeline(),
                NUMERICAL_COLUMNS
            ),
            (
                "binary",
                build_binary_pipeline(),
                BINARY_COLUMNS + SENIOR_CITIZEN_COLUMN
            ),
            (
                "multiclass",
                build_multiclass_pipeline(),
                MULTICLASS_COLUMNS
            )
        ],
        remainder="drop",       # Drop any columns not explicitly listed
        verbose_feature_names_out=False   # Cleaner feature names
    )

    return preprocessor


def build_full_pipeline() -> Pipeline:
    """
    Build the complete end-to-end pipeline including feature engineering
    and preprocessing.

    This pipeline takes raw cleaned data as input and outputs a fully
    preprocessed numerical matrix ready for model training.

    Note: SMOTE is NOT included here because it must only be applied
    to training data inside the cross validation loop. Adding SMOTE
    here would apply it to validation folds which would cause data
    leakage.

    Pipeline steps:
        1. FeatureEngineeringTransformer — creates 8 new features
        2. ColumnTransformer             — encodes and scales all features

    Returns
    -------
    Pipeline
        Complete preprocessing pipeline ready to be fitted on training data.
    """
    logger.info("Building full end-to-end preprocessing pipeline")

    pipeline = Pipeline(steps=[
        ("feature_engineering", FeatureEngineeringTransformer()),
        ("preprocessor", build_preprocessor())
    ])

    logger.info("Full pipeline built successfully")
    return pipeline


def fit_transform_pipeline(pipeline: Pipeline,
                           X_train: pd.DataFrame,
                           X_test: pd.DataFrame):
    """
    Fit the pipeline on training data only then transform both sets.

    This is the critical step that prevents data leakage.
    The pipeline learns its parameters (scaler means, encoder categories)
    exclusively from training data. The test set is only transformed
    using those learned parameters — never used to fit anything.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted preprocessing pipeline from build_full_pipeline().
    X_train : pd.DataFrame
        Training features. Pipeline is fitted on this.
    X_test : pd.DataFrame
        Test features. Only transformed, never fitted.

    Returns
    -------
    X_train_processed : np.ndarray
        Processed training features ready for SMOTE and model training.
    X_test_processed : np.ndarray
        Processed test features ready for final evaluation.
    pipeline : Pipeline
        Fitted pipeline — saved to disk for production use.
    """
    logger.info("Fitting pipeline on training data only")
    X_train_processed = pipeline.fit_transform(X_train)
    logger.info(f"Training data processed: {X_train_processed.shape}")

    logger.info("Transforming test data using fitted pipeline")
    X_test_processed = pipeline.transform(X_test)
    logger.info(f"Test data processed: {X_test_processed.shape}")

    return X_train_processed, X_test_processed, pipeline


def run_preprocessing_pipeline(df: pd.DataFrame):
    """
    Master function that runs the complete preprocessing pipeline.

    This is the main function called by train.py.

    Steps:
        1. Separate features and target
        2. Split into train and test sets
        3. Build the full pipeline
        4. Fit on training data, transform both sets

    Parameters
    ----------
    df : pd.DataFrame
        Fully cleaned dataframe from data_loader.load_and_clean_data().

    Returns
    -------
    X_train_processed : np.ndarray
        Processed training features.
    X_test_processed  : np.ndarray
        Processed test features.
    y_train           : pd.Series
        Training labels.
    y_test            : pd.Series
        Test labels.
    pipeline          : Pipeline
        Fitted preprocessing pipeline.
    """
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Separate features and target
    X, y = split_features_target(df)

    # Step 2: Split into train and test
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Step 3: Build the pipeline
    pipeline = build_full_pipeline()

    # Step 4: Fit on train, transform both
    X_train_processed, X_test_processed, pipeline = fit_transform_pipeline(
        pipeline, X_train, X_test
    )

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info(f"X_train_processed shape: {X_train_processed.shape}")
    logger.info(f"X_test_processed shape : {X_test_processed.shape}")
    logger.info(f"y_train shape          : {y_train.shape}")
    logger.info(f"y_test shape           : {y_test.shape}")
    logger.info("=" * 60)

    return X_train_processed, X_test_processed, y_train, y_test, pipeline
