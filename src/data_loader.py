
# =============================================================================
# data_loader.py
# Responsible for loading raw data and performing initial cleaning.
# No feature engineering happens here — just getting data into a clean state.
# Version 2: Added duplicate removal step
# =============================================================================

import logging
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw CSV data from disk.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file. Defaults to path in config.

    Returns
    -------
    pd.DataFrame
        Raw dataframe exactly as it appears in the CSV.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    """
    logger.info(f"Loading raw data from: {filepath}")

    if not os.path.exists(filepath):
        logger.error(f"Data file not found at: {filepath}")
        raise FileNotFoundError(f"No file found at {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Raw data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that carry no predictive value.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with unnecessary columns removed.
    """
    logger.info(f"Dropping columns: {config.DROP_COLUMNS}")

    cols_to_drop = [col for col in config.DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    logger.info(f"Remaining columns: {df.shape[1]}")
    return df


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert TotalCharges from string to float.

    TotalCharges is stored as object dtype because some rows contain
    blank spaces instead of numbers. These correspond to new customers
    with zero tenure who have never been billed. We replace blanks
    with 0 since that is the business correct value.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with TotalCharges as object dtype.

    Returns
    -------
    pd.DataFrame
        Dataframe with TotalCharges as float64.
    """
    logger.info("Fixing TotalCharges dtype from object to float")

    blank_count = df[df["TotalCharges"].str.strip() == ""]["TotalCharges"].count()
    logger.info(f"Found {blank_count} blank TotalCharges values — replacing with 0")

    df["TotalCharges"] = df["TotalCharges"].str.strip()
    df["TotalCharges"] = df["TotalCharges"].replace("", "0")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    remaining_nulls = df["TotalCharges"].isnull().sum()
    if remaining_nulls > 0:
        logger.warning(f"{remaining_nulls} NaN values remain in TotalCharges after conversion")
        logger.warning("Filling remaining NaN values with median")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    else:
        logger.info("TotalCharges conversion successful — no NaN values remaining")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataframe.

    Duplicate rows are customers with identical values across all columns.
    In a real customer dataset this almost certainly indicates data entry
    errors rather than genuinely identical customers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that may contain duplicate rows.

    Returns
    -------
    pd.DataFrame
        Dataframe with duplicate rows removed.
        Only the first occurrence of each duplicate is kept.
    """
    before = df.shape[0]
    df = df.drop_duplicates(keep="first")
    after = df.shape[0]
    removed = before - after

    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows: {before} -> {after} rows")
    else:
        logger.info("No duplicate rows found")

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert target column Churn from Yes/No strings to binary 1/0.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with Churn as object dtype containing Yes/No.

    Returns
    -------
    pd.DataFrame
        Dataframe with Churn as int64 containing 1/0.
    """
    logger.info("Encoding target column: Yes -> 1, No -> 0")

    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].map({"Yes": 1, "No": 0})

    unique_values = df[config.TARGET_COLUMN].unique()
    null_count = df[config.TARGET_COLUMN].isnull().sum()

    logger.info(f"Target column unique values after encoding: {unique_values}")
    logger.info(f"Churn rate: {df[config.TARGET_COLUMN].mean():.2%}")

    if null_count > 0:
        logger.error(f"Found {null_count} NaN values in target after encoding")
        raise ValueError(f"Target encoding failed — {null_count} unexpected values found")

    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Run basic validation checks on the cleaned dataframe.
    Logs warnings for any issues found but does not stop the pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe to validate.

    Returns
    -------
    None
    """
    logger.info("Running data validation checks")

    # Check 1: Expected number of rows
    if df.shape[0] < 7000:
        logger.warning(f"Fewer rows than expected: {df.shape[0]} (expected ~7043)")

    # Check 2: Check for missing values
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    if len(cols_with_nulls) > 0:
        logger.warning("Columns with missing values after cleaning:")
        for col, count in cols_with_nulls.items():
            logger.warning(f"  {col}: {count} missing values")
    else:
        logger.info("No missing values found after cleaning")

    # Check 3: Verify target column only contains 0 and 1
    valid_target_values = set(df[config.TARGET_COLUMN].unique())
    if not valid_target_values.issubset({0, 1}):
        logger.error(f"Target column contains unexpected values: {valid_target_values}")
    else:
        logger.info("Target column validation passed")

    # Check 4: Check for remaining duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate rows remaining")
    else:
        logger.info("No duplicate rows found")

    # Check 5: Verify TotalCharges is now numeric
    if df["TotalCharges"].dtype != "float64":
        logger.error(f"TotalCharges dtype is {df['TotalCharges'].dtype} — expected float64")
    else:
        logger.info("TotalCharges dtype validation passed")

    logger.info("Data validation complete")


def load_and_clean_data(filepath: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Master function that runs the complete data loading and cleaning pipeline.
    This is the main function called by other modules.

    Steps:
        1. Load raw CSV
        2. Drop unnecessary columns
        3. Fix TotalCharges dtype
        4. Remove duplicate rows
        5. Encode target variable
        6. Validate cleaned data

    Parameters
    ----------
    filepath : str
        Path to raw CSV file. Defaults to path in config.

    Returns
    -------
    pd.DataFrame
        Fully cleaned dataframe ready for feature engineering.
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA LOADING AND CLEANING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Load
    df = load_raw_data(filepath)

    # Step 2: Drop unnecessary columns
    df = drop_unnecessary_columns(df)

    # Step 3: Fix TotalCharges
    df = fix_total_charges(df)

    # Step 4: Remove duplicates
    df = remove_duplicates(df)

    # Step 5: Encode target
    df = encode_target(df)

    # Step 6: Validate
    validate_data(df)

    logger.info("=" * 60)
    logger.info(f"DATA LOADING COMPLETE: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info("=" * 60)

    return df
