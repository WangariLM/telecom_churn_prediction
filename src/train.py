
# =============================================================================
# train.py
# Handles SMOTE balancing, hyperparameter tuning via GridSearchCV,
# final model training, and saving the fitted pipeline to disk.
# =============================================================================

import logging
import numpy as np
import pandas as pd
import os
import sys
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def apply_smote(X_train: np.ndarray, y_train: pd.Series):
    """
    Apply SMOTE to balance the training data.

    SMOTE creates synthetic minority class samples by interpolating
    between existing minority class samples and their nearest neighbours.
    This gives the model more diverse churner examples to learn from.

    IMPORTANT: Only ever apply SMOTE to training data.
    Never apply to validation or test data.

    Parameters
    ----------
    X_train : np.ndarray
        Processed training features from preprocessing pipeline.
    y_train : pd.Series
        Training labels — imbalanced (roughly 74% / 26% split).

    Returns
    -------
    X_train_balanced : np.ndarray
        Balanced training features with synthetic samples added.
    y_train_balanced : np.ndarray
        Balanced training labels — now 50% / 50% split.
    """
    logger.info("Applying SMOTE to balance training data")
    logger.info(
        f"Before SMOTE — "
        f"Class 0: {(y_train==0).sum()}, "
        f"Class 1: {(y_train==1).sum()}"
    )

    smote = SMOTE(
        sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
        k_neighbors=config.SMOTE_K_NEIGHBOURS,
        random_state=config.RANDOM_SEED
    )

    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    logger.info(
        f"After SMOTE — "
        f"Class 0: {(y_train_balanced==0).sum()}, "
        f"Class 1: {(y_train_balanced==1).sum()}"
    )
    logger.info(f"Training set size after SMOTE: {X_train_balanced.shape[0]} rows")

    return X_train_balanced, y_train_balanced


def build_model_pipeline() -> ImbPipeline:
    """
    Build the model pipeline that includes SMOTE and LogisticRegression.

    We use imblearn Pipeline instead of sklearn Pipeline because
    imblearn Pipeline correctly handles resamplers like SMOTE.
    It applies SMOTE only during fit() and skips it during transform()
    and predict() — exactly what we need for safe cross validation.

    Pipeline steps:
        1. SMOTE              — balances classes on training folds only
        2. LogisticRegression — the classifier

    Returns
    -------
    ImbPipeline
        Unfitted model pipeline ready for GridSearchCV.
    """
    logger.info("Building model pipeline with SMOTE and LogisticRegression")

    model_pipeline = ImbPipeline(steps=[
        (
            "smote",
            SMOTE(
                sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
                k_neighbors=config.SMOTE_K_NEIGHBOURS,
                random_state=config.RANDOM_SEED
            )
        ),
        (
            "logisticregression",
            LogisticRegression(
                class_weight=config.CLASS_WEIGHT,
                max_iter=config.MAX_ITER,
                random_state=config.RANDOM_SEED
            )
        )
    ])

    return model_pipeline


def build_cross_validator() -> StratifiedKFold:
    """
    Build the cross validator.

    StratifiedKFold maintains the same class distribution in each fold
    as the full training set. This is essential for imbalanced datasets
    because random splits might produce folds with very few churners.

    Returns
    -------
    StratifiedKFold
        Cross validator with settings from config.
    """
    logger.info(
        f"Building StratifiedKFold cross validator — "
        f"{config.CV_FOLDS} folds"
    )

    return StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_SEED
    )


def run_hyperparameter_tuning(
    X_train_balanced: np.ndarray,
    y_train_balanced: np.ndarray
) -> GridSearchCV:
    """
    Run GridSearchCV to find the best hyperparameters.

    For each combination in HYPERPARAMETER_GRID runs 5-fold
    cross validation and records the average ROC AUC score.
    Picks the combination with the highest average score.

    Note: We pass balanced data from SMOTE here because GridSearchCV
    will run cross validation on whatever data we pass it. Since we
    are using imblearn Pipeline with SMOTE inside it, SMOTE will be
    applied correctly on each training fold during cross validation.
    This means we pass the original unbalanced training data here
    and let the pipeline handle SMOTE internally.

    Parameters
    ----------
    X_train_balanced : np.ndarray
        Processed training features — NOT yet SMOTE balanced.
        SMOTE is handled inside the pipeline during cross validation.
    y_train_balanced : np.ndarray
        Training labels — NOT yet SMOTE balanced.

    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object containing best estimator,
        best parameters, and full cross validation results.
    """
    logger.info("=" * 60)
    logger.info("STARTING HYPERPARAMETER TUNING")
    logger.info(f"Parameter grid: {config.HYPERPARAMETER_GRID}")
    logger.info(
        f"Total combinations: "
        f"{len(config.HYPERPARAMETER_GRID['logisticregression__C']) * len(config.HYPERPARAMETER_GRID['logisticregression__penalty']) * len(config.HYPERPARAMETER_GRID['logisticregression__solver'])}"
    )
    logger.info(f"Scoring metric: {config.SCORING_METRIC}")
    logger.info("=" * 60)

    model_pipeline = build_model_pipeline()
    cv = build_cross_validator()

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=config.HYPERPARAMETER_GRID,
        cv=cv,
        scoring=config.SCORING_METRIC,
        n_jobs=-1,          # Use all available CPU cores
        verbose=2,          # Show progress during tuning
        refit=True,         # Refit best model on full training data
        return_train_score=True  # Track training scores for overfitting check
    )

    grid_search.fit(X_train_balanced, y_train_balanced)

    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING COMPLETE")
    logger.info(f"Best parameters : {grid_search.best_params_}")
    logger.info(f"Best ROC AUC    : {grid_search.best_score_:.4f}")
    logger.info("=" * 60)

    return grid_search


def log_cv_results(grid_search: GridSearchCV) -> None:
    """
    Log the top cross validation results for transparency.

    Shows the top 5 parameter combinations ranked by mean test score.
    Also checks for overfitting by comparing train and test scores.

    Parameters
    ----------
    grid_search : GridSearchCV
        Fitted GridSearchCV object.

    Returns
    -------
    None
    """
    results = pd.DataFrame(grid_search.cv_results_)
    results = results.sort_values("mean_test_score", ascending=False)

    logger.info("Top 5 hyperparameter combinations:")
    top5 = results[[
        "param_logisticregression__C",
        "param_logisticregression__penalty",
        "param_logisticregression__solver",
        "mean_train_score",
        "mean_test_score",
        "std_test_score"
    ]].head()

    for _, row in top5.iterrows():
        logger.info(
            f"  C={row['param_logisticregression__C']:<8} "
            f"penalty={row['param_logisticregression__penalty']:<3} "
            f"solver={row['param_logisticregression__solver']:<12} "
            f"train={row['mean_train_score']:.4f} "
            f"val={row['mean_test_score']:.4f} "
            f"std={row['std_test_score']:.4f}"
        )

    # Overfitting check
    best_train = results.iloc[0]["mean_train_score"]
    best_val = results.iloc[0]["mean_test_score"]
    gap = best_train - best_val

    logger.info(f"Overfitting check — train: {best_train:.4f}, val: {best_val:.4f}, gap: {gap:.4f}")

    if gap > 0.05:
        logger.warning(f"Potential overfitting detected — train/val gap is {gap:.4f}")
    else:
        logger.info("No significant overfitting detected")


def save_pipeline(pipeline, path: str = config.MODEL_PATH) -> None:
    """
    Save the fitted pipeline to disk using joblib.

    joblib is preferred over pickle for sklearn objects because it
    handles numpy arrays more efficiently.

    The saved pipeline includes:
        - FeatureEngineeringTransformer
        - ColumnTransformer (with fitted scaler and encoders)
        - SMOTE
        - LogisticRegression (with fitted coefficients)

    Loading this single file in production gives you a complete
    end-to-end pipeline that takes raw customer data and returns
    a churn probability.

    Parameters
    ----------
    pipeline : fitted sklearn/imblearn Pipeline
        The complete fitted pipeline to save.
    path : str
        File path to save the pipeline. Defaults to config.MODEL_PATH.

    Returns
    -------
    None
    """
    # Create models directory if it does not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    joblib.dump(pipeline, path)
    logger.info(f"Pipeline saved to: {path}")

    # Log file size so we know how large the model is
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"Saved pipeline size: {file_size_mb:.2f} MB")


def run_training_pipeline(
    X_train_processed: np.ndarray,
    X_test_processed: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessing_pipeline
):
    """
    Master function that runs the complete training pipeline.

    This is the main function called by main.py.

    Steps:
        1. Run hyperparameter tuning with cross validation
        2. Log cross validation results
        3. Extract best model
        4. Save the best model pipeline to disk

    Note: We do NOT apply SMOTE before passing to GridSearchCV.
    SMOTE is inside the imblearn Pipeline and will be applied
    correctly on each training fold during cross validation.

    Parameters
    ----------
    X_train_processed : np.ndarray
        Processed training features from preprocessing pipeline.
    X_test_processed : np.ndarray
        Processed test features from preprocessing pipeline.
    y_train : pd.Series
        Training labels.
    y_test : pd.Series
        Test labels.
    preprocessing_pipeline : Pipeline
        Fitted preprocessing pipeline — saved alongside model.

    Returns
    -------
    best_model : fitted ImbPipeline
        Best model pipeline from GridSearchCV.
    grid_search : GridSearchCV
        Full GridSearchCV results for analysis.
    """
    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Run hyperparameter tuning
    # Pass unbalanced training data — SMOTE handled inside pipeline
    grid_search = run_hyperparameter_tuning(
        X_train_processed, y_train
    )

    # Step 2: Log results
    log_cv_results(grid_search)

    # Step 3: Extract best model
    best_model = grid_search.best_estimator_
    logger.info(f"Best model extracted: {best_model}")

    # Step 4: Save the best model
    save_pipeline(best_model, config.MODEL_PATH)

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)

    return best_model, grid_search
