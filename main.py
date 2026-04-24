
# =============================================================================
# main.py
# Master script that runs the complete telecom churn prediction pipeline
# from raw data to evaluation metrics and saved model.
#
# Usage:
#     python main.py
#
# This script assumes the raw data file exists at the path defined
# in config.py. All outputs are saved to the paths defined in config.py.
# =============================================================================

import os
import sys
import logging
import joblib

from sklearn.pipeline import Pipeline as SklearnPipeline

# Add src to path
SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_PATH)

import config
import data_loader
import preprocessing
import train
import evaluate
import predict

# =============================================================================
# SET UP LOGGING
# =============================================================================

logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    handlers=[
        # Log to console
        logging.StreamHandler(),
        # Log to file so we have a permanent record
        logging.FileHandler(
            os.path.join(config.BASE_DIR, "pipeline.log"),
            mode="w"
        )
    ]
)

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_full_prediction_pipeline(preprocessing_pipeline, best_model):
    """
    Combine preprocessing pipeline and trained model into one
    complete end to end pipeline for production predictions.

    The preprocessing pipeline handles feature engineering,
    encoding and scaling. The model handles the actual prediction.
    Together they take raw customer data and return churn probability.

    Parameters
    ----------
    preprocessing_pipeline : fitted Pipeline
        Preprocessing pipeline from preprocessing.py
    best_model : fitted ImbPipeline
        Best model from train.py containing SMOTE and LogisticRegression

    Returns
    -------
    full_pipeline : fitted SklearnPipeline
        Complete end to end prediction pipeline
    """
    logger.info("Building full end to end prediction pipeline")

    # Extract just the logistic regression — SMOTE not needed for prediction
    lr_model = best_model.named_steps["logisticregression"]

    full_pipeline = SklearnPipeline(steps=[
        ("preprocessing",      preprocessing_pipeline),
        ("logisticregression", lr_model)
    ])

    # Save full pipeline to disk
    joblib.dump(full_pipeline, config.FULL_PIPELINE_PATH)
    logger.info(f"Full prediction pipeline saved to: {config.FULL_PIPELINE_PATH}")

    return full_pipeline


def print_summary(metrics: dict, results_df) -> None:
    """
    Print a clean final summary of all results to the console.

    Parameters
    ----------
    metrics : dict
        Evaluation metrics from evaluate.py
    results_df : pd.DataFrame
        Batch prediction results from predict.py

    Returns
    -------
    None
    """
    print("\n")
    print("=" * 60)
    print("TELECOM CHURN PREDICTION — FINAL RESULTS SUMMARY")
    print("=" * 60)

    print("\nMODEL PERFORMANCE:")
    print(f"  ROC AUC Score     : {metrics['roc_auc_score']}")
    print(f"  Average Precision : {metrics['average_precision']}")
    print(f"  True Positives    : {metrics['confusion_matrix']['TP']}")
    print(f"  False Negatives   : {metrics['confusion_matrix']['FN']}")
    print(f"  False Positives   : {metrics['confusion_matrix']['FP']}")
    print(f"  True Negatives    : {metrics['confusion_matrix']['TN']}")

    print("\nCHURN CLASS METRICS (what matters most):")
    churn_metrics = metrics["classification_report"]["Churned (1)"]
    print(f"  Precision : {churn_metrics['precision']:.4f}")
    print(f"  Recall    : {churn_metrics['recall']:.4f}")
    print(f"  F1 Score  : {churn_metrics['f1-score']:.4f}")

    print("\nRISK CATEGORY DISTRIBUTION (test set):")
    dist = results_df["RiskCategory"].value_counts()
    for category, count in dist.items():
        pct = count / len(results_df) * 100
        print(f"  {category:<15}: {count} customers ({pct:.1f}%)")

    print("\nOUTPUT FILES:")
    print(f"  Model pipeline  : {config.FULL_PIPELINE_PATH}")
    print(f"  Evaluation JSON : {os.path.join(config.REPORTS_DIR, 'evaluation_results.json')}")
    print(f"  Predictions CSV : {os.path.join(config.REPORTS_DIR, 'predictions.csv')}")
    print(f"  Figures folder  : {config.FIGURES_DIR}")
    print(f"  Pipeline log    : {os.path.join(config.BASE_DIR, 'pipeline.log')}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Run the complete telecom churn prediction pipeline.

    Steps:
        1. Load and clean raw data
        2. Run preprocessing pipeline
        3. Train model or load from disk
        4. Build full prediction pipeline
        5. Run evaluation
        6. Run batch predictions
        7. Print final summary

    Returns
    -------
    None
    """

    logger.info("=" * 60)
    logger.info("TELECOM CHURN PREDICTION PIPELINE STARTING")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # STEP 1: Load and clean data
    # ------------------------------------------------------------------
    logger.info("STEP 1: Loading and cleaning data")
    df_clean = data_loader.load_and_clean_data()

    # ------------------------------------------------------------------
    # STEP 2: Run preprocessing pipeline
    # ------------------------------------------------------------------
    logger.info("STEP 2: Running preprocessing pipeline")
    X_train_processed, X_test_processed, y_train, y_test, pipeline = (
        preprocessing.run_preprocessing_pipeline(df_clean)
    )

    # ------------------------------------------------------------------
    # STEP 3: Train model or load from disk
    # ------------------------------------------------------------------
    logger.info("STEP 3: Training model or loading from disk")

    if os.path.exists(config.MODEL_PATH):
        logger.info("Saved model found — loading from disk")
        best_model = joblib.load(config.MODEL_PATH)
        grid_search = None
    else:
        logger.info("No saved model found — running GridSearchCV")
        best_model, grid_search = train.run_training_pipeline(
            X_train_processed,
            X_test_processed,
            y_train,
            y_test,
            pipeline
        )

    # ------------------------------------------------------------------
    # STEP 4: Build full prediction pipeline
    # ------------------------------------------------------------------
    logger.info("STEP 4: Building full prediction pipeline")
    full_pipeline = build_full_prediction_pipeline(pipeline, best_model)

    # ------------------------------------------------------------------
    # STEP 5: Run evaluation
    # ------------------------------------------------------------------
    logger.info("STEP 5: Running evaluation pipeline")
    metrics = evaluate.run_evaluation_pipeline(
        model=best_model,
        X_test_processed=X_test_processed,
        y_test=y_test,
        preprocessing_pipeline=pipeline,
        customer_index=0
    )

    # ------------------------------------------------------------------
    # STEP 6: Run batch predictions
    # ------------------------------------------------------------------
    logger.info("STEP 6: Running batch predictions")
    X, y = preprocessing.split_features_target(df_clean)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = (
        preprocessing.split_train_test(X, y)
    )

    results_df = predict.run_prediction_pipeline(
        full_pipeline, X_test_raw, y_test_raw
    )

    # ------------------------------------------------------------------
    # STEP 7: Print final summary
    # ------------------------------------------------------------------
    logger.info("STEP 7: Printing final summary")
    print_summary(metrics, results_df)

    logger.info("PIPELINE COMPLETE")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
