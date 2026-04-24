
# =============================================================================
# predict.py
# Handles all prediction logic for the trained churn model.
# Supports single customer prediction, batch prediction,
# and prediction explanation for the retention team.
# =============================================================================

import logging
import numpy as np
import pandas as pd
import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


# =============================================================================
# RISK CATEGORY DEFINITIONS
# =============================================================================

RISK_CATEGORIES = [
    (0.00, 0.30, "Low Risk",      "No action needed — customer is stable"),
    (0.30, 0.50, "Medium Risk",   "Monitor closely — schedule check-in call"),
    (0.50, 0.70, "High Risk",     "Proactive outreach — offer loyalty discount"),
    (0.70, 1.00, "Critical Risk", "Immediate intervention — escalate to retention team")
]


def load_model(model_path: str = config.MODEL_PATH):
    """
    Load the saved model pipeline from disk.

    Parameters
    ----------
    model_path : str
        Path to the saved .pkl model file.

    Returns
    -------
    pipeline : fitted Pipeline
        Complete end-to-end prediction pipeline.

    Raises
    ------
    FileNotFoundError
        If no saved model exists at the given path.
    """
    logger.info(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"No model found at: {model_path}")
        raise FileNotFoundError(
            f"No saved model found at {model_path}. "
            f"Please run train.py first."
        )

    pipeline = joblib.load(model_path)
    logger.info(f"Model loaded successfully: {type(pipeline).__name__}")

    return pipeline


def get_risk_category(probability: float) -> tuple:
    """
    Convert a churn probability into a business friendly risk category.

    Parameters
    ----------
    probability : float
        Churn probability between 0 and 1.

    Returns
    -------
    tuple
        (risk_category, recommendation) strings.
    """
    for low, high, category, recommendation in RISK_CATEGORIES:
        if low <= probability < high:
            return category, recommendation

    # Handle edge case of exactly 1.0
    return "Critical Risk", "Immediate intervention — escalate to retention team"


def validate_input(customer_data: dict) -> None:
    """
    Validate that input customer data contains all required columns.

    Parameters
    ----------
    customer_data : dict
        Raw customer data as a dictionary.

    Raises
    ------
    ValueError
        If any required columns are missing from the input.
    """
    required_columns = (
        config.NUMERICAL_FEATURES +
        config.BINARY_FEATURES +
        config.MULTICLASS_FEATURES +
        [config.SENIOR_CITIZEN_FEATURE]
    )

    missing = [col for col in required_columns if col not in customer_data]

    if missing:
        logger.error(f"Missing required fields: {missing}")
        raise ValueError(
            f"Customer data is missing required fields: {missing}"
        )

    logger.info("Input validation passed")


def predict_single_customer(
    customer_data: dict,
    model,
    threshold: float = config.CLASSIFICATION_THRESHOLD
) -> dict:
    """
    Predict churn probability for a single customer.

    Takes raw customer data exactly as it comes from the business
    system — no preprocessing required. The pipeline handles
    all feature engineering and preprocessing internally.

    Parameters
    ----------
    customer_data : dict
        Raw customer data as key-value pairs.
        Must contain all required fields.
    model : fitted Pipeline
        Loaded model pipeline.
    threshold : float
        Classification threshold. Default 0.5.
        Lower threshold catches more churners but increases false alarms.

    Returns
    -------
    dict
        Complete prediction result including:
        - churn_probability
        - churn_predicted
        - risk_category
        - recommendation
    """
    logger.info("Predicting churn for single customer")

    # Validate input
    validate_input(customer_data)

    # Convert dict to dataframe — pipeline expects a dataframe
    customer_df = pd.DataFrame([customer_data])

    # Get churn probability
    # predict_proba returns [[prob_stay, prob_churn]]
    # [:, 1] selects prob_churn
    churn_probability = model.predict_proba(customer_df)[:, 1][0]

    # Apply threshold to get binary prediction
    churn_predicted = int(churn_probability >= threshold)

    # Get risk category and recommendation
    risk_category, recommendation = get_risk_category(churn_probability)

    result = {
        "churn_probability" : round(float(churn_probability), 4),
        "churn_predicted"   : churn_predicted,
        "risk_category"     : risk_category,
        "recommendation"    : recommendation,
        "threshold_used"    : threshold
    }

    logger.info(f"Prediction complete:")
    logger.info(f"  Churn probability : {result['churn_probability']:.4f}")
    logger.info(f"  Risk category     : {result['risk_category']}")
    logger.info(f"  Recommendation    : {result['recommendation']}")

    return result


def predict_batch(
    customers_df: pd.DataFrame,
    model,
    threshold: float = config.CLASSIFICATION_THRESHOLD
) -> pd.DataFrame:
    """
    Predict churn probability for a batch of customers.

    Parameters
    ----------
    customers_df : pd.DataFrame
        Raw customer data. Each row is one customer.
        Must contain all required columns.
    model : fitted Pipeline
        Loaded model pipeline.
    threshold : float
        Classification threshold. Default 0.5.

    Returns
    -------
    pd.DataFrame
        Original dataframe with four new columns appended:
        - ChurnProbability
        - ChurnPredicted
        - RiskCategory
        - Recommendation
    """
    logger.info(f"Predicting churn for batch of {len(customers_df)} customers")

    # Work on a copy to avoid modifying the original
    results_df = customers_df.copy()

    # Get probabilities for all customers at once
    churn_probabilities = model.predict_proba(customers_df)[:, 1]

    # Apply threshold
    churn_predicted = (churn_probabilities >= threshold).astype(int)

    # Get risk categories for all customers
    risk_categories = []
    recommendations = []

    for prob in churn_probabilities:
        category, recommendation = get_risk_category(prob)
        risk_categories.append(category)
        recommendations.append(recommendation)

    # Append results to dataframe
    results_df["ChurnProbability"] = np.round(churn_probabilities, 4)
    results_df["ChurnPredicted"]   = churn_predicted
    results_df["RiskCategory"]     = risk_categories
    results_df["Recommendation"]   = recommendations

    # Log summary statistics
    logger.info(f"Batch prediction complete:")
    logger.info(f"  Total customers    : {len(results_df)}")
    logger.info(f"  Predicted churners : {churn_predicted.sum()}")
    logger.info(f"  Predicted churn rate: {churn_predicted.mean():.2%}")

    logger.info("Risk category distribution:")
    for category in ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]:
        count = (results_df["RiskCategory"] == category).sum()
        pct = count / len(results_df) * 100
        logger.info(f"  {category:<15}: {count} customers ({pct:.1f}%)")

    return results_df


def run_prediction_pipeline(
    model,
    X_test_raw: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Master function that runs batch prediction on the test set
    and produces a full results dataframe for analysis.

    This is the main function called by main.py.

    Parameters
    ----------
    model : fitted Pipeline
        Loaded model pipeline.
    X_test_raw : pd.DataFrame
        Raw unprocessed test features.
        Note: We use raw features here because the pipeline
        handles all preprocessing internally.
    y_test : pd.Series
        True test labels for comparison.

    Returns
    -------
    pd.DataFrame
        Complete results dataframe with predictions and actual labels.
    """
    logger.info("=" * 60)
    logger.info("STARTING PREDICTION PIPELINE")
    logger.info("=" * 60)

    # Run batch prediction
    results_df = predict_batch(X_test_raw, model)

    # Add actual labels for comparison
    results_df["ActualChurn"] = y_test.values

    # Add correct/incorrect flag for easy filtering
    results_df["CorrectPrediction"] = (
        results_df["ChurnPredicted"] == results_df["ActualChurn"]
    ).astype(int)

    # Save results to CSV for business use
    results_path = os.path.join(config.REPORTS_DIR, "predictions.csv")
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"Predictions saved to: {results_path}")

    logger.info("=" * 60)
    logger.info("PREDICTION PIPELINE COMPLETE")
    logger.info("=" * 60)

    return results_df
