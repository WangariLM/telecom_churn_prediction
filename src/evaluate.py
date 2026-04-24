
# =============================================================================
# evaluate.py
# Produces all evaluation metrics and plots for the trained model.
# Covers confusion matrix, classification report, ROC AUC curve,
# precision recall curve, and full SHAP interpretability analysis.
# =============================================================================

import logging
import numpy as np
import pandas as pd
import os
import sys
import joblib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

import shap

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Set consistent plot style across all figures
sns.set_theme(style="whitegrid", palette="husl")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_figure(fig, filename: str) -> None:
    """
    Save a matplotlib figure to the reports/figures directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filename : str
        Filename including extension e.g. confusion_matrix.png

    Returns
    -------
    None
    """
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    path = os.path.join(config.FIGURES_DIR, filename)
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    logger.info(f"Figure saved: {path}")
    plt.close(fig)


def get_predictions(model, X_test: np.ndarray):
    """
    Generate both class predictions and probability predictions.

    Parameters
    ----------
    model : fitted Pipeline
        Trained model pipeline.
    X_test : np.ndarray
        Processed test features.

    Returns
    -------
    y_pred : np.ndarray
        Binary class predictions using default threshold 0.5
    y_prob : np.ndarray
        Probability of churn for each customer.
        Used for ROC AUC and threshold analysis.
    """
    logger.info("Generating predictions on test set")

    # predict() uses threshold 0.5 by default
    y_pred = model.predict(X_test)

    # predict_proba() returns probabilities for both classes
    # [:, 1] selects the probability of class 1 (churn)
    y_prob = model.predict_proba(X_test)[:, 1]

    logger.info(f"Predictions generated for {len(y_pred)} customers")
    logger.info(f"Predicted churn rate: {y_pred.mean():.2%}")

    return y_pred, y_prob


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(y_test, y_pred) -> dict:
    """
    Plot and save the confusion matrix.

    Also logs the business interpretation of each cell:
        TN: correctly identified loyal customers
        FP: loyal customers incorrectly flagged for retention offers
        FN: churners who slipped through undetected (most costly mistake)
        TP: churners correctly identified for intervention

    Parameters
    ----------
    y_test : pd.Series
        True labels from test set.
    y_pred : np.ndarray
        Predicted labels from model.

    Returns
    -------
    dict
        Dictionary containing TN, FP, FN, TP values.
    """
    logger.info("Plotting confusion matrix")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    logger.info("Confusion matrix results:")
    logger.info(f"  True Negatives  (correctly identified loyal)   : {tn}")
    logger.info(f"  False Positives (loyal flagged as churner)      : {fp}")
    logger.info(f"  False Negatives (churners missed by model)      : {fn}")
    logger.info(f"  True Positives  (churners correctly identified) : {tp}")

    # Plot
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Stay", "Predicted Churn"],
        yticklabels=["Actually Stay", "Actually Churn"],
        ax=ax,
        linewidths=0.5
    )

    ax.set_title("Confusion Matrix — Telecom Churn Prediction", fontsize=14, pad=15)
    ax.set_ylabel("Actual Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    # Add business interpretation as text below the plot
    interpretation = (
        f"True Negatives: {tn} | False Positives: {fp} | "
        f"False Negatives: {fn} | True Positives: {tp}"
    )
    fig.text(
        0.5, -0.02, interpretation,
        ha="center", fontsize=10,
        style="italic", color="gray"
    )

    save_figure(fig, "confusion_matrix.png")

    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


# =============================================================================
# CLASSIFICATION REPORT
# =============================================================================

def log_classification_report(y_test, y_pred) -> dict:
    """
    Generate and log the full classification report.

    Parameters
    ----------
    y_test : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    dict
        Classification report as dictionary for saving to results file.
    """
    logger.info("Classification report:")

    report = classification_report(
        y_test, y_pred,
        target_names=["Stayed (0)", "Churned (1)"],
        output_dict=False
    )

    report_dict = classification_report(
        y_test, y_pred,
        target_names=["Stayed (0)", "Churned (1)"],
        output_dict=True
    )

    # Print full report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    print("=" * 60)

    # Log key metrics for churners specifically
    churn_precision = report_dict["Churned (1)"]["precision"]
    churn_recall    = report_dict["Churned (1)"]["recall"]
    churn_f1        = report_dict["Churned (1)"]["f1-score"]

    logger.info(f"Churn class metrics:")
    logger.info(f"  Precision : {churn_precision:.4f}")
    logger.info(f"  Recall    : {churn_recall:.4f}")
    logger.info(f"  F1 Score  : {churn_f1:.4f}")

    return report_dict


# =============================================================================
# ROC AUC CURVE
# =============================================================================

def plot_roc_curve(y_test, y_prob) -> float:
    """
    Plot and save the ROC AUC curve.

    The ROC curve plots true positive rate (recall) against false
    positive rate across all possible classification thresholds.
    The AUC summarises this as a single number between 0.5 and 1.0.

    Parameters
    ----------
    y_test : pd.Series
        True labels.
    y_prob : np.ndarray
        Predicted churn probabilities.

    Returns
    -------
    float
        ROC AUC score.
    """
    logger.info("Plotting ROC AUC curve")

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    logger.info(f"ROC AUC Score: {auc_score:.4f}")

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

    # Plot ROC curve
    ax.plot(
        fpr, tpr,
        color="steelblue",
        linewidth=2,
        label=f"Logistic Regression (AUC = {auc_score:.4f})"
    )

    # Plot random baseline
    ax.plot(
        [0, 1], [0, 1],
        color="gray",
        linewidth=1,
        linestyle="--",
        label="Random Classifier (AUC = 0.50)"
    )

    # Highlight the point closest to perfect classification
    optimal_idx = np.argmax(tpr - fpr)
    ax.scatter(
        fpr[optimal_idx], tpr[optimal_idx],
        color="red", s=100, zorder=5,
        label=f"Optimal threshold = {thresholds[optimal_idx]:.2f}"
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC AUC Curve — Telecom Churn Prediction", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    save_figure(fig, "roc_curve.png")

    return auc_score


# =============================================================================
# PRECISION RECALL CURVE
# =============================================================================

def plot_precision_recall_curve(y_test, y_prob) -> float:
    """
    Plot and save the precision recall curve.

    More informative than ROC for imbalanced datasets.
    Shows the tradeoff between precision and recall across thresholds.
    The business uses this to choose the operating threshold based
    on the cost of false positives vs false negatives.

    Parameters
    ----------
    y_test : pd.Series
        True labels.
    y_prob : np.ndarray
        Predicted churn probabilities.

    Returns
    -------
    float
        Average precision score (area under precision recall curve).
    """
    logger.info("Plotting precision recall curve")

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    # Baseline is the churn rate in the test set
    baseline = y_test.mean()

    logger.info(f"Average Precision Score: {avg_precision:.4f}")
    logger.info(f"Baseline (churn rate)  : {baseline:.4f}")

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

    ax.plot(
        recall, precision,
        color="steelblue",
        linewidth=2,
        label=f"Logistic Regression (AP = {avg_precision:.4f})"
    )

    # Baseline — a random classifier would achieve precision = churn rate
    ax.axhline(
        y=baseline,
        color="gray",
        linewidth=1,
        linestyle="--",
        label=f"Random Classifier (AP = {baseline:.4f})"
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision Recall Curve — Telecom Churn Prediction", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    save_figure(fig, "precision_recall_curve.png")

    return avg_precision


# =============================================================================
# SHAP ANALYSIS
# =============================================================================

def get_shap_values(model, X_test_processed: np.ndarray):
    """
    Calculate SHAP values for the test set.

    Uses LinearExplainer which is optimised for linear models like
    logistic regression. It is much faster than the general
    KernelExplainer for this use case.

    Parameters
    ----------
    model : fitted Pipeline
        Trained model pipeline containing logistic regression.
    X_test_processed : np.ndarray
        Processed test features — same array used for predictions.

    Returns
    -------
    shap_values : np.ndarray
        SHAP values for each feature and each customer.
        Shape: (n_customers, n_features)
    explainer : shap.LinearExplainer
        Fitted SHAP explainer object.
    """
    logger.info("Calculating SHAP values")

    # Extract the logistic regression step from the pipeline
    lr_model = model.named_steps["logisticregression"]

    # LinearExplainer is optimised for linear models
    explainer = shap.LinearExplainer(
        lr_model,
        X_test_processed,
        feature_perturbation="interventional"
    )

    shap_values = explainer.shap_values(X_test_processed)

    logger.info(f"SHAP values calculated: {shap_values.shape}")

    return shap_values, explainer


def get_feature_names(pipeline, preprocessing_pipeline) -> list:
    """
    Extract feature names from the preprocessing pipeline.

    After OneHotEncoding the feature names become complex.
    This function retrieves the exact names in the correct order
    so SHAP plots have meaningful labels.

    Parameters
    ----------
    pipeline : fitted ImbPipeline
        Trained model pipeline.
    preprocessing_pipeline : fitted Pipeline
        Preprocessing pipeline with ColumnTransformer.

    Returns
    -------
    list
        List of feature names in the same order as the processed array.
    """
    logger.info("Extracting feature names from preprocessing pipeline")

    try:
        feature_names = (
            preprocessing_pipeline
            .named_steps["preprocessor"]
            .get_feature_names_out()
            .tolist()
        )
        logger.info(f"Retrieved {len(feature_names)} feature names")
        return feature_names
    except Exception as e:
        logger.warning(f"Could not retrieve feature names: {e}")
        logger.warning("Using generic feature names instead")
        return [f"feature_{i}" for i in range(40)]


def plot_shap_summary(shap_values, X_test_processed, feature_names) -> None:
    """
    Plot and save the SHAP summary plot.

    Shows all features ranked by their average absolute SHAP value.
    Each dot represents one customer coloured by the feature value.
    Red = high feature value, Blue = low feature value.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values for all customers and features.
    X_test_processed : np.ndarray
        Processed test features.
    feature_names : list
        Names of all features.

    Returns
    -------
    None
    """
    logger.info("Plotting SHAP summary plot")

    fig, ax = plt.subplots(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X_test_processed,
        feature_names=feature_names,
        show=False,
        max_display=20,      # Show top 20 most important features
        plot_size=None
    )

    plt.title("SHAP Summary Plot — Feature Importance", fontsize=14, pad=15)
    plt.tight_layout()

    save_figure(plt.gcf(), "shap_summary.png")
    logger.info("SHAP summary plot saved")


def plot_shap_waterfall(shap_values, X_test_processed,
                        feature_names, explainer,
                        customer_index: int = 0) -> None:
    """
    Plot and save the SHAP waterfall plot for one specific customer.

    Shows exactly how each feature pushed the prediction up or down
    from the baseline expected value. This is the local explanation
    used by the retention team to understand why a specific customer
    was flagged.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values for all customers.
    X_test_processed : np.ndarray
        Processed test features.
    feature_names : list
        Names of all features.
    explainer : shap.LinearExplainer
        Fitted SHAP explainer.
    customer_index : int
        Index of the customer to explain. Default is 0 (first customer).

    Returns
    -------
    None
    """
    logger.info(f"Plotting SHAP waterfall plot for customer index {customer_index}")

    # Create SHAP Explanation object for the waterfall plot
    explanation = shap.Explanation(
        values=shap_values[customer_index],
        base_values=explainer.expected_value,
        data=X_test_processed[customer_index],
        feature_names=feature_names
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    shap.waterfall_plot(
        explanation,
        max_display=15,
        show=False
    )

    plt.title(
        f"SHAP Waterfall Plot — Customer {customer_index}",
        fontsize=14, pad=15
    )
    plt.tight_layout()

    save_figure(plt.gcf(), f"shap_waterfall_customer_{customer_index}.png")
    logger.info("SHAP waterfall plot saved")


def plot_shap_dependence(shap_values, X_test_processed,
                         feature_names) -> None:
    """
    Plot and save SHAP dependence plots for the top 3 features.

    Shows how the effect of one feature on churn probability
    changes across its range. Reveals non-linear relationships
    and interaction effects.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values for all customers.
    X_test_processed : np.ndarray
        Processed test features.
    feature_names : list
        Names of all features.

    Returns
    -------
    None
    """
    logger.info("Plotting SHAP dependence plots for top 3 features")

    # Find top 3 features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top3_indices = np.argsort(mean_abs_shap)[::-1][:3]
    top3_features = [feature_names[i] for i in top3_indices]

    logger.info(f"Top 3 features: {top3_features}")

    for i, feature in enumerate(top3_features):
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

        shap.dependence_plot(
            feature,
            shap_values,
            X_test_processed,
            feature_names=feature_names,
            ax=ax,
            show=False
        )

        ax.set_title(
            f"SHAP Dependence Plot — {feature}",
            fontsize=14, pad=15
        )
        plt.tight_layout()

        save_figure(fig, f"shap_dependence_{feature.replace(' ', '_')}.png")

    logger.info("SHAP dependence plots saved")


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(metrics: dict) -> None:
    """
    Save all evaluation metrics to a JSON file for future reference.

    Having results in a file means we never lose them between sessions.
    They can also be read programmatically for reporting.

    Parameters
    ----------
    metrics : dict
        Dictionary containing all evaluation metrics.

    Returns
    -------
    None
    """
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    results_path = os.path.join(config.REPORTS_DIR, "evaluation_results.json")

    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Evaluation results saved to: {results_path}")


# =============================================================================
# MASTER EVALUATION FUNCTION
# =============================================================================

def run_evaluation_pipeline(
    model,
    X_test_processed: np.ndarray,
    y_test,
    preprocessing_pipeline,
    customer_index: int = 0
) -> dict:
    """
    Master function that runs the complete evaluation pipeline.

    This is the main function called by main.py.

    Steps:
        1. Generate predictions
        2. Plot confusion matrix
        3. Log classification report
        4. Plot ROC AUC curve
        5. Plot precision recall curve
        6. Calculate SHAP values
        7. Plot SHAP summary
        8. Plot SHAP waterfall for one customer
        9. Plot SHAP dependence plots for top 3 features
        10. Save all metrics to JSON

    Parameters
    ----------
    model : fitted Pipeline
        Trained model pipeline.
    X_test_processed : np.ndarray
        Processed test features.
    y_test : pd.Series
        True test labels.
    preprocessing_pipeline : Pipeline
        Fitted preprocessing pipeline for feature name extraction.
    customer_index : int
        Customer index to use for waterfall plot. Default 0.

    Returns
    -------
    dict
        All evaluation metrics in one dictionary.
    """
    logger.info("=" * 60)
    logger.info("STARTING EVALUATION PIPELINE")
    logger.info("=" * 60)

    # Step 1: Generate predictions
    y_pred, y_prob = get_predictions(model, X_test_processed)

    # Step 2: Confusion matrix
    cm_results = plot_confusion_matrix(y_test, y_pred)

    # Step 3: Classification report
    report_dict = log_classification_report(y_test, y_pred)

    # Step 4: ROC AUC curve
    auc_score = plot_roc_curve(y_test, y_prob)

    # Step 5: Precision recall curve
    avg_precision = plot_precision_recall_curve(y_test, y_prob)

    # Step 6: Get feature names
    feature_names = get_feature_names(model, preprocessing_pipeline)

    # Step 7: Calculate SHAP values
    shap_values, explainer = get_shap_values(model, X_test_processed)

    # Step 8: SHAP summary plot
    plot_shap_summary(shap_values, X_test_processed, feature_names)

    # Step 9: SHAP waterfall plot
    plot_shap_waterfall(
        shap_values, X_test_processed,
        feature_names, explainer,
        customer_index
    )

    # Step 10: SHAP dependence plots
    plot_shap_dependence(shap_values, X_test_processed, feature_names)

    # Compile all metrics
    metrics = {
        "roc_auc_score"       : round(auc_score, 4),
        "average_precision"   : round(avg_precision, 4),
        "confusion_matrix"    : cm_results,
        "classification_report": report_dict
    }

    # Save results to JSON
    save_results(metrics)

    logger.info("=" * 60)
    logger.info("EVALUATION PIPELINE COMPLETE")
    logger.info(f"ROC AUC Score     : {auc_score:.4f}")
    logger.info(f"Average Precision : {avg_precision:.4f}")
    logger.info("All plots saved to reports/figures/")
    logger.info("=" * 60)

    return metrics
