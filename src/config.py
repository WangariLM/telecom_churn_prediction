
# =============================================================================
# config.py
# Central configuration file for the telecom churn prediction pipeline.
# All constants, paths, and settings are defined here.
# Never hardcode these values anywhere else in the codebase.
# =============================================================================

import os

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base project path — all other paths are relative to this
BASE_DIR = '/content/drive/MyDrive/telecom_churn_prediction'

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'telco_churn.csv')

# Model output path
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_pipeline.pkl')

# Complete end to end pipeline for production predictions
FULL_PIPELINE_PATH = os.path.join(MODEL_DIR, 'full_prediction_pipeline.pkl')

# Complete end to end pipeline for production predictions
FULL_PIPELINE_PATH = os.path.join(MODEL_DIR, 'full_prediction_pipeline.pkl')

# Reports and figures path
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

# Fixed random seed used everywhere — ensures identical results on every run
RANDOM_SEED = 42

# =============================================================================
# DATA SETTINGS
# =============================================================================

# The column we are predicting
TARGET_COLUMN = 'Churn'

# Column to drop — just an identifier, no predictive value
DROP_COLUMNS = ['customerID']

# Numerical features in the raw dataset
NUMERICAL_FEATURES = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

# Binary categorical features — only two possible values (Yes/No)
BINARY_FEATURES = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling'
]

# Multi-class categorical features — more than two possible values
MULTICLASS_FEATURES = [
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaymentMethod'
]

# SeniorCitizen is stored as int (0/1) but is categorical in nature
# We handle it separately from other numerical features
SENIOR_CITIZEN_FEATURE = 'SeniorCitizen'

# =============================================================================
# ENGINEERED FEATURE NAMES
# =============================================================================

# These are the new columns our feature engineering transformer will create
ENGINEERED_NUMERICAL_FEATURES = [
    'TotalServices',
    'SpendPerService',
    'ChargesRatio',
    'ContractRiskScore',
    'TenureContractInteraction'
]

ENGINEERED_BINARY_FEATURES = [
    'HasPremiumServices',
    'IsAutomatedPayment'
]

ENGINEERED_CATEGORICAL_FEATURES = [
    'TenureGroup'
]

# =============================================================================
# DATA SPLIT SETTINGS
# =============================================================================

# Proportion of data reserved for final testing — never touched during training
TEST_SIZE = 0.2

# Proportion of training data used for validation during development
VALIDATION_SIZE = 0.2

# =============================================================================
# CROSS VALIDATION SETTINGS
# =============================================================================

# Number of folds for stratified k-fold cross validation
# 5 is the production standard — balances reliability and computation time
CV_FOLDS = 5

# Primary metric used to select the best model during cross validation
# roc_auc is robust to class imbalance and measures overall discrimination
SCORING_METRIC = 'roc_auc'

# =============================================================================
# HYPERPARAMETER TUNING GRID
# =============================================================================

# C is the inverse of regularization strength (C = 1/lambda)
# Small C = strong regularization = simpler model
# Large C = weak regularization = more complex model
# We search across several orders of magnitude to find the sweet spot

HYPERPARAMETER_GRID = {
    # C values to try — covering a wide range from strong to weak regularization
    'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100],

    # Penalty type — L1 can zero out features, L2 shrinks them
    'logisticregression__penalty': ['l1', 'l2'],

    # Solver must be compatible with penalty type
    # liblinear: works with both L1 and L2, good for small datasets
    # saga: works with both L1 and L2, better for large datasets
    'logisticregression__solver': ['liblinear', 'saga']
}

# =============================================================================
# MODEL TRAINING SETTINGS
# =============================================================================

# Maximum iterations for the logistic regression solver to converge
# 1000 is generous — prevents convergence warnings on complex data
MAX_ITER = 3000

# Class weight setting
# balanced: automatically adjusts weights inversely proportional to class frequency
# This is a second line of defence against class imbalance alongside SMOTE
CLASS_WEIGHT = 'balanced'

# =============================================================================
# SMOTE SETTINGS
# =============================================================================

# Sampling strategy for SMOTE
# auto: resamples minority class to match majority class exactly
SMOTE_SAMPLING_STRATEGY = 'auto'

# Number of nearest neighbours SMOTE uses to generate synthetic samples
# 5 is the default and works well in most cases
SMOTE_K_NEIGHBOURS = 5

# =============================================================================
# EVALUATION SETTINGS
# =============================================================================

# Default classification threshold
# We expose this as a config value because the business may want to adjust it
# Lower threshold = higher recall = catch more churners but more false alarms
# Higher threshold = higher precision = fewer false alarms but miss more churners
CLASSIFICATION_THRESHOLD = 0.5

# =============================================================================
# LOGGING SETTINGS
# =============================================================================

# Log level — INFO shows progress, DEBUG shows everything
LOG_LEVEL = 'INFO'

# Log format — timestamp, module name, level, message
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# =============================================================================
# FIGURE SETTINGS
# =============================================================================

# DPI for saved figures — 300 is publication quality
FIGURE_DPI = 300

# Default figure size
FIGURE_SIZE = (10, 6)
