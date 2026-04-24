
# =============================================================================
# feature_engineering.py
# Custom sklearn transformer that creates all engineered features.
# Inherits from BaseEstimator and TransformerMixin so it plugs directly
# into any sklearn Pipeline without any special handling.
# =============================================================================

import logging
import numpy as np
import pandas as pd
import os
import sys

from sklearn.base import BaseEstimator, TransformerMixin

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer that engineers all new features.

    Inheriting from BaseEstimator gives us get_params() and set_params()
    for free which are required for GridSearchCV to work correctly.

    Inheriting from TransformerMixin gives us fit_transform() for free
    which is required for sklearn Pipeline compatibility.

    This transformer is stateless — it does not learn anything from
    the training data. It just applies deterministic calculations.
    This means fit() does nothing and transform() does all the work.
    This is intentional and correct for feature engineering operations
    that are based purely on business logic and domain knowledge.

    Parameters
    ----------
    None

    Attributes
    ----------
    feature_names_out_ : list
        Names of all columns in the transformed dataframe.
        Set during fit() call.
    """

    # Define which service columns count towards TotalServices
    SERVICE_COLUMNS = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    # Define which values in service columns count as active
    # Anything not in this list (No, No phone service, No internet service)
    # is treated as inactive
    ACTIVE_SERVICE_VALUES = [
        "Yes",
        "DSL",
        "Fiber optic"
    ]

    # Define premium service columns
    PREMIUM_SERVICE_COLUMNS = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport"
    ]

    # Define automated payment methods
    AUTOMATED_PAYMENT_METHODS = [
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    # Define contract risk scores — higher means higher churn risk
    CONTRACT_RISK_MAP = {
        "Month-to-month": 3,
        "One year": 2,
        "Two year": 1
    }

    # Define tenure group boundaries and labels
    TENURE_BINS = [0, 12, 24, 48, 72]
    TENURE_LABELS = ["New", "Developing", "Established", "Loyal"]


    def fit(self, X, y=None):
        """
        Fit method required by sklearn API.

        This transformer is stateless so fit() simply validates the input
        and records feature names. No parameters are learned from data.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe. Must contain all required columns.
        y : ignored
            Not used. Present for sklearn API compatibility.

        Returns
        -------
        self
            Returns self to allow method chaining (fit().transform()).
        """
        logger.info("Fitting FeatureEngineeringTransformer")

        # Validate that all required columns are present
        self._validate_columns(X)

        # Record input feature names for reference
        self.feature_names_in_ = list(X.columns)

        logger.info("FeatureEngineeringTransformer fit complete")
        return self


    def transform(self, X, y=None):
        """
        Apply all feature engineering transformations to the input dataframe.

        Creates 8 new features on top of the existing columns:
            1. TenureGroup
            2. TotalServices
            3. SpendPerService
            4. ChargesRatio
            5. HasPremiumServices
            6. IsAutomatedPayment
            7. ContractRiskScore
            8. TenureContractInteraction

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe. Must contain all required columns.
        y : ignored
            Not used. Present for sklearn API compatibility.

        Returns
        -------
        pd.DataFrame
            Transformed dataframe with 8 additional feature columns.
            Original columns are preserved.
        """
        logger.info("Starting feature engineering transformations")

        # Always work on a copy to avoid modifying the original dataframe
        X = X.copy()

        # Apply each feature engineering step
        X = self._create_tenure_group(X)
        X = self._create_total_services(X)
        X = self._create_spend_per_service(X)
        X = self._create_charges_ratio(X)
        X = self._create_has_premium_services(X)
        X = self._create_is_automated_payment(X)
        X = self._create_contract_risk_score(X)
        X = self._create_tenure_contract_interaction(X)

        # Record output feature names
        self.feature_names_out_ = list(X.columns)

        logger.info(
            f"Feature engineering complete: "
            f"{len(self.feature_names_in_)} input features -> "
            f"{len(self.feature_names_out_)} output features"
        )

        return X


    def _validate_columns(self, X):
        """
        Validate that all required columns are present in the input dataframe.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe to validate.

        Raises
        ------
        ValueError
            If any required columns are missing from the input dataframe.
        """
        required_columns = (
            self.SERVICE_COLUMNS +
            self.PREMIUM_SERVICE_COLUMNS +
            ["PaymentMethod", "Contract", "tenure",
             "MonthlyCharges", "TotalCharges"]
        )

        # Remove duplicates from required columns list
        required_columns = list(set(required_columns))

        missing = [col for col in required_columns if col not in X.columns]

        if missing:
            logger.error(f"Missing required columns: {missing}")
            raise ValueError(
                f"FeatureEngineeringTransformer requires columns: {missing}"
            )

        logger.info("Column validation passed")


    def _create_tenure_group(self, X):
        """
        Group tenure months into meaningful business categories.

        Bins:
            0-12  months -> New
            13-24 months -> Developing
            25-48 months -> Established
            49-72 months -> Loyal

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe containing tenure column.

        Returns
        -------
        pd.DataFrame
            Dataframe with TenureGroup column added.
        """
        logger.info("Creating TenureGroup")

        X["TenureGroup"] = pd.cut(
            X["tenure"],
            bins=self.TENURE_BINS,
            labels=self.TENURE_LABELS,
            include_lowest=True
        )

        # Log distribution so we can see how customers are spread
        distribution = X["TenureGroup"].value_counts().to_dict()
        logger.info(f"TenureGroup distribution: {distribution}")

        return X


    def _create_total_services(self, X):
        """
        Count how many services each customer actively uses.

        Active means the value is Yes, DSL, or Fiber optic.
        Values like No, No phone service, No internet service
        are treated as inactive and contribute 0 to the count.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe containing all service columns.

        Returns
        -------
        pd.DataFrame
            Dataframe with TotalServices column added.
        """
        logger.info("Creating TotalServices")

        # For each service column create a temporary binary column
        # 1 if the service is active, 0 otherwise
        # Then sum across all service columns for each row
        X["TotalServices"] = sum(
            X[col].isin(self.ACTIVE_SERVICE_VALUES).astype(int)
            for col in self.SERVICE_COLUMNS
        )

        logger.info(
            f"TotalServices — "
            f"min: {X['TotalServices'].min()}, "
            f"max: {X['TotalServices'].max()}, "
            f"mean: {X['TotalServices'].mean():.2f}"
        )

        return X


    def _create_spend_per_service(self, X):
        """
        Calculate monthly spend per active service.

        Formula:
            SpendPerService = MonthlyCharges / (TotalServices + 1)

        Adding 1 to TotalServices prevents division by zero for
        customers with no active services.

        Note: TotalServices must be created before calling this method.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe. Must already contain TotalServices column.

        Returns
        -------
        pd.DataFrame
            Dataframe with SpendPerService column added.
        """
        logger.info("Creating SpendPerService")

        X["SpendPerService"] = (
            X["MonthlyCharges"] / (X["TotalServices"] + 1)
        )

        logger.info(
            f"SpendPerService — "
            f"min: {X['SpendPerService'].min():.2f}, "
            f"max: {X['SpendPerService'].max():.2f}, "
            f"mean: {X['SpendPerService'].mean():.2f}"
        )

        return X


    def _create_charges_ratio(self, X):
        """
        Calculate ratio of monthly charges to total lifetime charges.

        Formula:
            ChargesRatio = MonthlyCharges / (TotalCharges + 1)

        Adding 1 to TotalCharges prevents division by zero for brand
        new customers who have never been billed (TotalCharges = 0).

        A high ratio indicates the customer is new or was recently
        upsold to a more expensive plan — both increase churn risk.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe containing MonthlyCharges and TotalCharges.

        Returns
        -------
        pd.DataFrame
            Dataframe with ChargesRatio column added.
        """
        logger.info("Creating ChargesRatio")

        X["ChargesRatio"] = (
            X["MonthlyCharges"] / (X["TotalCharges"] + 1)
        )

        logger.info(
            f"ChargesRatio — "
            f"min: {X['ChargesRatio'].min():.4f}, "
            f"max: {X['ChargesRatio'].max():.4f}, "
            f"mean: {X['ChargesRatio'].mean():.4f}"
        )

        return X


    def _create_has_premium_services(self, X):
        """
        Create binary flag for whether customer has any premium add-ons.

        Premium services are OnlineSecurity, OnlineBackup,
        DeviceProtection, and TechSupport.

        A value of Yes in any of these columns means the customer
        has invested in premium services and is less likely to churn.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe containing premium service columns.

        Returns
        -------
        pd.DataFrame
            Dataframe with HasPremiumServices column added.
            1 = has at least one premium service
            0 = has no premium services
        """
        logger.info("Creating HasPremiumServices")

        X["HasPremiumServices"] = (
            X[self.PREMIUM_SERVICE_COLUMNS]
            .eq("Yes")
            .any(axis=1)
            .astype(int)
        )

        distribution = X["HasPremiumServices"].value_counts().to_dict()
        logger.info(f"HasPremiumServices distribution: {distribution}")

        return X


    def _create_is_automated_payment(self, X):
        """
        Create binary flag for whether customer pays automatically.

        Automated payment methods are:
            Bank transfer (automatic)
            Credit card (automatic)

        Manual payment methods are:
            Electronic check
            Mailed check

        Customers on automated payment are passively committed
        and less likely to churn.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe containing PaymentMethod column.

        Returns
        -------
        pd.DataFrame
            Dataframe with IsAutomatedPayment column added.
            1 = automated payment
            0 = manual payment
        """
        logger.info("Creating IsAutomatedPayment")

        X["IsAutomatedPayment"] = (
            X["PaymentMethod"]
            .isin(self.AUTOMATED_PAYMENT_METHODS)
            .astype(int)
        )

        distribution = X["IsAutomatedPayment"].value_counts().to_dict()
        logger.info(f"IsAutomatedPayment distribution: {distribution}")

        return X


    def _create_contract_risk_score(self, X):
        """
        Encode contract type as an ordinal churn risk score.

        Risk scores:
            Month-to-month -> 3  (highest churn risk)
            One year       -> 2  (medium churn risk)
            Two year       -> 1  (lowest churn risk)

        This encoding uses domain knowledge to impose a meaningful
        ordering on contract types rather than treating them as
        unordered categories.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe containing Contract column.

        Returns
        -------
        pd.DataFrame
            Dataframe with ContractRiskScore column added.

        Raises
        ------
        ValueError
            If any unexpected contract values are found.
        """
        logger.info("Creating ContractRiskScore")

        # Check for unexpected contract values before mapping
        unexpected = set(X["Contract"].unique()) - set(self.CONTRACT_RISK_MAP.keys())
        if unexpected:
            logger.error(f"Unexpected contract values found: {unexpected}")
            raise ValueError(f"Unknown contract types: {unexpected}")

        X["ContractRiskScore"] = X["Contract"].map(self.CONTRACT_RISK_MAP)

        distribution = X["ContractRiskScore"].value_counts().to_dict()
        logger.info(f"ContractRiskScore distribution: {distribution}")

        return X


    def _create_tenure_contract_interaction(self, X):
        """
        Create interaction feature combining tenure and contract risk.

        Formula:
            TenureContractInteraction = tenure * ContractRiskScore

        This captures the combined effect of loyalty and commitment.

        High tenure + low risk score (Two year) = very safe customer
        Low tenure  + high risk score (Month-to-month) = very risky customer

        Note: ContractRiskScore must be created before calling this method.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe. Must already contain ContractRiskScore.

        Returns
        -------
        pd.DataFrame
            Dataframe with TenureContractInteraction column added.
        """
        logger.info("Creating TenureContractInteraction")

        X["TenureContractInteraction"] = (
            X["tenure"] * X["ContractRiskScore"]
        )

        logger.info(
            f"TenureContractInteraction — "
            f"min: {X['TenureContractInteraction'].min()}, "
            f"max: {X['TenureContractInteraction'].max()}, "
            f"mean: {X['TenureContractInteraction'].mean():.2f}"
        )

        return X


    def get_feature_names_out(self):
        """
        Return names of all output features after transformation.

        Required by sklearn API for pipeline compatibility.

        Returns
        -------
        list
            List of all column names in the transformed dataframe.

        Raises
        ------
        AttributeError
            If transform() has not been called yet.
        """
        if not hasattr(self, "feature_names_out_"):
            raise AttributeError(
                "feature_names_out_ is not set. "
                "Call fit_transform() or transform() first."
            )
        return self.feature_names_out_
