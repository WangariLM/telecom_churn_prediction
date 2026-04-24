# Telecom Customer Churn Prediction

A production level machine learning project that predicts which telecom customers are likely to cancel their subscription. Built with logistic regression, SHAP interpretability, and a modular sklearn pipeline designed for real world deployment.

---

## The Business Problem

Telecom companies lose thousands of customers every month to competitors. Acquiring a new customer costs five to seven times more than keeping an existing one. The question every telecom business wants answered is simple — which customers are about to leave, and why?

This project builds a machine learning model that answers exactly that question. Given a customer profile the model outputs a churn probability and a risk category that the retention team can act on before it is too late.

---

## The Dataset

The IBM Telco Customer Churn dataset contains 7043 customers and 21 features covering demographics, service subscriptions, contract details, and billing information. After cleaning and removing 22 duplicate rows the final dataset used for modelling contains 7021 customers.

The dataset has a natural class imbalance with roughly 73.5% of customers staying and 26.5% churning. This reflects the real world distribution you would find in any telecom company.

Source: IBM Sample Data Sets via Kaggle

---

## Project Structure

    telecom_churn_prediction/
    |
    |-- data/
    |   |-- telco_churn.csv
    |
    |-- notebooks/
    |   |-- 01_setup_and_first_look.ipynb
    |
    |-- src/
    |   |-- config.py
    |   |-- data_loader.py
    |   |-- feature_engineering.py
    |   |-- preprocessing.py
    |   |-- train.py
    |   |-- evaluate.py
    |   |-- predict.py
    |
    |-- models/
    |   |-- logistic_regression_pipeline.pkl
    |   |-- full_prediction_pipeline.pkl
    |
    |-- reports/
    |   |-- evaluation_results.json
    |   |-- predictions.csv
    |   |-- figures/
    |       |-- confusion_matrix.png
    |       |-- roc_curve.png
    |       |-- precision_recall_curve.png
    |       |-- shap_summary.png
    |       |-- shap_waterfall_customer_0.png
    |
    |-- tests/
    |-- requirements.txt
    |-- main.py
    |-- README.md

---

## How I Built This

### Data Cleaning

The raw dataset had two issues worth addressing before any modelling could begin. The TotalCharges column was stored as a string because 11 brand new customers had blank values instead of numbers. These customers had never been billed so I replaced the blanks with zero before converting the column to float. I also found 22 duplicate rows which were removed since no two real customers should be identical across 20 features.

### Feature Engineering

The raw features tell part of the story but not all of it. I built eight new features on top of the originals each grounded in a specific business question.

TenureGroup groups customers into four loyalty stages. New covers 0 to 12 months, Developing covers 13 to 24 months, Established covers 25 to 48 months, and Loyal covers 49 to 72 months. The relationship between tenure and churn is not linear and these groups capture the natural breakpoints in churn risk.

TotalServices counts how many of the nine available services each customer actively uses. A customer using seven services has a much higher switching cost than one using two. This is what keeps customers from leaving.

SpendPerService divides monthly charges by total active services. A customer paying eighty dollars for eight services feels good value. The same spend for two services often signals dissatisfaction.

ChargesRatio divides monthly charges by total lifetime charges. A high ratio means the customer is either new or was recently upsold, both of which increase churn risk.

HasPremiumServices flags whether the customer has any of the four premium add-ons including online security, backup, device protection, and tech support. Customers who have invested in these services have more to lose by switching.

IsAutomatedPayment flags customers on bank transfer or credit card automatic payment. These customers have committed to the relationship passively and are less likely to actively reconsider it each month.

ContractRiskScore encodes contract type as an ordinal risk score. Month to month gets a 3, one year gets a 2, and two year gets a 1. This is domain knowledge encoded directly rather than leaving it to the model to figure out from dummy variables.

TenureContractInteraction multiplies tenure by contract risk score. A brand new month to month customer scores very high. A long tenured two year customer scores very low. This single number captures the combined loyalty and commitment risk better than either feature does alone.

### Preprocessing Pipeline

I built the preprocessing as a custom sklearn transformer so that feature engineering, encoding, and scaling are all encapsulated in a single reusable pipeline object. This eliminates data leakage and makes the pipeline safe for production use.

Numerical features are imputed with the median and scaled with StandardScaler. Binary categorical features are ordinally encoded. Multi class categorical features are one hot encoded with the first category dropped to prevent the dummy variable trap. The full pipeline takes raw customer data as input and outputs a scaled numerical matrix ready for the model.

### Handling Class Imbalance

With 73.5% of customers staying and 26.5% churning a naive model could achieve 73.5% accuracy by predicting everyone stays. That model would be completely useless in practice.

I used SMOTE (Synthetic Minority Oversampling Technique) to balance the training data. Rather than simply copying existing churner records SMOTE creates realistic synthetic customers by interpolating between real churners and their nearest neighbours. Importantly SMOTE is applied inside the cross validation loop using an imblearn Pipeline so synthetic customers never leak into validation or test folds.

I also set class_weight to balanced in the logistic regression which provides a second layer of protection against class imbalance.

### Model Training and Hyperparameter Tuning

I used logistic regression for this project deliberately. It is interpretable, fast, and well understood by business stakeholders. A model that can be explained to a retention manager is far more valuable than a black box with marginally better metrics.

Hyperparameter tuning was done with GridSearchCV across 24 combinations of regularization strength, penalty type, and solver. Each combination was evaluated with 5 fold stratified cross validation using ROC AUC as the selection criterion. The best combination found was C equal to 1 with L1 penalty and the saga solver.

L1 regularization is particularly interesting here because it pushes some feature coefficients to exactly zero, effectively selecting the most important features automatically. This gives us a sparse interpretable model.

### Model Evaluation

The final model achieves a ROC AUC score of 0.8413 on the held out test set. This means the model correctly ranks a churner above a non churner 84.13% of the time.

For the churn class specifically the model achieves 75.27% recall meaning it catches three out of every four customers who would have churned. Precision is 52.34% meaning roughly half of customers flagged for retention offers are genuine churn risks. For a telecom business this tradeoff is acceptable since the cost of a missed churner far exceeds the cost of an unnecessary retention offer.

| Metric | Score |
|---|---|
| ROC AUC | 0.8413 |
| Churn Recall | 75.27% |
| Churn Precision | 52.34% |
| Churn F1 | 0.6174 |
| True Positives | 280 |
| False Negatives | 92 |

### Model Interpretability with SHAP

Beyond the standard metrics I used SHAP (SHapley Additive exPlanations) to understand what drives the model predictions at both a global and individual customer level.

The SHAP summary plot shows which features matter most across all customers. Tenure, contract type, and monthly charges consistently appear as the strongest predictors which aligns with the business intuition we built into the feature engineering.

The SHAP waterfall plot shows for one specific customer exactly how each feature pushed the prediction up or down from the baseline. This is what makes the model actionable for a retention team. Instead of just saying this customer will churn you can say this customer will churn because they have only been with us for two months, are on a month to month contract, and are paying above average monthly charges.

---

## Results

The model identifies 280 out of 372 churners in the test set while maintaining a false alarm rate that is commercially viable. In a real deployment with a customer base of tens of thousands this translates to a significant amount of recoverable revenue that would otherwise be lost.

Risk categories give the business a practical way to prioritise interventions without needing to understand the underlying model:

| Risk Category | Customers | Action |
|---|---|---|
| Low Risk | 615 (43.8%) | No action needed |
| Medium Risk | 255 (18.1%) | Schedule check in call |
| High Risk | 213 (15.2%) | Offer loyalty discount |
| Critical Risk | 322 (22.9%) | Escalate to retention team |

---

## How to Run This Project

Clone the repository and install the dependencies:

    pip install -r requirements.txt

Download the IBM Telco Customer Churn dataset from Kaggle and place it at data/telco_churn.csv

Then run the full pipeline with a single command:

    python main.py

This will clean the data, engineer features, train the model with cross validation, evaluate performance, generate all plots, and save everything to disk.

---

## Making Predictions on New Customers

Load the saved pipeline and pass in raw customer data:

    import joblib
    import pandas as pd

    pipeline = joblib.load('models/full_prediction_pipeline.pkl')

    customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.50,
        'TotalCharges': 171.00
    }

    customer_df = pd.DataFrame([customer])
    churn_probability = pipeline.predict_proba(customer_df)[:, 1][0]
    print(f'Churn probability: {churn_probability:.4f}')

---

## Dependencies

    pandas==2.2.2
    numpy==2.0.2
    scikit-learn==1.6.1
    imbalanced-learn==0.14.1
    shap==0.51.0
    matplotlib==3.10.0
    seaborn==0.13.2
    joblib==1.5.3
    scipy==1.16.3

---

## What I Would Add Next

This project covers the full modelling pipeline but a production deployment would need several additional components. Data validation with Great Expectations would catch upstream data quality issues before they reach the model. Experiment tracking with MLflow would log every training run automatically so no results are ever lost. A FastAPI wrapper would expose the model as a REST endpoint that any internal system could call. Docker containerisation would ensure the environment is identical across development and production. And a monitoring system like Evidently AI would track model drift over time and trigger retraining when performance degrades.

These are the natural next steps for taking this from a portfolio project to a fully deployed production system.

---

## About

Built as a demonstration of production level machine learning practices applied to a real business problem. Every design decision from the modular pipeline structure to the choice of evaluation metrics was made with production deployment in mind.
