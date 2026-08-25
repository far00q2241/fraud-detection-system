# Fraud Detection ML

A Machine Learning project to detect fraudulent transactions using Logistic Regression.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## Model

- Logistic Regression
- StandardScaler
- Class Weight: Balanced

## Performance

- ROC-AUC: 0.9344
- Recall: 0.81
- F1 Score: 0.19

## Features

The model uses transaction-related information such as:

- Merchant Category
- Card Type
- Authentication Method
- Channel
- Device Type
- Transaction Amount
- Foreign Transaction
- IP Country Mismatch
- Billing & Shipping Mismatch
- VPN Usage

## Files

- `fraud_detection.ipynb` - Model development
- `fraud_detection_model.pkl` - Trained model
- `app.py` - Streamlit application
- `requirements.txt` - Required packages

## Run the Application

```bash
pip install -r requirements.txt
streamlit run app.py
