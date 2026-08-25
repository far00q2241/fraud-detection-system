import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

model_path = os.path.join(
    os.path.dirname(__file__),
    "fraud_detection_model.pkl"
)

model = joblib.load(model_path)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Predict whether a transaction is **Fraud** or **Not Fraud**."
)

st.subheader("Enter Transaction Details")

# --------------------------------------------------
# Numerical Inputs
# --------------------------------------------------

amount_usd = st.number_input(
    "Transaction Amount (USD)",
    min_value=0.0,
    value=100.0
)

hours_since_last_txn = st.number_input(
    "Hours Since Last Transaction",
    min_value=0.0,
    value=5.0
)

txn_count_last_24h = st.number_input(
    "Transaction Count Last 24 Hours",
    min_value=0,
    value=2
)

distance_from_home_km = st.number_input(
    "Distance From Home (km)",
    min_value=0.0,
    value=5.0
)

card_age_months = st.number_input(
    "Card Age (Months)",
    min_value=0,
    value=24
)

customer_age = st.number_input(
    "Customer Age",
    min_value=18,
    max_value=100,
    value=30
)

account_balance_usd = st.number_input(
    "Account Balance (USD)",
    min_value=0.0,
    value=5000.0
)

cvv_retry_count = st.number_input(
    "CVV Retry Count",
    min_value=0,
    value=0
)

velocity_score = st.number_input(
    "Velocity Score",
    min_value=0.0,
    value=1.0
)

time_of_day_hour = st.slider(
    "Time of Day (Hour)",
    min_value=0,
    max_value=23,
    value=12
)

day_of_week = st.slider(
    "Day of Week",
    min_value=0,
    max_value=6,
    value=2
)

merchant_risk_score = st.number_input(
    "Merchant Risk Score",
    min_value=0.0,
    value=0.5
)

prior_disputes = st.number_input(
    "Prior Disputes",
    min_value=0,
    value=0
)

# --------------------------------------------------
# Binary Inputs
# --------------------------------------------------

binary_map = {
    "No": 0,
    "Yes": 1
}

is_foreign_transaction = st.selectbox(
    "Foreign Transaction",
    ["No", "Yes"]
)

is_new_merchant = st.selectbox(
    "New Merchant",
    ["No", "Yes"]
)

used_vpn = st.selectbox(
    "Used VPN",
    ["No", "Yes"]
)

ip_country_mismatch = st.selectbox(
    "IP Country Mismatch",
    ["No", "Yes"]
)

billing_shipping_mismatch = st.selectbox(
    "Billing & Shipping Mismatch",
    ["No", "Yes"]
)

is_ai_generated_scam_attempt = st.selectbox(
    "AI Generated Scam Attempt",
    ["No", "Yes"]
)

# --------------------------------------------------
# Categorical Inputs
# --------------------------------------------------

merchant_category = st.selectbox(
    "Merchant Category",
    [
        "Online Retail",
        "Groceries",
        "Restaurants",
        "Electronics",
        "Fuel",
        "Travel",
        "Utilities",
        "Healthcare",
        "Gaming",
        "Streaming",
        "Gift Cards",
        "Crypto Exchange"
    ]
)

card_type = st.selectbox(
    "Card Type",
    [
        "Visa",
        "Mastercard",
        "Amex",
        "RuPay",
        "Discover"
    ]
)

auth_method = st.selectbox(
    "Authentication Method",
    [
        "3D Secure",
        "OTP",
        "PIN",
        "Biometric",
        "No Authentication"
    ]
)

channel = st.selectbox(
    "Transaction Channel",
    [
        "Online",
        "POS",
        "Contactless",
        "In-App",
        "ATM"
    ]
)

device_type = st.selectbox(
    "Device Type",
    [
        "Android Phone",
        "iPhone",
        "POS Terminal",
        "Windows PC",
        "Mac",
        "Tablet",
        "Smart Watch",
        "ATM Machine"
    ]
)

# --------------------------------------------------
# Create Input Data
# --------------------------------------------------

input_data = pd.DataFrame({
    "amount_usd": [amount_usd],
    "is_foreign_transaction": [
        binary_map[is_foreign_transaction]
    ],
    "hours_since_last_txn": [hours_since_last_txn],
    "txn_count_last_24h": [txn_count_last_24h],
    "distance_from_home_km": [distance_from_home_km],
    "card_age_months": [card_age_months],
    "customer_age": [customer_age],
    "account_balance_usd": [account_balance_usd],
    "is_new_merchant": [
        binary_map[is_new_merchant]
    ],
    "used_vpn": [
        binary_map[used_vpn]
    ],
    "ip_country_mismatch": [
        binary_map[ip_country_mismatch]
    ],
    "billing_shipping_mismatch": [
        binary_map[billing_shipping_mismatch]
    ],
    "cvv_retry_count": [cvv_retry_count],
    "velocity_score": [velocity_score],
    "time_of_day_hour": [time_of_day_hour],
    "day_of_week": [day_of_week],
    "is_ai_generated_scam_attempt": [
        binary_map[is_ai_generated_scam_attempt]
    ],
    "merchant_risk_score": [merchant_risk_score],
    "prior_disputes": [prior_disputes]
})

# --------------------------------------------------
# One-Hot Encoding
# --------------------------------------------------

# Add expected categorical columns
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

# Set selected categorical values to 1

categorical_columns = [
    f"merchant_category_{merchant_category}",
    f"card_type_{card_type}",
    f"auth_method_{auth_method}",
    f"channel_{channel}",
    f"device_type_{device_type}"
]

for col in categorical_columns:
    if col in input_data.columns:
        input_data[col] = 1

# --------------------------------------------------
# Ensure Exact Feature Order
# --------------------------------------------------

input_data = input_data[
    model.feature_names_in_
]

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Fraud"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ Fraudulent Transaction Detected"
        )

    else:

        st.success(
            "✅ Transaction appears legitimate"
        )

    st.metric(
        "Fraud Probability",
        f"{probability:.2%}"
    )
