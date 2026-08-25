import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fraud_detection_model.pkl")

st.set_page_config(page_title="Fraud Detection System", page_icon="💳")

st.title("💳 Credit Card Fraud Detection")
st.write("Predict whether a transaction is **Fraud** or **Not Fraud**.")

st.subheader("Enter Transaction Details")

# Numerical inputs
transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=500.0)
customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
account_age_days = st.number_input("Account Age (Days)", min_value=0, value=365)
transaction_hour = st.slider("Transaction Hour", 0, 23, 12)
hours_since_last_txn = st.number_input("Hours Since Last Transaction", min_value=0.0, value=5.0)

# Binary inputs
is_foreign_transaction = st.selectbox("Foreign Transaction", ["No", "Yes"])
ip_country_mismatch = st.selectbox("IP Country Mismatch", ["No", "Yes"])
billing_shipping_mismatch = st.selectbox("Billing & Shipping Mismatch", ["No", "Yes"])
is_new_merchant = st.selectbox("New Merchant", ["No", "Yes"])
used_vpn = st.selectbox("Used VPN", ["No", "Yes"])

# Categorical inputs
merchant_category = st.selectbox(
    "Merchant Category",
    [
        "Online Retail", "Groceries", "Restaurants", "Electronics",
        "Fuel", "Travel", "Utilities", "Healthcare",
        "Gaming", "Streaming", "Gift Cards", "Crypto Exchange"
    ]
)

card_type = st.selectbox(
    "Card Type",
    ["Visa", "Mastercard", "Amex", "RuPay", "Discover"]
)

auth_method = st.selectbox(
    "Authentication Method",
    ["3D Secure", "OTP", "PIN", "Biometric", "No Authentication"]
)

channel = st.selectbox(
    "Transaction Channel",
    ["Online", "POS", "Contactless", "In-App", "ATM"]
)

device_type = st.selectbox(
    "Device Type",
    [
        "Android Phone", "iPhone", "POS Terminal", "Windows PC",
        "Mac", "Tablet", "Smart Watch", "ATM Machine"
    ]
)

# Convert Yes/No to 1/0
binary_map = {"No": 0, "Yes": 1}

input_data = pd.DataFrame({
    "transaction_amount": [transaction_amount],
    "customer_age": [customer_age],
    "account_age_days": [account_age_days],
    "transaction_hour": [transaction_hour],
    "hours_since_last_txn": [hours_since_last_txn],
    "is_foreign_transaction": [binary_map[is_foreign_transaction]],
    "ip_country_mismatch": [binary_map[ip_country_mismatch]],
    "billing_shipping_mismatch": [binary_map[billing_shipping_mismatch]],
    "is_new_merchant": [binary_map[is_new_merchant]],
    "used_vpn": [binary_map[used_vpn]],
    "merchant_category": [merchant_category],
    "card_type": [card_type],
    "auth_method": [auth_method],
    "channel": [channel],
    "device_type": [device_type]
})

# Prediction
if st.button("Predict Fraud"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

    st.metric("Fraud Probability", f"{probability:.2%}")
