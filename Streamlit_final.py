import streamlit as st
import pandas as pd
import joblib

# Title & description di tengah
st.markdown("<h1 style='text-align: center;'>Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This app predicts whether a customer will churn or not</p>", unsafe_allow_html=True)

# Load model
model = joblib.load("model_Final.sav")

# Feature names yang dipakai model (dari training)
expected_features = [
    'SeniorCitizen', 'gender_Male', 'Partner_Yes', 'Dependents_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

# Inference function
def get_prediction(data: pd.DataFrame):
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    return pred, pred_proba

# Layout input
left, right = st.columns(2, gap="medium", border=True)

# Left column → 4 input
SeniorCitizen = left.selectbox("SeniorCitizen", ["No", "Yes"])
Gender = left.selectbox("Gender", ["Male", "Female"])
Partner = left.selectbox("Partner", ["Yes", "No"])
Dependents = left.selectbox("Dependents", ["Yes", "No"])

# Right column → 3 input
PaperlessBilling = right.selectbox("PaperlessBilling", ["Yes", "No"])
Contract = right.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
PaymentMethod = right.selectbox("Payment Method", [
    "Electronic check", 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])

# Mapping SeniorCitizen Yes/No ke 0/1
senior_val = 1 if SeniorCitizen == "Yes" else 0

# Buat dataframe mentah
data_raw = pd.DataFrame({
    "SeniorCitizen": [senior_val],
    "Gender": [Gender],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "PaperlessBillling": [PaperlessBilling],
    "Contract": [Contract],
    "PaymentMethod": [PaymentMethod]
})

# One-hot encode manual sesuai expected_features
data_encoded = pd.get_dummies(data_raw, drop_first=False)

# Tambahkan kolom yang hilang dengan 0
for col in expected_features:
    if col not in data_encoded.columns:
        data_encoded[col] = 0

# Urutkan kolom agar sama dengan training
data_encoded = data_encoded[expected_features]

st.subheader("Data Input (Encoded)")
st.dataframe(data_encoded, use_container_width=True, hide_index=True)

# Prediction Button
if st.button("Prediksi Customer", use_container_width=True):
    pred, pred_proba = get_prediction(data_encoded)
    label_map = {0: "Loyal", 1: "Churn"}
    label_pred = label_map[pred[0]]
    label_proba = pred_proba[0][1]
    output = f"Probabilitas customer untuk churn: {label_proba:.0%}, Prediksi Customer: {label_pred}"
    st.write(output)
