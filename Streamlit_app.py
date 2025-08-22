import streamlit as st
import pandas as pd
import joblib

st.title("Churn Prediction")
st.write("This app predicts whether a customer will churn or not")

# Inference Function
model = joblib.load("Model_Porto.sav")
def get_prediction(data:pd.DataFrame):
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    return pred, pred_proba

# User Input
left, right = st.columns(2, gap="medium", border=True)

# Numerical Input
left.subheader("Fitur Numerik")
age = left.slider("Age", min_value=0, max_value=92, value=1)
tenure = left.slider("Tenure", min_value=0, max_value=10, value=1)
num_of_products = left.slider("Number of Products", min_value=0, max_value=4, value=2)
has_cr_card = left.slider("Has Credit Card", min_value=0, max_value=1, value=1)
is_active_member = left.slider("Is Active Member", min_value=0, max_value=1, value=1)
credit_score = left.number_input("Credit Score", min_value=350, max_value=850, value=500, step=1)
balance = left.number_input("Balance", min_value=0.0, value=80000.0, step=100.0)
estimated_salary = left.number_input("Estimated Salary", min_value=0.0, value=200.0, step=100.0)

# Categorical Input
right.subheader("Fitur Kategorik")
gender = right.selectbox("Gender", ["Male", "Female"])
geography = right.selectbox("Geography", ["France", "Germany", "Spain"])

data = pd.DataFrame({"Age": [age],
                    "Tenure": [tenure], 
                    "NumOfProducts": [num_of_products], 
                    "HasCrCard": [has_cr_card],
                    "IsActiveMember": [is_active_member],
                    "CreditScore": [credit_score],
                    "Balance": [balance],
                    "EstimatedSalary": [estimated_salary],
                    "Gender": gender,
                    "Geography": geography})
st.dataframe(data, use_container_width=True, hide_index=True)

# Prediction Button
button = st.button("Prediksi Customer", use_container_width=True)
if button:
    st.write("Prediksi Berhasil !")
    pred, pred_proba = get_prediction(data)

    label_map = {0: "Loyal", 1: "Churn"}
    
    label_pred = label_map[pred[0]]
    label_proba = pred_proba[0][1]
    output = f"Probabilitas customer untuk churn: {label_proba:.0%}, Prediksi Customer: {label_pred}"
    st.write(output)