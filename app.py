import streamlit as st
import pandas as pd
import joblib

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

def main():
    st.title("Diabetes Prediction App")

    with st.sidebar:
        st.subheader("Model Settings")
        model_path = st.text_input("Model path", "diabetes_model.pkl")

    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Could not load model at '{model_path}'. Error: {e}")
        st.stop()

    st.write("Provide the patient features:")

    cols = st.columns(2)
    inputs = {}
    for i, feat in enumerate(FEATURES):
        with cols[i % 2]:
            # sensible defaults based on dataset ranges
            default = 0.0
            inputs[feat] = st.number_input(feat, value=float(default), step=0.1)

    if st.button("Predict"):
        input_df = pd.DataFrame([inputs])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0, 1]

        label = "Diabetic" if pred == 1 else "Non-diabetic"
        st.success(f"Prediction: **{label}**")
        st.info(f"Probability of diabetes: **{proba:.3f}**")

if __name__ == "__main__":
    main()
