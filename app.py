# ---------------------------------------------------------
# Heart Disease Predictor - Full Streamlit App
# ---------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")


@st.cache_data(show_spinner=False)
def load_model(model_path="final_svm_model.joblib", meta_path="model_meta.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)

    meta = {}
    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)

    return model, meta


def make_row_from_inputs(inputs: dict, cols: list, dtypes: dict, fill_value=np.nan):
    """Construct a one-row DataFrame that exactly matches training columns."""
    row = pd.DataFrame([inputs])

    for c in cols:
        if c not in row.columns:
            row[c] = fill_value

    row = row.loc[:, cols]

    for c in cols:
        try:
            dtype = dtypes.get(c, None)
            if dtype is None:
                continue
            if pd.api.types.is_integer_dtype(dtype):
                row[c] = pd.to_numeric(row[c], errors="coerce").astype("Int64")
            elif pd.api.types.is_float_dtype(dtype):
                row[c] = pd.to_numeric(row[c], errors="coerce").astype(float)
            else:
                row[c] = row[c].astype(object)
        except:
            pass

    return row

#UI Layout
st.title("💓 Heart Disease Prediction App")

st.write("Enter patient details to get prediction from the trained SVM model.")

try:
    model, meta = load_model()
    st.success("Model & metadata loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model/meta: {e}")
    st.stop()

original_cols = meta.get("original_cols", [])
original_dtypes = meta.get("original_dtypes", {})

#Input Section
st.header("Patient Details")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 130)
serum_chol = st.number_input("Serum Cholesterol (mg/dl)", 50, 600, 240)
max_hr = st.number_input("Max Heart Rate Achieved", 40, 250, 150)
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0, step=0.1)

sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
chest_pain_type = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
slope = st.selectbox("Slope of Peak Exercise ST Segment", [1, 2, 3])
resting_ekg = st.selectbox("Resting EKG Results", [0, 1, 2])
exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1])
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
num_major_vessels = st.selectbox("Num Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal", ["normal", "fixed_defect", "reversible_defect"])

input_dict = {
    "age": age,
    "resting_blood_pressure": resting_bp,
    "serum_cholesterol_mg_per_dl": serum_chol,
    "max_heart_rate_achieved": max_hr,
    "oldpeak_eq_st_depression": oldpeak,
    "chest_pain_type": chest_pain_type,
    "slope_of_peak_exercise_st_segment": slope,
    "exercise_induced_angina": exercise_angina,
    "fasting_blood_sugar_gt_120_mg_per_dl": fasting_bs,
    "resting_ekg_results": resting_ekg,
    "sex": sex,
    "num_major_vessels": num_major_vessels,
    "thal": thal
}

#Prediction part
st.markdown("### Prediction")

chosen_threshold = st.slider(
    "Probability threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(meta.get("threshold", 0.5)),
    step=0.01,
)

if st.button("Predict"):
    try:
        row = make_row_from_inputs(input_dict, original_cols, original_dtypes)
        proba = model.predict_proba(row)[:, 1][0]
        pred = int(proba >= chosen_threshold)

        st.write("---")
        st.metric("Probability of Heart Disease", f"{proba:.3f}")
        st.metric("Predicted Class", "1 (Disease)" if pred == 1 else "0 (No Disease)")

        with st.expander("Show Processed Input Data"):
            st.dataframe(row.convert_dtypes().T)

    except Exception as e:
        st.error("Prediction failed: " + str(e))
        st.exception(e)


st.write("---")
st.info("Model trained using SVM + calibrated probabilities + GridSearchCV tuning.")
