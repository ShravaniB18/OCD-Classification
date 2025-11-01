import streamlit as st
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

# ------------------------
# Load models
# ------------------------
model = joblib.load("models/ocd_calibrated_pipeline.joblib")
reg_pipeline = joblib.load("models/ocd_reg_pipeline.joblib")

# ------------------------
# Display names for UI
# ------------------------
display_names = {
    "age": "Age (in years)",
    "gender": "Gender",
    "ethnicity": "Ethnic Background",
    "marital_status": "Marital Status",
    "education_level": "Education Level",
    "symptom_duration_months": "Symptom Duration (Months)",
    "family_history": "Family History of OCD",
    "ybocs_obs": "Obsessions Score (Y-BOCS)",
    "ybocs_comp": "Compulsions Score (Y-BOCS)",
    "depression_dx": "Depression Diagnosis",
    "anxiety_dx": "Anxiety Diagnosis",
    "num_medications": "Number of Medications"
}

# ------------------------
# Load training data
# ------------------------
with open("models/X_train.pkl", "rb") as f:
    X_train = joblib.load(f)

numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

# ------------------------
# Clinical features used for classification
# ------------------------
model_features = ['ybocs_obs', 'ybocs_comp', 'symptom_duration_months',
                  'family_history', 'depression_dx', 'anxiety_dx']

# ------------------------
# Demographic info (for display only)
# ------------------------
demographic_features = ['age', 'gender', 'ethnicity', 'marital_status', 'education_level']

# ------------------------
# SHAP setup
# ------------------------
if isinstance(model, CalibratedClassifierCV):
    rf_model = model.estimator.named_steps['model']
    pre = model.estimator.named_steps['pre']
else:
    rf_model = model.named_steps['model']
    pre = model.named_steps['pre']

explainer = shap.TreeExplainer(rf_model)
feature_names = pre.get_feature_names_out()

# ------------------------
# Streamlit UI
# ------------------------
st.title("🧠 OCD Classifier with Explainability")
st.write("Enter patient details below to classify OCD likelihood and generate a downloadable report.")

# ------------------------
# Step 1: Collect demographic info (not used in model)
# ------------------------
st.header("🩺 Patient Demographic Information")
patient_info = {}

patient_info["name"] = st.text_input("👤 Patient Name", placeholder="Enter full name")

for col in demographic_features:
    label = display_names.get(col, col)
    if col == "age":
        patient_info[col] = st.number_input(label, min_value=1, max_value=100, value=25)
    elif col in ["gender", "ethnicity", "marital_status", "education_level"]:
        options = X_train[col].dropna().unique().tolist() if col in X_train.columns else []
        options.append("Other")
        patient_info[col] = st.selectbox(label, options)

# ------------------------
# Step 2: Collect clinical features (used in model)
# ------------------------
st.header("🧬 Clinical Details for OCD Classification")
patient_data = {}

for col in model_features:
    label = display_names.get(col, col)
    col_dtype = X_train[col].dtype

    if col_dtype == "object" or str(col_dtype) == "category":
        unique_vals = X_train[col].dropna().unique().tolist()
        unique_vals = ["Select"] + unique_vals + ["Other"]
        user_input = st.selectbox(f"{label}", unique_vals)
        patient_data[col] = None if user_input == "Select" else user_input
    elif np.issubdtype(col_dtype, np.number):
        # Start with blank by using a very small placeholder value (or 0)
        patient_data[col] = st.number_input(
            f"{label}",
            min_value=0.0,
            max_value=float(np.nanmax(X_train[col])),
            value=float(np.nanmin(X_train[col])),
            step=1.0
        )
        patient_data[col] = patient_data[col]
# ------------------------
# Step 3: Prediction
# ------------------------
patient_df = pd.DataFrame([patient_data])

# Fill missing expected features
all_features = pre.feature_names_in_
for col in all_features:
    if col not in patient_df.columns:
        if col in numeric_features:
            patient_df[col] = X_train[col].mean()
        else:
            patient_df[col] = "Other"
    else:
        # If user entered None for clinical features, replace with mean
        if patient_df[col].isnull().iloc[0] and col in numeric_features:
            patient_df[col] = X_train[col].mean()
        elif patient_df[col].isnull().iloc[0]:
            patient_df[col] = "Other"

patient_df = patient_df[all_features]

if st.button("🔍 Predict OCD Risk"):
    # --- Classification ---
    y_proba = model.predict_proba(patient_df)[:, 1][0]
    y_pred = int(y_proba >= 0.5)
    # if y_proba >= 0.45:   # lower threshold for borderline/high symptom cases
    #     y_pred = 1
    # else:
    #     y_pred = 0

    # --- Severity prediction ---
    severity_pred = reg_pipeline.predict(patient_df)[0]
    severity_pred = np.clip(severity_pred, 0, 10)

    # if y_pred == 1 and y_proba < 0.55:
    #     severity_pred = max(severity_pred, 5.0)
    # elif y_pred == 0 and y_proba > 0.35:
    #     severity_pred = min(severity_pred, 4.0)

    # --- Display main result ---
    if y_pred == 1:
        result = "🧠 OCD Positive"
        color = "red"
    else:
        result = "✅ OCD Negative"
        color = "green"

    st.markdown(f"### Prediction: <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)
    st.write(f"**Probability:** {y_proba:.2f}")
    st.write(f"**Predicted OCD Severity (0-10):** {severity_pred:.2f}")

    # ------------------------
    # Step 4: Show Report Table + Download Option
    # ------------------------
    st.subheader("📋 Patient Report Summary")

    report_data = {
        "Feature": [
             "Patient Name","Age", "Gender", "Ethnicity", "Marital Status", "Education Level",
            "Obsessions Score (Y-BOCS)", "Compulsions Score (Y-BOCS)", "Symptom Duration (Months)",
            "Family History of OCD", "Depression Diagnosis", "Anxiety Diagnosis",
            "Predicted OCD Probability", "OCD Classification", "Predicted OCD Severity (0–10)"
        ],
        "Value": [
            patient_info["name"],patient_info["age"], patient_info["gender"], patient_info["ethnicity"], patient_info["marital_status"],
            patient_info["education_level"], patient_data["ybocs_obs"], patient_data["ybocs_comp"],
            patient_data["symptom_duration_months"], patient_data["family_history"], patient_data["depression_dx"],
            patient_data["anxiety_dx"], f"{y_proba:.2f}", result, f"{severity_pred:.2f}"
        ]
    }

    report_df = pd.DataFrame(report_data)
    st.table(report_df)

    # CSV download
    csv = report_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Patient Report (CSV)",
        data=csv,
        file_name="ocd_patient_report.csv",
        mime="text/csv"
    )

    # ------------------------
    # Step 5: Visual Interpretation (Simplified)
    # ------------------------
    st.subheader("📊 Visual Interpretation of Prediction")

    # --- Probability Bar ---
    st.markdown("#### 🔹 Predicted OCD Probability")
    fig1, ax1 = plt.subplots(figsize=(6, 1.2))
    ax1.barh(["OCD Probability"], [y_proba], color="red" if y_pred == 1 else "green")
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Probability")
    for i, v in enumerate([y_proba]):
        ax1.text(v + 0.02 if v < 0.95 else v - 0.07, i, f"{v:.2f}", color='black', fontweight='bold')
    ax1.set_yticks([])
    st.pyplot(fig1)

    # Risk label
    if y_proba < 0.3:
        st.success("🟢 Low OCD Risk")
    elif y_proba < 0.6:
        st.warning("🟡 Moderate OCD Risk — consider screening")
    else:
        st.error("🔴 High OCD Risk — clinical attention advised")

    # --- 2. Severity Level ---
    st.markdown("#### 🔹 Predicted OCD Severity (0–10 Scale)")
    fig2, ax2 = plt.subplots(figsize=(6, 1.2))
    ax2.barh(["OCD Severity"], [severity_pred], color="orange")
    ax2.set_xlim(0, 10)
    ax2.set_xlabel("Severity Level")
    for i, v in enumerate([severity_pred]):
        ax2.text(v + 0.2 if v < 9.5 else v - 0.6, i, f"{v:.2f}", color='black', fontweight='bold')
    ax2.set_yticks([])
    st.pyplot(fig2)