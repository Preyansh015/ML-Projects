from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Employee Attrition Prediction", page_icon="👥")

# Look for the CSV in the same folder as this file
CSV_FILE = Path(__file__).parent / "WA_Fn-UseC_-HR-Employee-Attrition.csv"


# ---------- Same preprocessing as the notebook ----------
def preprocess(data):
    data = data.drop(columns=["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"])
    data["Gender"] = data["Gender"].map({"Female": 1, "Male": 0})
    data["OverTime"] = data["OverTime"].map({"Yes": 1, "No": 0})
    if "Attrition" in data.columns:
        data["Attrition"] = data["Attrition"].map({"Yes": 1, "No": 0})
    text_cols = data.select_dtypes(exclude="number").columns.tolist()
    return pd.get_dummies(data, columns=text_cols, drop_first=True, dtype=int)


@st.cache_data
def load_data():
    return pd.read_csv(CSV_FILE)


@st.cache_resource
def train_model():
    raw = load_data()
    df = preprocess(raw.copy())

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_train_s, y_train)

    pred = model.predict(X_test_s)
    scores = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    }
    return model, scaler, list(X.columns), scores


raw = load_data()
model, scaler, feature_cols, scores = train_model()

# ---------- Page ----------
st.title("👥 Employee Attrition Prediction")
st.write("Enter employee details to see the chance that the employee will leave the company.")

tab1, tab2 = st.tabs(["Predict", "About the model"])

# ---------- Tab 1: Prediction ----------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 60, 35)
        monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000, step=500)
        total_years = st.slider("Total Working Years", 0, 40, 8)
        years_company = st.slider("Years At Company", 0, 40, 5)
        distance = st.slider("Distance From Home (km)", 1, 29, 8)

    with col2:
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        department = st.selectbox("Department", sorted(raw["Department"].unique()))
        job_role = st.selectbox("Job Role", sorted(raw["JobRole"].unique()))
        travel = st.selectbox("Business Travel", sorted(raw["BusinessTravel"].unique()))
        marital = st.selectbox("Marital Status", sorted(raw["MaritalStatus"].unique()))
        job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        wlb = st.slider("Work Life Balance (1-4)", 1, 4, 3)

    if st.button("Predict", type="primary"):
        # Start from a typical employee (median / most common values), then
        # replace the values the user entered.
        row = {}
        for c in raw.columns:
            if c == "Attrition":
                continue
            row[c] = raw[c].median() if pd.api.types.is_numeric_dtype(raw[c]) else raw[c].mode()[0]

        row.update({
            "Age": age,
            "MonthlyIncome": monthly_income,
            "TotalWorkingYears": total_years,
            "YearsAtCompany": years_company,
            "DistanceFromHome": distance,
            "OverTime": overtime,
            "Department": department,
            "JobRole": job_role,
            "BusinessTravel": travel,
            "MaritalStatus": marital,
            "JobSatisfaction": job_sat,
            "WorkLifeBalance": wlb,
        })

        # Add the new row to the data so encoding gives the same columns
        combined = pd.concat([raw.drop(columns=["Attrition"]), pd.DataFrame([row])], ignore_index=True)
        X_new = preprocess(combined).iloc[[-1]][feature_cols]

        prob = model.predict_proba(scaler.transform(X_new))[0][1]

        st.subheader(f"Chance of leaving: {prob * 100:.1f}%")
        st.progress(float(prob))
        if prob >= 0.5:
            st.error("⚠️ High risk: this employee is likely to leave.")
        else:
            st.success("✅ Low risk: this employee is likely to stay.")

# ---------- Tab 2: About ----------
with tab2:
    st.write(
        "Model: **Logistic Regression** with `class_weight='balanced'`, "
        "trained on the IBM HR Attrition dataset (1,470 employees)."
    )
    st.write("Scores on the test set (class = leaves):")
    st.table(pd.DataFrame(scores, index=["Score"]).round(2))
    st.caption(
        "Recall is higher than precision, so the model finds most leavers "
        "but also gives some false alarms."
    )

    st.write("Attrition in the data:")
    st.bar_chart(raw["Attrition"].value_counts())
