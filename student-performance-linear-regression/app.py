from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
)


@st.cache_resource
def train_model():
    csv_path = Path(__file__).resolve().parent / "Student_Performance.csv"
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().reset_index(drop=True)
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map(
        {"Yes": 1, "No": 0}
    )

    x = df.drop(columns=["Performance Index"])
    y = df["Performance Index"]

    model = LinearRegression()
    model.fit(x, y)
    return model, list(x.columns), len(df)


model, columns, row_count = train_model()

st.title("🎓 Student Performance Predictor")
st.markdown(
    "Predict a student's **Performance Index** using a **Multiple Linear Regression** "
    f"model trained on **{row_count:,} cleaned records** from `Student_Performance.csv`."
)

with st.expander("About this project"):
    st.write(
        "This app uses historical student data—including study hours, prior scores, "
        "extracurricular participation, sleep hours, and sample papers practiced—to "
        "estimate an expected Performance Index. Previous scores and study hours carry "
        "the strongest influence in the dataset."
    )

st.subheader("Student Details")

col1, col2 = st.columns(2)

with col1:
    hours = st.slider("Hours Studied", min_value=1, max_value=9, value=5)
    prev_score = st.slider("Previous Scores", min_value=40, max_value=100, value=70)
    sleep = st.slider("Sleep Hours", min_value=4, max_value=9, value=7)

with col2:
    extra = st.selectbox("Extracurricular Activities", options=["No", "Yes"])
    papers = st.slider(
        "Sample Question Papers Practiced", min_value=0, max_value=9, value=4
    )

st.divider()

if st.button("Predict Performance Index", type="primary", use_container_width=True):
    input_df = pd.DataFrame(
        {
            "Hours Studied": [hours],
            "Previous Scores": [prev_score],
            "Extracurricular Activities": [1 if extra == "Yes" else 0],
            "Sleep Hours": [sleep],
            "Sample Question Papers Practiced": [papers],
        }
    )[columns]

    prediction = float(model.predict(input_df)[0])
    clamped_prediction = max(0.0, min(100.0, prediction))

    st.success(f"**Predicted Performance Index:** {clamped_prediction:.2f} / 100")
    st.caption(
        "Note: Predictions are statistical estimates based on linear regression and "
        "historical patterns, not a guaranteed exam outcome."
    )
```[cite: 1]

---

### How to Put This Into Your Project
1. Save the **`README.md`** text inside your project folder on GitHub.
2. Replace your **`app.py`** with the code above and commit.
3. Make sure **`requirements.txt`** in that same folder has just `streamlit`, `pandas`, and `scikit-learn` (one per line), then reboot your app on Streamlit Cloud.

Would you also like to add model evaluation metrics (like $R^2$ score or MAE) or feature weights to the app?
