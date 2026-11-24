from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")


@st.cache_resource
def train_model():
    csv_path = Path(__file__).resolve().parent / "Student_Performance.csv"
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().reset_index(drop=True)
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

    x = df.drop(columns=["Performance Index"])
    y = df["Performance Index"]

    model = LinearRegression()
    model.fit(x, y)
    return model, list(x.columns)


model, columns = train_model()

st.title("🎓 Student Performance Predictor")
st.write("Enter the student's details to predict the Performance Index.")

hours = st.slider("Hours Studied", 1, 9, 5)
prev_score = st.slider("Previous Scores", 40, 100, 70)
extra = st.selectbox("Extracurricular Activities", ["No", "Yes"])
sleep = st.slider("Sleep Hours", 4, 9, 7)
papers = st.slider("Sample Question Papers Practiced", 0, 9, 4)

if st.button("Predict"):
    input_df = pd.DataFrame({
        "Hours Studied": [hours],
        "Previous Scores": [prev_score],
        "Extracurricular Activities": [1 if extra == "Yes" else 0],
        "Sleep Hours": [sleep],
        "Sample Question Papers Practiced": [papers],
    })[columns]

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Performance Index: {prediction:.2f}")
