# 🎓 Student Performance Predictor (Linear Regression)

An interactive Streamlit web application that predicts a student's **Performance Index** using multiple linear regression.

## 📌 Overview
This project explores how study habits, academic history, sleep, and practice tests relate to overall student performance, trains a linear regression model on the dataset, and provides an interactive interface for real-time predictions.

## 📊 Dataset & Preprocessing
The dataset (`Student_Performance.csv`) contains **10,000 records** across 6 columns with no missing values:
- **Hours Studied** (1 to 9)
- **Previous Scores** (40 to 99)
- **Extracurricular Activities** (`Yes` / `No`, mapped to `1` / `0`)
- **Sleep Hours** (4 to 9)
- **Sample Question Papers Practiced** (0 to 9)
- **Performance Index** (Target variable, ~10 to 100)

During preprocessing, **127 duplicate rows** are removed, leaving **9,873 unique rows** for model training. From exploratory data analysis, **Previous Scores** shows the strongest positive correlation with Performance Index, followed by **Hours Studied**.

## 🚀 Live App
- **Streamlit URL:** `https://student-performance-preyansh-app.streamlit.app`

## 🛠️ Tech Stack
- **Python**, **Pandas**, **Scikit-learn**, **Streamlit**
