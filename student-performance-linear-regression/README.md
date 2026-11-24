<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:4F46E5,50:7C3AED,100:EC4899&height=160&section=header&text=Student%20Performance%20Predictor&fontSize=34&fontColor=ffffff&animation=twinkling" width="100%" alt="Student Performance Predictor Banner" />

<a href="https://student-performance-preyansh.streamlit.app/">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=18&pause=1000&color=7C3AED&center=true&vCenter=true&width=520&lines=Multiple+Linear+Regression+Model;Predict+Student+Performance+Index+(0-100);Interactive+Streamlit+Web+App" alt="Typing SVG" />
</a>

<br/>

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://student-performance-preyansh.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-LinearRegression-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)

</div>

---

## Live Demo

Try the live application here: **[https://student-performance-preyansh.streamlit.app/](https://student-performance-preyansh.streamlit.app/)**

---

## Project Overview

The **Student Performance Predictor** is a machine learning web application built with **Streamlit** and **Scikit-learn**. It uses a **Multiple Linear Regression** model to estimate a student's **Performance Index** based on study habits, prior academic scores, extracurricular involvement, sleep duration, and practice test habits.

---

## Dataset & Data Preprocessing

The model is trained on `Student_Performance.csv`, which starts with **10,000 rows** and **6 columns** with **no missing values**[cite: 1].

### Features & Target
* **Hours Studied:** Numeric input (`1` to `9` hours)[cite: 1]
* **Previous Scores:** Numeric input (`40` to `99`)[cite: 1]
* **Extracurricular Activities:** Categorical (`Yes` / `No`), encoded as `1` / `0`[cite: 1]
* **Sleep Hours:** Numeric input (`4` to `9` hours)[cite: 1]
* **Sample Question Papers Practiced:** Numeric input (`0` to `9`)[cite: 1]
* **Performance Index:** Target variable (`~10` to `100`)[cite: 1]

### Cleaning & Exploratory Insights
* **Duplicate Removal:** Drops **127 duplicate records**, resulting in **9,873 clean rows** for training[cite: 1].
* **Key Correlations:** **Previous Scores** has a strong positive correlation (`~0.92`) with **Performance Index**, followed by **Hours Studied** (`~0.38`), while **Sleep Hours** and **Sample Question Papers Practiced** show weaker direct linear correlation (`~0.05`)[cite: 1].

---

## Project Structure

```text
student-performance-linear-regression/
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── Student_Performance.csv             # Dataset used to train the model
├── Student_Performance_Prediction.ipynb# Exploratory analysis & modeling notebook
└── README.md                           # Project documentation
```[cite: 1, 2]

---

## Run Locally

1. **Clone the repository**
   ```bash
   git clone [https://github.com/Preyansh015/ML-Projects.git](https://github.com/Preyansh015/ML-Projects.git)
   cd ML-Projects/student-performance-linear-regression
