# 🎓 Graduate Admission Chance Predictor

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)

## 📄 Overview
This project is a Machine Learning web application that predicts a student's chances of getting admitted into a Master's program.

By analyzing historical data of applicants (GRE scores, TOEFL scores, CGPA, etc.), the model calculates a probability percentage, helping students understand where they stand in a competitive market.

## 🔗 Live Demo
👉 **[Click here to view the Live App](https://admission-predictor-fzdhmrw9g5spzeluyayyxo.streamlit.app/)**

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn (Linear Regression)
* **Data Manipulation:** Pandas, NumPy
* **IDE:** VS Code

## 📊 Dataset
The model is trained on the **Graduate Admission 2** dataset from Kaggle.
* **Inputs:** GRE Score, TOEFL Score, University Rating, SOP Strength, LOR Strength, CGPA, Research Experience.
* **Target:** Chance of Admit (0 to 1).

## 🚀 How to Run Locally

If you want to run this project on your own machine, follow these steps:

**1. Clone the repository**


Create a virtual Environment - 
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate


Install Dependencies - 
pip install -r requirements.txt


Run The App - 
streamlit run app.py


Author - 
Priyanshu Soni
```bash
git clone [https://github.com/YourUsername/admission-predictor.git](https://github.com/YourUsername/admission-predictor.git)
cd admission-predictor
