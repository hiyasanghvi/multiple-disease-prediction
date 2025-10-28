# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 11:46:32 2025
@author: hiyas
"""

import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# ----------------------------
# Load the saved models (using relative paths)
# ----------------------------
base_path = os.path.dirname(__file__)
diabetes_model_path = os.path.join(base_path, "saved_models", "diabetes_model.sav")
heart_model_path = os.path.join(base_path, "saved_models", "heart_disease_model.sav")
parkinsons_model_path = os.path.join(base_path, "saved_models", "parkinsons_model.sav")

# ------------------------------------------------
# Load the saved models
# ------------------------------------------------
with open(diabetes_model_path, "rb") as f:
    diabetes_model = pickle.load(f)
with open(heart_model_path, "rb") as f:
    heart_model = pickle.load(f)
with open(parkinsons_model_path, "rb") as f:
    parkinsons_model = pickle.load(f)


# ----------------------------
# Sidebar navigation
# ----------------------------
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# ----------------------------
# Diabetes Prediction Page
# ----------------------------
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction Using Machine Learning')

    # Input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Pregnancies', min_value=0)
        SkinThickness = st.number_input('Skin Thickness')
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function')
    with col2:
        Glucose = st.number_input('Glucose Level')
        Insulin = st.number_input('Insulin Level')
        Age = st.number_input('Age')
    with col3:
        BloodPressure = st.number_input('Blood Pressure')
        BMI = st.number_input('BMI')
        Outcome = st.selectbox('Known Outcome (Optional)', ['Unknown', '0', '1'])

    # Prediction
    diab_diagnosis = ''
    if st.button('🔍 Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure,
                                                   SkinThickness, Insulin, BMI,
                                                   DiabetesPedigreeFunction, Age]])
        diab_diagnosis = '✅ The person is diabetic' if diab_prediction[0] == 1 else '❎ The person is not diabetic'

    st.success(diab_diagnosis)

# ----------------------------
# Heart Disease Prediction Page
# ----------------------------
elif selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction Using Machine Learning')

    # Input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=1)
        trestbps = st.number_input('Resting Blood Pressure')
        fbs = st.number_input('Fasting Blood Sugar (1=True, 0=False)', min_value=0, max_value=1)
        exang = st.number_input('Exercise Induced Angina (1=True, 0=False)', min_value=0, max_value=1)
    with col2:
        sex = st.number_input('Sex (1=Male, 0=Female)', min_value=0, max_value=1)
        chol = st.number_input('Serum Cholesterol (mg/dl)')
        restecg = st.number_input('Resting ECG Results (0–2)', min_value=0, max_value=2)
        oldpeak = st.number_input('ST depression induced by exercise')
    with col3:
        cp = st.number_input('Chest Pain Type (0–3)', min_value=0, max_value=3)
        thalach = st.number_input('Maximum Heart Rate Achieved')
        slope = st.number_input('Slope of Peak Exercise ST Segment (0–2)', min_value=0, max_value=2)
        ca = st.number_input('Major Vessels Colored (0–4)', min_value=0, max_value=4)
        thal = st.number_input('Thal (1=Normal, 2=Fixed defect, 3=Reversible defect)', min_value=1, max_value=3)

    # Prediction
    heart_diagnosis = ''
    if st.button('💓 Heart Disease Test Result'):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs,
                                                 restecg, thalach, exang, oldpeak,
                                                 slope, ca, thal]])
        heart_diagnosis = '⚠️ The person has heart disease' if heart_prediction[0] == 1 else '✅ The person does not have heart disease'

    st.success(heart_diagnosis)

# ----------------------------
# Parkinson’s Prediction Page
# ----------------------------
elif selected == 'Parkinsons Prediction':
    st.title('🧠 Parkinson’s Disease Prediction Using Machine Learning')

    parkinsons_features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]

    inputs = []
    for feature in parkinsons_features:
        inputs.append(st.number_input(feature, format="%.6f"))

    parkinsons_diagnosis = ''
    if st.button('🧬 Parkinson’s Test Result'):
        parkinsons_prediction = parkinsons_model.predict([inputs])
        parkinsons_diagnosis = '⚠️ The person has Parkinson’s disease' if parkinsons_prediction[0] == 1 else '✅ The person does not have Parkinson’s disease'

    st.success(parkinsons_diagnosis)

