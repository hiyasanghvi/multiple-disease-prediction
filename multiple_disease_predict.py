# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 11:46:32 2025
@author: hiyas
"""

import os
import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# ------------------------------------------------
# Dynamically set model file paths
# ------------------------------------------------
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

# ------------------------------------------------
# Sidebar navigation
# ------------------------------------------------
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# =================================================================
# 🩸 Diabetes Prediction Page
# =================================================================
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction Using Machine Learning')

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input('Pregnancies', min_value=0)
        SkinThickness = st.number_input('Skin Thickness', min_value=0)
        BMI = st.number_input('BMI', min_value=0.0)
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0)
        Insulin = st.number_input('Insulin Level', min_value=0)
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0)
    with col3:
        BloodPressure = st.number_input('Blood Pressure', min_value=0)
        Age = st.number_input('Age', min_value=0)

    diab_diagnosis = ''
    if st.button('Predict Diabetes'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure,
                                                    SkinThickness, Insulin, BMI,
                                                    DiabetesPedigreeFunction, Age]])
        diab_diagnosis = '⚠️ Diabetic' if diab_prediction[0] == 1 else '✅ Not Diabetic'
    st.success(diab_diagnosis)

# =================================================================
# ❤️ Heart Disease Prediction Page
# =================================================================
elif selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction Using Machine Learning')

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', min_value=0)
        cp = st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3)
        chol = st.number_input('Cholesterol (mg/dl)', min_value=0)
        restecg = st.number_input('Resting ECG (0-2)', min_value=0, max_value=2)
        slope = st.number_input('Slope (0-2)', min_value=0, max_value=2)
    with col2:
        sex = st.number_input('Sex (1=Male, 0=Female)', min_value=0, max_value=1)
        trestbps = st.number_input('Resting BP (mm Hg)', min_value=0)
        fbs = st.number_input('Fasting BS > 120 (1=True, 0=False)', min_value=0, max_value=1)
        exang = st.number_input('Exercise Induced Angina (1=Yes, 0=No)', min_value=0, max_value=1)
        ca = st.number_input('Major Vessels (0-3)', min_value=0, max_value=3)
    with col3:
        thalach = st.number_input('Max Heart Rate', min_value=0)
        oldpeak = st.number_input('Oldpeak', min_value=0.0)
        thal = st.number_input('Thal (0=Normal,1=Fixed,2=Reversible)', min_value=0, max_value=2)

    heart_diagnosis = ''
    if st.button('Predict Heart Disease'):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs,
                                                 restecg, thalach, exang, oldpeak,
                                                 slope, ca, thal]])
        heart_diagnosis = '⚠️ Heart Disease Detected' if heart_prediction[0] == 1 else '✅ No Heart Disease'
    st.success(heart_diagnosis)

# =================================================================
# 🧠 Parkinson’s Prediction Page
# =================================================================
elif selected == 'Parkinsons Prediction':
    st.title('🧠 Parkinson’s Disease Prediction Using Machine Learning')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fo = st.number_input('MDVP:Fo(Hz)')
        rap = st.number_input('MDVP:RAP')
        shimmer = st.number_input('MDVP:Shimmer')
        hnr = st.number_input('HNR')
        spread1 = st.number_input('spread1')
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)')
        ppq = st.number_input('MDVP:PPQ')
        shimmer_db = st.number_input('MDVP:Shimmer(dB)')
        status = st.number_input('Status (0=Healthy,1=Parkinsons)')
        spread2 = st.number_input('spread2')
    with col3:
        flo = st.number_input('MDVP:Flo(Hz)')
        ddp = st.number_input('Jitter:DDP')
        apq = st.number_input('MDVP:APQ')
        rpde = st.number_input('RPDE')
        D2 = st.number_input('D2')
    with col4:
        jitter_perc = st.number_input('MDVP:Jitter(%)')
        jitter_abs = st.number_input('MDVP:Jitter(Abs)')
        shimmer_apq3 = st.number_input('Shimmer:APQ3')
        DFA = st.number_input('DFA')
        PPE = st.number_input('PPE')

    parkinsons_diagnosis = ''
    if st.button("Predict Parkinson’s"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, jitter_perc, jitter_abs, rap,
                                                           ppq, ddp, shimmer, shimmer_db, shimmer_apq3,
                                                           apq, hnr, rpde, DFA, spread1, spread2, D2, PPE]])
        parkinsons_diagnosis = '⚠️ Parkinson’s Detected' if parkinsons_prediction[0] == 1 else '✅ No Parkinson’s'
    st.success(parkinsons_diagnosis)

