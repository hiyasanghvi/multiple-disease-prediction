# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 11:46:32 2025
@author: hiyas
"""

import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# ----------------------------
# Load the saved models
# ----------------------------
diabetes_model = pickle.load(open("C:/Users/hiyas/OneDrive/Desktop/multiple disease prediction/saved models/diabetes_model.sav", "rb"))
heart_model = pickle.load(open("C:/Users/hiyas/OneDrive/Desktop/multiple disease prediction/saved models/heart_disease_model.sav", "rb"))
parkinsons_model = pickle.load(open("C:/Users/hiyas/OneDrive/Desktop/multiple disease prediction/saved models/parkinsons_model.sav", "rb"))

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

# =================================================================
# 🩸 Diabetes Prediction Page
# =================================================================
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction Using Machine Learning')

    # Input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0)
        SkinThickness = st.number_input('Skin Thickness value', min_value=0)
        BMI = st.number_input('BMI value', min_value=0.0)
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0)
        Insulin = st.number_input('Insulin Level', min_value=0)
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0)
    with col3:
        BloodPressure = st.number_input('Blood Pressure value', min_value=0)
        Age = st.number_input('Age of the Person', min_value=0)

    # Prediction
    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure,
                                                    SkinThickness, Insulin, BMI,
                                                    DiabetesPedigreeFunction, Age]])
        if diab_prediction[0] == 1:
            diab_diagnosis = '⚠️ The person is likely Diabetic.'
        else:
            diab_diagnosis = '✅ The person is not Diabetic.'
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
        chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0)
        restecg = st.number_input('Resting ECG Results (0-2)', min_value=0, max_value=2)
        slope = st.number_input('Slope of Peak Exercise ST Segment (0-2)', min_value=0, max_value=2)
    with col2:
        sex = st.number_input('Sex (1=Male, 0=Female)', min_value=0, max_value=1)
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0)
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', min_value=0, max_value=1)
        exang = st.number_input('Exercise Induced Angina (1=Yes, 0=No)', min_value=0, max_value=1)
        ca = st.number_input('Major Vessels Colored by Fluoroscopy (0-3)', min_value=0, max_value=3)
    with col3:
        thalach = st.number_input('Max Heart Rate Achieved', min_value=0)
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0)
        thal = st.number_input('Thal (0=Normal, 1=Fixed Defect, 2=Reversible Defect)', min_value=0, max_value=2)

    # Prediction
    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs,
                                                 restecg, thalach, exang, oldpeak,
                                                 slope, ca, thal]])
        if heart_prediction[0] == 1:
            heart_diagnosis = '⚠️ The person is likely to have Heart Disease.'
        else:
            heart_diagnosis = '✅ The person does not have Heart Disease.'
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
        status = st.number_input('Status (0=Healthy, 1=Parkinsons)')
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

    # Prediction
    parkinsons_diagnosis = ''
    if st.button("Parkinson’s Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, jitter_perc, jitter_abs, rap,
                                                           ppq, ddp, shimmer, shimmer_db, shimmer_apq3,
                                                           apq, hnr, rpde, DFA, spread1, spread2, D2, PPE]])
        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = '⚠️ The person has Parkinson’s Disease.'
        else:
            parkinsons_diagnosis = '✅ The person does not have Parkinson’s Disease.'
    st.success(parkinsons_diagnosis)
