# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:51:43 2026

@author: cathe
"""


# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import streamlit as st

# Get the directory containing this Python file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define model file paths
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.sav")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.sav")

# Check whether the files exist before loading
if not os.path.exists(MODEL_PATH):
    st.error("trained_model.sav is missing from the project repository.")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error("scaler.sav is missing from the project repository.")
    st.stop()

# Load the saved model and scaler
with open(MODEL_PATH, "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


# Prediction function
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(
        input_data, dtype=float
    ).reshape(1, -1)

    std_data = scaler.transform(input_data_as_numpy_array)
    prediction = loaded_model.predict(std_data)

    if prediction[0] == 0:
        return "The model predicts a negative diabetes classification."
    else:
        return "The model predicts a positive diabetes classification."


# Streamlit application
def main():
    st.title("Diabetes Prediction")

    pregnancies = st.number_input(
        "Number of pregnancies", min_value=0, step=1
    )
    glucose = st.number_input("Glucose level", min_value=0.0)
    blood_pressure = st.number_input(
        "Blood pressure value", min_value=0.0
    )
    skin_thickness = st.number_input(
        "Skin thickness value", min_value=0.0
    )
    insulin = st.number_input("Insulin level value", min_value=0.0)
    bmi = st.number_input("BMI value", min_value=0.0)
    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0
    )
    age = st.number_input("Age", min_value=1, step=1)

    if st.button("Run prediction"):
        input_data = [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age,
        ]

        diagnosis = diabetes_prediction(input_data)
        st.success(diagnosis)
        st.caption(
            "This is a machine-learning prediction, not a medical diagnosis."
        )


if __name__ == "__main__":
    main()
