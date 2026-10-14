# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:51:43 2026

@author: cathe
"""

import os
import numpy as np
import pickle
import streamlit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Loading the saved model

loaded_model = pickle.load(open(os.path.join(BASE_DIR, 'trained_model.sav'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'scaler.sav'), 'rb'))


#loaded_model = pickle.load(open('trained_model.sav', 'rb'))


#loaded_model = pickle.load(open(r'C:\Users\cathe\Documents\DATA SCIENCE MASTERS\PROJECTS\trained_model.sav', 'rb'))
#scaler = pickle.load(open('scaler.sav', 'rb'))   # <-- this was missing


#creating a function for prediction

def diabetes_prediction(input_data):
    
    input_data = (7, 196, 90, 0, 0, 39.8, 0.451, 41)
    input_data_as_numpy_array = np.array(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    std_data = scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(std_data)
    
    if prediction[0] == 0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"


def main():
    streamlit.title("Diabetes Prediction")
    
    Pregnancies = streamlit.text_input("Number of pregnancies")
    Glucose = streamlit.text_input("Glucose level")
    BloodPressure = streamlit.text_input("Blood pressure value")
    SkinThickness = streamlit.text_input("Skinthickness value")
    Insulin = streamlit.text_input("Insulin level value")
    BMI = streamlit.text_input("BMI value")
    DiabetesPedigreeFunction = streamlit.text_input("DiabetesPedigreeFunction value")
    Age = streamlit.text_input("Age value of the person")
    
    if streamlit.button("Diabetes test Button"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness,
                                          Insulin, BMI, DiabetesPedigreeFunction, Age])
        streamlit.success(diagnosis)   # now only runs after the button is clicked


if __name__ == '__main__':
    main()

   
    
    
    
    