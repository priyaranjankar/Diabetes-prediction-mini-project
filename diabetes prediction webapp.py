    # -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:39:52 2023

@author: priya
"""

import numpy as np
import pickle as pkl
import streamlit as st



#load the saved model -rb means read binary
loaded_model = pkl.load(open("D:/AMPBA/Personal projects/Diabetes prediction/trained_model.sav",'rb'))

#creating a function for prediction:
def diabetes_prediction(input_data):

    #converting it to a numpy array & reshaping it for the model to understand it as 1 record as it was trained on multiple records
    input_data_reshaped = np.asarray(input_data).reshape(1,-1)
    #standardizing the values as the training happened on standardized values- need to perform the same for our input data
    
    #predict the label
    pred = loaded_model.predict(input_data_reshaped)

    if (pred[0] == 1):
      return 'This person is diabetic.'
    else:
      return 'This person is non-diabetic.'
  
#function for streamlit 
def main():
    
    #giving a title(of our web page):
    st.title('Diabetes Prediction Web App')
    
    #getting the inout_data from users:
    
    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure level(mm Hg)')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin level(mm)')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age(years)')
    
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction:
    if st.button('Diabetes test result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness,
                                         Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    