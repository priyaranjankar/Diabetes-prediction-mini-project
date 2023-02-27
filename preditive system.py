# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:35:03 2023

@author: priya
"""

import numpy as np
import pickle as pkl
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load the saved model -rb means read binary
loaded_model = pkl.load(open('D:/AMPBA/Personal projects/Diabetes prediction/trained_model.sav','rb'))


#lets pick one row of input data from our dataset as a list( exculding the label data as that needs to be predicted by system):
input_data = (1,85,66,29,0,26.6,0.351,31)
#converting it to a numpy array & reshaping it for the model to understand it as 1 record as it was trained on multiple records
input_data_reshaped = np.asarray(input_data).reshape(1,-1)
#standardizing the values as the training happened on standardized values- need to perform the same for our input data
input_data_reshaped_std = scaler.transform(input_data_reshaped)

#predict the label
pred = loaded_model.predict(input_data_reshaped_std)

if (pred[0] == 1):
  print('This person is diabetic.')
else:
  print('This person is non-diabetic.')