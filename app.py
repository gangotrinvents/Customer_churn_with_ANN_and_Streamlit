import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

## Load trained model
model = tf.keras.models.load_model('model.h5')

## Scaler encoder
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Gender label encoder
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

## OneHot encoder
with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

## Streamlit app
st.title('Customer Churn Predicition')


# User input
credit_score = st.number_input('Credit Score')
geo = encoder.categories_[0]
geography = st.selectbox('Geography', geo)
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure', 0,12)
balance = st.number_input('Balance')
num_of_products = st.slider('NumOfProducts', 1, 4)
has_cr_card = st.selectbox('Har Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Memmber', [0, 1])
estimated_salary = st.number_input('Estimated Salary')

input_data = pd.DataFrame({
   'CreditScore' : [credit_score],
    'Geography' : [geography],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
    })

## One-hot encode 'Geography'
encoder_geo = encoder.transform([[geography]])
encoder_geo_df = pd.DataFrame(encoder_geo, columns= encoder.get_feature_names(['Geography']))

## Combine one-hot encode 'Geography' with main data
input_data = pd.concat([input_data.reset_index(drop=True),encoder_geo_df], axis=1)

## Drop columns: 'Geography'
input_data = input_data.drop(['Geography'], axis= 1)

## Scale input data
input_data_scaled = scaler.transform(input_data)

## Predict churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

if prediction_prob > 0.5:
    st.write('Customers are likely to churn with probability of {0}'.format(prediction_prob*100))
else:
    st.write('Customers are not likely to churn with probability of {0}'.format(prediction_prob*100))