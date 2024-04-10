import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pipeline_ import predict_survival

st.title('Titanic Survival Prediction')
st.write('This is a simple web app to predict the survival of passengers on the Titanic.')
st.header('Passenger Details')

passenger_id = st.number_input('Passenger ID', min_value=0)
pclass = st.selectbox('Passenger Class', ['1', '2', '3'])
name = st.text_input('Name')

sex = st.selectbox('Sex',["male","female"])

age = st.number_input('Age', min_value=0)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0)
parch = st.number_input('Number of Parents/Children Aboard', min_value=0)
ticket = st.text_input('Ticket Number')
fare = st.number_input('Fare', min_value=0.0)
cabin = st.text_input('Cabin')
embarked = st.selectbox('Embarked', ['C', 'Q', 'S','N'])

data = pd.DataFrame({'PassengerId': passenger_id, 'Pclass': pclass , 'Name': name,
                     'Sex':sex, 'Age': age, 'SibSp': sibsp, 'Parch': parch,
                        'Ticket': ticket, 'Fare': fare, 'Cabin': cabin, 'Embarked': embarked}, index=[0])
prediction = 0

if st.button('Predict Survival'):
    print(data)
    prediction = predict_survival(data)
    print(prediction)

st.header('Prediction')
if prediction == 1:
    st.write('The passenger is likely to survive.')
else:

    st.write('The passenger is unlikely to survive.')



