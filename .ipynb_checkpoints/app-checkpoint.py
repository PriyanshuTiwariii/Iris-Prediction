import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('iris_model.pkl')

st.title('Iris Species Prediction')

# Define input fields
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0)

# Prediction
if st.button('Predict'):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    species = iris.target_names[prediction][0]
    st.write(f'The predicted species is {species}')
