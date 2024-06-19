import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load model
model = joblib.load('iris_model.pkl')

# Load iris data for target names
iris = load_iris()

st.title('Hello Shatakshi!! Welcome to my Iris Species Prediction model')

st.write("Enter the features to predict the Iris species:")

# Define input fields
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0)

# Prediction
if st.button('Predict'):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    
    # Ensure prediction is an array and access the first element
    predicted_class = prediction[0]
    species = iris.target_names[predicted_class]
    
    st.write(f'The predicted species is {species}')
