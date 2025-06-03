import streamlit as st
import numpy as np
import joblib

# Load your trained model
model = joblib.load('irirUi.pkl')

# Title and intro
st.title("🌸 Iris Flower Species Predictor")
st.write("Adjust the sliders below to input the flower's measurements and predict its species.")

# Sliders for input features
sepal_length = st.slider('🌿 Sepal Length (cm)', min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.slider('🌿 Sepal Width (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.slider('🌺 Petal Length (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.slider('🌺 Petal Width (cm)', min_value=0.0, max_value=10.0, step=0.1)

# Prediction on button click
if st.button('🔍 Predict Species'):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    # Map numeric prediction to actual species name
    species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    predicted_species = species_mapping[prediction[0]]

    st.success(f"✅ The predicted species is: **{predicted_species}**")
