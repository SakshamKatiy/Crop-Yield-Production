import streamlit as st
import numpy as np
import pickle

try:
    # Loading models
    dtr = pickle.load(open('dtr.pkl', 'rb'))
    preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading models: {e}")
    dtr = None
    preprocessor = None

st.title('Crop Yield Prediction Per Country')

st.markdown('## Crop Features Here')

year = st.number_input('Year')
average_rain_fall_mm_per_year = st.number_input('Average Rainfall (mm per year)')
pesticides_tonnes = st.number_input('Pesticides (tonnes)')
avg_temp = st.number_input('Average Temperature')

area_options = ["Albania", "Algeria", "Angola","Argentina","Bahamas","Bahrain","Bangladesh","Belarus","Canada","Egypt","India","Nepal","New Zealand","Pakistan"]  # Add your actual options here
area = st.selectbox('Area', area_options)

item_option = ["Maize","Potatoes","Wheat","Rice, paddy","Soybeans","Sweet potatoes","Cassava","Yams","Plantains and others"] 
item = st.selectbox('Item', item_option)

if st.button('Predict'):
    # Separate encoding for numerical and categorical features
    numerical_features = np.array([[year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]])
    categorical_features = np.array([[area, item]])

    # Transform numerical features
    numerical_transformed_features = preprocessor.named_transformers_['num'].transform(numerical_features)

    # Transform categorical features
    categorical_transformed_features = preprocessor.named_transformers_['cat'].transform(categorical_features).toarray()

    # Combine numerical and categorical features
    features = np.concatenate([categorical_transformed_features, numerical_transformed_features], axis=1)

    predicted_value = dtr.predict(features).reshape(1, -1)

    st.markdown('## Predicted Yield Productions:')
    st.write(predicted_value)


