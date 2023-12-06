import streamlit as st
import numpy as np
import pickle

try:
    # Loading models
    dtr = pickle.load(open('dtr.pkl', 'rb'))
    preprocesser = pickle.load(open('preprocesser.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading models: {e}")
    dtr = None
    preprocesser = None

st.title('Crop Yield Prediction Per Country')

st.markdown('## Crop Features Here')

year = st.slider('Year', min_value=2000, max_value=2023)
average_rain_fall_mm_per_year = st.number_input('Average Rainfall (mm per year)')
pesticides_tonnes = st.number_input('Pesticides (tonnes)')
avg_temp = st.number_input('Average Temperature')


area_options = ["Albania", "Algeria", "Angola","Argentina","Bahamas","Bahrain","Bangladesh","Belarus","Canada","Egypt","India","Nepal","New Zealand","Pakistan"]  # Add your actual options here

area = st.selectbox('Area', area_options)

item_option = ["Maize","Potatoes","Wheat","Rice, paddy","Soybeans","Sweet potatoes","Cassava","Yams","Plantains and others"] 
item= st.selectbox('Item',item_option)

if st.button('Predict'):
    raw_features = np.array([[area, item, year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]])
    
    # Assuming preprocessor is already fitted
    preprocessed_features = preprocessor.transform(raw_features)
    
    try:
        predicted_value = dtr.predict(preprocessed_features)
        st.markdown('## Predicted Yield Productions:')
        st.write(predicted_value)
    except Exception as e:
        st.error(f"Error predicting: {e}")







