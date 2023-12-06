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

area_options = ["Albania", "Algeria", "Angola", "Argentina", "Bahamas", "Bahrain", "Bangladesh", "Belarus", "Canada", "Egypt", "India", "Nepal", "New Zealand", "Pakistan"]
area = st.selectbox('Area', area_options)

item_option = ["Maize", "Potatoes", "Wheat", "Rice, paddy", "Soybeans", "Sweet potatoes", "Cassava", "Yams", "Plantains and others"]
item = st.selectbox('Item', item_option)

if st.button('Predict'):
    if preprocessor is not None:
        # Separate encoding for numerical and categorical features
        features = np.array([[year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, area, item]])

        try:
            print("Original Features:", features)

            # Transform features
            transformed_features = preprocessor.transform(features)
            print("Transformed Features:", transformed_features)

            predicted_value = dtr.predict(transformed_features).reshape(1, -1)

            st.markdown('## Predicted Yield Productions:')
            st.write(predicted_value)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Add these lines to print information about the loaded models
print("Decision Tree Regressor:", dtr)
print("Preprocessor:", preprocessor)





