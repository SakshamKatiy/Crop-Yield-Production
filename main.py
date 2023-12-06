import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Assume X_train is your training data
X_train = np.array([[...]])  # Your actual training data here

# Create and fit the preprocessor
preprocessor = StandardScaler()
preprocessor.fit(X_train)

# Create and fit the model
dtr = RandomForestRegressor()
dtr.fit(X_train, y_train)  # Assuming y_train is your target variable

# Streamlit app
st.title('Yield Production Prediction App')

# Input features
area = st.slider('Area', min_value=0, max_value=1000)
item = st.selectbox('Item', ['Wheat', 'Corn', 'Rice'])
year = st.slider('Year', min_value=2000, max_value=2023)
average_rain_fall_mm_per_year = st.slider('Average Rainfall (mm/year)', min_value=0, max_value=2000)
pesticides_tonnes = st.slider('Pesticides (tonnes)', min_value=0, max_value=100)
avg_temp = st.slider('Average Temperature', min_value=-10, max_value=40)

# Prediction button
if st.button('Predict'):
    # Collect input features
    features = np.array([[area, item, year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]])

    # Transform features using the preprocessor
    transformed_features = preprocessor.transform(features)

    # Make prediction
    predicted_value = dtr.predict(transformed_features)

    # Display prediction
    st.markdown('## Predicted Yield Productions:')
    st.write(predicted_value)






