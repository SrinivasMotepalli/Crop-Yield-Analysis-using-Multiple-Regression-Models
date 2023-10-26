import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"C:\Users\srini\Desktop\crop_yield_dataset.csv")

# Select relevant features and target
features = ['Crop', 'Rainfall', 'Area', 'Fertilizer_Name', 'Fertilizer_Used', 'Humidity', 'Temperature']
target = 'Produce'

# Filter the DataFrame
data = df[features + [target]]

# Convert categorical features to numerical using one-hot encoding
data = pd.get_dummies(data)

# Separate features and target
X = data.drop(target, axis=1)
y = data[target]

# Create and train a Random Forest Regressor on the entire dataset
model = LinearRegression()
model.fit(X, y)

# Streamlit app
st.title('Crop Produce Prediction')

# Sliders for numerical columns
rainfall = st.slider('Rainfall (mm)', 100.0, 700.0, step=1.0)
area = st.slider('Area (hectares)', 1.0, 90.0, step=1.0)
fertilizer_used = st.slider('Fertilizer Used (kg/ha)', 5.0, 170.0, step=1.0)
humidity = st.slider('Humidity (%)', 8.0, 42.0, step=1.0)
temperature = st.slider('Temperature (°C)', 12.0, 38.0, step=1.0)

# Dropdowns for categorical columns
crop_name = st.selectbox('Select Crop', ['Rice', 'Maize', 'Cotton', 'Wheat'])
fertilizer_name = st.selectbox('Select Fertilizer', ['Urea', 'NPK', 'DAP', 'Potash'])

# Create a DataFrame for the input data
input_data = pd.DataFrame([[crop_name, rainfall, area, fertilizer_name, fertilizer_used, humidity, temperature]],
                           columns=['Crop', 'Rainfall', 'Area', 'Fertilizer_Name', 'Fertilizer_Used', 'Humidity', 'Temperature'])

# Convert categorical features to numerical using one-hot encoding
input_data = pd.get_dummies(input_data)

# Align the columns of input_data with the columns of X
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Make a prediction
predicted_produce = model.predict(input_data)

# Display the prediction
st.subheader('Predicted Produce:')
st.write(predicted_produce[0])

