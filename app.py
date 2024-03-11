import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('https://drive.google.com/uc?export=download&id=1-xMqX1B6LQN8ikHvGIyzL9h9yYzQEhTC')
def load_model():
    xgb_model = joblib.load('xgboost_vehicle_price_prediction_model.pkl')
    return xgb_model
regressor = load_model()

st.title('Vehicle Price Prediction')
st.subheader('Dataset Link: https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data/data')
st.write('Jovan Amarta Liem')

col1, col2 = st.columns(2)

with col1:
    Year = st.selectbox('Select the year of purchase', df['year'].unique())

with col2:
    Make = st.selectbox('Select who make the product', df['make'].unique())

with col1:
    Model = st.selectbox('Select model', df['model'].unique())
    
with col2:
    Trim = st.selectbox('Select trim', df['trim'].unique())
    
with col1:
    Body = st.selectbox('Select body', df['body'].unique())
    
with col2:
    Transmission = st.selectbox('Select transmission', ('Automatic', 'Manual'))

with col1: 
    State = st.selectbox('Select state', ('ca', 'tx', 'pa', 'mn', 'az', 'wi', 'tn', 'md', 'az', 'wi', 'tn', 'md', 'fl', 'ne', 'nj', 'nv', 'oh',
                                          'mi', 'ga', 'va', 'sc', 'nc', 'in', 'il', 'co', 'ut', 'mo', 'ny', 'ma', 'pr', 'or', 'la', 'wa', 'hi', 
                                          'qc', 'ab', 'on', 'ok', 'ms', 'nm', 'al', 'ns'))
    
with col2:
    Condition = st.number_input('Input condition')

with col1: 
    Odometer = st.number_input('Input odometer')
    
with col2:
    Color = st.selectbox('Select color', ('white', 'gray', 'black', 'red', 'silver', 'blue', 'brown', 'beige', 'purple', 'burgundy',
                                          'gold', 'yellow', 'green', 'charcoal', 'orange', 'turqoise', 'pink', 'lime'))

with col1:
    Interior = st.selectbox('Select interior', df['interior'].unique())

with col2: 
    Seller = st.selectbox('Select seller', df['seller'].unique())

with col1: 
    Mmr = st.number_input('Input MMR')

predict_btn = st.button('Predict')

user_input = pd.DataFrame({
    'year': [Year],
    'make': [Make],
    'model': [Model],
    'trim': [Trim],
    'body': [Body],
    'transmission': [Transmission],
    'state': [State],
    'condition': [Condition],
    'odometer': [Odometer],
    'color': [Color],
    'interior': [Interior],
    'seller': [Seller],
    'mmr': [Mmr]
})

def predict_vehicle_price(features):
    prediction_res = regressor.predict(features)
    return prediction_res[0]

def encoding_categorical_data(user_input):
    category_features = user_input.select_dtypes(include=['object']).columns
    label = LabelEncoder()
    encoded_input = user_input.copy()
    for feature in user_input[category_features]:
        encoded_input[feature] = label.fit_transform(encoded_input[feature]) 
    return encoded_input

if predict_btn:
    user_input = encoding_categorical_data(user_input)
    prediction = predict_vehicle_price(user_input)
    st.write(f'Predicted Vehicle Price {prediction:.2f}')