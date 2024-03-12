import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

def load_model():
    xgb_model = joblib.load('xgboost_vehicle_price_prediction_model.pkl')
    return xgb_model
regressor = load_model()

st.title('Vehicle Price Prediction')
st.subheader('Dataset Link: https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data/data')
st.write('Jovan Amarta Liem')

col1, col2 = st.columns(2)

with col1:
    Year = st.selectbox('Select the year of purchase', ('2015', '2014', '2013', '2012',
                                                        '2011', '2010', '2009', '2008',
                                                        '2007', '2006', '2005', '2004',
                                                        '2003', '2002', '2001', '2000',
                                                        '1999', '1998'))

with col2:
    Make = st.selectbox('Select who make the product', ('Kia', 'BMW', 'Volvo', 'Nissan', 'Chevrolet', 'Audi', 'Ford',
                                                        'Hyundai', 'Buick', 'Cadillac', 'Acura', 'Lexus', 'Infiniti',
                                                        'Jeep', 'Mercedes-Benz', 'Mitsubishi', 'Mazda', 'MINI',
                                                        'Land Rover', 'Lincoln', 'Jaguar', 'Volkswagen',
                                                        'Toyota', 'Subaru', 'Scion', 'Porsche', 'Dodge', 'FIAT',
                                                        'Chrysler', 'Ferrari', 'Honda', 'GMC', 'Ram',
                                                        'Smart', 'Bentley', 'Pontiac','Saturn', 'Maserati', 'Mercury', 
                                                        'Saab', 'Suzuki', 'HUMMER','Oldsmobile', 'Isuzu', 'Geo', 
                                                        'Rolls-Royce', 'Daewoo', 'Plymouth', 'Tesla', 'Lotus',
                                                        'Airstream', 'Dot', 'Aston Martin', 'Fisker', 'Lamborghini'))
    
with col1:
    Body = st.selectbox('Select body', ('SUV', 'Sedan', 'Convertible', 'Coupe', 'Wagon', 'Hatchback',
                        'Crew Cab', 'G Coupe', 'G Sedan', 'Elantra Coupe', 'Genesis Coupe',
                        'Minivan', 'Van', 'Double Cab', 'CrewMax Cab', 'Access Cab',
                        'King Cab', 'SuperCrew', 'CTS Coupe', 'Extended Cab',
                        'E-Series Van', 'SuperCab', 'Regular Cab', 'G Convertible', 'Koup',
                        'Quad Cab', 'CTS-V Coupe', 'G37 Convertible', 'Club Cab',
                        'Xtracab', 'Q60 Convertible', 'CTS Wagon', 'convertible',
                        'G37 Coupe', 'Mega Cab', 'Cab Plus 4', 'Q60 Coupe', 'Cab Plus',
                        'Beetle Convertible', 'TSX Sport Wagon', 'Promaster Cargo Van',
                        'GranTurismo Convertible', 'CTS-V Wagon', 'Ram Van',
                        'Transit Van', 'Navitgation'))
    
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
    Interior = st.selectbox('Select interior', ('black', 'beige', 'tan', 'gray', 'brown', 'burgundy', 'white',
                                                'silver', 'off-white', 'blue', 'red', 'yellow', 'green', 'purple',
                                                'orange', 'gold'))

with col2: 
    Mmr = st.number_input('Input MMR')

predict_btn = st.button('Predict')

user_input = pd.DataFrame({
    'year': [Year],
    'make': [Make],
    'body': [Body],
    'transmission': [Transmission],
    'state': [State],
    'condition': [Condition],
    'odometer': [Odometer],
    'color': [Color],
    'interior': [Interior],
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