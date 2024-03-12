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
    Model = st.selectbox('Select model', ('Sorento', '3 Series', 'S60', '6 Series Gran Coupe', 'Altima',
                        'M5', 'Cruze', 'A4', 'Camaro', 'A6', 'Optima', 'Fusion', 'Sonata',
                        'Q5', '6 Series', 'Impala', '5 Series', 'A3', 'XC70', 'X5', 'SQ5',
                        'S5', 'Verano', 'Suburban', 'ELR', 'V60', 'X6', 'ILX', 'K900',
                        'Malibu', 'RX 350', 'Versa', 'Elantra', 'Versa Note', 'A8', 'X1',
                        'Enclave', 'TTS', '4 Series', 'Silverado 2500HD', 'MDX',
                        'Silverado 1500', 'SRX', 'G Coupe', 'G Sedan', 'FX', 'Santa Fe',
                        'Genesis', 'Equus', 'Sonata Hybrid', 'Accent', 'Veloster',
                        'Elantra Coupe', 'Azera', 'Tucson', 'Genesis Coupe', 'Wrangler',
                        'S-Class', 'GS 350', 'Outlander', 'C-Class', 'Mazda2', 'Rio', 'M',
                        '370Z', 'Soul', 'Outlander Sport', 'SLK-Class', 'ES 350',
                        'E-Class', 'Mazda3', 'Cooper Clubman', 'Cooper', 'CX-9', 'Forte',
                        'Compass', 'JX', 'RX 450h', 'LR4', 'Mazda5', 'Range Rover Evoque',
                        'LS 460', 'GLK-Class', 'Sportage', 'Grand Cherokee', 'MKX', 'mkt',
                        'XF', 'GL-Class', 'M-Class', 'Cooper Countryman', 'Lancer',
                        'Range Rover Sport', 'Passat', 'Corolla', 'XC60', 'Sienna', 'Juke',
                        'Yaris', 'Sentra', 'Rogue', 'NV', 'CC', 'Leaf', 'Camry', 'Tacoma',
                        'Jetta', 'Impreza WRX', 'FJ Cruiser', 'Beetle', 'Avalon', 'FR-S',
                        'NV200', 'RAV4', 'Quest', 'Tundra', 'tC', 'Maxima', 'Cayenne',
                        '911', 'Xterra', 'Prius', 'S80', 'Frontier', 'Boxster',
                        'Camry Hybrid', 'xB', 'Cube', 'Jetta SportWagen', '4Runner',
                        'Sequoia', 'Legacy', 'Armada', 'Venza', 'Murano', 'Pathfinder',
                        'Panamera', 'Forester', 'Highlander', 'Impreza', '750i', 'TSX',
                        '7 Series', '1 Series', 'TL', '750li', 'S4', 'A7', 'A5', 'RDX',
                        'M3', 'Cooper Coupe', 'ZDX', 'R8', 'X3', 'Avenger',
                        'E-Series Wagon', 'Escape', 'Edge', 'Focus', 'Flex', 'Z4',
                        'Traverse', 'F-350 Super Duty', 'Fiesta', '500', '200', 'Journey',
                        'Charger', 'e350', 'Equinox', '300', 'F-150', 'Explorer',
                        'Captiva Sport', 'Escalade', 'Grand Caravan', 'CTS Coupe',
                        'Town and Country', 'E-Series Van', 'Volt', 'Express Cargo',
                        'e150', 'X5 M', 'Expedition', 'Colorado', 'Express', 'California',
                        'Escalade ESV', 'Sonic', 'Accord', 'CR-V', 'Mustang', 'Civic',
                        'Fit', 'Pilot', 'Odyssey', 'Crosstour', 'Transit Connect',
                        'Terrain', 'Taurus', 'G Convertible', 'Yukon', 'Veracruz', 'XJ',
                        'Liberty', 'IS 250', 'XK', 'QX', 'CT 200h', 'Mazda6', 'MKZ',
                        'Navigator', 'Range Rover', 'SL-Class', 'Sedona', 'IS 350',
                        'Patriot', 'galant', '1500', 'GT-R', '2500', 'Galant', 'fortwo',
                        'GLI', '5 Series Gran Turismo', 'XC90', 'Tiguan', 'GTI', 'Q7',
                        'Highlander Hybrid', 'Prius Plug-in', 'CR-Z', 'EX', 'Sierra 1500',
                        'LaCrosse', 'HHR', 'Accord Crosstour', 'CTS', 'Nitro', 'Tahoe',
                        'Challenger', 'CTS-V', 'Escape Hybrid', 'X6 M', 'Ranger',
                        'Insight', 'Fusion Hybrid', 'CTS-V Coupe', 'F-250 Super Duty',
                        'Acadia', 'Impala Limited', 'Dart', 'Spark', 'M37', 'Sprinter',
                        'Town Car', 'CLS-Class', 'CX-7', 'MKT', 'QX56', 'Aveo', 'Outback',
                        'Caliber', 'Routan', 'g1500', 'Sebring', 'Corvette',
                        'Continental GT Speed', 'malibu', 'Land Cruiser', 'town', 'V50',
                        'Commander', 'Altima Hybrid', 'G37 Convertible', 'g6',
                        'New Beetle', 'Golf', 'LR2', 'Lancer Sportback', 'G5', 'Yukon XL',
                        'Escalade Hybrid', 'Avalanche', 'Titan', 'Spectra', 'Rondo',
                        'Borrego', 'G-Class', 'MKS', 'CLK-Class', 'Tahoe Hybrid',
                        'Econoline Cargo', 'Econoline Wagon', 'PT Cruiser', 'STS',
                        'Ridgeline', 'F-450 Super Duty', 'Magnum', 'Durango', 'S40',
                        'Malibu Classic', 'TT', 'Taurus X', 'Explorer Sport Trac',
                        'Ram Pickup 1500', 'impala', 'Cobalt', 'Pacifica', 'S6', 'Rabbit',
                        'C70', 'Sierra 2500HD', 'C30', 'VUE', 'GranTurismo', 'G6',
                        'Grand Prix', '350Z', 'Raider', 'Mazdaspeed Mazda3', 'Solstice',
                        'Milan', 'GX 470', 'Aura', 'RX 400h', 'Matrix', 'H3', 'CL-Class',
                        'Outlook', '7', 'G37', 'IS F', 'Touareg 2', 'Lancer Evolution',
                        'G35', 'xD', 'XJ-Series', 'G8', 'hhr', 'H2', 'DTS', 'lr3', 'sts',
                        'Silverado 1500 Classic', 'M45', 'Uplander', 'GS 450h', 'corvette',
                        'rangerover', 'Rendezvous', 'Monte Carlo', 'FX35', 'range', 'ION',
                        'R-Class', 'lancer', 'Eclipse', 'c230wz', 'cx-7', 'B9 Tribeca',
                        'matrix', 'tundra', 'RSX', 'mazda5', 'Mariner', 'gx',
                        'Five Hundred', 'Envoy XL', 'S-Type', 'Element',
                        'Continental Flying Spur', 'S2000', 'FX45', 'sr', 'pilot',
                        'GS 430', 'Cayman S', 'Mark LT', 'ES 330', 'GS 300', '350z',
                        'Camry Solara', 'rx8', 'Touareg', 'Relay', 'lx', 'allroad quattro',
                        '9-3', '500L', 'C-Max Hybrid', 'pacifica', 'Freestyle',
                        'Ram Pickup 3500', 'Sprinter Cargo', 'DeVille', 'H2 SUT',
                        'TrailBlazer', 'Canyon', 'srx', 'Dakota', 'Continental GT', 'Neon',
                        'Stratus', 'Q45', 'Freestar', 'Montana', 'Grand Marquis', 'XLR',
                        'Aviator', 'g55', 'MPV', 'LS 430', 'Verona', 'Forenza', 'RX 330',
                        '300M', 'SC 430', 'discovery', 'Excursion', 'Envoy XUV', 'Envoy',
                        'Concorde', 'Monterey', 'stratus', 'Mountaineer', 'Amanti',
                        'Malibu Maxx', 'Celica', 'Grand Am', 'Endeavor', 'Marauder',
                        'escape', 'QX4', 'LS', 'Blazer', 'Ram Pickup 2500', 'LeSabre',
                        'V40', 'Mazdaspeed Protege', 'Montero', 'ES 300', 'focus',
                        'Thunderbird', 'Century', 'Cavalier', 'Venture', 'S-10', 's55',
                        'Cougar', 'XL-7', 'Windstar', 'Silverado 1500HD', 'Explorer Sport',
                        'Savana Cargo', 'X-Type', 'Sonoma', 'IS 300', 'forester',
                        'Protege5', 'sprinter', 'RL', 'Alero', 'Grand Vitara', 'RX 300',
                        'L-Series', 'V70', 'Intrigue', 'XC', 'Discovery Series II',
                        'S-Series', 'alero', 'santa', 'ECHO', 'MX-5 Miata', 'Continental',
                        'Seville', 'camry', 'Park Avenue', 'Millenia', 'I30', 'gr',
                        'sienna', 'camaro', 'Cherokee', 'Z3', 'civic', 'ram', 'odyssey',
                        'taurus', 'expedition', 'Prizm', 'Escort', 's10', 'LHS',
                        'windstar', 'f250', 'Regal', 'explorer', 'G20', 'Bonneville',
                        'Eldorado', 'voyager', 'venture', 'durango', 'Intrepid', 'Contour',
                        'S90', 'Sunfire', 'mpv', 'caravan', '200SX', 'Rodeo', 'wrangler',
                        'f150', 'Tercel', 'S70', 'Discovery', 'Mustang SVT Cobra', '300e',
                        'pickup', 'ciera', 'Legend', 'LS 400', 'Cutlass Ciera',
                        'Santa Fe Sport', 'Cadenza', 'Q50', 'Elantra GT', 'F-TYPE',
                        'Shelby GT500', 'QX70', 'QX60', 'Q60 Convertible',
                        'Cooper Roadster', 'CX-5', 'Cooper Paceman', 'Rogue Select',
                        'Cayman', 'CLA-Class', 'allroad', 'ATS', 'Prius v',
                        'Continental GTC', 'XV Crosstrek', '3500', 'C-Max Energi',
                        'Focus ST', 'RS 7', 'GX 460', 'CTS Wagon', 'SLS AMG', 'Aspen',
                        'Eclipse Spyder', 'Vibe', 'Eos', 'Entourage', 'expeditn', 'rl',
                        'Caravan', 'Quattroporte', 'M35', '9-5', 'SSR', 'Astro Cargo',
                        'Safari Cargo', 'passat', 'Tribute', 'Diamante', 'Sable',
                        'Silverado 3500', 'Phaeton', 'R32', 'I35', 'g500', 'Bravada',
                        'Tahoe Limited/Z71', 'Truck', 'C/K 1500 Series', 'grand', 'SC 300',
                        'Roadmaster', 'SC 400', '420sel', 'LX 570', 'QX80', 'RS 5',
                        'Jetta GLI', 'ES 300h', 'capt', 'M4', 'SX4', 'iQ', 'Kizashi',
                        'C/V Cargo Van', 'Prius c', '750lxi', 'alp', 'Lucerne',
                        'Escalade EXT', 'Silverado 3500HD', 'Crown Victoria',
                        'Sierra 3500HD', 'M56', 'rio', 'IS 250 C', '3', 'endeavor',
                        'corolla', 'jetta', 'ActiveHybrid X6', 'a4', 'dts', 'g3500',
                        'colorado', 'sebring', 'e250', 'police', 'Elantra Touring',
                        'G37 Coupe', 'HS 250h', 'journey', 'Mazdaspeed3', 'Milan Hybrid',
                        'Ghost', 'Silverado 1500 Hybrid', 'crown', 'Yukon Hybrid',
                        'elantra', 'optima', 'borrego', 'mazda6', '6', 'Mariner Hybrid',
                        'montana', 'Torrent', 'VUE Hybrid', 'G3', '9-7X', 'vibe',
                        'Expedition EL', 'Tiburon', 'patriot', 'LR3', 'Navigator L',
                        'Astra', 'Tribeca', 'XL7', 'sx4', 'Sky', 'Reno', 'M6', 'S8',
                        'Terraza', 'Silverado 2500HD Classic', 'uplander', 'ram3500',
                        'Sierra 1500 Classic', 'Sierra 2500HD Classic', 'Fusion Energi',
                        'XK-Series', 'quattroporte', 'B-Series Truck', 'mazda3', 'b200',
                        'Mazdaspeed Mazda6', 'Montego', 'g5', 'yaris', 'Rainier',
                        'TrailBlazer EXT', 'optra', 'Crossfire', 'magnum', 'savana',
                        'Sierra 1500HD', 'ridgelin', 'Savana', 'Zephyr', 'rrs', 'tribute',
                        'carrera', 'Montana SV6', 'wave', 'tt', 'Classic', 'pt',
                        'freestyle', 'Ascender', 'XG350', 'Q60 Coupe', 'Q70', 'QX50',
                        'pursuit', 'x-trail', 'GTO', 'xA', 'L300', 'Baja', '9-2X', 'Aerio',
                        'el', 'rainier', 'Silverado 2500', 'Astro', 'Tracker', 'intrepid',
                        'F-150 Heritage', 'expedit', 'accord', 'Freelander', 'c240w',
                        'MR2 Spyder', 'RS 6', '320i', 'Voyager', 'Axiom', 'cl55',
                        'Montero Sport', 'sl55', 'Protege', 'Silhouette', 'b1500',
                        'concorde', '626', 'Blackwood', 'Rodeo Sport', 'LX 470',
                        'Villager', 'Firebird', 'Aztek', 'Aurora', 'CL', 'EuroVan',
                        'Catera', 'Leganza', 'XG300', 'Prelude', 'Trooper', 'Prowler',
                        'Cabrio', 'Integra', 'cavalier', 'astro', 'excurs', 'dakota',
                        'Cirrus', 'twn&country', 'Jimmy', 'safari', 'ranger', 'sonoma',
                        'Sierra 2500', 'yukon', 'Sephia', 'Passport', 'Mirage', 'breeze',
                        'silhouette', 'villager', 'beetle', 'suburban', 'Lumina', 'Amigo',
                        'mountaineer', 'Esteem', 'pathfinder', 'quest', 'Breeze', 'e300dt',
                        'intrigue', 'lumina', 'Cabriolet', 'envoy', 'Cutlass', 'GS 400',
                        'E-150', 'Regency', 'thunderbird', 'B-Series Pickup', 'legacy',
                        'corsica', 'bronco', 'Le Baron', 'Caprice', 'Pickup', 'century',
                        '500-Class', '300-Class', 'previa', 'Murano CrossCabriolet',
                        'NV Cargo', 'Jetta Hybrid', 'S7', 'Encore', 'XTS',
                        'Black Diamond Avalanche', 'RLX', '2 Series', 'M6 Gran Coupe',
                        'regal', 'versa', 'BRZ', 'C/V Tradesman', 'Model S',
                        'Beetle Convertible', 'Golf R', 'routan', 'TSX Sport Wagon',
                        'interstate', 'a6', 'Corvette Stingray', 'SS', 'Mazdaspeed 3',
                        'i-MiEV', 'F430', 'EX35', 'swift', 'GranSport', 'escalade', 'neon',
                        'f350', 'Yukon Denali', 'Ghibli', '960', 'Cutlass Supreme',
                        'cougar', 'IS 350 C', 'lacrosse', 'tucson', '323i', 'cobalt', 'x3',
                        'Mazdaspeed MX-5 Miata', 'Ram Cargo', 'Safari', 'Eighty-Eight',
                        '850', 'J30', 'Promaster Cargo Van', 'sonic', 'rr', '1',
                        'Malibu Hybrid', '350', 'twn/cntry', 'Spirit', 'Accord Hybrid',
                        '3 Series Gran Turismo', 'e', 'cruze', 'c230', 'crossfire',
                        'Viper', 'Riviera', 'compass', 'Avalon Hybrid', 'RX-8',
                        'V8 Vantage', 'Equator', 'sportage', 'C/K 3500 Series',
                        'Mark VIII', 'charger', 'GranTurismo Convertible', 'avenger',
                        'equinox', 'allure', 'c240s', 'Vitara', 'avalon', 'eurovan',
                        'siera', 'pathfind', 'Eighty-Eight Royale', 'cherokee',
                        'ActiveHybrid 7', 'aveo', 'GS 460', 'Tribute Hybrid',
                        'Aura Hybrid', 'tahoe', 'g2500', '300ZX', 'Golf GTI', 'i-Series',
                        'golf', 'MKZ Hybrid', 'Macan', 'FX50', 'comm', 'STS-V',
                        'Windstar Cargo', 'CTS-V Wagon', 'Karma', '328i', 'Z4 M', '42c',
                        'subrbn', 'b2300', 'mountnr', 'Coupe', 'uplandr', 'Ram Van',
                        'Tempo', 'Tracer', 'CV Tradesman', 'DB9', 'C/K 2500 Series', 'i8',
                        'Rapide', 'Nubira', 'Corsica', 'NV Passenger', 'Spyder',
                        'LS 600h L', '400-Class', 'H3T', 'LX 450', 'WRX',
                        'Silverado 3500 Classic', '500e', 'Continental Supersports',
                        'Sierra 3500', 'Mystique', 'F-150 SVT Lightning', '190-Class',
                        'MKC', 'Aspire', '940', 'Gallardo',
                        'Continental Flying Spur Speed', '3000GT', 'TT RS', 'B-Series',
                        'ActiveHybrid 5', 'Sierra 1500 Hybrid', 'ML55 AMG', 'S-10 Blazer',
                        'RS 4', 'T100', 'Continental GTC Speed', 'mdx', 'Transit Van',
                        'F-250', 'Sidekick', 'E-250', '8 Series', '420-Class', 'E-350',
                        'Achieva', 'B-Class Electric Drive', 'Fleetwood', 'Paseo',
                        'Civic del Sol', 'Exige', 'X4', 'Spark EV', 'Transit Wagon', 'H1',
                        'SLS AMG GT', 'Flying Spur', 'Metro', 'Grand Cherokee SRT', 'RC F',
                        'Q3', '4 Series Gran Coupe', 'RC 350', '360', 'GLA-Class', 'TLX',
                        '458 Italia'))
    
with col2:
    Trim = st.selectbox('Select trim', df['trim'].unique())
    
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