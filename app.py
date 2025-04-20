import streamlit as st
import base64
import pandas as pd
import pickle
import requests

# Load the saved model and scaler
best_model = pickle.load(open('pickle/nn_model.pkl', 'rb'))
scaler = pickle.load(open('pickle/scaler.pkl', 'rb'))

# Load the dictionaries for driver confidence and constructor reliability
driver_confidence_dict = pickle.load(open('pickle/driver_confidence_dict.pkl', 'rb'))
constructor_reliability_dict = pickle.load(open('pickle/constructor_reliability_dict.pkl', 'rb'))

# Load the LabelEncoder objects
le_gp = pickle.load(open('pickle/gp_label_encoder.pkl', 'rb'))
le_d = pickle.load(open('pickle/d_label_encoder.pkl', 'rb'))
le_c = pickle.load(open('pickle/c_label_encoder.pkl', 'rb'))

# Define driver, constructor, and circuit dropdowns
driver_names = le_d.inverse_transform(list(driver_confidence_dict.keys()))
constructor_names = le_c.inverse_transform(list(constructor_reliability_dict.keys()))
gp_names = le_gp.inverse_transform(range(len(le_gp.classes_)))
qualifying_positions = list(range(1, 21))

st.set_page_config(page_title="F1 Predictor", page_icon="🏎️", layout="wide")
# Add at the top, just after st.set_page_config
st.markdown("""
    <style>
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        .main {
            animation: fadeIn 1s ease-in;
        }
        button[kind="primary"] {
            transition: 0.3s ease-in-out;
            background-color: #ff4b4b !important;
        }
        button[kind="primary"]:hover {
            background-color: #ff7777 !important;
            transform: scale(1.03);
        }
        .css-1d391kg {  /* container for sidebar */
            background-color: #111111;
        }
        .css-1v0mbdj.edgvbvh3 {
            padding-top: 2rem;
        }
        h1, h2, h3, h4 {
            color: #ff4b4b;
        }
    </style>
""", unsafe_allow_html=True)

# Top layout: logo + credits
col1, col2 = st.columns([1, 3])
with col1:
    st.image("logo.png", use_container_width=True)

with col2:
    st.markdown(
        """
        <div style='padding-top: 15px;'>
            <h3>Project Credits</h3>
            <ul style='line-height: 1.6;'>
                <li><b>Shashank Jain</b> - 221IT062</li>
                <li><b>Vineet Jain</b> - 221IT081</li>
                <li><b>Divyam Gupta</b> - 221IT025</li>
                <li><b>Tejas Pratap</b> - 221IT072</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Utility functions
def get_weather(city):
    api_key = "56bae0fe92162262a26656aa0ec852b1"
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {'q': city, 'appid': api_key, 'units': 'metric'}
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        if response.status_code == 200:
            return {
                'Weather': data['weather'][0]['main'],
                'Description': data['weather'][0]['description'],
                'Temperature (°C)': data['main']['temp'],
                'Humidity (%)': data['main']['humidity'],
                'Wind Speed (m/s)': data['wind']['speed']
            }
    except:
        return None

def fetch_constructor_news(constructor_name):
    api_key = '7730dfb2545e4b5cbfa7ee6f92277a59'
    query = f"F1 {constructor_name} standings"
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if data['status'] == 'ok' and data['totalResults'] > 0:
            return data['articles'][:5]
    except:
        return None

# GP to city mapping
gp_to_city = {
    "Monaco Grand Prix": "Monaco", "British Grand Prix": "Silverstone",
    "Italian Grand Prix": "Monza", "Canadian Grand Prix": "Montreal",
    "Japanese Grand Prix": "Suzuka", "United States Grand Prix": "Austin",
    "Abu Dhabi Grand Prix": "Abu Dhabi", "Australian Grand Prix": "Melbourne",
    "Circuit Paul Ricard": "Le Castellet", "Hungarian Grand Prix": "Budapest",
    "Belgian Grand Prix": "Spa-Francorchamps", "Singapore Grand Prix": "Singapore",
    "Brazilian Grand Prix": "São Paulo", "Saudi Arabian Grand Prix": "Jeddah",
    "Qatar Grand Prix": "Lusail", "Austrian Grand Prix": "Spielberg",
    "French Grand Prix": "Le Castellet", "Dutch Grand Prix": "Zandvoort",
}

# User Inputs
st.header("🏁 F1 Final Grid Position Predictor")
season = st.text_input("Enter season (e.g., 2023):")
driver_name = st.selectbox("Select driver's name:", driver_names)
constructor_name = st.selectbox("Select constructor's name:", constructor_names)

st.subheader(f"📰 Latest News on {constructor_name}")
news_articles = fetch_constructor_news(constructor_name)
if news_articles:
    for article in news_articles:
        st.markdown(f"**[{article['title']}]({article['url']})**")
        st.markdown(f"*{article['source']['name']} - {article['publishedAt']}*")
        st.markdown(f"{article['description']}")
        st.markdown("---")
else:
    st.write("No recent news articles found for this constructor.")

gp_name = st.selectbox("Select circuit's name:", gp_names)
selected_city = gp_to_city.get(gp_name, None)

if selected_city:
    st.subheader(f"🌦️ Current Weather at {gp_name} ({selected_city})")
    weather_info = get_weather(selected_city)
    st.write(weather_info if weather_info else "Weather data not available.")
else:
    st.info("Weather data is unavailable for this Grand Prix.")

qualifying_position = st.selectbox("Enter driver's qualifying position:", qualifying_positions)

# Encode inputs
driver_encoded = le_d.transform([driver_name])[0]
constructor_encoded = le_c.transform([constructor_name])[0]
gp_encoded = le_gp.transform([gp_name])[0]

# Predict button
if st.button("Predict"):
    if all([season, driver_name, constructor_name, gp_name]):
        input_data = pd.DataFrame({
            'GP_name': [gp_encoded],
            'quali_pos': [qualifying_position],
            'constructor': [constructor_encoded],
            'driver': [driver_encoded],
            'driver_confidence': driver_confidence_dict[driver_encoded],
            'constructor_relaiblity': constructor_reliability_dict[constructor_encoded],
            'season': [season]
        })

        data_scaled = scaler.transform(input_data)
        predicted_position = best_model.predict(data_scaled)

        st.divider()
        st.success("SUCCESS!")
        st.subheader(f'Predicted final grid position of the driver: **{int(predicted_position[0])}**')
        st.divider()

        # Graph for all quali positions
        all_input_data = pd.DataFrame({
            'GP_name': [gp_encoded] * 22,
            'quali_pos': list(range(1, 23)),
            'constructor': [constructor_encoded] * 22,
            'driver': [driver_encoded] * 22,
            'driver_confidence': [driver_confidence_dict[driver_encoded]] * 22,
            'constructor_relaiblity': [constructor_reliability_dict[constructor_encoded]] * 22,
            'season': [season] * 22
        })

        all_scaled = scaler.transform(all_input_data)
        all_preds = best_model.predict(all_scaled)

        result_df = pd.DataFrame({
            'Possible Qualifying position': range(1, 23),
            'Predicted Final Grid Position': all_preds.astype(int)
        })

        st.subheader(f"📈 Final Position Predictions vs Qualifying")
        col1, col2 = st.columns([3, 1], gap="medium")
        col1.line_chart(result_df, x='Possible Qualifying position', y='Predicted Final Grid Position')
        col2.dataframe(result_df, use_container_width=True)
    else:
        st.error("Please fill out all required fields.")
