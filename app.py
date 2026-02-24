import numpy as np
import streamlit as st
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1560518883-ce09059eeffa");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Remove white container */
.block-container {
    background: transparent;
    padding-top: 2rem;
}

/* Center content */
section.main > div {
    display: flex;
    flex-direction: column;
    align-items: center;
}

/* Title */
h1 {
    color: black !important;
    font-size: 50px !important;
    font-weight: 900 !important;
    text-align: center;
}

/* Subtitle */
p {
    color: black !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    text-align: center;
}

/* Labels */
label {
    color: black !important;
    font-size: 18px !important;
    font-weight: 800 !important;
}

/* Input box */
div[data-baseweb="input"] input {
    background: rgba(255,255,255,0.25) !important;
    backdrop-filter: blur(5px);
    border-radius: 8px !important;
    height: 40px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    color: black !important;
    text-align: center;
}

/* Button */
.stButton>button {
    background-color: #8B4513 !important;
    color: black !important;
    font-size: 22px !important;
    font-weight: 900 !important;
    padding: 12px 60px !important;
    border-radius: 10px !important;
    margin-top: 30px;
}

.stButton>button:hover {
    background-color: #A0522D !important;
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
with open("lasso_model.pkl", "rb") as f:
    L_model = pickle.load(f)

with open("poly_transform.pkl", "rb") as f:
    poly = pickle.load(f)

# ---------------- TITLE ----------------
st.title("🏠 House Price Prediction Model")
st.write("Enter House Details Below")

# ---------------- INPUT FIELDS ----------------

# Row 1
col1, col2, col3 = st.columns(3)
with col1:
    crime_rate = st.number_input("Crime Rate (in 1 to 100)", key="crime")
with col2:
    large_plot_area = st.number_input("Large Plot Area", key="plot")
with col3:
    industrial_area = st.number_input("Industrial Area Distance (in KM)", key="industry")

# Row 2
col4, col5, col6 = st.columns(3)
with col4:
    near_river = st.number_input("Near River (in KM)", key="river")
with col5:
    pollution_level = st.number_input("Pollution Level (in 1 to 100)", key="pollution")
with col6:
    number_of_rooms = st.number_input("Number of Rooms", key="rooms")

# Row 3
col7, col8, col9 = st.columns(3)
with col7:
    house_age = st.number_input("House Age (in Years)", key="age")
with col8:
    distance_to_city = st.number_input("Distance to City (in KM)", key="city")
with col9:
    highway_access = st.number_input("Highway Access (in KM)", key="highway")

# Row 4
col10, col11, col12 = st.columns(3)
with col10:
    property_tax = st.number_input("Property Tax (in USD)", key="tax")
with col11:
    distance_to_school = st.number_input("Distance to School (in KM)", key="school")
with col12:
    population_factor = st.number_input("Population Factor", key="population")

# Last Row (centered)
col13, col14, col15 = st.columns([1,2,1])
with col14:
    low_income_percentage = st.number_input("Low Income Percentage", key="income")

# Center Button
col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
with col_btn2:
    predict = st.button("Predict House Price", key="predict_btn")

# ---------------- PREDICTION ----------------

if predict:

    user_input = np.array([[crime_rate, large_plot_area, industrial_area,
                            near_river, pollution_level, number_of_rooms,
                            house_age, distance_to_city, highway_access,
                            property_tax, distance_to_school,
                            population_factor, low_income_percentage]])

    user_input_poly = poly.transform(user_input)
    prediction = L_model.predict(user_input_poly)
    prediction = np.maximum(prediction, 0)


    st.success(f"🏡 Estimated House Price: ${prediction[0]:,.2f} USD")
