import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
BASEURL = os.getenv('BASEURL')


@st.dialog("result")
def result_modal(res):
    if (res["status"] == False):
        st.write("failed")

    st.write(data["result"])


submitted = False
st.markdown("# Flight Price Prediction App")
with st.form("my_form"):
    source_city = st.selectbox(
        "Select a source city",
        ('Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'),
        placeholder="Select source city",
    )

    destination_city = st.selectbox(
        "Select a destination city",
        ('Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'),
        placeholder="Select destination city",
    )

    airline = st.selectbox(
        "Select a airline",
        ('SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India'),
        placeholder="Select source city",
    )

    days_left = st.number_input(
        "Type how many days till the flight", max_value=50, min_value=1)

    departure_time = st.selectbox("Select a departure time slot", (
        "Early Morning", "Morning", "Afternoon", "Evening", "Night", "Late Night"))
    arrival_time = st.selectbox("Select a arrival time slot", (
        "Early Morning", "Morning", "Afternoon", "Evening", "Night", "Late Night"))
    stops = st.selectbox("Select number of stops in between",
                         ("zero", "one", "two_or_more"))

    haul_type = st.selectbox("Select a haul",
                             ("short_haul", "medium_haul", "long_haul", "ultra_long_haul"))

    submitted = st.form_submit_button("Submit")

url = BASEURL+"/predict"
if (submitted is True):
    inp = {
        "source_city": source_city,
        "destination_city": destination_city,
        "departure_time": departure_time,
        "arrival_time": arrival_time,
        "stops": stops,
        "days_left": days_left,
        "airline": airline,
        "haul_type": haul_type
    }

    # inpp = json.dumps(inp)

    # st.write(type(inp))
    with st.spinner("Wait for it..."):
        data = requests.post(url, json=inp, timeout=5).json()
        st.write(data)
