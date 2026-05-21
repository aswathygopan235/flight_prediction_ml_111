import streamlit as st
import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()
BASEURL = os.getenv('BASEURL')
OPENAPI = os.getenv('OPENAPI')


@st.dialog("Result", dismissible=False, width="small")
def result_modal(res):
    """modal to show result"""
    with st.container():
        if res["success"] is False:
            st.badge("Fail", icon=":material/cancel:", color="red")
            err = construct_error_message(res)
            st.write(err)
        else:
            st.badge("Success", icon=":material/check:", color="green")
            st.write(res["result"])

        if (st.button("close")):
            st.rerun()


def construct_error_message(result):
    """create colour coded error message"""
    error_log = result["errors"]
    msg = ""
    for error in error_log:
        log = f'\n:gray[{error["field"]}]: :red[{error["msg"]}]'
        msg = msg+log
    return msg


def call_price_api():
    """API called to predict price"""
    url = BASEURL+"/predict"
    inp = {
        "source_city": st.session_state["source_city_sb"],
        "destination_city": st.session_state["destination_city_sb"],
        "departure_time": st.session_state["departure_time_sb"],
        "arrival_time": st.session_state["arrival_time_sb"],
        "stops": st.session_state["stops_sb"],
        "days_left": st.session_state["days_left_inp"],
        "airline": st.session_state["airline_sb"],
        "haul_type": st.session_state["haul_type_sb"]
    }

    with st.spinner("Calculating"):
        data = requests.post(url, json=inp, timeout=60).json()
        time.sleep(5)

    if (result_modal not in st.session_state):
        result_modal(data)


# st.session_state["submit"] = False

def main():
    """Main"""

    full = st.empty()

    with full, st.container():
        col1, col2, col3 = st.columns(
            [.6, .2, .2], vertical_alignment="center")
        with col2:
            # st.html("<a href='https://github.com/aswathygopan235/flight_prediction_ml_111'><img alt='View on github' src='https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white' /></a>")
            st.link_button(label="view on github",
                           url="https://github.com/aswathygopan235/flight_prediction_ml_111", type="secondary")

        with col3:
            st.link_button(label="Try it out", url=OPENAPI +
                           "/docs", type="secondary")

    with st.container():

        st.markdown("# Flight Price Prediction App")

        with st.form(key="flight_data_submit_form"):
            st.selectbox(
                "Select a source city",
                ('Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'),
                placeholder="Select source city", key="source_city_sb"
            )

            st.selectbox(
                "Select a destination city",
                ('Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'),
                placeholder="Select destination city", key="destination_city_sb"
            )

            st.selectbox(
                "Select a airline",
                ('SpiceJet', 'AirAsia', 'Vistara',
                 'GO_FIRST', 'Indigo', 'Air_India'),
                placeholder="Select airline", key="airline_sb"
            )

            st.number_input(
                "Type how many days till the flight", max_value=50, min_value=1, key="days_left_inp")

            st.selectbox("Select a departure time slot", (
                "Early Morning", "Morning", "Afternoon", "Evening", "Night", "Late Night"), key="departure_time_sb")

            st.selectbox("Select a arrival time slot", (
                "Early Morning", "Morning", "Afternoon", "Evening", "Night", "Late Night"), key="arrival_time_sb")
            st.selectbox("Select number of stops in between",
                         ("zero", "one", "two_or_more"), key="stops_sb")

            st.selectbox("Select a haul",
                         ("short_haul", "medium_haul", "long_haul", "ultra_long_haul"), key="haul_type_sb")

            st.form_submit_button(
                "Submit", on_click=call_price_api, type="primary")


# st.write(type(inp))
# with st.spinner("Wait for it..."):
if __name__ == "__main__":
    main()
