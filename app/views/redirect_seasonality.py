"""Old URL /Seasonality: this page is now a tab of Demand explorer."""
import streamlit as st

st.switch_page(st.session_state["pp_pages"]["explorer"])
