import os
import streamlit as st
import logging
from anthropic import Anthropic

# Fetch API Key from Streamlit secrets
api_key = st.secrets["general"].get("ANTHROPIC_API_KEY", None)
st.write("DEBUG: API Key:", api_key)  # Remove this after testing


if not api_key:
    st.error("❌ AI service is not properly configured. Please check your API key settings in Streamlit Cloud.")
    logging.error("Missing Anthropic API Key! Make sure it's set in Streamlit secrets.")
    client = None
else:
    try:
        client = Anthropic(api_key=api_key)
    except Exception as e:
        st.error("❌ Failed to initialize AI service. Check your API key.")
        logging.error(f"Failed to initialize Anthropic client: {str(e)}")
        client = None
