import os
import streamlit as st
import logging
from anthropic import Anthropic

# Load API key from Streamlit secrets
api_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")

# Debugging Output (Remove later)
if not api_key:
    st.error("❌ API Key is missing! Check Streamlit secrets.")
    logging.error("Missing Anthropic API Key! Ensure it's set in Streamlit secrets.")
else:
    st.success("✅ API Key loaded successfully!")  # Remove this after testing

# Initialize AI Client
try:
    client = Anthropic(api_key=api_key)
except Exception as e:
    st.error("❌ Failed to initialize AI service. Check your API key.")
    logging.error(f"Anthropic Client Error: {str(e)}")
    client = None
