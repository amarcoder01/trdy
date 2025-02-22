import os
import streamlit as st
import logging
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)

# Fetch API Key securely
api_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")

# Check API Key availability
if not api_key:
    st.error("❌ API Key is missing! Please set it in Streamlit secrets.")
    logging.error("Missing Anthropic API Key! Ensure it's set in Streamlit secrets.")
    client = None  # Prevent further execution
else:
    st.success("✅ API Key loaded successfully!")

    # Try initializing the AI client
    try:
        client = Anthropic(api_key=api_key)
        logging.info("Anthropic AI Client initialized successfully.")
    except Exception as e:
        st.error("❌ Failed to initialize AI service. Check your API key.")
        logging.error(f"Failed to initialize Anthropic client: {str(e)}")
        client = None  # Prevent usage of uninitialized client
