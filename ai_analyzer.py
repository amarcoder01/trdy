
import os
import streamlit as st

# Try fetching the API key from Streamlit secrets
api_key = st.secrets["general"]["ANTHROPIC_API_KEY"]
st.write("DEBUG: Secrets loaded ->", st.secrets)


if not api_key:
    st.error("❌ API Key is missing! Please check Streamlit secrets or environment variables.")
else:
    st.success("✅ API Key loaded successfully!")
