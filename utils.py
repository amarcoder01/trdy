import streamlit as st

def initialize_session_state():
    """Initialize session state variables"""
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'risks' not in st.session_state:
        st.session_state.risks = None
