import streamlit as st
import os
import logging
from pdf_processor import extract_text_from_pdf
from ai_analyzer import summarize_document, identify_risks
from utils import initialize_session_state

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    st.set_page_config(
        page_title="Alan AI Legal Document Summarizer",
        page_icon="⚖️",
        layout="wide"
    )

    initialize_session_state()

    st.title("Alan AI Legal Document Summarizer")
    st.markdown("""
    Upload your legal documents for AI-powered analysis and risk identification.
    Supported format: PDF files
    """)

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])

    if uploaded_file:
        # Display file info
        st.info(f"Processing file: {uploaded_file.name} ({uploaded_file.size} bytes)")

        try:
            with st.spinner('Extracting text from PDF...'):
                logging.info(f"Starting text extraction for file: {uploaded_file.name}")
                extracted_text = extract_text_from_pdf(uploaded_file)

                if not extracted_text or len(extracted_text.strip()) == 0:
                    st.error("No text could be extracted from the PDF. The file might be scanned, protected, or corrupted.")
                    return

                st.session_state.extracted_text = extracted_text
                logging.info(f"Successfully extracted {len(extracted_text)} characters from PDF")
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            logging.error(f"PDF extraction error: {str(e)}")
            return

        # Display extracted text
        st.subheader("Extracted Text")
        with st.expander("Show extracted text", expanded=True):
            st.text_area("Document Content", st.session_state.extracted_text, height=200)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📝 Summarize Document"):
                try:
                    with st.spinner('Generating summary...'):
                        summary = summarize_document(st.session_state.extracted_text)
                        st.session_state.summary = summary if summary else "Could not generate summary."
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    st.session_state.summary = "Error occurred during summarization."

        with col2:
            if st.button("🔍 Identify Risks"):
                try:
                    with st.spinner('Analyzing risks...'):
                        risks = identify_risks(st.session_state.extracted_text)
                        st.session_state.risks = risks if risks else []
                except Exception as e:
                    st.error(f"Error identifying risks: {str(e)}")
                    st.session_state.risks = []

        # Display results
        if 'summary' in st.session_state:
            st.subheader("Document Summary")
            st.write(st.session_state.summary)

        if 'risks' in st.session_state:
            st.subheader("Identified Risks")
            risks = st.session_state.risks
            if risks:
                for idx, risk in enumerate(risks, 1):
                    st.warning(f"Risk {idx}: {risk}")
            else:
                st.info("No risks were identified in the document.")

if __name__ == "__main__":
    main()