
import os
import streamlit as st
import anthropic

# Load API Key
api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
if not api_key:
    st.error("❌ API Key is missing! Please check Streamlit secrets or environment variables.")
    st.stop()

# Initialize Anthropic Client
client = anthropic.Anthropic(api_key=api_key)

# Function to summarize legal documents
def summarize_document(text):
    try:
        response = client.messages.create(
            model="claude-2",  # Change model if needed
            max_tokens=500,
            messages=[
                {"role": "user", "content": f"Summarize this legal document: {text}"}
            ]
        )
        return response.content[0].text if response.content else "No summary generated."
    except Exception as e:
        return f"Error in summarization: {str(e)}"

# Function to identify risks in legal documents
def identify_risks(text):
    try:
        response = client.messages.create(
            model="claude-2",  # Change model if needed
            max_tokens=500,
            messages=[
                {"role": "user", "content": f"Identify potential legal risks in this document: {text}"}
            ]
        )
        return response.content[0].text if response.content else "No risks identified."
    except Exception as e:
        return f"Error in risk analysis: {str(e)}"

# Streamlit UI
st.title("📜 Legal Document Analyzer")
st.markdown("Analyze legal documents for summaries and potential risks using AI.")

document_text = st.text_area("📄 Paste your legal document here:")
if st.button("Analyze Document"):
    if document_text.strip():
        with st.spinner("Analyzing document..."):
            summary = summarize_document(document_text)
            risks = identify_risks(document_text)
            
        st.subheader("📝 Summary")
        st.write(summary)
        
        st.subheader("⚠️ Potential Risks")
        st.write(risks)
    else:
        st.warning("⚠️ Please enter some text to analyze.")
