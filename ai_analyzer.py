import os
import anthropic
import json
import logging
import time
import streamlit as st
from anthropic import Anthropic
from requests.exceptions import ConnectionError, Timeout

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Fetch API Key safely
api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")

if not api_key:
    st.error("⚠️ API Key is missing! Please set it in environment variables or Streamlit secrets.")
    logging.error("API Key not found! Set it in Streamlit secrets or as an environment variable.")
    client = None
else:
    try:
        client = Anthropic(api_key=api_key)
    except Exception as e:
        logging.error(f"Failed to initialize Anthropic client: {str(e)}")
        client = None

# Constants
MAX_CHUNK_SIZE = 4000
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
REQUEST_TIMEOUT = 30  # seconds


def check_api_availability():
    """Check if API client is initialized properly"""
    if client is None:
        raise RuntimeError("❌ AI service is not properly configured. Please check your API key settings.")


def make_api_request(func, *args, **kwargs):
    """Make API request with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except (ConnectionError, Timeout) as e:
            if attempt == MAX_RETRIES - 1:
                logging.error(f"⚠️ Connection failed after {MAX_RETRIES} attempts: {str(e)}")
                raise RuntimeError("Unable to connect to the AI service. Please check your internet connection.")
            time.sleep(RETRY_DELAY * (attempt + 1))
        except anthropic.APIError as api_err:
            logging.error(f"🛑 API Error: {str(api_err)}")
            return None
        except anthropic.RateLimitError:
            logging.error("⏳ Rate limit exceeded. Try again later.")
            return None
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return None


def summarize_document(text):
    """Generate a summary of the document using Anthropic Claude"""
    if not text:
        return "⚠️ No text provided for summarization."

    try:
        check_api_availability()
        chunks = chunk_text(text)
        if not chunks:
            return "⚠️ Text is too short or empty."

        summaries = []

        for chunk in chunks:
            response = make_api_request(
                client.messages.create,
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                timeout=REQUEST_TIMEOUT,
                messages=[
                    {"role": "system", "content": "You are a legal document summarizer. Provide a concise summary of the text while maintaining key legal points."},
                    {"role": "user", "content": chunk}
                ]
            )

            if response and hasattr(response, "content") and response.content:
                summaries.append(response.content[0].text)
            else:
                logging.warning("⚠️ Received empty response from API.")

        return " ".join(summaries) if summaries else "⚠️ No summary generated."

    except RuntimeError as e:
        return str(e)
    except Exception as e:
        logging.error(f"Unexpected error in summarize_document: {str(e)}")
        return f"An unexpected error occurred: {str(e)}"


def identify_risks(text):
    """Identify potential legal risks in the document using Anthropic Claude"""
    if not text:
        return ["⚠️ No text provided for risk analysis."]

    try:
        check_api_availability()
        chunks = chunk_text(text)
        if not chunks:
            return ["⚠️ Text is too short or empty."]

        all_risks = []

        for chunk in chunks:
            response = make_api_request(
                client.messages.create,
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                timeout=REQUEST_TIMEOUT,
                messages=[
                    {"role": "system", "content": "You are a legal risk analyst. Identify potential legal risks and concerns in the text."},
                    {"role": "user", "content": chunk}
                ]
            )

            if response and hasattr(response, "content") and response.content:
                risk_text = response.content[0].text
                risks = [risk.strip() for risk in risk_text.split("\n") if risk.strip()]
                all_risks.extend(risks)

        if not all_risks:
            return ["⚠️ No risks identified."]

        unique_risks = list(set(all_risks))

        # Prioritize risks if more than 10 are identified
        if len(unique_risks) > 10:
            response = make_api_request(
                client.messages.create,
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                timeout=REQUEST_TIMEOUT,
                messages=[
                    {"role": "system", "content": "Analyze these risks and return the 10 most critical ones."},
                    {"role": "user", "content": "\n".join(unique_risks)}
                ]
            )

            if response and hasattr(response, "content") and response.content:
                return [risk.strip() for risk in response.content[0].text.split("\n") if risk.strip()][:10]

        return unique_risks

    except RuntimeError as e:
        return [str(e)]
    except Exception as e:
        logging.error(f"Unexpected error in identify_risks: {str(e)}")
        return [f"⚠️ An unexpected error occurred: {str(e)}"]


def chunk_text(text, max_length=MAX_CHUNK_SIZE):
    """Split text into chunks while preserving words"""
    if not text:
        return []

    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
