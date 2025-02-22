import os
import anthropic
import json
import logging
import time
import streamlit as st
from anthropic import Anthropic
from requests.exceptions import ConnectionError, Timeout

# Fetch API Key from environment variables or Streamlit secrets
api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets["ANTHROPIC_API_KEY"]

# Initialize the Anthropic client
try:
    client = Anthropic(api_key=api_key)
except Exception as e:
    logging.error(f"Failed to initialize Anthropic client: {str(e)}")
    client = None

# Constants for text processing
MAX_CHUNK_SIZE = 4000
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
REQUEST_TIMEOUT = 30  # seconds

def check_api_availability():
    """Check if the API client is properly initialized"""
    if client is None:
        raise RuntimeError("AI service is not properly configured. Please check your API settings.")

def make_api_request(func, *args, **kwargs):
    """Make API request with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except (ConnectionError, Timeout) as e:
            if attempt == MAX_RETRIES - 1:
                logging.error(f"Connection failed after {MAX_RETRIES} attempts: {str(e)}")
                raise RuntimeError("Unable to connect to the AI service. Please check your internet connection and try again.")
            time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            raise e

def summarize_document(text):
    """Generate a summary of the document using Anthropic Claude"""
    if not text:
        return "No text provided for summarization."

    try:
        check_api_availability()
        chunks = chunk_text(text)
        if not chunks:
            return "Text is too short or empty."

        summaries = []

        for chunk in chunks:
            try:
                message = make_api_request(
                    client.messages.create,
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    timeout=REQUEST_TIMEOUT,
                    messages=[
                        {"role": "system", "content": "You are a legal document summarizer. Provide a concise summary of the text while maintaining key legal points."},
                        {"role": "user", "content": chunk}
                    ]
                )

                if message and message.content:
                    summaries.append(message.content[0].text)
                else:
                    logging.error("Received empty response from API")
            except Exception as e:
                logging.error(f"Error processing chunk: {str(e)}")

        if not summaries:
            return "No summary generated."

        final_summary = " ".join(summaries)

        return final_summary

    except RuntimeError as e:
        return str(e)
    except Exception as e:
        logging.error(f"Unexpected error in summarize_document: {str(e)}")
        return f"An unexpected error occurred: {str(e)}"

def identify_risks(text):
    """Identify potential legal risks in the document using Anthropic Claude"""
    if not text:
        return []

    try:
        check_api_availability()
        chunks = chunk_text(text)
        if not chunks:
            return []

        all_risks = []

        for chunk in chunks:
            try:
                message = make_api_request(
                    client.messages.create,
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    timeout=REQUEST_TIMEOUT,
                    messages=[
                        {"role": "system", "content": "You are a legal risk analyst. Identify potential legal risks and concerns in the text."},
                        {"role": "user", "content": chunk}
                    ]
                )

                risk_text = message.content[0].text
                risks = [risk.strip() for risk in risk_text.split('\n') if risk.strip()]
                if risks:
                    all_risks.extend(risks)
            except anthropic.APIError as api_err:
                logging.error(f"API error during risk identification: {str(api_err)}")
                return ["Service temporarily unavailable. Please try again later."]
            except anthropic.RateLimitError:
                logging.error("Rate limit exceeded")
                return ["Service is busy. Please wait a moment and try again."]
            except ConnectionError as conn_err:
                logging.error(f"Connection error: {str(conn_err)}")
                return ["Network connection issue. Please check your internet connection."]
            except Exception as e:
                logging.error(f"Error processing chunk for risks: {str(e)}")
                continue

        if not all_risks:
            return []

        unique_risks = list(set(all_risks))

        if len(unique_risks) > 10:
            try:
                message = make_api_request(
                    client.messages.create,
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    timeout=REQUEST_TIMEOUT,
                    messages=[
                        {"role": "system", "content": "Analyze these risks and return the 10 most critical ones."},
                        {"role": "user", "content": "\n".join(unique_risks)}
                    ]
                )
                prioritized_risks = [risk.strip() for risk in message.content[0].text.split('\n') if risk.strip()]
                return prioritized_risks[:10]
            except Exception as e:
                logging.error(f"Error in risk prioritization: {str(e)}")
                return unique_risks[:10]

        return unique_risks

    except RuntimeError as e:
        return [str(e)]
    except Exception as e:
        logging.error(f"Unexpected error in identify_risks: {str(e)}")
        return [f"An unexpected error occurred: {str(e)}"]

def chunk_text(text, max_length=MAX_CHUNK_SIZE):
    """Split text into chunks of maximum length while preserving words"""
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
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
