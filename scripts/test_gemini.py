"""
test_gemini.py

Quick sanity check that your Gemini API key is working before
wiring it into the full RAG pipeline.

Usage:
    python scripts/test_gemini.py
"""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()  # reads .env in the project root

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Make sure you have a .env file "
        "at your project root with: GEMINI_API_KEY=your_key_here"
    )

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-flash-latest",
    contents="In one sentence, what is required to register a private limited company in India?",
)

print("\n--- Gemini response ---\n")
print(response.text)
print("\n--- Test successful. Your API key works. ---\n")