"""
Centralized environment configuration.
Loads environment variables and validates required keys.
"""

import os
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

# OpenAI API key required by LangChain / OpenAI SDK
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Fail fast if key is missing
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")

print("OPENAI_API_KEY found.")
