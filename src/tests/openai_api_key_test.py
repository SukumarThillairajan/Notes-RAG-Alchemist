import os
from openai import OpenAI
from dotenv import load_dotenv

# --- CRITICAL ---
# Load environment variables from .env BEFORE you do anything else
load_dotenv()

print("Environment key loaded...")

# --- THIS IS THE SAFE WAY ---
# The client automatically finds the key in your environment.
# DO NOT paste your key here.
try:
    client = OpenAI()

    print("Client created. Making API call...")

    # This is the correct function for a chat model
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use a real, common model
        messages=[
            {"role": "user", "content": "write a haiku about ai"}
        ]
    )

    print("--- SUCCESS! ---")
    print(response.choices[0].message.content)

except Exception as e:
    print("--- ERROR ---")
    print(f"An error occurred: {e}")
