"""Quick test of google-genai SDK directly."""
import os
from dotenv import load_dotenv
load_dotenv()

from google import genai

key = os.getenv("ACTIVE_SCHOLAR_GOOGLE_API_KEY")
client = genai.Client(api_key=key)

# List available models that support generateContent
print("=== Models supporting generateContent ===")
for m in client.models.list():
    methods = getattr(m, 'supported_generation_methods', []) or []
    if 'generateContent' in methods:
        print(f"  {m.name}")

# Try generating content
for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]:
    try:
        print(f"\nTesting {model_name}...")
        response = client.models.generate_content(
            model=model_name,
            contents="Say hello in one sentence."
        )
        print(f"  SUCCESS: {response.text[:100]}")
        break
    except Exception as e:
        print(f"  FAILED: {e}")
