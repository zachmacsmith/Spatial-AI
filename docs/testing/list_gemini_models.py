"""
Quick script to list available Gemini models
"""

import google.generativeai as genai
import config

genai.configure(api_key=config.GEMINI_API_KEY)

print("Available Gemini models:")
print("="*60)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✓ {model.name}")
        print(f"  Display name: {model.display_name}")
        print(f"  Description: {model.description[:100] if model.description else 'N/A'}...")
        print()
