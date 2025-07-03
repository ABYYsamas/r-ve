import os
import openai
from groq import Groq
import json
from transformers import pipeline

# ✅ Gestion hybride des clés API
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

try:
    import streamlit as st
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except:
    pass

# 🔊 Transcription audio avec l’API Groq (Whisper)
def speech_to_text(audio_path: str, language: str = "fr") -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3-turbo",
            prompt="Extrait le texte de l'audio de la manière la plus factuelle possible",
            response_format="text",
            timestamp_granularities=["word", "segment"],
            language=language,
            temperature=0.0
        )
    return transcription

# 🖼 Génération d’image via OpenAI
def generate_image(prompt: str) -> str:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )
    return response["data"][0]["url"]

# 💭 Détection d’émotion via modèle HuggingFace
emotion_classifier = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")

def detect_emotion(text: str) -> str:
    result = emotion_classifier(text)
    return max(result, key=lambda x: x['score'])['label']

