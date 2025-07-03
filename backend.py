import os
import requests
import base64
from groq import Groq
from transformers import pipeline

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

try:
    import streamlit as st
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]
    os.environ["CLIPDROP_API_KEY"] = st.secrets["CLIPDROP_API_KEY"]
except:
    pass

def speech_to_text(audio_path: str, language: str = "fr") -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3-turbo",
            prompt="Extrait le texte de l'audio de la manière la plus factuelle possible",
            response_format="text",
            language=language,
            temperature=0.0
        )
    return transcription


def generate_image(prompt: str) -> str:
    url = "https://clipdrop-api.co/text-to-image/v1"
    headers = {
        "x-api-key": os.environ["CLIPDROP_API_KEY"]
    }
    data = {
        "prompt": prompt
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        image_bytes = response.content
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{image_base64}"
    else:
        raise Exception(f"ClipDrop Error {response.status_code}: {response.text}")

emotion_classifier = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")

def detect_emotion(text: str) -> str:
    result = emotion_classifier(text)
    return max(result, key=lambda x: x['score'])['label']
