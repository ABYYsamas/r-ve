import streamlit as st
from backend import speech_to_text, generate_image, detect_emotion

st.set_page_config(page_title="Synthétiseur de rêves", page_icon="🌙")

st.title("🌙 Synthétiseur de rêves")
st.markdown(
    "Et si ta nuit n'était pas terminée...  \n"
    "🎤 Envoie un enregistrement de ton rêve...  \n"
    "🖼 Et on te génère une image fidèle à ton subconscient."
)

uploaded_file = st.file_uploader("📂 Upload un fichier audio (format .wav, .mp3 ou .m4a)", type=["wav", "mp3", "m4a"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")
    if st.button("✨ Synthétiser mon rêve"):
        with st.spinner("🧠 Analyse en cours..."):
            try:
                with open("temp_audio.m4a", "wb") as f:
                    f.write(uploaded_file.read())

                transcription = speech_to_text("temp_audio.m4a", language="fr")
                st.markdown("### 📝 Texte transcrit :")
                st.code(transcription)

                emotion = detect_emotion(transcription)
                st.markdown("### 💭 Émotion détectée :")
                st.success(emotion)

                image_url = generate_image(transcription)
                st.markdown("### 🎨 Image générée :")
                st.image(image_url, caption="Interprétation onirique", use_container_width=True)

            except Exception as e:
                st.error(f"Erreur lors du traitement : {e}")
else:
    st.info("Si tu veux que ta nuit continue, commence par uploader un fichier audio.")
