import streamlit as st
import random
import torch
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
import tempfile
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------
# CARGA DEL MODELO
# -------------------------

@st.cache_resource
def cargar_modelo_emociones_es():
    modelo = AutoModelForSequenceClassification.from_pretrained(
        "pysentimiento/robertuito-emotion-analysis"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "pysentimiento/robertuito-emotion-analysis"
    )
    return modelo, tokenizer

modelo_emociones, tokenizer_emociones = cargar_modelo_emociones_es()

# -------------------------
# DETECCIÓN DE EMOCIÓN
# -------------------------

def detectar_emocion_espanol(texto):
    inputs = tokenizer_emociones(texto, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = modelo_emociones(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    etiquetas = modelo_emociones.config.id2label
    emocion = etiquetas[pred_id]
    return emocion.lower()

# -------------------------
# RESPUESTAS EMOCIONALES
# -------------------------

respuestas_emocionales = {
    "joy": ["¡Qué bueno escuchar eso! 😊", "¡Me alegra mucho! Cuéntame más.", "¡Tu energía positiva se siente desde aquí!"],
    "sadness": ["Lo siento mucho... ¿quieres hablar de ello? 😔", "Estoy aquí para ti. A veces compartir ayuda.", "No estás solo. Si te puedo apoyar, dime cómo."],
    "anger": ["Vaya, eso suena molesto. ¿Qué pasó?", "Tu enojo es válido. ¿Quieres desahogarte?", "Respirar profundo ayuda, pero también expresarlo. Estoy contigo."],
    "fear": ["Parece que eso te inquieta. ¿Qué te preocupa?", "No estás solo. Estoy aquí para escucharte.", "El miedo se reduce al compartirlo. Puedes contar conmigo."],
    "disgust": ["Uf, entiendo por qué eso te causa rechazo.", "A veces hay cosas que simplemente nos repelen.", "Si quieres sacarlo de tu sistema, aquí estoy."],
    "surprise": ["¡Qué sorpresa! 😮 ¿Qué pasó?", "¡Eso sí que no lo veía venir!", "Wow, eso suena inesperado."],
    "neutral": ["Estoy aquí si quieres platicar de cualquier cosa.", "A veces lo normal también tiene valor.", "¿Cómo te fue hoy?"]
}

def responder_emocionalmente(emocion):
    return random.choice(respuestas_emocionales.get(emocion, respuestas_emocionales["neutral"]))

# -------------------------
# AUDIO: TEXTO A VOZ
# -------------------------

def reproducir_audio(texto):
    tts = gTTS(text=texto, lang='es')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        audio = AudioSegment.from_file(tmp.name, format="mp3")
        audio.export("respuesta.wav", format="wav")
        st.audio("respuesta.wav", format='audio/wav')

# -------------------------
# INTERFAZ STREAMLIT
# -------------------------

st.set_page_config(page_title="Agente Emocional", page_icon="🧠")
st.title("🧠 Agente Emocional en Español")
st.markdown("Este agente detecta emociones en tus mensajes y responde con empatía.")

if "historial" not in st.session_state:
    st.session_state.historial = []

modo_respuesta = st.radio("Modo de respuesta:", ["Texto", "Bocina (audio)"], horizontal=True)

if st.button("🔄 Reiniciar sesión"):
    st.session_state.historial = []
    st.success("Sesión emocional reiniciada.")

# Entrada por texto
texto_usuario = st.text_input("Escribe tu mensaje aquí:")
procesar = st.button("Enviar")

# -------------------------
# PROCESAR MENSAJE
# -------------------------

if procesar and texto_usuario:
    emocion = detectar_emocion_espanol(texto_usuario)
    respuesta = responder_emocionalmente(emocion)

    st.session_state.historial.append({
        "usuario": texto_usuario,
        "emocion": emocion,
        "respuesta": respuesta
    })

    st.markdown(f"**Emoción detectada:** `{emocion}`")
    if modo_respuesta == "Texto":
        st.markdown(f"**Agente:** {respuesta}")
    else:
        reproducir_audio(respuesta)

# -------------------------
# HISTORIAL
# -------------------------

if st.session_state.historial:
    st.markdown("### 🗂️ Historial de esta sesión:")
    for entrada in reversed(st.session_state.historial):
        st.markdown(f"**Tú:** {entrada['usuario']}")
        st.markdown(f"*Emoción detectada:* `{entrada['emocion']}`")
        st.markdown(f"**Agente:** {entrada['respuesta']}")
        st.markdown("---")
