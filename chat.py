import streamlit as st
import random
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pydub import AudioSegment
import tempfile

# -------- CARGAR MODELO --------
@st.cache_resource
def cargar_modelo_emociones_es():
    modelo = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-base-finetuned-emotion-spanish")
    tokenizer = T5Tokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion-spanish")
    return modelo, tokenizer

modelo_emociones, tokenizer_emociones = cargar_modelo_emociones_es()

# -------- RESPUESTAS --------
respuestas_emocionales = {
    "alegría": ["¡Qué bueno escuchar eso! 😊", "¡Me alegra mucho! Cuéntame más.", "¡Tu energía positiva se siente desde aquí!"],
    "tristeza": ["Lo siento mucho... ¿quieres hablar de ello? 😔", "Estoy aquí para ti. A veces compartir ayuda.", "No estás solo. Si te puedo apoyar, dime cómo."],
    "enojo": ["Vaya, eso suena molesto. ¿Qué pasó?", "Tu enojo es válido. ¿Quieres desahogarte?", "Respirar profundo ayuda, pero también expresarlo. Estoy contigo."],
    "asco": ["Uf, entiendo por qué eso te causa rechazo.", "A veces hay cosas que simplemente nos repelen.", "Si quieres sacarlo de tu sistema, aquí estoy."],
    "miedo": ["Parece que eso te inquieta. ¿Qué te preocupa?", "No estás solo. Estoy aquí para escucharte.", "El miedo se reduce al compartirlo. Puedes contar conmigo."],
    "sorpresa": ["¡Qué sorpresa! 😮 ¿Qué pasó?", "¡Eso sí que no lo veía venir!", "Wow, eso suena inesperado."],
    "desconocido": ["No estoy seguro de cómo te sientes, pero te escucho.", "A veces los sentimientos son confusos, y está bien.", "Aquí estoy si necesitas hablar o pensar en voz alta."]
}

# -------- DETECTAR EMOCIÓN --------
def detectar_emocion_espanol(texto):
    entrada = "emocion: " + texto
    inputs = tokenizer_emociones.encode(entrada, return_tensors="pt", max_length=512, truncation=True)
    output = modelo_emociones.generate(inputs, max_length=8)
    emocion = tokenizer_emociones.decode(output[0], skip_special_tokens=True)
    return emocion if emocion in respuestas_emocionales else "desconocido"

# -------- RESPUESTA EMOCIONAL --------
def responder_emocionalmente(emocion):
    return random.choice(respuestas_emocionales.get(emocion, respuestas_emocionales["desconocido"]))

# -------- RECONOCIMIENTO DE VOZ --------
def transcribir_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Escuchando... habla ahora.")
        audio = recognizer.listen(source, timeout=5)
    try:
        texto = recognizer.recognize_google(audio, language="es-MX")
        st.success(f"🗣️ Dijiste: {texto}")
        return texto
    except sr.UnknownValueError:
        st.error("No pude entender el audio.")
        return None
    except sr.RequestError:
        st.error("Error de conexión con el servicio de reconocimiento.")
        return None

# -------- TEXTO A VOZ --------
def reproducir_audio(texto):
    tts = gTTS(text=texto, lang='es')
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        audio = AudioSegment.from_file(tmp.name, format="mp3")
        audio.export("respuesta.wav", format="wav")
        audio_file = open("respuesta.wav", "rb")
        st.audio(audio_file.read(), format='audio/wav')

# -------- INICIALIZAR SESIÓN --------
if "historial" not in st.session_state:
    st.session_state.historial = []

# -------- INTERFAZ --------
st.set_page_config(page_title="Agente Emocional con Voz", page_icon="🧠")
st.title("🧠 Agente Emocional con Entrada de Voz o Texto")
st.markdown("Este agente detecta tu emoción en español y responde por texto o con voz.")

# Entradas del usuario
modo_entrada = st.radio("Modo de entrada:", ["Texto", "Micrófono"], horizontal=True)
modo_respuesta = st.radio("Modo de respuesta:", ["Texto", "Bocina"], horizontal=True)

# Reiniciar
if st.button("🔄 Reiniciar sesión"):
    st.session_state.historial = []
    st.success("Sesión emocional reiniciada.")

# Obtener entrada
texto_usuario = ""
if modo_entrada == "Texto":
    texto_usuario = st.text_input("Escribe tu mensaje aquí:")
    procesar = st.button("Enviar")
else:
    if st.button("🎙️ Grabar y procesar voz"):
        texto_usuario = transcribir_audio()
        procesar = True
    else:
        procesar = False

# Procesar entrada
if procesar and texto_usuario:
    emocion = detectar_emocion_espanol(texto_usuario)
    respuesta = responder_emocionalmente(emocion)

    st.session_state.historial.append({
        "usuario": texto_usuario,
        "emocion": emocion,
        "respuesta": respuesta
    })

    if modo_respuesta == "Texto":
        st.markdown(f"**Agente ({emocion}):** {respuesta}")
    else:
        reproducir_audio(respuesta)

# Mostrar historial
if st.session_state.historial:
    st.markdown("### 🗂️ Historial de esta sesión:")
    for entrada in reversed(st.session_state.historial):
        st.markdown(f"**Tú:** {entrada['usuario']}")
        st.markdown(f"*Emoción detectada:* `{entrada['emocion']}`")
        st.markdown(f"**Agente:** {entrada['respuesta']}")
        st.markdown("---")
