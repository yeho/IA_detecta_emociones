import streamlit as st
import random
from transformers import pipeline

# Configurar página
st.set_page_config(page_title="Agente Emocional", page_icon="💬")

# Cargar modelos con cache
@st.cache_resource
def cargar_modelos():
    traductor = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
    clasificador = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
    return traductor, clasificador

traductor, clasificador_emociones = cargar_modelos()

# Diccionario de respuestas por emoción
respuestas_por_emocion = {
    "joy": [
        "¡Me alegra mucho escuchar eso! 😊",
        "¡Qué bien suena eso! ¿Quieres contarme más?",
        "¡Tu alegría es contagiosa!"
    ],
    "sadness": [
        "Siento que estés así. Si necesitas hablar, estoy aquí. 😔",
        "A veces hablar ayuda. ¿Quieres contarme qué pasa?",
        "No estás solo. Estoy contigo."
    ],
    "anger": [
        "Parece que estás molesto. ¿Qué ocurrió?",
        "Desahógate si lo necesitas. Te escucho.",
        "Tu enojo es válido. Estoy aquí para apoyarte."
    ],
    "fear": [
        "Parece que algo te preocupa. ¿Quieres contármelo?",
        "Estoy aquí para ayudarte. No estás solo.",
        "A veces compartir tus miedos los hace más pequeños."
    ],
    "surprise": [
        "¡Vaya, eso sí que es una sorpresa! 😮",
        "¡Qué inesperado! Cuéntame más.",
        "¡Eso no me lo esperaba!"
    ],
    "disgust": [
        "Eso suena desagradable. ¿Qué fue lo que pasó?",
        "Uf, entiendo tu reacción.",
        "A veces algo simplemente nos causa rechazo, y está bien."
    ],
    "neutral": [
        "¿Cómo va tu día?",
        "Estoy aquí si quieres charlar o simplemente pasar el rato.",
        "A veces lo cotidiano también merece ser contado."
    ]
}

# Inicializar variables de sesión
if "historial" not in st.session_state:
    st.session_state.historial = []

# Función para traducir texto
def traducir_a_ingles(texto):
    return traductor(texto)[0]["translation_text"]

# Función para detectar emoción
def detectar_emocion(texto):
    texto_en = traducir_a_ingles(texto)
    resultado = clasificador_emociones(texto_en)[0]
    return resultado["label"].lower()

# Función para obtener respuesta
def generar_respuesta(emocion):
    return random.choice(respuestas_por_emocion.get(emocion, ["Estoy aquí para ti."]))

# Título e instrucciones
st.title("🧠 Agente Conversacional Emocional")
st.markdown("Este agente detecta tu emoción y ajusta su tono para responderte de forma empática.")
st.markdown("Escribe lo que sientas. Haz clic en 'Reiniciar sesión' para borrar el historial emocional.")

# Botón para reiniciar la sesión
if st.button("🔄 Reiniciar sesión emocional"):
    st.session_state.historial = []
    st.success("Sesión reiniciada.")

# Formulario de mensaje
with st.form("form_mensaje"):
    mensaje_usuario = st.text_input("Tu mensaje:", "")
    enviado = st.form_submit_button("Enviar")

if enviado and mensaje_usuario.strip():
    emocion = detectar_emocion(mensaje_usuario)
    respuesta = generar_respuesta(emocion)
    
    # Guardar en historial
    st.session_state.historial.append({
        "usuario": mensaje_usuario,
        "emocion": emocion,
        "respuesta": respuesta
    })

# Mostrar historial emocional
if st.session_state.historial:
    st.markdown("### 🗂️ Historial emocional de esta sesión:")
    for i, entrada in enumerate(reversed(st.session_state.historial), 1):
        st.markdown(f"**Tú:** {entrada['usuario']}")
        st.markdown(f"*Emoción detectada:* `{entrada['emocion']}`")
        st.markdown(f"**Agente:** {entrada['respuesta']}")
        st.markdown("---")
