# 🧠 Agente Emocional en Español

Este es un agente conversacional simple basado en inteligencia artificial que **detecta la emoción de un mensaje de texto en español** y responde de forma empática. Funciona completamente en navegador a través de [Streamlit](https://streamlit.io/).

🔗 **Prueba la demo en línea aquí:**  
👉 [https://iadetectaemociones-3mjvl3gxp6xz36wg2j76cd.streamlit.app](https://iadetectaemociones-3mjvl3gxp6xz36wg2j76cd.streamlit.app/)

---

## 🤖 ¿Qué emociones puede detectar?

El modelo puede identificar las siguientes emociones básicas (basadas en Ekman + neutral):

- 😃 **Alegría** (`joy`)  
- 😔 **Tristeza** (`sadness`)  
- 😠 **Enojo** (`anger`)  
- 😱 **Miedo** (`fear`)  
- 🤢 **Asco** (`disgust`)  
- 😮 **Sorpresa** (`surprise`)  
- 😐 **Neutral** (`neutral`)

---

## 🧰 Tecnologías utilizadas

- [Streamlit](https://streamlit.io) – interfaz web
- [Hugging Face Transformers](https://huggingface.co/transformers/) – para el modelo RoBERTuito
- [pysentimiento/robertuito-emotion-analysis](https://huggingface.co/pysentimiento/robertuito-emotion-analysis) – modelo de detección de emociones en español
- PyTorch – backend del modelo

---

## 🚀 ¿Cómo usarlo localmente?

1. Clona este repositorio:

   ```bash
   git clone https://github.com/tu-usuario/agente-emocional-espanol.git
   cd agente-emocional-espanol

   
2.	Instala las dependencias:
     ```bash
     pip install -r requirements.txt

     
3.	Ejecuta la app:
     ```bash
      streamlit run chat.py

  
## 📦 Requisitos

requirements.txt:
  ```bash
    streamlit
    transformers
    torch
```

## 🙌 Créditos

Proyecto desarrollado con fines educativos, de experimentación y exploración de interfaces afectivas en español.

Modelo por pysentimiento.
Frontend y UX con Streamlit

   

