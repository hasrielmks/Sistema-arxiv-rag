import streamlit as st
import pandas as pd
import faiss
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from google.genai import types

# 1. Configuración de seguridad de la API Key (Requerimiento H)
# Intentará leer la clave secreta desde las variables de entorno del servidor en la nube
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Configuración visual de la aplicación web
st.set_page_config(page_title="arXiv Research RAG", page_icon="🔬", layout="centered")

st.title("Asistente Científico arXiv RAG")
st.markdown("Consulta el corpus de resúmenes científicos mediante IA y verifica las evidencias utilizadas.")

# Validar que la API key exista antes de renderizar la app
if not GEMINI_API_KEY:
    st.error("Error de Configuración: Falta la variable de entorno 'GEMINI_API_KEY' en el servidor.")
    st.stop()

# Inicializar cliente de Gemini de forma segura
client = genai.Client(api_key=GEMINI_API_KEY)

# 2. Carga optimizada de Modelos y Datos con caché de Streamlit
@st.cache_resource
def inicializar_componentes():
    # Modelos de recuperación y ordenamiento
    embed_mod = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    rerank_mod = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # Carga de los archivos que exportamos desde Colab
    datos_df = pd.read_pickle("arxiv_clean_df.pkl")
    indice_faiss = faiss.read_index("faiss_arxiv_index.bin")
    return embed_mod, rerk_mod, datos_df, indice_faiss

embedding_model, reranker_model, df, index = inicializar_componentes()

# 3. Gestión del historial del Chat (Requerimiento G)
if "historial" not in st.session_state:
    st.session_state.historial = []

# Mostrar mensajes anteriores en la pantalla si existen
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])
        if "evidencias" in mensaje:
            with st.expander("Ver Evidencias Científicas Utilizadas"):
                for ev in mensaje["evidencias"]:
                    st.write(f"**Título:** {ev['titulo']}")
                    st.caption(f"Score de Relevancia (Cross-Encoder): {ev['score']:.4f}")

# 4. Procesamiento de Nuevas Consultas
if consulta_usuario := st.chat_input("¿Qué deseas investigar hoy? (ej. Applications of Graph Neural Networks)"):
    
    # Mostrar la pregunta del usuario inmediatamente
    with st.chat_message("user"):
        st.markdown(consulta_usuario)
    st.session_state.historial.append({"rol": "user", "contenido": consulta_usuario})
    
    # Activar animación de carga mientras procesa el pipeline
    with st.spinner("Buscando en la base de datos vectorial y analizando papers..."):
        # A. Recuperación densa en FAISS
        query_vector = embedding_model.encode([consulta_usuario]).astype('float32')
        _, indices = index.search(query_vector, 10)
        candidatos = df.iloc[indices[0]].copy()
        
        # B. Re-ranking de alta precisión
        pares = [[consulta_usuario, doc] for doc in candidatos['content'].tolist()]
        candidatos['rerank_score'] = reranker_model.predict(pares)
        documentos_top = candidatos.sort_values(by='rerank_score', ascending=False).head(3)
        
        # C. Construcción del Contexto para el LLM
        contexto = "\n\n".join([
            f"Documento [{i+1}]:\n{row['content']}" 
            for i, row in enumerate(documentos_top.to_dict(orient='records'))
        ])
        
        # Instrucciones estrictas del sistema para el control de información insuficiente
        system_prompt = (
        "Eres un asistente de investigación científica de IA riguroso y objetivo.\n"
        "Tu tarea es responder la consulta del usuario utilizando EXCLUSIVAMENTE el contexto proporcionado abajo.\n"
        "Reglas de respuesta obligatorias:\n"
        "1. Si el contexto NO contiene suficiente información para responder con certeza, "
        "debes responder EXACTAMENTE: 'Lo siento, el corpus no contiene información para responder'\n"
        "2. No inventes datos, no asumas hipótesis y no uses tu conocimiento externo.\n"
        "3. Respalda siempre tus respuestas citando explícitamente el documento de origen utilizando el formato [Documento X]."
        )
        
        # D. Generación con Gemini 2.5 Flash
        try:
            config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.0)
            respuesta_modelo = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Contexto:\n{contexto}\n\nConsulta: {consulta_usuario}",
                config=config
            )
            respuesta_final = respuesta_modelo.text
        except Exception as error_api:
            respuesta_final = f"Error al conectar con el LLM: {error_api}"
        
        # Estructurar la metadata de evidencias para la interfaz
        lista_evidencias = [
            {"titulo": row['titles'], "score": row['rerank_score']} 
            for _, row in documentos_top.iterrows()
        ]
        
    # Mostrar la respuesta del asistente en la interfaz web
    with st.chat_message("assistant"):
        st.markdown(respuesta_final)
        with st.expander("Ver Evidencias Científicas Utilizadas"):
            for ev in lista_evidencias:
                st.write(f"**Título:** {ev['titulo']}")
                st.caption(f"Score de Relevancia (Cross-Encoder): {ev['score']:.4f}")
                
    # Guardar la respuesta y sus evidencias en el estado de la sesión
    st.session_state.historial.append({
        "rol": "assistant", 
        "contenido": respuesta_final, 
        "evidencias": lista_evidencias
    })