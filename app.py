import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from pypdf import PdfReader
import tempfile
import os

# Configuración de la página
st.set_page_config(page_title="Asistente IMSS - CCT y Estatutos", layout="wide")
st.title("🤖 Asistente Experto en el CCT y Estatutos del IMSS")
st.markdown("Sube documentos oficiales del IMSS (PDF) y haz preguntas sobre ellos.")

# === Entrada de API Key ===
groq_api_key = st.sidebar.text_input(
    "🔑 Clave API de Groq", 
    type="password",
    help="Obtén tu clave en https://console.groq.com/"
)

# === Carga de PDFs ===
uploaded_files = st.file_uploader(
    "📂 Sube uno o varios PDFs del IMSS", 
    type=["pdf"], 
    accept_multiple_files=True
)

# === Inicialización del vectorstore en session_state ===
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# === Procesamiento de PDFs ===
if uploaded_files and groq_api_key:
    with st.spinner("Procesando documentos..."):
        try:
            all_docs = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name

                reader = PdfReader(tmp_path)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
                os.unlink(tmp_path)  # Elimina archivo temporal

                if text.strip():
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200
                    )
                    chunks = splitter.split_text(text)
                    all_docs.extend([Document(page_content=chunk) for chunk in chunks])

            if not all_docs:
                st.warning("⚠️ No se extrajo texto de los PDFs. Verifica que no estén escaneados.")
            else:
                with st.spinner("Generando base de conocimiento..."):
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    st.session_state.vectorstore = FAISS.from_documents(all_docs, embeddings)
                st.success(f"✅ Procesados {len(all_docs)} fragmentos de texto.")
        except Exception as e:
            st.error(f"❌ Error al procesar los PDFs: {str(e)}")

# === Pregunta del usuario ===
st.divider()
user_question = st.text_input("💬 ¿Qué deseas saber del CCT o Estatutos del IMSS?")

if st.button("Enviar pregunta"):
    if not groq_api_key:
        st.error("⚠️ Por favor, ingresa tu clave API de Groq en el panel lateral.")
    elif not st.session_state.vectorstore:
        st.error("⚠️ Primero sube al menos un PDF para consultar.")
    elif not user_question.strip():
        st.warning("⚠️ Escribe una pregunta antes de enviar.")
    else:
        try:
            with st.spinner("Pensando..."):
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name="llama3-70b-8192",
                    temperature=0.2,
                    max_tokens=1024
                )
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}),
                    return_source_documents=False
                )
                response = qa_chain.invoke({"query": user_question})
                st.markdown("### 📌 Respuesta:")
                st.write(response["result"])
        except Exception as e:
            st.error(f"❌ Error al generar la respuesta: {str(e)}")
            st.info("💡 ¿Olvidaste activar tu clave API o el modelo está temporalmente no disponible?")

# === Pie de página ===
st.divider()
st.caption("🔒 Tu clave API nunca se almacena. Todo el procesamiento se hace en tiempo real.")
