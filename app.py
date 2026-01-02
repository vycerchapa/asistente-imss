import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Configuración de la interfaz
st.set_page_config(page_title="Consultor CCT IMSS-SNTSS", layout="wide")
st.title("🤖 Asistente Experto en CCT y Estatutos IMSS")
st.markdown("---")

# Barra lateral para configuración
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Introduce tu Groq API Key:", type="password")
    uploaded_files = st.file_uploader("Sube los archivos PDF (CCT/Estatutos)", accept_multiple_files=True, type="pdf")
    st.info("Nota: Los archivos se procesan localmente en la sesión.")

# Lógica principal
if uploaded_files and api_key:
    try:
        with st.status("Procesando documentos..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                # Guardar temporalmente para que PyPDFLoader lo lea
                temp_file = f"temp_{uploaded_file.name}"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = PyPDFLoader(temp_file)
                all_docs.extend(loader.load())
                os.remove(temp_file) # Limpiar archivo temporal

            # Dividir texto en fragmentos
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(all_docs)

            # Crear base de datos vectorial
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)

            # Configurar el modelo Llama 3
            llm = ChatGroq(groq_api_key=api_key, model_name="llama3-70b-8192", temperature=0)

            # Crear cadena de consulta
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever()
            )

        # Campo de consulta
        st.success("¡Documentos listos! Ya puedes preguntar.")
        user_question = st.text_input("Escribe tu duda sobre el CCT o Estatutos:")
        
        if user_question:
            with st.spinner("Buscando en los documentos..."):
                response = qa_chain.invoke({"query": user_question})
                st.write("### Respuesta:")
                st.info(response["result"])

    except Exception as e:
        st.error(f"Se produjo un error técnico: {e}")
else:
    st.warning("👈 Por favor, ingresa tu API Key y sube al menos un PDF en la barra lateral.")
