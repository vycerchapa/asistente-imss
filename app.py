import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Configuración de la página
st.set_page_config(page_title="Consultor CCT IMSS-SNTSS", layout="wide")
st.title("🤖 Asistente Experto en CCT y Estatutos IMSS")

# Sidebar para configuración
with st.sidebar:
    api_key = st.text_input("Introduce tu Groq API Key:", type="password")
    uploaded_files = st.file_uploader("Sube los PDF del CCT y Estatutos", accept_multiple_files=True, type="pdf")

if uploaded_files and api_key:
    # 1. Procesamiento de documentos
    documents = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(uploaded_file.name)
        documents.extend(loader.load())

    # 2. Fragmentación (Chunking) para precisión
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # 3. Creación de Base de Datos Vectorial (Gratis con HuggingFace)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 4. Configuración del Modelo (Llama 3 de Groq)
    llm = ChatGroq(groq_api_key=api_key, model_name="llama3-70b-8192", temperature=0)

    # 5. Cadena de respuesta
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Interfaz de Chat
    user_question = st.text_input("Haz una pregunta sobre el contrato o estatutos:")
    if user_question:
        response = qa_chain.invoke(user_question)
        st.write("### Respuesta:")
        st.info(response["result"])