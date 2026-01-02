import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Consultor CCT IMSS-SNTSS", layout="wide")
st.title("🤖 Asistente Experto en CCT y Estatutos IMSS")
st.markdown("---")

with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Introduce tu Groq API Key:", type="password")
    uploaded_files = st.file_uploader("Sube los archivos PDF (CCT/Estatutos)", accept_multiple_files=True, type="pdf")

if uploaded_files and api_key:
    try:
        with st.status("Procesando documentos..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                temp_file = f"temp_{uploaded_file.name}"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = PyPDFLoader(temp_file)
                all_docs.extend(loader.load())
                os.remove(temp_file)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(all_docs)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)

            llm = ChatGroq(groq_api_key=api_key, model_name="llama3-70b-8192", temperature=0)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever()
            )

        st.success("¡Documentos listos!")
        user_question = st.text_input("Escribe tu duda:")
        
        if user_question:
            with st.spinner("Buscando..."):
                response = qa_chain.invoke({"query": user_question})
                st.write("### Respuesta:")
                st.info(response["result"])

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.warning("👈 Ingresa tu API Key y sube un PDF.")
