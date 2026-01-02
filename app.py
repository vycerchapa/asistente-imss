import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Consultor CCT IMSS", layout="wide")
st.title("🤖 Asistente CCT y Estatutos IMSS")

with st.sidebar:
    key = st.text_input("Groq API Key:", type="password")
    files = st.file_uploader("Sube los PDF", accept_multiple_files=True, type="pdf")

if files and key:
    try:
        docs = []
        for f in files:
            with open(f.name, "wb") as temp:
                temp.write(f.getbuffer())
            loader = PyPDFLoader(f.name)
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeds = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embeds)

        llm = ChatGroq(groq_api_key=key, model_name="llama3-70b-8192", temperature=0)
        
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

        pregunta = st.text_input("¿Qué deseas consultar?")
        if pregunta:
            res = qa.invoke({"query": pregunta})
            st.info(res["result"])
    except Exception as e:
        st.error(f"Error: {e}")
