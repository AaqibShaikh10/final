import os
import warnings
import logging
from datetime import datetime

import streamlit as st

# LangChain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Disable warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Streamlit page config
st.set_page_config(page_title="RAG Chatbot", layout="centered")

# --- Custom CSS (optional) ---
st.markdown("""
    <style>
    body { background-color: #D3D3D3; }
    .main { background-color: #C0C0C0; border-radius: 5px; padding: 20px; }
    .stApp h1 { color: #000000; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- App title ---
st.title("RAG Chatbot")

# --- Chat history ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# --- Load Vector Store ---
@st.cache_resource
def get_vectorstore():
    with st.spinner("Loading PDF and creating index..."):
        loader = PyPDFLoader("document.pdf")
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        ).from_loaders([loader])
        return index.vectorstore

# --- Input prompt ---
prompt = st.chat_input("Ask something about the PDF...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    groq_chat = ChatGroq(
        groq_api_key=st.secrets["groq"]["api_key"],
        model_name="llama3-8b-8192"
    )

    vectorstore = get_vectorstore()
    chain = RetrievalQA.from_chain_type(
        llm=groq_chat,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=False
    )

    response = chain({"query": prompt})["result"]
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})

# --- Export history ---
if st.session_state.get("messages"):
    def format_chat():
        chat_lines = []
        for msg in st.session_state.messages:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            chat_lines.append(f"[{timestamp}] {msg['role'].capitalize()}: {msg['content']}")
        return "\n\n".join(chat_lines)

    st.download_button(
        "Save Chat History",
        data=format_chat(),
        file_name="chat_history.txt",
        mime="text/plain"
    )
