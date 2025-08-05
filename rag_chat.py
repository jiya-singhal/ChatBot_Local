# rag_chat.py
import os
import faiss
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document
from fuzzywuzzy import fuzz

# Load embedding model once
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load Ollama LLM once
llm = Ollama(model="llama3")

# Chunk splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

def load_vectorstore_from_file(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    
    # Optional: Save index locally
    vectorstore.save_local("index_store")
    return vectorstore

def load_vectorstore_from_text(text):
    doc = Document(page_content=text)
    chunks = splitter.split_documents([doc])
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    
    # Optional: Save index locally
    vectorstore.save_local("index_store")
    return vectorstore

def fuzzy_filter(query, docs, threshold=70):
    return [doc for doc in docs if fuzz.partial_ratio(query.lower(), doc.page_content.lower()) > threshold]

def get_llm_response(query, vectorstore=None, system_prompt="You are a helpful assistant."):
    from langchain_community.chat_models import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage

    model = ChatOllama(model="llama3", temperature=0.4)

    context = ""
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join(doc.page_content for doc in docs)

    messages = [
        SystemMessage(content=system_prompt + "\nUse the following context to help answer:\n" + context),
        HumanMessage(content=query)
    ]

    response = model.invoke(messages)
    return response.content

