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
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        docs = retriever.get_relevant_documents(query)
        filtered_docs = fuzzy_filter(query, docs)

        if filtered_docs:
            filtered_vectorstore = FAISS.from_documents(filtered_docs, embedding_model)
            retriever = filtered_vectorstore.as_retriever()
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={
                "prompt": {
                    "input_variables": ["context", "question"],
                    "template": (
                        system_prompt + "\n\n"
                        "Context:\n{context}\n\n"
                        "Question: {question}\nAnswer:"
                    )
                }
            })
            return qa.run(query)
        else:
            return llm.invoke(system_prompt + "\n\nUser: " + query)
    else:
        return llm.invoke(system_prompt + "\n\nUser: " + query)



