# To run this, launch a command prompt and activate the 'rag-chatbot' conda environment
# using conda activate rag-chatbot
# In the directory containing app.py run: streamlit run app.py
# Ensure Ollama servers are running

import streamlit as st
import os
import sys


# Import standard libraries for file handling and text processing
import os, pathlib, textwrap, glob

# Load documents from various sources (URLs, text files, PDFs)
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader, PyPDFLoader

# Split long texts into smaller, manageable chunks for embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store to store and retrieve embeddings efficiently using FAISS
from langchain.vectorstores import FAISS

# Generate text embeddings using OpenAI or Hugging Face models
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings

# Use local LLMs (e.g., via Ollama) for response generation
from langchain.llms import Ollama

# Build a retrieval chain that combines a retriever, a prompt, and an LLM
from langchain.chains import ConversationalRetrievalChain

# Create prompts for the RAG system
from langchain.prompts import PromptTemplate

def main():

    print("✅ Libraries imported! You're good to go!")

    pdf_paths = glob.glob("C:/rajesh/docs/personal/career/learning/bytebytegoAI/ai-eng-projects/project_2/data/Everstorm_*.pdf")
    raw_docs = []

    # Load each PDF file using PyPDFLoader
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        raw_docs.extend(loader.load())

    print(f"Loaded {len(raw_docs)} PDF pages from {len(pdf_paths)} files.")

    # Create text splitter with 300 token chunks and 30 token overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = text_splitter.split_documents(raw_docs)

    print(f"✅ {len(chunks)} chunks ready for embedding")

    # Embed the sentence "Hello world!" and store it in an embedding_vector.
    from langchain_community.embeddings import HuggingFaceInstructEmbeddings

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding_vector = embeddings.embed_query("Hello world!")

    print(len(embedding_vector))

    # Expected steps:
        # 1. Build the FAISS index from the list of document chunks and their embeddings.
        # 2. Create a retriever object with a suitable k value (e.g., 8).
        # 3. Save the vector store locally (e.g., under "faiss_index").
        # 4. Print a short confirmation showing how many embeddings were stored.

     # Build FAISS vector store from document chunks
    vectordb = FAISS.from_documents(documents=chunks, embedding=embeddings)

    # Create retriever with k=8 (retrieve top 8 most similar chunks)
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})

    # Save the vector store locally
    vectordb.save_local("faiss_index")

    print("✅ Vector store with", vectordb.index.ntotal, "embeddings")

    llm = Ollama(model="gemma3:1b", temperature=0.1)

    # Test the model with a simple prompt
    response = llm.invoke("What is the capital of India?")
    print(response)


    SYSTEM_TEMPLATE = """
    You are a **Customer Support Chatbot**. Use only the information in CONTEXT to answer.
    If the answer is not in CONTEXT, respond with “I'm not sure from the docs.”

    Rules:
    1) Use ONLY the provided <context> to answer.
    2) If the answer is not in the context, say: "I don't know based on the retrieved documents."
    3) Be concise and accurate. Prefer quoting key phrases from the context.

    CONTEXT:
    {context}

    USER:
    {question}
    """

    prompt = PromptTemplate(template=SYSTEM_TEMPLATE, input_variables=["context", "question"])

    # Initialize LLM with Ollama (reuse the existing llm variable if already defined)
    if 'llm' not in locals():
        llm = Ollama(model="gemma3:1b", temperature=0.1)

    # Build the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    print("✅ RAG chain created successfully!")

    # Set page config
    st.set_page_config(
        page_title="Everstorm Customer Support Chatbot",
        page_icon="🛍️",
        layout="wide"
    )

    # Title and description
    st.title("🛍️ Everstorm Customer Support Chatbot")
    st.markdown("Ask me anything about Everstorm Outfitters policies, shipping, returns, and more!")

    # Check if required variables are available
    print("Globals:", globals().keys())  # Debugging line to check globals
    if chain is None:
        st.error("❌ RAG chain not found! Please run the notebook cells first to initialize the chain.")
        st.stop()

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about Everstorm policies..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from RAG chain
                    response = chain({"question": prompt, "chat_history": []})
                    answer = response['answer']
                    
                    # Display assistant response
                    st.markdown(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Sidebar with information
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This chatbot can help you with:
        - **Refund & Return Policies**
        - **Shipping & Delivery Information**
        - **Product Sizing & Care Guides**
        - **Payment & Security Policies**
        - **General Customer Support**
        """)
        
        st.header("🔧 Technical Info")
        st.markdown("""
        - **Model**: Gemma3:1b (via Ollama)
        - **Vector Store**: FAISS
        - **Embeddings**: SentenceTransformers
        - **Documents**: Everstorm PDFs
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip**: Ask specific questions for the best results!")

# import streamlit as st

# def main():
#     print("Hello from app_test.py!")
#     st.title("Test Chat Interface")

if __name__ == "__main__":
    main()

