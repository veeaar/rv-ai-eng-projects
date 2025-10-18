# import streamlit as st
# import os
# import sys

# # Add the current directory to Python path to import variables from notebook
# sys.path.append('C:/rajesh/docs/personal/career/learning/bytebytegoAI/ai-eng-projects/project_2')

# # Set page config
# st.set_page_config(
#     page_title="Everstorm Customer Support Chatbot",
#     page_icon="🛍️",
#     layout="wide"
# )

# # Title and description
# st.title("🛍️ Everstorm Customer Support Chatbot")
# st.markdown("Ask me anything about Everstorm Outfitters policies, shipping, returns, and more!")

# # Check if required variables are available
# if 'chain' not in globals():
#     st.error("❌ RAG chain not found! Please run the notebook cells first to initialize the chain.")
#     st.stop()

# # Initialize session state for chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Chat input
# if prompt := st.chat_input("Ask me about Everstorm policies..."):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # Generate response
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             try:
#                 # Get response from RAG chain
#                 response = chain({"question": prompt, "chat_history": []})
#                 answer = response['answer']
                
#                 # Display assistant response
#                 st.markdown(answer)
                
#                 # Add assistant response to chat history
#                 st.session_state.messages.append({"role": "assistant", "content": answer})
                
#             except Exception as e:
#                 error_msg = f"Sorry, I encountered an error: {str(e)}"
#                 st.error(error_msg)
#                 st.session_state.messages.append({"role": "assistant", "content": error_msg})

# # Sidebar with information
# with st.sidebar:
#     st.header("📋 About")
#     st.markdown("""
#     This chatbot can help you with:
#     - **Refund & Return Policies**
#     - **Shipping & Delivery Information**
#     - **Product Sizing & Care Guides**
#     - **Payment & Security Policies**
#     - **General Customer Support**
#     """)
    
#     st.header("🔧 Technical Info")
#     st.markdown("""
#     - **Model**: Gemma3:1b (via Ollama)
#     - **Vector Store**: FAISS
#     - **Embeddings**: SentenceTransformers
#     - **Documents**: Everstorm PDFs
#     """)
    
#     # Clear chat button
#     if st.button("🗑️ Clear Chat History"):
#         st.session_state.messages = []
#         st.rerun()

# # Footer
# st.markdown("---")
# st.markdown("💡 **Tip**: Ask specific questions for the best results!")

import streamlit as st

def main():
    print("Hello from app_test.py!")
    st.title("Test Chat Interface")

if __name__ == "__main__":
    main()

