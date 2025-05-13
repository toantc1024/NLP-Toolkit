import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader,
    WebBaseLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
import sys

# Import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.Embedding import hf
from config import settings

def app():
    st.title("AI Chatbot with Knowledge Base")
    
    # Initialize session state for messages and vector database
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    # Move sidebar content to a new tab and create three tabs
    tab1, tab2, tab3 = st.tabs(["Configuration", "Update Document", "Chatbot Interface"])
    
    # Tab 1: Configuration (formerly in sidebar)
    with tab1:
        st.header("Configuration")
        
        # API Key options - use system key from config or custom key
        api_key_option = st.radio(
            "API Key Source", 
            ["Use system .env key", "Enter custom key"],
            index=0 if settings.GOOGLE_GEN_AI_API_KEY else 1
        )
        
        if api_key_option == "Use system .env key":
            if settings.GOOGLE_GEN_AI_API_KEY:
                api_key = settings.GOOGLE_GEN_AI_API_KEY
                st.success("Using system API key from .env file")
            else:
                st.error("No API key found in system .env file. Please set GOOGLE_GEN_AI_API_KEY in .env or enter a custom key.")
                api_key = None
        else:
            # Custom API Key for Google Gemini
            api_key = st.text_input("Enter Google Gemini API Key", type="password")
            
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        
        # Model selection
        model_options = ["gemini-2.0-flash", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-03-25"]
        selected_model = st.selectbox("Select Gemini Model", model_options, index=0)
        
        # Toggle for using knowledge base
        use_kb = st.checkbox("Use Knowledge Base for Answers", value=True)
        
        # Temperature slider
        temperature = st.slider("Response Creativity", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        
        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    
    # Tab 2: Update Document (formerly tab1)
    with tab2:
        st.header("Update Knowledge Base")
        
        # Multi-file upload section
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader("Choose files", type=["pdf", "txt", "docx"], accept_multiple_files=True)
        
        # URL input
        st.subheader("Or Enter a URL")
        url_input = st.text_input("Enter URL to extract content")
        
        # Process document button
        process_doc = st.button("Process Documents")
        
        # Process documents when button is clicked
        if process_doc and (uploaded_files or url_input):
            with st.spinner("Processing documents..."):
                try:
                    documents = []
                    
                    # Handle uploaded files
                    for uploaded_file in uploaded_files:
                        # Save uploaded file to temp location
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_file_path = tmp_file.name
                        
                        # Load document based on file type
                        file_extension = uploaded_file.name.split('.')[-1].lower()
                        if file_extension == 'pdf':
                            loader = PyPDFLoader(temp_file_path)
                        elif file_extension == 'txt':
                            loader = TextLoader(temp_file_path)
                        elif file_extension == 'docx':
                            loader = UnstructuredWordDocumentLoader(temp_file_path)
                        
                        documents.extend(loader.load())
                        os.unlink(temp_file_path)  # Remove temp file
                    
                    # Handle URL input
                    if url_input:
                        loader = WebBaseLoader(url_input)
                        documents.extend(loader.load())
                    
                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(documents)
                    
                    # Create or update vector store
                    if st.session_state.vector_store is None:
                        # Initialize in-memory Qdrant store
                        st.session_state.vector_store = Qdrant.from_documents(
                            documents=splits,
                            embedding=hf,
                            location=":memory:",
                            collection_name="document_store"
                        )
                    else:
                        # Add to existing store
                        st.session_state.vector_store.add_documents(splits)
                    
                    st.success(f"Successfully processed {len(uploaded_files) + (1 if url_input else 0)} document(s) with {len(splits)} chunks!")
                    
                    # Display document status
                    if st.session_state.vector_store:
                        st.info(f"Knowledge base is active with documents loaded.")
                
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
        
            st.markdown("""
            **Supported File Types:**
            - PDF (.pdf)
            - Text (.txt)
            - Word (.docx)
            
            **Processing Steps:**
            1. Documents are uploaded and processed
            2. Text is extracted from each document
            3. Documents are split into smaller chunks
            4. BGE-M3 embeddings are created for each chunk
            5. Chunks are stored in a vector database for semantic search
            
            The knowledge base is stored in memory and will be reset when you refresh the page.
            """)
    
    # Tab 3: Chatbot Interface (formerly tab2)
    with tab3:
        # Create a layout with fixed chat input at bottom
        chat_area = st.container()
        input_area = st.container()
        
        # Knowledge base status indicator
        if st.session_state.vector_store:
            st.success("Knowledge base is active and ready for queries")
        else:
            st.warning("No documents loaded. Add documents in the 'Update Document' tab for better responses.")
        
        
        # Handle chat input in bottom container first (but render later)
        with input_area:
            prompt = st.chat_input("Ask a question about your documents...")
            if prompt:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Generate response
                try:
                    if api_key:
                        if use_kb and st.session_state.vector_store:
                            # Use knowledge base for response
                            llm = ChatGoogleGenerativeAI(
                                model=selected_model,
                                temperature=temperature,
                                convert_system_message_to_human=True
                            )
                            
                            # Create conversation chain
                            retriever = st.session_state.vector_store.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": 5}
                            )
                            
                            qa_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=retriever,
                                return_source_documents=True
                            )
                            
                            # Get chat history in the format expected by the chain
                            chat_history = [(m["content"], st.session_state.messages[i+1]["content"]) 
                                           for i, m in enumerate(st.session_state.messages[:-1]) 
                                           if m["role"] == "user" and i+1 < len(st.session_state.messages)]
                            
                            # Get response
                            result = qa_chain({"question": prompt, "chat_history": chat_history})
                            response = result["answer"]
                            
                            # Store source documents for next render
                            st.session_state.source_documents = result["source_documents"]
                            
                        else:
                            # Direct LLM response without knowledge base
                            llm = ChatGoogleGenerativeAI(
                                model=selected_model,
                                temperature=temperature,
                                convert_system_message_to_human=True
                            )
                            
                            response = llm.invoke(prompt).content
                            st.session_state.source_documents = None
                            
                        # Add assistant message to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Store token usage if available
                        try:
                            st.session_state.token_usage = llm.last_response_metadata.get("usage_metadata")
                        except:
                            st.session_state.token_usage = None
                            
                    else:
                        st.error("Please enter Google Gemini API Key in the 'Configuration' tab to continue.")
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                
                # Rerun to update UI with new messages
                st.rerun()
                
        # Display chat messages in scrollable container
        with chat_area:
            # Create scrollable container for messages
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Display all messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show source documents if available (outside the scrollable area)
            if "source_documents" in st.session_state and st.session_state.source_documents:
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(st.session_state.source_documents):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(doc.page_content)
                        st.markdown("---")
                        
            # Show token usage if available (outside the scrollable area)
            if "token_usage" in st.session_state and st.session_state.token_usage:
                usage = st.session_state.token_usage
                st.caption(f"Tokens used: {usage['total_tokens']} (Input: {usage['input_tokens']}, Output: {usage['output_tokens']})")
