import os
import streamlit as st
import google.generativeai as genai

from src.ui import (
    init_session_state, 
    render_sidebar, 
    render_chat_interface, 
    display_chat_history,
    display_sources
)
from src.services import (
    init_qdrant, 
    process_pdf, 
    process_web, 
    create_vector_store,
    get_indexed_documents
)
from src.agents import (
    get_query_rewriter_agent, 
    get_web_search_agent, 
    get_rag_agent
)

def main():
    st.title("🤔 Agentic RAG with Gemini Thinking and Agno")
    
    # Initialize session state
    init_session_state()
    
    # Render Sidebar
    search_domains = render_sidebar()
    
    # Debug: Show current storage mode
    st.sidebar.info(f"🔍 Current mode: {st.session_state.get('qdrant_storage_mode', 'Not set')}")
    
    # Init Qdrant client (can work without API key in Local mode)
    qdrant_client = init_qdrant()
    
    # Debug: Show client status
    if qdrant_client:
        st.sidebar.success("✅ Qdrant client connected")
        
        # Load existing documents from vector DB on first run (without needing API key)
        if not st.session_state.documents_loaded:
            with st.spinner('📚 Loading existing documents from database...'):
                existing_docs = get_indexed_documents(qdrant_client)
                if existing_docs:
                    st.session_state.processed_documents = existing_docs
                    st.sidebar.info(f"✅ Loaded {len(existing_docs)} existing document(s)")
            st.session_state.documents_loaded = True
    else:
        st.sidebar.error("❌ Qdrant client not connected")
    
    # Display Processed Sources (always show if we have documents)
    if st.session_state.processed_documents:
        st.sidebar.header("📚 Processed Sources")
        for source in st.session_state.processed_documents:
            icon = "📄" if source.endswith('.pdf') else "🌐"
            st.sidebar.text(f"{icon} {source}")


    
    # API Configuration Check
    if st.session_state.google_api_key:
        os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
        genai.configure(api_key=st.session_state.google_api_key)
        
        # Initialize vector store if not already done
        if qdrant_client and st.session_state.vector_store is None and st.session_state.processed_documents:
            try:
                from src.config import COLLECTION_NAME
                from src.models import GeminiEmbedder
                from langchain_qdrant import QdrantVectorStore
                st.session_state.vector_store = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name=COLLECTION_NAME,
                    embedding=GeminiEmbedder()
                )
            except Exception as e:
                st.warning(f"⚠️ Could not initialize vector store: {e}")
        
        # Data Upload Section in Sidebar


        st.sidebar.header("📁 Data Upload")
        uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
        web_url = st.sidebar.text_input("Or enter URL")
        
        # Process Uploads
        if uploaded_file and uploaded_file.name not in st.session_state.processed_documents:
            with st.spinner('Processing PDF...'):
                texts = process_pdf(uploaded_file)
                if texts and qdrant_client:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(texts)
                    else:
                        st.session_state.vector_store = create_vector_store(qdrant_client, texts)
                    st.session_state.processed_documents.append(uploaded_file.name)
                    st.success(f"✅ Added PDF: {uploaded_file.name}")

        if web_url and web_url not in st.session_state.processed_documents:
            with st.spinner('Processing URL...'):
                texts = process_web(web_url)
                if texts and qdrant_client:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(texts)
                    else:
                        st.session_state.vector_store = create_vector_store(qdrant_client, texts)
                    st.session_state.processed_documents.append(web_url)
                    st.success(f"✅ Added URL: {web_url}")

        # Chat Area
        prompt = render_chat_interface()
        
        if prompt:
            st.session_state.history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # 1. Rewriting Query
            rewritten_query = prompt
            with st.spinner("🤔 Reformulating query..."):
                try:
                    query_rewriter = get_query_rewriter_agent()
                    rewritten_query = query_rewriter.run(prompt).content
                    with st.expander("🔄 See rewritten query"):
                        st.write(f"Original: {prompt}")
                        st.write(f"Rewritten: {rewritten_query}")
                except Exception as e:
                    st.warning(f"⚠️ Query rewriting failed: {e}")

            # 2. Search Strategy
            context = ""
            docs = []
            if not st.session_state.force_web_search and st.session_state.vector_store:
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "k": 5, 
                        "score_threshold": st.session_state.similarity_threshold
                    }
                )
                docs = retriever.invoke(rewritten_query)
                if docs:
                    context = "\n\n".join([d.page_content for d in docs])
                    st.info(f"📊 Found {len(docs)} relevant documents")
                elif st.session_state.use_web_search:
                    st.info("🔄 No documents found, checking web...")

            # 3. Web Search
            if (st.session_state.force_web_search or not context) and st.session_state.use_web_search and st.session_state.exa_api_key:
                with st.spinner("🔍 Searching the web..."):
                    try:
                        web_search_agent = get_web_search_agent(search_domains)
                        web_results = web_search_agent.run(rewritten_query).content
                        if web_results:
                            context = f"Web Search Results:\n{web_results}"
                            st.info("ℹ️ Using web search results")
                    except Exception as e:
                        st.warning(f"⚠️ Web search failed: {e}")

            # 4. Generate Response
            with st.spinner("🤖 Thinking..."):
                try:
                    rag_agent = get_rag_agent()
                    full_prompt = f"Context: {context}\n\nQuestion: {prompt}\nRewritten: {rewritten_query}" if context else f"Question: {prompt}"
                    
                    response = rag_agent.run(full_prompt)
                    
                    st.session_state.history.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    
                    with st.chat_message("assistant"):
                        st.write(response.content)
                        if not st.session_state.force_web_search and docs:
                            display_sources(docs)
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    else:
        st.warning("⚠️ Please enter your Google API Key to continue")
        st.info("""
        **💡 Gemini API Tips**:
        - Free quota: 15 RPM, 1M TPM, 1.5K RPD
        - Check usage: [Google AI Studio](https://ai.dev/usage)
        """)

if __name__ == "__main__":
    main()
