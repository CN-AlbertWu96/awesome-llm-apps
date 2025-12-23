import streamlit as st

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'google_api_key': "",
        'qdrant_api_key': "",
        'qdrant_url': "",
        'vector_store': None,
        'processed_documents': [],
        'history': [],
        'exa_api_key': "",
        'use_web_search': False,
        'force_web_search': False,
        'similarity_threshold': 0.7
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_sidebar():
    """Render the sidebar configuration."""
    st.sidebar.header("🔑 API Configuration")
    
    st.session_state.google_api_key = st.sidebar.text_input(
        "Google API Key", 
        type="password", 
        value=st.session_state.google_api_key
    )
    st.session_state.qdrant_api_key = st.sidebar.text_input(
        "Qdrant API Key", 
        type="password", 
        value=st.session_state.qdrant_api_key
    )
    st.session_state.qdrant_url = st.sidebar.text_input(
        "Qdrant URL",
        placeholder="https://your-cluster.cloud.qdrant.io:6333",
        value=st.session_state.qdrant_url
    )

    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.history = []
        st.rerun()

    st.sidebar.header("🌐 Web Search Configuration")
    st.session_state.use_web_search = st.sidebar.checkbox(
        "Enable Web Search Fallback", 
        value=st.session_state.use_web_search
    )

    search_domains = []
    if st.session_state.use_web_search:
        st.session_state.exa_api_key = st.sidebar.text_input(
            "Exa AI API Key",
            type="password",
            value=st.session_state.exa_api_key,
            help="Required for web search fallback when no relevant documents are found"
        )
        
        default_domains = ["arxiv.org", "wikipedia.org", "github.com", "medium.com"]
        custom_domains = st.sidebar.text_input(
            "Custom domains (comma-separated)",
            value=",".join(default_domains),
            help="Enter domains to search from, e.g.: arxiv.org,wikipedia.org"
        )
        search_domains = [d.strip() for d in custom_domains.split(",") if d.strip()]

    st.sidebar.header("🎯 Search Configuration")
    st.session_state.similarity_threshold = st.sidebar.slider(
        "Document Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.similarity_threshold,
        help="Lower values will return more documents but might be less relevant. Higher values are more strict."
    )
    
    return search_domains

def render_chat_interface():
    """Render the chat input and toggle."""
    chat_col, toggle_col = st.columns([0.9, 0.1])
    
    with chat_col:
        prompt = st.chat_input("Ask about your documents...")
        
    with toggle_col:
        st.session_state.force_web_search = st.toggle('🌐', help="Force web search")
        
    return prompt

def display_chat_history():
    """Display the chat history."""
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def display_sources(docs):
    """Display document sources in an expander."""
    with st.expander("🔍 See document sources"):
        for i, doc in enumerate(docs, 1):
            source_type = doc.metadata.get("source_type", "unknown")
            source_icon = "📄" if source_type == "pdf" else "🌐"
            source_name = doc.metadata.get("file_name" if source_type == "pdf" else "url", "unknown")
            st.write(f"{source_icon} Source {i} from {source_name}:")
            st.write(f"{doc.page_content[:200]}...")
