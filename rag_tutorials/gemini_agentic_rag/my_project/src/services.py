import tempfile
from datetime import datetime
from typing import List, Optional

import streamlit as st
import bs4
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .config import COLLECTION_NAME, VECTOR_SIZE, CHUNK_SIZE, CHUNK_OVERLAP, QDRANT_PATH
from .models import GeminiEmbedder

@st.cache_resource
def get_local_qdrant_client() -> Optional[QdrantClient]:
    """Get or create a cached local Qdrant client."""
    try:
        return QdrantClient(path=QDRANT_PATH)
    except Exception as e:
        st.error(f"🔴 Local Qdrant connection failed: {str(e)}")
        return None

@st.cache_resource
def get_cloud_qdrant_client(_api_key: str, _url: str) -> Optional[QdrantClient]:
    """Get or create a cached cloud Qdrant client."""
    try:
        if not _api_key or not _url:
            return None
        return QdrantClient(url=_url, api_key=_api_key, timeout=60)
    except Exception as e:
        st.error(f"🔴 Cloud Qdrant connection failed: {str(e)}")
        return None

def init_qdrant() -> Optional[QdrantClient]:
    """Initialize Qdrant client with configured settings."""
    storage_mode = st.session_state.get('qdrant_storage_mode', 'Cloud')
    
    if storage_mode == 'Local':
        return get_local_qdrant_client()
    else:
        api_key = st.session_state.get('qdrant_api_key', '')
        url = st.session_state.get('qdrant_url', '')
        if not api_key or not url:
            return None
        return get_cloud_qdrant_client(api_key, url)

def get_indexed_documents(client: QdrantClient) -> List[str]:
    """Retrieve list of already indexed documents from Qdrant."""
    try:
        # Check if collection exists
        if not client.collection_exists(COLLECTION_NAME):
            st.info(f"📋 Collection '{COLLECTION_NAME}' does not exist yet")
            return []
        
        # Get collection info
        collection_info = client.get_collection(COLLECTION_NAME)
        st.info(f"📊 Collection has {collection_info.points_count} points")
        
        if collection_info.points_count == 0:
            return []
        
        # Scroll through all points to get unique source names
        sources = set()
        offset = None
        all_keys = set()  # 用于调试
        
        while True:
            points, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=None,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                break
                
            for point in points:
                if point.payload:
                    # 收集所有键用于调试
                    all_keys.update(point.payload.keys())
                    
                    # 尝试多种可能的字段名
                    source_name = None
                    
                    # 方法1: 直接字段
                    if 'file_name' in point.payload:
                        source_name = point.payload['file_name']
                    elif 'url' in point.payload:
                        source_name = point.payload['url']
                    elif 'source' in point.payload:
                        source_name = point.payload['source']
                    # 方法2: 嵌套在 metadata 中
                    elif 'metadata' in point.payload:
                        metadata = point.payload['metadata']
                        if isinstance(metadata, dict):
                            source_name = (metadata.get('file_name') or 
                                         metadata.get('url') or 
                                         metadata.get('source'))
                    
                    if source_name:
                        sources.add(source_name)
            
            # Check if there are more points
            if next_offset is None:
                break
            offset = next_offset
        
        # 调试信息
        if sources:
            st.success(f"✅ Found {len(sources)} unique document(s)")
        else:
            st.warning(f"⚠️ No document sources found. Payload keys: {sorted(all_keys)}")
        
        return sorted(list(sources))  # Sort for consistent display
    except Exception as e:
        st.error(f"❌ Could not retrieve indexed documents: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return []


def process_pdf(file) -> List:
    """Process PDF file and add source metadata."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat()
                })
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"📄 PDF processing error: {str(e)}")
        return []

def process_web(url: str) -> List:
    """Process web URL and add source metadata."""
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header", "content", "main")
                )
            )
        )
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata.update({
                "source_type": "url",
                "url": url,
                "timestamp": datetime.now().isoformat()
            })
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"🌐 Web processing error: {str(e)}")
        return []

def create_vector_store(client: QdrantClient, texts: List) -> Optional[QdrantVectorStore]:
    """Create and initialize vector store with documents."""
    try:
        # Create collection if needed
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            st.success(f"📚 Created new collection: {COLLECTION_NAME}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e
        
        # Initialize vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=GeminiEmbedder()
        )
        
        # Add documents
        with st.spinner('📤 Uploading documents to Qdrant...'):
            vector_store.add_documents(texts)
            st.success("✅ Documents stored successfully!")
            return vector_store
            
    except Exception as e:
        st.error(f"🔴 Vector store error: {str(e)}")
        return None

def check_document_relevance(query: str, vector_store, threshold: float = 0.7) -> tuple[bool, List]:
    """Check if documents in vector store are relevant to the query."""
    if not vector_store:
        return False, []
        
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": threshold}
    )
    docs = retriever.invoke(query)
    return bool(docs), docs
