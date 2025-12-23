import google.generativeai as genai
from langchain_core.embeddings import Embeddings
from typing import List
import streamlit as st
from .config import EMBEDDING_MODEL_ID

class GeminiEmbedder(Embeddings):
    """
    Custom Embedder using Google Gemini's text-embedding-004 model.
    """
    def __init__(self, model_name=EMBEDDING_MODEL_ID):
        # Ensure API key is configured from session state before using
        if st.session_state.get('google_api_key'):
             genai.configure(api_key=st.session_state.google_api_key)
        self.model = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        try:
             response = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
             return response['embedding']
        except Exception as e:
            # Return empty list or handle error appropriately in production
            st.error(f"Embedding error: {e}")
            return []
