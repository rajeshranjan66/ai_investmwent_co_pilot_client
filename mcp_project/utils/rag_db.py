from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_core.documents import Document
import numpy as np
import logging
import chromadb
import os
from typing import List
from langchain_core.documents import Document
import tiktoken
import streamlit as st

LOCAL_MODEL_ENABLED = os.getenv("LOCAL_MODEL_ENABLED", "false").lower() == "true"
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "phi3:mini")  # Ollama model name

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FIX: Define an absolute path to the ChromaDB directory ---
# This ensures the path is correct regardless of the script's entry point.
# It assumes 'chroma_db' is in the project root, one level up from the 'utils' directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "chroma_db")


# def _get_token_count(text: str, model_name: str = "gpt-4") -> int:
#     """Helper to count tokens in a string."""
#     try:
#         encoding = tiktoken.encoding_for_model(model_name)
#         return len(encoding.encode(text))
#     except Exception:
#         return len(text.split())

# Replace your token counting function
def _get_token_count(text: str, model_name: str = None) -> int:
    """Get token count for text, handling both local and OpenAI models"""
    if LOCAL_MODEL_ENABLED:
        # Simple character-based estimation for local models (approx 4 chars per token)
        return len(str(text)) // 4
    else:
        # Original OpenAI token counting
        try:
            import tiktoken
            model_for_encoding = model_name or "gpt-4"
            encoding = tiktoken.encoding_for_model(model_for_encoding)
            return len(encoding.encode(text))
        except Exception:
            # Fallback to character estimation
            return len(str(text)) // 4


def _process_results(results, query, max_tokens):
    """Process and format retrieval results."""
    context_parts = []
    total_tokens = 0
    for doc in results:
        doc_tokens = _get_token_count(doc.page_content)
        if total_tokens + doc_tokens <= max_tokens:
            context_parts.append(f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}")
            total_tokens += doc_tokens
        else:
            break
    return "\n\n---\n\n".join(context_parts)


def get_context_from_chroma(prompt: str) -> str:
    """
    Enhanced ChromaDB context retrieval with multiple retrieval strategies.
    Simplified signature - only prompt required, other parameters handled internally.

    Args:
        prompt: Search query/user prompt

    Returns:
        Context string from relevant documents
    """
    # Configuration - you can adjust these as needed
    k = 4
    retrieval_method = "compression"  # Options: "similarity", "mmr", "compression"
    score_threshold = 0.65
    max_tokens = 2000

    try:
        # --- FIX: Use the absolute path defined above ---
        if not os.path.exists(PERSIST_DIRECTORY):
            logging.error(f"ChromaDB directory not found at: {PERSIST_DIRECTORY}")
            st.error(f"Error: The research database was not found. Please ensure it has been created at `{PERSIST_DIRECTORY}`.")
            return ""

        # Initialize embeddings and database
        embeddings = OpenAIEmbeddings()
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

        # Choose retrieval method
        if retrieval_method == "similarity":
            results = db.similarity_search(prompt, k=k)

        elif retrieval_method == "mmr":
            results = db.max_marginal_relevance_search(prompt, k=k, fetch_k=k * 2)

        elif retrieval_method == "compression":
            # This is the recommended method for better relevance
            compressor = EmbeddingsFilter(
                embeddings=embeddings,
                similarity_threshold=score_threshold
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=db.as_retriever(search_kwargs={"k": k * 2})
            )
            results = compression_retriever.get_relevant_documents(prompt)

        else:
            # Default to similarity search
            results = db.similarity_search(prompt, k=k)

        # Process and format results
        context = _process_results(results, prompt, max_tokens)

        logging.debug(f"Retrieved {len(results)} documents for query: '{prompt}'")
        if results and hasattr(results[0], 'metadata'):
            logging.debug(f"Top result source: {results[0].metadata.get('source', 'Unknown')}")

        return context

    except Exception as e:
        logging.error(f"Error retrieving context from ChromaDB: {str(e)}")
        return ""


# def get_context_from_chroma(prompt: str) -> str:
#     """
#     Enhanced ChromaDB context retrieval with multiple retrieval strategies.
#     Simplified signature - only prompt required, other parameters handled internally.
#
#     Args:
#         prompt: Search query/user prompt
#
#     Returns:
#         Context string from relevant documents
#     """
#     # Configuration - you can adjust these as needed
#     persist_directory = "chroma_db"
#     k = 4
#     retrieval_method = "compression"  # Options: "similarity", "mmr", "compression"
#     score_threshold = 0.65
#     max_tokens = 2000
#
#     try:
#         # Initialize embeddings and database
#         embeddings = OpenAIEmbeddings()
#         db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
#
#         # Choose retrieval method
#         if retrieval_method == "similarity":
#             results = db.similarity_search(prompt, k=k)
#
#         elif retrieval_method == "mmr":
#             results = db.max_marginal_relevance_search(prompt, k=k, fetch_k=k * 2)
#
#         elif retrieval_method == "compression":
#             # This is the recommended method for better relevance
#             compressor = EmbeddingsFilter(
#                 embeddings=embeddings,
#                 similarity_threshold=score_threshold
#             )
#             compression_retriever = ContextualCompressionRetriever(
#                 base_compressor=compressor,
#                 base_retriever=db.as_retriever(search_kwargs={"k": k * 2})
#             )
#             results = compression_retriever.get_relevant_documents(prompt)
#
#         else:
#             # Default to similarity search
#             results = db.similarity_search(prompt, k=k)
#
#         # Process and format results
#         context = _process_results(results, prompt, max_tokens)
#
#         logging.debug(f"Retrieved {len(results)} documents for query: '{prompt}'")
#         if results and hasattr(results[0], 'metadata'):
#             logging.debug(f"Top result source: {results[0].metadata.get('source', 'Unknown')}")
#
#         return context
#
#     except Exception as e:
#         logging.error(f"Error retrieving context from ChromaDB: {str(e)}")
#         return ""


# def _process_results(results: List[Document], prompt: str, max_tokens: int) -> str:
#     """Process and format retrieval results"""
#     if not results:
#         return ""
#
#     context_parts = []
#     current_length = 0
#
#     for i, doc in enumerate(results):
#         content = doc.page_content
#
#         # Add metadata information for better context
#         source = doc.metadata.get('source', 'Unknown')
#         chunk_id = doc.metadata.get('chunk_id', 'N/A')
#
#         formatted_content = f"[From {source}, section {chunk_id}]:\n{content}\n\n"
#
#         # Rough token estimation (4 chars ≈ 1 token)
#         content_tokens = len(formatted_content) // 4
#
#         if current_length + content_tokens > max_tokens:
#             break
#
#         context_parts.append(formatted_content)
#         current_length += content_tokens
#
#     return "\n".join(context_parts)


# Simple similarity-based fallback (your original approach)
def get_context_simple(prompt: str, persist_directory: str = "chroma_db", k: int = 4) -> str:
    """Fallback to simple similarity search if needed"""
    try:
        embeddings = OpenAIEmbeddings()
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        results = db.similarity_search(prompt, k=k)
        context = "\n".join([doc.page_content for doc in results])
        return context
    except Exception as e:
        logging.error(f"Error in simple context retrieval: {str(e)}")
        return ""