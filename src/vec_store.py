import os
import logging
from typing import List, Optional, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

def get_embedding_model() -> OpenAIEmbeddings:
    '''
    Helper function to ensure the embedding model is initialized with API key checks.
    '''
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('OPENAI_API_KEY not found in environment variables.')
    return OpenAIEmbeddings(model='text-embedding-3-small')

def vector_store_creator(
        text_chunks: List[Document],
        persist_directory: str ='vector_db'
) -> Chroma:
    '''
    Takes text chunks from the document, makes embeddings, and saves them to a local Chroma DB.
    '''
    if not text_chunks:
        logger.error('No text chunks are provided. Cannot create vector store.')
        raise ValueError('text_chunks list is empty.')
    logger.info(f'Creating embeddings for {len(text_chunks)} chunks.')

    try:
        embeddings = get_embedding_model()

        vector_db = Chroma.from_documents(
            documents=text_chunks,
            embedding = embeddings, 
            persist_directory=persist_directory
        )
        logger.info(f"Vector store is successfully created and saved to '{persist_directory}/'.")
        return vector_db
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise

def vector_store_loader(persist_directory: str ='vector_db') -> Optional[Chroma]:
    '''
    This will load an existing vector store from disk.
    '''
    if not os.path.exists(persist_directory):
        logger.error(f"Vector store directory '{persist_directory}' doesn't exist. Please run the creator first.")
        raise FileNotFoundError(f"Missing directory: {persist_directory}")
    
    logger.info(f"Loading existing vector store from '{persist_directory}'")

    try:
        embeddings = get_embedding_model()
        vector_db = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embeddings
        )
        logger.info("Vector store has loaded successfully.")
        return vector_db
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise
    
if __name__ == '__main__':
    try:
        from doc_load import load_financial_docs

        logger.info("Testing vector store pipeline")
        sample_chunks = load_financial_docs()

        if sample_chunks:
            db = vector_store_creator(sample_chunks)
            logger.info("Test is complete. Vector store is fully operational.")

    except Exception as e:
        logger.error("Vector store pipeline test has failed.")