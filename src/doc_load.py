import os
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(
    level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_financial_docs(
        data_path : str = 'data/',
        chunk_size: int = 1000,
        chunk_overlap: int = 150
) -> List[Document]:
    '''
    Uploads all PDFs from the specified directory and splits the document.
    '''
    
    if not os.path.exists(data_path):
        logger.error(f"Directory not found: {data_path}")
        raise FileNotFoundError(f"The directory '{data_path}' does not exist.")
    
    logger.info(f"Scanning '{data_path}' for PDF documents")

    try:
        loader = DirectoryLoader(data_path, glob='./*.pdf', loader_cls = PyPDFLoader)
        raw_documents = loader.load()

        if not raw_documents:
            logger.warning(f"No PDFs found in{data_path}. The vector store will be empty.")
            return []
        
        logger.info(f"Successfully loaded {len(raw_documents)} documents. Splitting text.")

        text_split = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators = ['\n\n', '\n', ' ', '']
        )

        text_chunks = text_split.split_documents(raw_documents)
        logger.info(f"Documents successfuly split into {len(text_chunks)} individual chunks.")
        return text_chunks
    
    except Exception as e:
        logger.error(f"An error occured while processing the documents: {e}")
        raise

if __name__ == '__main__':
    logger.info("Starting document ingestion pipeline")
    try:
        docs = load_financial_docs()
        if docs:
            source_file = docs[0].metadata.get('source', 'Unknown File')
            logger.info(f"Preview of the first chunk from {source_file}:")
            logger.info(f"\n{docs[0].page_content[:200]}...\n")
    except Exception as e:
        logger.error("Ingestion pipeline has failed.")