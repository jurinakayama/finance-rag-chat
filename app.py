import os
import sys
import shutil
import argparse
import logging
from dotenv import load_dotenv

from src.doc_load import load_financial_docs
from src.vec_store import vector_store_creator, vector_store_loader
from src.langmodel_engine import get_qa_chain

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def parse_args():
    '''
    Sets up command-line arguments.'''
    parser = argparse.ArgumentParser(description="Finance RAG Chat Agent")
    parser.add_argument(
        '--reset',
        action = 'store_true',
        help = 'Deletes the existing vector store and rebuilds it from the data folder.'
    )
    return parser.parse_args()
    
def main():
    args = parse_args()
    persist_di = 'vector_db'

    print('==============================')
    print('The Finance RAG Chat Agent is Initializing...')
    print('==============================\n')

    if args.reset and os.path.exists(persist_di):
        logger.info(f"reset flag has been detected. Deleting existing '{persist_di}'directory")
        shutil.rmtree(persist_di)

    if not os.path.exists(persist_di):
        logger.info("No vector store is found. Initializing ingestion pipeline")
        chunk = load_financial_docs()
        if not chunk:
            logger.error("Error: There are no PDF files in 'data/'. Please add files to process and restart the agent.")
            sys.exit(1)
        vector_db = vector_store_creator(chunk, persist_directory=persist_di)
    else:
        logger.info("Loading the existing vector store from disk.")
        vector_db = vector_store_loader(persist_directory=persist_di)

    logger.info("Booting up financial reasoning engine")
    try:
        qa_chain = get_qa_chain(vector_db)
    except Exception as e:
        logger.error(f"Failed to initialize AI engine: {e}")
        sys.exit(1)

    print("\nThe system is ready! Type 'exit' or 'quit' to stop the agent.")

    while True:
        try:
            query = input("\nUsername: ").strip()

            if query.lower() in ["exit", "quit"]:
                print("Shutting down. Thank you for using the agent.")
                break

            if not query:
                continue

            print("Analyzing the documents")
        
            response = qa_chain.invoke({"input": query})

            answer = response.get('answer', 'I cannot generate an answer.')

            retrieved_docs = response.get('context', [])
            sources = set([doc.metadata.get('source', 'Unknown file') for doc in retrieved_docs])

            print(f"\n AI Analyst: {answer}")
            if sources:
                print(f"[Sources: {', '.join(sources)}]")

        except KeyboardInterrupt:
            print('\n\n Process is interrupted by the user. Shutting down.')
            break
        except Exception as e:
            logger.error(f"\n An error has occured during generation of answer: {e}")

if __name__ == '__main__':
    main()