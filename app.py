import os
import sys
from dotenv import load_dotenv

from src.doc_load import load_financial_docs
from src.vec_store import vector_store_creator, vector_store_loader
from src.langmodel_engine import get_qa_chain

load_dotenv()

def main():
    print('Hello, this is the Finance RAG Chat Agent.')

    persist_di = 'vector_db'

    if not os.path.exists(persist_di):
        print("No vector store is found. Initializing from the 'data' folder.")
        chunk = load_financial_docs()
        if not chunk:
            print("Error: There are no PDF files in 'data/'. Please add files to process and restart the agent.")
            return
        vector_db = vector_store_creator(chunk, persist_directory=persist_di)
    else:
        print("Loading the existing vector store.")
        vector_db = vector_store_loader(persist_directory=persist_di)

    print("Setting up the financial reasoning engine")
    qa_chain = get_qa_chain(vector_db)

    print("\nThe system is ready! Type 'exit' or 'quit' to stop the agent.")

    while True:
        query = input("\nUsername: ")

        if query.lower() in ["exit", "quit"]:
            break

        if not query.strip():
            continue

        print("Analyzing the documents")
        try:
            response = qa_chain.invoke({"query": query})

            print(f"\nAI Analyst: {response['result']}")
            sources = set([doc.metadata.get('source', 'Unknown') for doc in response['source_documents']])
            print(f"\n[Sources: {', '.join(sources)}]")

        except Exception as e:
            print(f"An error has occured: {e}")

if __name__ == "__main__":
    main()
