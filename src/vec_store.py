import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def vector_store_creator(text_chunks, persist_directory='vector_db'):
    '''
    Takes text chunks from the document, makes embeddings, and saves them to a local Chroma DB.
    '''
    print(f'Creating embeddings for {len(text_chunks)} chunks.')

    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')

    vector_db = Chroma.from_documents(
        documents=text_chunks,
        embedding = embeddings, 
        persist_directory=persist_directory
    )
    print(f'The vector is saved to {persist_directory}/')
    return vector_db


def vector_store_loader(persist_directory='vector_db'):
    '''
    This will load an existing vector store from disk.
    '''
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vector_db

if __name__ == '__main__':
    from doc_load import load_financial_docs
    sample_chunks = load_financial_docs()
    if sample_chunks:
        db = vector_store_creator(sample_chunks)
        print('Test complete: Vector store is ready.')