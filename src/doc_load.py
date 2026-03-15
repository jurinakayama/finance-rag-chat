import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_financial_docs(data_path = 'data/'):
    '''
    Uploads all PDFs from the data folder and splits the document.
    '''
    print(f'Uploading documents from {data_path}...')


    loader = DirectoryLoader(data_path, glob='./*.pdf', loader_cls = PyPDFLoader)
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150,
        separators = ['\n\n', '\n', ' ', '']
    )

    text_chunks = text_splitter.split_documents(raw_documents)

    print(f'Document has been split into {len(text_chunks)} chunks.')
    return text_chunks

if __name__ == '__main__':
    docs = load_financial_docs()
    if docs:
        print(f'A sample content from the first chunk is \n{docs[0].page_content[:200]}...')
