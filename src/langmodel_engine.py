import os 
import logging
from typing import Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def get_qa_chain(vector_db):
    '''
    Creates a RetrievalQA chain that connects the LLM to our vector database.
    '''

    # Initializing the LLM (Using temperature=0 for finance to keep answers consistent)
    llm = ChatOpenAI(model_name='gpt-4-turbo', temperature=0)

    # Defining a custom prompt to guide the AI 
    template = '''
    You are a professional financial analyst. Use the following pieces of context to answer the user's question.
    
    If you can't derive an answer based on the context, replay that you do not know.
    Don't make up an answer. Keep the answer concise and professional.
    Always cite the source or page number if it is available in the context.
    
    Context: {context}
    Question: {question}
    
    Helpful Answer:'''

    qa_chain_prompt = PromptTemplate(
        input_variables = ['context', 'question'],
        template = template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff', retriever= vector_db.as_retriever(search_kwargs={'k':5}), 
        return_source_documents=True, chain_type_kwargs={'prompt':qa_chain_prompt}
    )

    return qa_chain

if __name__ == '__main__':
    from vec_store import vector_store_loader

    db = vector_store_loader()
    chain = get_qa_chain(db)

    query = 'What are the key financial risks from the documents?'
    response = chain.invoke({'query': query})

    print(f'\nAnswer: {response['result']}')








