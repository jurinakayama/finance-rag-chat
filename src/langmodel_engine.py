import os 
import logging
from typing import Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

def get_qa_chain(
        vector_store: Any,
        model_name: str = 'gpt-4-turbo',
        temperature: float = 0.0,
        k_retrievals: int = 5
) -> Any:
    '''
    Creates a RetrievalQA (RAG) chain using LCEL.

    Will return: Any: invoked runnable chain to process queries
    '''
    logger.info(f'Initializing QA Chain with model: {model_name}.')

    try:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not found in the environment variables.")
        
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)

        system_prompt = (
            "You are a professional financial analyst. Use the following pieces of context to answer the user's question."
            "If you cannot derive an answer based on the context, reply that you do not know."
            "Do not make up an answer. Keep the answer concise and professional."
            "Always cite the source or page number if it is available in the context. \n\n"
            "Context: {context}"
        )

        prompt= ChatPromptTemplate.from_message([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector_store.as_retriever(search_kwargs={'k': k_retrievals})
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        logger.info("RAG Chain has been successfully created.")
        return rag_chain
    
    except Exception as e:
        logger.error(f"Failed to initialize QA chain: {e}")
        raise

if __name__ == '__main__':
    from vec_store import vector_store_loader

    logger.info("Testing the Language Model Engine...")
    try:
        db = vector_store_loader()
        chain = get_qa_chain(db)

        user_query = "What are the key financial risks from the documents?"
        logger.info(f"Invoking chain with query: '{user_query}'")
        response = chain.invoke({'input': user_query})

        print('RESPONSE')
        print(response.get('answer', 'No answer has been generated.'))

    except Exception as e:
        logger.error('Engine test')

        