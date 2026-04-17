# Finance RAG Chat Agent

## Overview

This Finance RAG (Retrieval-Agumented Generation) Chat Agent is created as a terminal-based AI supporting assistant that instantly analyzes complex financial documents. It securely processes local PDFs to build a searchable vector database and uses a large language model to answer queries by the user. The agent generates answers with precise, evidence-based, and professional information.

## Business Problem Statement

This agent deals with the struggle within the financial field regarding the speed and accuracy of document analysis:
* **Time Consuming Process of Dcoument Analysis:** Financial analysts use a great amount of their time manually parsing dense, extensive reports to find specific metrics or risk factors.  
* **AI Assumptions Error:** Standard LLMs often invent financial figures even when they lack context, which makes them dangerous for making essential financial decisions. 
To address these problems, we constructed an agent that anchors all of the responses strictly to the attached document context by citing its sources. This reduces the research time, assumptions, and generalizations to maintain factual integrity. 

## Dataset
**Local Document Storing ('data/')**
This chat agent loads and analyzes the PDF files provided by the user. This agent is best for testing the following types of records:
* 10–K or 10–Q Reports
* Federal Reserve Policy Releases
* Corporate Business Call Transcripts

## Methods Used
1. **Analyzing the Document:** Goes through dense PDF structures and takes raw text from the document securely without exposing essential documents to public internet scraping
2. **Embedding of Vector:** Uses (`text-embedding-3-small`) to change text chunks into vector representations for semantic search
3. **LCEL:** Uses modern LCEL (LangChain Expression Language) to take the top 5 most relevant document chunks based on the user's inquiry, feeding them into LLM for response generation.

### Libraries Used
* `langchain` (Core, Community, OpenAI, and Chroma partner packages)
* `chromadb`
* `pypdf`
* `python-dotenv`

## Output Preview

1. Initial Setup & Uploading the Document

```text
==============================
The Finance RAG Chat Agent is Initializing...
==============================

reset flag has been detected. Deleting existing 'vector_db' directory
No vector is found. Initializing ingestion pipeline
Scanning 'data/' for PDF documents
Successfully loaded 1 documents. Splitting text.
Documents successfully split into 482 individual chunks.
Creating embeddings for 482 chunks.
Vector store is successfully created and saved to 'vector_db/'.
Booting up financial reasoning engine
Initializing QA Chain with model: gpt-4-turbo.
RAG Chain has been successfully created.

The system is ready! Type 'exit' or 'quit' to stop the agent.

Username: What are the primary factors on manufacturing and production?
Analyzing the documents

AI Analyst: According to the 10-K, Tesla faces significant manufacturing risks including the complexity of scaling production at Gigafactories in Texas and Berlin. Key challenges involve potential delays in the ramp-up of new vehicle architectures like the Cypertruck, reliance on single-source suppliers for battery cells, and volatility in the costs of raw materials such as lithium and cobalt.
[Sources: Tesla 10-k.pdf]

Username: exit
Shutting down. Thank you for using the agent. 
```

2. Standard Start (When Using an Existing Database)

```text
==============================
The Finance RAG Chat Agent is Initializing...
==============================

Loading the existing vector store from disk.
Loading existing vector store from 'vector_db'
Vector store has loaded successfully.
Booting up financial reasoning engine
Initializing QA Chain with model: gpt-4-turbo.
RAG Chain has been successfully created.

The system is ready! Type 'exit' or 'quit' to stop the agent.

Username: Who signed the report and when was the annual report signed? 
Analyzing the documents

AI Analyst: The report was signed by Elon Musk (Cheif Executive Officer), Vaibhav Taneja (Cheif Financial Officer), and Robyn Denholm (Director), and others. The annual report was signed on January 28, 2026. 
[Sources: Tesla 10-k.pdf]

Username: quit
Shutting down. Thank you for using the agent.
```

## Project Structure

finance-rag-chat
│
├─ data 
│   └─ online_retail_2010_2011.csv
│
├─ app
│   └─ dashboard.py
│
├─ models 
│   └─ demand_curve_85123A.png
│
├─ src  
│   ├─ doc_load.py
│   ├─ langmodel_engine.py
│   └─ vec_store.py
│
├─ app.py
├─ README.md
└─ requirements.txt 