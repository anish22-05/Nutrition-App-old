# Nutrition Chat App

A RAG-based nutrition question-answering app built with FastAPI, Streamlit, OpenAI embeddings, and Supabase pgvector.

## Overview

This application lets users upload nutrition PDFs, chunk and embed the text, store vectors in Supabase, and ask questions over the uploaded document. The app returns relevant chunks and generates a grounded answer using an LLM.

## Features

- Upload PDF nutrition documents.
- Chunk and embed document text.
- Store embeddings in Supabase.
- Search for relevant chunks by semantic similarity.
- Generate answers using retrieved context.
- Streamlit frontend for easy interaction.

## Tech Stack

- Python
- FastAPI
- Streamlit
- Supabase
- OpenAI API
- pgvector
- PyMuPDF
- tiktoken

## Project Structure

```text
nutrition-app/
├── api.py
├── db.py
├── front.py
├── ingest.py
├── .env
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/nutrition-app.git
cd nutrition-app
```

### 2. Create a virtual environment

```bash
python -m venv .myenv
source .myenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

## Running the App

### 1. Start the FastAPI backend

```bash
uvicorn api:app --reload
```

### 2. Start the Streamlit frontend

```bash
streamlit run front.py
```

## Usage

1. Open the Streamlit app in your browser.
2. Upload a PDF document.
3. Wait for the upload to finish.
4. Go to the Search tab to inspect relevant chunks.
5. Go to the Ask AI tab to get a final answer grounded in the document.

## Notes

- The app currently works best with nutrition-related PDFs.
- Search quality depends on chunking and retrieval quality.
- `doc_id` can be used to filter results for a specific document.

## License

Add your chosen license here, or keep this project private until you're ready to release it.
## Explaination:
App overview

This is a RAG-based nutrition assistant. It lets a user upload a nutrition PDF, splits the document into chunks, creates embeddings, stores them in Supabase, retrieves the most relevant chunks for a query, and uses an LLM to generate the final answer.
ingest.py

This file handles document processing. It cleans the PDF text, splits it into sentence-based chunks, and creates OpenAI embeddings for each chunk so the content can be searched semantically later.
db.py

This file handles database operations with Supabase. It deletes old chunks for a document, inserts new chunk rows in batches, and runs the match_documents RPC function to search for the most relevant chunks using vector similarity.
api.py

This is the backend brain of the app. It receives uploaded PDFs, calls the ingest functions, stores chunks in Supabase, handles semantic search requests, and also sends retrieved chunks plus the user query to the LLM for a final grounded answer.
front.py

This is the Streamlit frontend. It gives the user three tabs: upload a PDF, search for relevant chunks, and ask the AI for a final answer. It calls the FastAPI endpoints and displays both the retrieved chunks and the generated response.
Short flow explanation

    User uploads a PDF in Streamlit.

    api.py sends the file to ingest.py.

    ingest.py chunks and embeds the text.

    db.py stores the chunks in Supabase.

    User asks a question.

    db.py retrieves similar chunks.

    api.py sends those chunks to the LLM.

    Streamlit shows the final answer and the supporting chunks