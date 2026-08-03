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