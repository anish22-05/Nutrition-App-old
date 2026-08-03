import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from ingest import build_chunks_from_pdf, embed_chunks
from db import insert_chunks, delete_doc_chunks, search_chunks
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMBED_MODEL = "text-embedding-3-small"
llm_client = OpenAI(api_key = OPENAI_API_KEY)
CHAT_MODEL = "gpt-4o-mini"

app = FastAPI(title= "Nutrition API")
@app.get("/")
def root():
    return {"status": "ok", "message": "Nutrition API is running"}
class SearchRequest(BaseModel):
    query: str
    top_k: int=5
    doc_id: str | None = None
class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_id: str | None = None
    
@app.get("/health")
def health():
    return {"status": "ok"}
@app.post("/upload")
async def upload(file: UploadFile = File(...), doc_id: str = Form(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code = 400, detail= "Only PDF files are supported.")
    file_bytes = await file.read()
    chunks, metas = build_chunks_from_pdf(file_bytes, file.filename)
    if not chunks:
        return {"status": "empty", "message": "No text found in PDF."}
    vectors = embed_chunks(chunks, EMBED_MODEL, OPENAI_API_KEY)
    delete_doc_chunks(doc_id)
    rows = []
    for idx, (content,emb,meta) in enumerate(zip(chunks, vectors, metas)):
        rows.append({
            "doc_id": doc_id,
            "chunk_index": idx,
            "content": content,
            "metadata": meta,
            "embedding": emb,
        })
    insert_chunks(rows)
    return {"status": "success", "chunks_inserted": len(rows), "doc_id": doc_id}
@app.post("/search")
async def search(req: SearchRequest):
    query_vec = embed_chunks([req.query], EMBED_MODEL, OPENAI_API_KEY)[0]
    result = search_chunks(query_vec, top_k = req.top_k, filter_obj={"doc_id": req.doc_id} if req.doc_id else {})
    return result.data
@app.post("/chat")
async def chat(req: ChatRequest):
    query_vec = embed_chunks([req.query], EMBED_MODEL, OPENAI_API_KEY)[0]

    result = search_chunks(
        query_vec,
        top_k=min(req.top_k, 3),
        filter_obj={"doc_id": req.doc_id} if req.doc_id else {}
    )

    matches = result.data or []

    if not matches:
        return {
            "answer": "I couldn't find relevant information in the uploaded document.",
            "chunks": []
        }

    matches = [m for m in matches if m.get("similarity", 0) >= 0.35]
    if not matches:
        return {
            "answer": "I found only weak matches, so I can't answer confidently from the document.",
            "chunks": []
        }

    context = "\n\n---\n\n".join(
        f"Chunk {m['chunk_index']}:\n{m['content']}"
        for m in matches
    )

    system_prompt = (
        "You are a helpful nutrition assistant. "
        "Answer the question using only the provided context. "
        "If the answer is not in context, then you may use your wisdom and try what you can prduce from context. "
        "Be concise and factual."
    )

    user_prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question: {req.query}

Answer:
"""

    completion = llm_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = completion.choices[0].message.content
    return {
        "answer": answer,
        "chunks": matches,
    }
