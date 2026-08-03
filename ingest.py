# # pip3 install -q pymupdf tiktoken supabase openai tqdm python-dotenv
# import os, uuid, re
# import fitz # Pymupdf
# import tiktoken
# from supabase import create_client, Client
# from openai import OpenAI
# from tqdm import tqdm
# from dotenv import load_dotenv, find_dotenv

# # #----Load environment
# load_dotenv(find_dotenv(usecwd=True))
# SUPABASE_URL = os.environ['SUPABASE_URL']
# SUPABASE_SERVICE_ROLE_KEY = os.environ['SUPABASE_SERVICE_ROLE_KEY']
# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# #-------Config
# PDF_PATH = 'Human-Nutrition-2020.pdf'
# DOC_ID = 'nutrition-v1'
# EMBED_MODEL = 'text-embedding-3-small'
# BATCH_EMBED = 100
# BATCH_INSERT = 200
# #---Sentence chunking params.
# SENTS_PER_CHUNK = 10
# SENT_OVERLAP = 2
# MAX_TOKENS = 1300
# MIN_TOKENS = 50

# enc = tiktoken.get_encoding("cl100k_base") # matches openAI embeddings tokens

# def clean_text(t: str) -> str:
#     # normalize whitespace and fix hyphenation across line breaks
#     t = t.replace("\r"," ")
#     t = re.sub(r"-\s*\n\s*","",t)#join "nutri-\n tion" => 'nutrition'
#     t = re.sub(r"\s+\n","\n",t)
#     t = t.replace("\n"," ").strip()
#     return t
# def split_sentences(text: str):
#     # simple sentence splitter (good for prose)
#     sents = re.split(r'(?<=[.!?])\s+',text.strip())
#     return [s.strip() for s in sents if s.strip()]
# def chunk_page_by_sentences(text:str, sents_per_chunk: int = SENTS_PER_CHUNK,overlap: int=SENT_OVERLAP,max_tokens: int=MAX_TOKENS,min_tokens: int = MIN_TOKENS):
#     sents = split_sentences(text)
#     i = 0
#     step = max(1,sents_per_chunk - overlap)
#     while i < len(sents):
#         piece = sents[i:i + sents_per_chunk]
#         if not piece:
#             break
#         chunk = " ".join(piece)
#         # enforce token ceiling
#         ids = enc.encode(chunk)
#         while max_tokens and len(ids) > max_tokens and len(piece) > 1:
#             piece = piece[:-1]
#             chunk = " ".join(piece)
#             ids = enc.encode(chunk)
#         if len(ids) >= min_tokens:
#             yield chunk
#         i += step

# def pdf_pages(path: str):
#     """Yield (page_number_1based,cleaned_text)."""
#     doc = fitz.open(path)
#     try:
#         for i in range(len(doc)):
#             txt = doc[i].get_text("text") or ""
#             yield (i+1,clean_text(txt))
#     finally:
#         doc.close()
# def main():
#     sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
#     client = OpenAI(api_key = OPENAI_API_KEY)
#     # Optional: keep the table clean for this document.
#     sb.table("chunks").delete().eq("doc_id",DOC_ID).execute()
#     print("Reading PDF by pages...")
#     pages = list(pdf_pages(PDF_PATH))
#     #Build chunks with page metadata
#     inputs, metas = [],[]
#     print("Chunking (10 sentences per chunk, 2 overlap)...")
#     for page,text in pages:
#         if not text:
#             continue
#         for chunk in chunk_page_by_sentences(text):
#             inputs.append(chunk)
#             metas.append({'page': page,'source':PDF_PATH})
#     print(f"Built  {len(inputs)} chunks from {PDF_PATH}")
#     # Generate embedding
#     vectors = []
#     print('Generating embedding...')
#     for i in tqdm(range(0,len(inputs), BATCH_EMBED), desc='Embedding'):
#         batch = inputs[i:i+BATCH_EMBED]
#         resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
#         vectors.extend([d.embedding for d in resp.data])
#     # Prepare rows
#     rows = []
#     for idx, (content, emb, meta) in enumerate(zip(inputs,vectors,metas)):
#         rows.append({
#             "doc_id": DOC_ID,
#             "chunk_index": idx,
#             "content": content,
#             "metadata": meta,
#             "embedding": emb
#         })
#     print('Uploading to supabase...')
#     for j in tqdm(range(0,len(rows), BATCH_INSERT),desc = "Uploading"):
#         sb.table("chunks").insert(rows[j:j+BATCH_INSERT]).execute()
#     print(f"Done! Inserted {len(rows)} chunks for doc_id={DOC_ID}")
# if __name__ == "__main__":
#     main()

#-----------------------Production grade file for file preprocessing and chunking------------------
import re
import uuid
import fitz
import tiktoken
from openai import OpenAI

enc = tiktoken.get_encoding("cl100k_base")

def clean_text(t: str) -> str:
    t = t.replace("\r", " ")
    t = re.sub(r"-\s*\n\s*", "", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = t.replace("\n", " ").strip()
    return t

def split_sentences(text: str):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]

def chunk_page_by_sentences(text: str, sents_per_chunk=10, overlap=2, max_tokens=1300, min_tokens=50):
    sents = split_sentences(text)
    i = 0
    step = max(1, sents_per_chunk - overlap)
    while i < len(sents):
        piece = sents[i:i + sents_per_chunk]
        if not piece:
            break
        chunk = " ".join(piece)
        ids = enc.encode(chunk)
        while max_tokens and len(ids) > max_tokens and len(piece) > 1:
            piece = piece[:-1]
            chunk = " ".join(piece)
            ids = enc.encode(chunk)
        if len(ids) >= min_tokens:
            yield chunk
        i += step

def pdf_pages_from_bytes(file_bytes: bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        for i in range(len(doc)):
            txt = doc[i].get_text("text") or ""
            yield (i + 1, clean_text(txt))
    finally:
        doc.close()

def build_chunks_from_pdf(file_bytes: bytes, filename: str):
    pages = list(pdf_pages_from_bytes(file_bytes))
    inputs, metas = [], []

    for page_num, text in pages:
        if not text:
            continue
        for chunk_index, chunk in enumerate(chunk_page_by_sentences(text)):
            inputs.append(chunk)
            metas.append({
                "page": page_num,
                "source": filename,
                "chunk_index": chunk_index,
            })

    return inputs, metas

def embed_chunks(chunks: list[str], embed_model: str, openai_api_key: str):
    client = OpenAI(api_key=openai_api_key)
    vectors = []
    for i in range(0, len(chunks), 100):
        batch = chunks[i:i + 100]
        resp = client.embeddings.create(model=embed_model, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return vectors

            









