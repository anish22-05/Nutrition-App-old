import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
def delete_doc_chunks(doc_id: str):
    return supabase.table("chunks").delete().eq("doc_id",doc_id).execute()
def insert_chunks(rows: list[dict], batch_size: int=50):
    if not rows:
        return None
    last = None
    for i in range(0,len(rows), batch_size):
        batch = rows[i:i + batch_size]
        last = supabase.table("chunks").insert(batch).execute()
    # return supabase.table("chunks").insert(rows).execute()
    return last
def search_chunks(query_embedding: list[float], top_k: int = 5, filter_obj:dict | None = None):
    payload = {
        "query_embedding": query_embedding,
        "match_count": top_k,
        "filter": filter_obj or {},
    }
    return supabase.rpc("match_documents", payload).execute()
