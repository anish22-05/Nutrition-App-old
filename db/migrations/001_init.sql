-- Run this once in the Supabase SQL editor. It is idempotent and NON-destructive:
-- existing chunks are kept and migrated to the new versioned schema.

create extension if not exists vector;

-- ---------------------------------------------------------------------------
-- documents: one row per logical document. `active_batch` points at the set of
-- chunks that is currently live. Re-uploading a doc writes a NEW batch first and
-- only then flips this pointer, so a failed upload can never destroy good data.
-- ---------------------------------------------------------------------------
create table if not exists public.documents (
  doc_id        text primary key,
  active_batch  uuid,
  filename      text,
  num_chunks    int default 0,
  updated_at    timestamptz not null default now()
);

create table if not exists public.chunks (
  id           bigserial primary key,
  doc_id       text not null,
  batch_id     uuid not null,
  chunk_index  int  not null,
  content      text not null,
  metadata     jsonb not null default '{}'::jsonb,
  embedding    vector(1536) not null   -- text-embedding-3-small
);

-- Migrate a pre-existing `chunks` table (created by the old supabase_code.txt).
alter table public.chunks add column if not exists batch_id uuid;

insert into public.documents (doc_id, active_batch)
select doc_id, gen_random_uuid()
from public.chunks
group by doc_id
on conflict (doc_id) do nothing;

update public.chunks c
set batch_id = d.active_batch
from public.documents d
where c.doc_id = d.doc_id and c.batch_id is null;

alter table public.chunks alter column batch_id set not null;

create index if not exists idx_chunks_doc_batch on public.chunks (doc_id, batch_id);

-- HNSW instead of IVFFlat: IVFFlat clusters are computed from the rows present at
-- build time (useless on an empty table) while HNSW needs no training step and
-- gives better recall/latency.
drop index if exists public.idx_chunks_embedding;
create index if not exists idx_chunks_embedding_hnsw
  on public.chunks using hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- Similarity search. Only chunks from each document's ACTIVE batch are visible.
-- `filter_doc_id` filters on the real doc_id COLUMN (the old version filtered
-- on metadata, where doc_id doesn't exist, so it always returned nothing).
-- ---------------------------------------------------------------------------
drop function if exists public.match_documents(vector, integer, jsonb);
drop function if exists public.match_documents(vector, integer, text, double precision);

create or replace function public.match_documents(
  query_embedding vector(1536),
  match_count     int              default 5,
  filter_doc_id   text             default null,
  min_similarity  double precision default 0
)
returns table (
  id          bigint,
  doc_id      text,
  chunk_index int,
  content     text,
  metadata    jsonb,
  similarity  double precision
)
language sql
stable
set search_path = public, extensions   -- find the pgvector `<=>` operator wherever it is installed
as $$
  select
    c.id,
    c.doc_id,
    c.chunk_index,
    c.content,
    c.metadata,
    1 - (c.embedding <=> query_embedding) as similarity
  from public.chunks c
  join public.documents d
    on d.doc_id = c.doc_id and d.active_batch = c.batch_id
  where (filter_doc_id is null or c.doc_id = filter_doc_id)
    and 1 - (c.embedding <=> query_embedding) >= min_similarity
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

-- ---------------------------------------------------------------------------
-- Lock the tables down. Tables made through SQL are readable/writable by the public
-- `anon` key unless row-level security is on. Our backend uses the service_role key,
-- which bypasses RLS, so enabling RLS with NO policies = "only the backend may touch this".
-- ---------------------------------------------------------------------------
alter table public.documents enable row level security;
alter table public.chunks    enable row level security;

notify pgrst, 'reload schema';

-- ---------------------------------------------------------------------------
-- Verification: the editor shows the LAST statement's result. Expect one row:
--   documents >= 1, chunks_without_batch = 0, hnsw_index = 1, ivfflat_left = 0,
--   match_documents_overloads = 1, rls_on = 2
-- ---------------------------------------------------------------------------
select
  (select count(*) from public.documents)                                         as documents,
  (select count(*) from public.chunks)                                            as chunks,
  (select count(*) from public.chunks where batch_id is null)                     as chunks_without_batch,
  (select count(*) from pg_indexes where indexname = 'idx_chunks_embedding_hnsw') as hnsw_index,
  (select count(*) from pg_indexes where indexname = 'idx_chunks_embedding')      as ivfflat_left,
  (select count(*) from pg_proc where proname = 'match_documents')                as match_documents_overloads,
  (select count(*) from pg_class
    where relname in ('documents', 'chunks') and relrowsecurity)                  as rls_on;
