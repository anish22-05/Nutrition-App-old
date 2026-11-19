-- Run once in the Supabase SQL editor, AFTER 001_init.sql. Idempotent and non-destructive.
--
-- Adds the core-vs-private split:
--   scope = 'core'  -> curated knowledge base, visible to everyone (managed by `python -m app.admin`)
--   scope = 'user'  -> private upload, visible only to documents.owner_id
-- Everything that already exists (e.g. 'nutrition-v1') becomes 'core' automatically.

alter table public.documents add column if not exists scope     text not null default 'core';
alter table public.documents add column if not exists owner_id  text;
alter table public.documents add column if not exists file_hash text;   -- sha256 of the PDF, to skip duplicate uploads

alter table public.documents drop constraint if exists documents_scope_check;
alter table public.documents add  constraint documents_scope_check check (scope in ('core', 'user'));

alter table public.documents drop constraint if exists documents_owner_check;
alter table public.documents add  constraint documents_owner_check check (scope = 'core' or owner_id is not null);

create index if not exists idx_documents_owner on public.documents (owner_id);
create index if not exists idx_documents_hash  on public.documents (file_hash);

-- ---------------------------------------------------------------------------
-- Search with isolation enforced HERE, not in the UI:
--   filter_scope 'all'  -> core + the caller's own documents
--              'core' -> shared knowledge base only
--              'mine' -> the caller's documents only
--   no filter_owner    -> core only (private data is never returned without an owner)
--
-- Two branches on purpose. Core is large, so it uses the HNSW index. A user's data is small
-- and a selective filter on top of an approximate index can silently return too few rows, so the
-- private branch does an exact scan (`+ 0` stops the planner from using the index for ordering).
-- ---------------------------------------------------------------------------
drop function if exists public.match_documents(vector, integer, text, double precision);
drop function if exists public.match_documents(vector, integer, text, double precision, text, text);

create or replace function public.match_documents(
  query_embedding vector(1536),
  match_count     int              default 5,
  filter_doc_id   text             default null,
  min_similarity  double precision default 0,
  filter_owner    text             default null,
  filter_scope    text             default 'all'
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
set search_path = public, extensions
as $$
  with core_hits as (
    select c.id, c.doc_id, c.chunk_index, c.content, c.metadata,
           1 - (c.embedding <=> query_embedding) as similarity
    from public.chunks c
    join public.documents d on d.doc_id = c.doc_id and d.active_batch = c.batch_id
    where filter_scope in ('all', 'core')
      and d.scope = 'core'
      and (filter_doc_id is null or c.doc_id = filter_doc_id)
    order by c.embedding <=> query_embedding
    limit match_count
  ),
  mine_hits as (
    select c.id, c.doc_id, c.chunk_index, c.content, c.metadata,
           1 - (c.embedding <=> query_embedding) as similarity
    from public.chunks c
    join public.documents d on d.doc_id = c.doc_id and d.active_batch = c.batch_id
    where filter_scope in ('all', 'mine')
      and filter_owner is not null
      and d.scope = 'user'
      and d.owner_id = filter_owner
      and (filter_doc_id is null or c.doc_id = filter_doc_id)
    order by (c.embedding <=> query_embedding) + 0
    limit match_count
  )
  select h.id, h.doc_id, h.chunk_index, h.content, h.metadata, h.similarity
  from (select * from core_hits union all select * from mine_hits) h
  where h.similarity >= min_similarity
  order by h.similarity desc
  limit match_count;
$$;

notify pgrst, 'reload schema';

-- Verification - expect: new_columns = 3, scope_checks = 2, match_documents_overloads = 1,
-- has_owner_param = 1, core_docs >= 1, user_docs = 0 (until someone uploads)
select
  (select count(*) from information_schema.columns
    where table_schema = 'public' and table_name = 'documents'
      and column_name in ('scope', 'owner_id', 'file_hash'))                       as new_columns,
  (select count(*) from pg_constraint
    where conname in ('documents_scope_check', 'documents_owner_check'))           as scope_checks,
  (select count(*) from pg_proc where proname = 'match_documents')                 as match_documents_overloads,
  (select count(*) from pg_proc p
    where p.proname = 'match_documents' and pg_get_function_arguments(p.oid) like '%filter_owner%') as has_owner_param,
  (select count(*) from public.documents where scope = 'core')                     as core_docs,
  (select count(*) from public.documents where scope = 'user')                     as user_docs;
