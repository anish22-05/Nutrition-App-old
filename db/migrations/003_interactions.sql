-- Run once in the Supabase SQL editor, AFTER 002_scopes.sql. Idempotent and non-destructive.
--
-- One row per answered chat request, so real user questions (and thumbs up/down on the
-- answers) can be reviewed and turned into evaluation cases (`python -m eval.from_feedback`).
-- Only the backend (service key, which bypasses RLS) can read or write it.

create table if not exists public.interactions (
  request_id   text primary key,
  created_at   timestamptz not null default now(),
  owner_id     text,                       -- null for anonymous visitors
  question     text not null,              -- what the user typed
  standalone   text,                       -- the rewritten, self-contained question used for retrieval
  answer       text not null,
  sources      jsonb not null default '[]', -- [{doc_id, chunk_index, page, similarity}]
  tools        jsonb not null default '[]', -- [{name, arguments}]
  safety       text,                       -- null | 'medical' | 'eating_disorder' | 'emergency'
  cached       boolean not null default false,
  latency_ms   integer,
  cost_usd     double precision,
  rating       smallint check (rating in (-1, 1)),  -- thumbs down / up
  comment      text,
  rated_at     timestamptz
);

create index if not exists idx_interactions_created on public.interactions (created_at desc);
create index if not exists idx_interactions_rating  on public.interactions (rating) where rating is not null;

alter table public.interactions enable row level security;  -- no policies: backend only

notify pgrst, 'reload schema';

-- Verification - expect: interactions_table = 1, rls_on = 1
select
  (select count(*) from information_schema.tables
    where table_schema = 'public' and table_name = 'interactions')              as interactions_table,
  (select count(*) from pg_class where relname = 'interactions' and relrowsecurity) as rls_on;
