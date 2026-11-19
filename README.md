# Nutrition Chat App

A RAG-based nutrition question-answering app built with FastAPI, Streamlit, OpenAI embeddings, and Supabase pgvector.

## Overview

A shared, curated **nutrition library** that the maintainer keeps updating, plus **private documents** that each user can upload. Questions are answered from both, with citations, and the app says so when the documents don't contain the answer.

## Features

- Shared nutrition library (admin-managed) + private per-user uploads, searched together or separately.
- Upload checks: rejects cut-off, scanned, password-protected, garbled or oversized PDFs with a reason; skips duplicates.
- Multi-turn chat with follow-up rewriting and streamed, cited answers; opt-in cross-encoder reranking.
- Nutrition tools the model can call: calorie needs (Mifflin-St Jeor), official recommended intakes (DRI), and food nutrients from USDA FoodData Central, cited as [T1], [T2]...
- Prompt-injection defence for uploaded documents (randomised delimiters, instruction redaction, canary), measured by an attack eval.
- Health-safety layer: emergencies get an immediate fixed message; medical and eating-disorder questions get guard-rails and a notice.
- Semantic answer cache, per-request step timings, chat logging with thumbs up/down that feeds the evaluation set.
- Background upload jobs with progress; a local storage mode that runs the whole app without a database.
- Chunk and embed document text; versioned, atomic storage in Supabase.
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
rag_nutrition/
├── app/
│   ├── api.py          # FastAPI routes: /upload /documents /search /chat /health
│   ├── ingest_service.py  # the one ingestion pipeline: validate -> dedupe -> chunk -> embed -> store
│   ├── retrieval.py    # BM25, rank fusion (RRF), cross-encoder rerank
│   ├── ratelimit.py    # per-client sliding-window limiter for the costly endpoints
│   ├── pdf_check.py    # is this PDF worth ingesting? (truncated / scanned / locked / garbled)
│   ├── admin.py        # CLI to manage the shared library: add | list | remove
│   ├── verify_db.py    # python -m app.verify_db : is the Supabase schema right?
│   ├── ingestion.py    # PDF -> clean text -> sentence chunks (pure, unit-tested)
│   ├── llm.py          # embeddings + grounded, cited answer generation
│   ├── rag.py          # retrieve() + answer(): the one code path the API and eval share
│   ├── db.py           # Supabase access, versioned (atomic) uploads, interaction log
│   ├── local_store.py  # VECTOR_STORE=local: same functions backed by a file (no database needed)
│   ├── tools.py        # energy_needs / recommended_intake / lookup_food (USDA) for the model
│   ├── guard.py        # prompt-injection defence for retrieved text
│   ├── safety.py       # emergency / medical / eating-disorder rules
│   ├── cache.py        # semantic answer cache
│   ├── jobs.py         # background upload jobs
│   ├── trace.py        # per-request step timings
│   └── config.py       # typed settings from env / .env
├── db/migrations/
│   ├── 001_init.sql    # schema, versioned chunks, HNSW index, row-level security
│   ├── 002_scopes.sql  # core vs private documents, owner isolation inside match_documents()
│   └── 003_interactions.sql  # answered questions + thumbs up/down
├── data/               # source PDFs (git-ignored, see Data below)
├── eval/               # evaluation harness (see Build log, Phase 1)
│   ├── data/golden.jsonl        # 61 curated questions with gold evidence
│   ├── generate_dataset.py      # LLM drafts questions, code validates them
│   ├── curate.py                # the human review step, as reproducible code
│   ├── metrics.py  judge.py     # Recall/MRR/bootstrap CI  |  faithfulness/correctness judges
│   ├── corpus.py                # in-memory index built from the PDF
│   ├── run.py                   # python -m eval.run  (--contextual / --decompose switches)
│   ├── injection.py             # python -m eval.injection: attacks vs naive and hardened prompts
│   ├── from_feedback.py         # rated real answers -> draft eval cases
│   ├── baseline.json            # numbers to beat / not regress below
│   └── results/latest.md        # last report
├── tests/
├── front.py            # Streamlit UI
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env            # then fill in the 3 keys
```

In the Supabase SQL editor run `db/migrations/001_init.sql`, then `002_scopes.sql`, then `003_interactions.sql` (all safe to re-run and keep existing data). Check with `python -m app.verify_db`.

**No database? Run it locally.** With `VECTOR_STORE=local` everything (shared library, private uploads, chat log) lives in `data/local_store.pkl`, with the same owner isolation, and only `OPENAI_API_KEY` is needed:

```bash
set VECTOR_STORE=local            # PowerShell: $env:VECTOR_STORE="local"   bash: export VECTOR_STORE=local
python -m app.admin add data/Human-Nutrition-2020.pdf --doc-id human-nutrition-2020 --title "Human Nutrition (2020)"
uvicorn app.api:app --port 8000
streamlit run front.py
```

## Running

```bash
uvicorn app.api:app --reload      # backend  -> http://localhost:8000/docs
streamlit run front.py            # frontend (set API_BASE if the backend is elsewhere)
pytest                            # unit tests
python -m app.verify_db           # is the database schema correct?
python -m eval.run                # evaluation report -> eval/results/latest.md
python -m eval.injection          # prompt-injection report -> eval/results/injection.md
python -m eval.from_feedback      # rated answers -> eval/data/feedback_draft.jsonl
```

API endpoints: `/search`, `/chat`, `/chat/stream`, `/upload`, `/upload/async` + `/jobs/{id}`, `/documents`, `/feedback`, `/health` (see `/docs`).

## Usage

**Users:** open the Streamlit app, optionally upload a PDF (it stays private to that browser link), then use *Search* to inspect chunks or *Ask AI* for a cited answer. The scope selector chooses the shared library, your documents, or both.

**Maintainer (shared library):**

```bash
python -m app.admin add data/Human-Nutrition-2020.pdf --doc-id human-nutrition-2020 --title "Human Nutrition (2020)"
python -m app.admin list
python -m app.admin remove human-nutrition-2020
```

Re-running `add` with the same `--doc-id` publishes a new version atomically; the old one stays live until the new one is fully stored.

Built the library in local mode first? `python -m app.admin import-local` copies its core documents, embeddings included, into Supabase: no second embedding bill.

**Keeping the free Supabase project awake:** `.github/workflows/keepalive.yml` runs a one-row REST query on Mondays and Thursdays (a free project pauses after 7 idle days). Add the repository secrets `SUPABASE_URL` and `SUPABASE_ANON_KEY` (optionally `APP_URL` for the deployed API's `/health`). The job fails, and GitHub emails you, if the project is already paused. Test it from the Actions tab with *Run workflow*.

## Data

PDFs live in `data/` and are git-ignored. The shared library source is the open-licence *Human Nutrition: 2020 Edition*: <https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf>. Use the publisher's download, not a browser "Print to PDF", which truncates very large pages into unreadable files.

## Notes

- The app currently works best with nutrition-related PDFs.
- Search quality depends on chunking and retrieval quality.
- `doc_id` can be used to filter results for a specific document.
- There is no login yet: a browser's private link (`?uid=...`) is its identity. See Phase 1.5 for what that does and does not protect.

## License

Add your chosen license here, or keep this project private until you're ready to release it.
## How a question flows through

1. **Upload:** PDF -> cleaned text -> overlapping sentence chunks -> OpenAI embeddings -> Supabase (new batch written first, then switched live, so a failed re-upload never loses the old version).
2. **Ask:** the question is embedded, the nearest chunks above a similarity floor are fetched, and `gpt-4o-mini` answers *only* from them with `[S1]`-style citations. If nothing clears the floor, the app refuses without calling the LLM.

---

# Build log

Short notes on what was built, why, and what the numbers said. Newest last.

## Phase 0 - a RAG app that is actually correct

The first version worked on the happy path but had real bugs, which I fixed before building anything on top:

- Re-uploading a document deleted the old chunks *before* the new ones were safely stored. It is now versioned: insert under a fresh `batch_id`, flip a pointer, then clean up.
- `filter_doc_id` filtered on a metadata key that did not exist, so filtering returned nothing. It now filters on the real column.
- The chunker silently dropped sentences when it trimmed a window and re-emitted the tail as a duplicate chunk.
- Blocking OpenAI/Supabase calls sat inside `async def` endpoints and froze the server; they are plain `def` now so FastAPI uses its threadpool.
- The IVFFlat index (needs training data, useless on an empty table) was replaced by HNSW.

## Phase 1 - Evaluation harness

**Why bother.** A RAG system fails in two separate places: retrieval (the right passage never reaches the model) and generation (the model ignores, distorts or invents beyond the passage). A fluent wrong answer reads exactly like a right one, so eyeballing a few chats tells you almost nothing, and every later change (chunk size, reranker, prompt, model) can quietly make things worse. Without a fixed test set you are guessing; with one, every change becomes an experiment with a before/after number.

**What it measures**

| Stage | Metric | Plain meaning |
|---|---|---|
| Retrieval | Recall@k | Was the passage holding the answer in the top k? |
| Retrieval | MRR | How high up was it? (1.0 = always first) |
| Generation | Faithfulness | Share of the answer's claims the retrieved text actually supports (hallucination check) |
| Generation | Correctness | Does it match a reference answer? (LLM judge: correct / partial / incorrect) |
| Generation | Citation validity | Does it cite sources that exist? |
| Safety | Abstention accuracy / false-refusal rate | Does it say "I don't know" when the document has no answer, without refusing answerable questions? |

**Design decisions worth knowing**

- **Relevance is judged by evidence text, not chunk id.** Each question stores the verbatim sentence that answers it, and a retrieved chunk counts as relevant if it contains that sentence (compared after stripping punctuation and PDF glyph noise). So the test set survives re-chunking, which Phase 2 will do a lot.
- **Same code path as production.** `/chat` and the eval both call `app/rag.py`, so a score describes what users get.
- **Own in-memory index instead of Supabase.** Reproducible, no production data touched, and it can re-chunk freely. Brute-force cosine is exact at this scale.
- **Confidence intervals, not just means.** With ~50 questions, 0.84 vs 0.88 is often noise; every score carries a 95% bootstrap interval.
- **Golden set:** 61 questions: 46 single-chunk, 4 multi-chunk, 11 unanswerable. An LLM drafted the answerable ones; code rejected any whose evidence was not a verbatim quote or whose wording leaked ("according to the slide..."). I then read all of them and dropped 12 (near-duplicate slides, contrived multi-hop pairs) and added 7 by hand that paraphrase instead of echoing the slides. Unanswerable questions are keyword-checked against the corpus so none is secretly answerable.
- **Regression gate:** `python -m eval.run --check` exits 1 if a headline metric drops more than 0.05 below `eval/baseline.json`. Ready to drop into CI.

**Baseline** (`text-embedding-3-small`, `gpt-4o-mini`, top-5, similarity floor 0.35, judge `gpt-4o-mini`):

| | score |
|---|---|
| Recall@1 / @3 / @5 / @10 | 0.74 / 0.93 / 0.98 / 1.00 |
| MRR@10 | 0.87 (95% CI 0.79-0.93) |
| Answer correctness | 0.92 (0.87-0.97) |
| Faithfulness | 0.99 |
| Abstention accuracy / false refusals | 100% / 0% |
| Similarity as answerable-vs-not detector (AUC) | 1.00 |

**What the numbers do and don't say - read before trusting them**

- The corpus is the 77-page slide deck: only **62 chunks**, so top-5 covers 8% of everything and Recall@5 is nearly guaranteed. Recall@1 and MRR are the numbers that can still move. A bigger corpus is the single best upgrade to this benchmark.
- LLM-drafted questions share vocabulary with their source passage, which inflates retrieval scores; the hand-written ones counter that, but there are only 7.
- The multi-chunk set has n=4, so its interval is huge (0.50-1.00). Treat it as a smoke test.
- The judge is the same model family as the answerer and is not perfect: some "partial" verdicts penalise harmless extra detail. Correctness is good for spotting *changes*, less so for absolute claims. Run-to-run noise at temperature 0 was small (headline numbers identical across two runs; one citation check flipped).
- AUC 1.0 comes from easy unanswerable questions (off-topic or clearly absent). Near-miss questions ("what causes rickets?" when the deck only mentions it) would be a harder, more honest test of the 0.35 threshold.

**Bugs the eval found (the point of having one)**

- My first abstention check was a substring match, so a correct partial answer ending in the refusal sentence was scored as a refusal (false-refusal rate read 2%). It is now an exact match; refusal-plus-hallucination also no longer counts as a clean abstention.
- Every slide chunk starts with the same "EU-India Capacity Building Initiative ..." header. That boilerplate dilutes every embedding. Not fixed yet on purpose: it is a Phase 2 experiment with a measurable delta.
- The chunker drops pages under 50 tokens: 77 slides became 62 chunks.

**Data note:** the first `Human-Nutrition-2020.pdf` copy was corrupt (1,208 pages, zero extractable text, zlib stream errors), so the eval uses the slide deck. A good copy has since been downloaded from the publisher (see Data). The harness takes `--pdf`, so a good copy of a larger textbook can be dropped in and the golden set regenerated with `python -m eval.generate_dataset`.

## Phase 1.5 - Upload hardening and a shared + private library

Before building retrieval improvements on top, the upload path needed fixing. Two real problems showed up while testing with actual files.

**1. A bad PDF was "successfully" ingested as nothing.** A browser "Print to PDF" of a 1,200-page web page was cut off mid-file: 48 MB on disk, **0 readable pages**. The old upload accepted anything starting with `%PDF`. Now `app/pdf_check.py` rejects, with a message the user can act on: not a PDF, unreadable/corrupt, truncated (0 pages), password-protected, over the page limit, scanned (under 30% of pages yield text), garbled text, or a document that would produce too many chunks. Pages that merely have no text are skipped and reported as a warning. Every cheap check runs *before* the expensive step (embedding), so a rejected file costs nothing. An identical file (SHA-256) uploaded twice is detected and not re-embedded.

**2. Users could overwrite, and see, each other's data.** `doc_id` was user-chosen, so uploading as `nutrition-v1` replaced the curated knowledge base, and all documents shared one search pool. Now:

| | Core library | Private upload |
|---|---|---|
| Who adds it | the maintainer, `python -m app.admin` (needs the service key) | any user, `POST /upload` |
| Who sees it | everyone | only the owner |
| `doc_id` | chosen by the maintainer | chosen by the server: `u-` + hash(owner, filename); a user can't name, and so can't clobber, a core id |
| Limits | 50 MB, 3000 pages | 25 MB, 400 pages, 1500 chunks, 5 documents |

- **Isolation lives in SQL, not the UI.** `match_documents()` takes `filter_owner` and `filter_scope`; with no owner it returns core only. `python -m app.verify_db` includes a check that a stranger gets nothing back from private data.
- **A pgvector pitfall I designed around.** Filtering an approximate (HNSW) index afterwards can quietly return too few rows when the filter is selective, which is exactly the "my documents only" case in a table dominated by the core library. The private branch of the query therefore does an exact scan (`order by distance + 0` stops the planner using the index); that's cheap because a user's data is capped small. The core branch keeps the index.
- **Delete returns the same 404 for "missing" and "not yours"**, so document ids can't be probed. Users can delete their own documents; only the maintainer CLI removes core ones.
- **Identity is a stopgap.** There's no login: each browser gets a random UUID kept in the page URL (`?uid=...`) and sends it as `X-Owner-Id`. It behaves like an unguessable private link, so anyone holding the URL sees that user's documents, and it can't stop abuse such as scripted uploads. Real authentication (Supabase Auth + signed tokens, plus rate limiting) is planned before this is exposed publicly.

**Calibrated on real files, not guesses:** the corrupt textbook is rejected, the slide deck (58 of 77 pages carry text) and the 1,208-page textbook (1,112 pages with text, 1,797 chunks) are accepted, and neither has any garbled characters. 56 unit tests cover the checker (using real generated PDFs, including a password-protected and a truncated one), the ingestion service with the database mocked, and the API's authorization rules.

## Phase 2 - Retrieval experiments (and what I decided *not* to ship)

The plan was hybrid search plus a reranker. Before touching production I needed an eval that could show a difference, and the slide deck couldn't: it was already at Recall@5 0.98 on 62 chunks.

**A harder, honest test set.** I regenerated the golden set on the 1,208-page textbook (1,900 chunks): 100 single-chunk, 13 multi-chunk and 13 unanswerable questions. Drafts come from an LLM, then I read every one and dropped 22 (duplicates across chapters, contrived multi-chunk pairings, vague "advice" questions; each drop and its reason is in `eval/curate_textbook.py`). The unanswerable set includes near-misses (a drug dose, a branded drink, a diet the book never covers). Seven of my first candidates turned out to *be* in the book, so the script now checks each against the corpus text.

**A flaw in my own metric, found by reading the failures.** Correctness was stuck near 0.92, and 17 of the "not fully correct" answers were being marked down for adding true detail the short reference didn't mention. The judge prompt already said "extra detail is fine"; the small judge model ignored it. Fix: the judge now only checks each *reference fact* (present / contradicted) and code computes the verdict, so extra content can't lower a score. Correctness moved 0.925 -> 0.965 on identical chunks with no change to the app, which is a reminder to read the failures before believing the number.

**Results** (113 answerable questions, retrieval only, `python -m eval.ablate`):

| strategy | R@1 | R@5 | MRR | latency/query |
|---|---|---|---|---|
| vector (what ships) | 0.730 | 0.965 | 0.865 | <1 ms |
| BM25 only | 0.721 | 0.916 | 0.840 | 2 ms |
| hybrid (vector + BM25, RRF) | 0.730 | 0.956 | 0.872 | 1 ms |
| vector + cross-encoder (MiniLM-L6) | 0.792 | 0.965 | 0.905 | ~900 ms |
| hybrid + cross-encoder (MiniLM-L12) | 0.792 | 0.982 | 0.913 | ~2000 ms |

What I took from it:

- **Hybrid:** +0.007 MRR, well inside the confidence interval (about +/-0.05 at this sample size). Not worth a second index and a new SQL migration. BM25 is strong here partly because the questions are LLM-written and echo the source's wording, so I'd re-test on real user queries before dismissing it. The code is in `app/retrieval.py` and `eval/`, ready if that changes.
- **Reranker:** it does what it should at the top of the list (Recall@1 +6 points, MRR +0.04) but it **did not improve answers**: correctness 0.951 -> 0.920 and faithfulness 1.0 -> 0.989, both within noise but pointing the wrong way. The multi-chunk questions got worse (Recall@5 0.92 -> 0.85) because a cross-encoder promotes the single best passage and pushes out the second one a comparison question needs. It also adds ~1 s per query on CPU. So it ships as an **opt-in setting** (`RERANK_ENABLED=true`), off by default, wired through the same `rag.retrieve()` the eval measures.
- **Lesson:** better ranking at the top of a top-5 list doesn't automatically mean better answers; measure the end-to-end outcome.

**Two ingestion fixes that did pay off.**

1. **Repeated headers and footers are stripped** (`find_boilerplate`): any line that appears on 30%+ of pages, with digits normalised so "Page 12" and "Page 13" match. On the slide deck this removed 24% of the text (the "EU-India Capacity Building Initiative..." banner on every slide). The textbook lost nothing, so no false positives.
2. **A silent data-loss bug.** Chunks under 50 tokens were discarded, and once the banner was gone several short slides fell under that line and vanished from the index; Recall@5 on the deck *dropped* from 0.98 to 0.93 until I traced it. The floor is now 30 tokens (checked at 10/20/30/40 on both corpora). Deck result with both fixes: Recall@5 1.00, MRR 0.866 -> 0.952. Textbook: unchanged.

Baseline is re-saved with the final settings: Recall@5 0.965, MRR 0.865, correctness 0.951, faithfulness 1.00, abstention 100% (13/13), false refusals 0%.

## Phase 3 - Conversation and streaming

- **Follow-ups work.** "And how much do children need?" is useless as a search query. `/chat` now takes the last few turns and rewrites the question into a standalone one (`llm.condense_question`) *before* retrieval; the UI shows "Searched for: ..." so the rewrite is visible. If the rewrite call fails, chat falls back to the raw question rather than breaking.
- **Streaming.** `POST /chat/stream` sends newline-delimited JSON: sources first (so they can be shown while the answer is still being written), then tokens, then `done`. A mid-stream failure becomes an `error` event rather than a broken connection, and an empty retrieval refuses without calling the LLM, exactly as in `/chat`.
- The Streamlit tab is now a real conversation (history, sources per answer, "New conversation"), with a render smoke test using Streamlit's `AppTest`.

## Phase 4 - Abuse protection, observability and real auth

- **Rate limits** on the endpoints that cost money (`/upload` 5/min, `/chat` 20/min by default), keyed on both client IP and owner id, because the owner id is a UUID anyone can mint. It's an in-process sliding window: fine for one container, and the README note applies that it should move to Redis/the gateway when scaled out. Returns 429 with `Retry-After`.
- **Request log** line per call: method, path, status, latency.
- **Cost per request** is logged (`app/usage.py`): OpenAI calls, tokens and an estimated dollar figure, attributed through a request-scoped counter. For streamed answers it is logged when the stream ends, because the HTTP middleware returns before the tokens exist. Prices are a table in the code, so they need updating when OpenAI changes them.
- **Real authentication** (`app/auth.py`). The URL-id stopgap from Phase 1.5 is now dev-only. With `AUTH_REQUIRED=true` the server ignores `X-Owner-Id` completely; the owner is the user id from a Supabase access token (`Authorization: Bearer`), validated by Supabase Auth (cached 5 minutes to avoid a network call per request). Anonymous visitors can still search the shared library; upload and delete return 401. The Streamlit sidebar has sign in / sign up / sign out and uses only the public anon key. Setup: enable the Email provider in the Supabase dashboard (turn off "Confirm email" for local testing), then set `SUPABASE_ANON_KEY` and `AUTH_REQUIRED=true`.
- Limits I know about: the token cache means a revoked session stays valid for up to 5 minutes, and the rate limiter is per-process.

## Phase 5 - Packaging and CI

- **Docker** (written but not yet built on my machine: Docker Desktop wasn't running, so treat it as untested until `docker compose up --build` has been run): one image runs both processes (`docker compose up --build` -> API :8000, UI :8501). Dependencies are installed before the code is copied so edits don't rebuild the slow layer; the app runs as a non-root user; secrets come from `.env` at runtime and are never baked in (`.dockerignore` also keeps `.env`, the PDFs and the eval cache out of the image).
- **CI** (`.github/workflows/ci.yml`): every push runs `ruff` and the unit tests (all network calls are mocked, so it needs no secrets and costs nothing). The evaluation gate (`python -m eval.run --check`, fails the build if a headline metric drops more than 0.05 below `eval/baseline.json`) is a **manual** job because it spends a few cents of OpenAI credit and needs the textbook, which isn't committed.

## Phase 6 - Tools, safety, injection defence, and two ideas the numbers rejected

**Tools.** Some questions need a number, not a passage: "how many calories do I need?", "how much protein is in 200 g of lentils?". The model can now call three tools (`app/tools.py`): `energy_needs` (Mifflin-St Jeor x activity factor), `recommended_intake` (US DRI values for 17 nutrients, ages 14+, incl. pregnancy/lactation; outside that it says so instead of guessing) and `lookup_food` (USDA FoodData Central, live). Results are cited as `[T1]` and shown in the UI with their inputs. Grounding is enforced **in code**: with no retrieved sources and no tool call, whatever the model wrote is replaced by the refusal sentence. When nothing is retrieved, the model is only called at all if the question looks like a tool question, so off-topic questions still cost nothing. The textbook contains DRI tables, so for "what is the RDA of X" the model usually (correctly) cites the book; tools fill the gaps.

**Prompt-injection defence** (`app/guard.py`). Private uploads are untrusted text pasted into the prompt. Layers: sources sit inside delimiters carrying a random per-request nonce (a document can't close them); invisible/bidi characters and fake tags are stripped; sentences that read as instructions to an AI are *removed* before the model sees them and the source is quoted and labelled; a canary in the system prompt withholds any answer that leaks it; uploads with such text get a warning. Measured, not assumed (`python -m eval.injection`, 6 attacks x 3 questions, a poisoned chunk next to a genuine one):

| | naive prompt | hardened (shipped) |
|---|---|---|
| attacks that worked | 3/18 | **0/18** |

The first hardened version (delimiters + prompt rules only) still obeyed a fake `SYSTEM:` line 3 times out of 3, then 1 out of 3 after the prompt wording was tightened; only removing the instruction sentences in code got it to 0. Wording alone isn't a defence.

**Health safety** (`app/safety.py`, deterministic rules, tested). First-person emergencies (overdose, anaphylaxis, chest pain, suicidal thoughts) get an immediate fixed message pointing to emergency help, with no search and no model call; it's checked before the follow-up rewrite so it costs nothing and can't be delayed. Eating-disorder intent ("how can I make myself throw up") and medical questions (doses, medication, diagnosis) are answered from the sources with extra prompt rules and a visible notice. Informational questions ("what are the health effects of purging?") are answered normally: the rules look for first-person intent.

**Cache, tracing, feedback, jobs.** A semantic cache reuses an answer when a new (rewritten) question is >= 0.97 cosine-similar under the same visibility key (scope, filters, tools, safety category, and the owner whenever private documents could be involved), so one user's private answer is never served to another; uploads/deletes drop that owner's entries. Live: an identical question went 3.2 s -> 0.5 s. Every response carries step timings (`condense`, `embed`, `search`, `generate`...). Answers are logged (`db/migrations/003_interactions.sql`) with thumbs up/down from the UI, and `python -m eval.from_feedback` turns rated answers (thumbs-down first) into draft eval cases, so the test set can grow from real questions instead of LLM-written ones. Uploads can run as background jobs (`/upload/async` + `/jobs/{id}`), and the UI shows a progress bar.

**Local mode.** `VECTOR_STORE=local` serves every database function from a file with the same semantics (versioned replace, scope + owner isolation in search, similarity gate), so the full app runs with only an OpenAI key. Same caveat as the rate limiter: single machine.

**Two retrieval ideas I measured and did not turn on** (113 answerable questions, retrieval only):

| setting | R@1 | R@5 | MRR | R@5 multi-chunk (n=13) |
|---|---|---|---|---|
| baseline (what ships) | 0.730 | 0.965 | 0.865 | 0.923 |
| section-aware embeddings | 0.730 | 0.965 | 0.865 | 0.923 |
| query decomposition | 0.690 | 0.956 | 0.833 | 0.846 |
| both | 0.681 | 0.943 | 0.828 | 0.885 |

- *Section-aware embeddings* (prefix each chunk with "Chapter > Section" from the PDF outline): no change. The textbook's chunks are already self-explanatory at 10 sentences. Off by default (`CONTEXTUAL_EMBEDDINGS=true` to enable); section titles are still stored and shown with every source.
- *Query decomposition* (an LLM splits "compare X and Y" into sub-queries, results merged round-robin): **worse**, including on the multi-chunk questions it was meant to help. The planner split 53 of 113 questions, most of which didn't need it, and splitting pushed the single best passage down. Keeping the original query's results first didn't recover it. Off by default (`QUERY_DECOMPOSITION=true`).

**End-to-end after all of this** (`python -m eval.run --check`, same 126 questions): correctness 0.969 (was 0.951), faithfulness 0.993, abstention 100%, false refusals 0%, retrieval unchanged; the regression gate passes. 172 unit tests.

**Not done, and why:** OCR for scanned PDFs (needs Tesseract installed), a Redis rate limiter/job queue (needs Redis; the in-process versions are fine for one container), a React front end, and deployment (needs accounts). Docker is still unverified: Docker Desktop wasn't running.
