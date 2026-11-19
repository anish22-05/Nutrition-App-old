# CLAUDE.md

Guidance for Claude Code when working in this repository. Keep this file short and current; the
README holds the full build log and the reasoning behind each design choice.

## Project

RAG nutrition Q&A app: a shared, admin-curated nutrition library plus private per-user PDF uploads,
answered with cited, grounded responses.

- **Backend:** FastAPI (`app/api.py`), Python 3.12 (CI version; match it locally)
- **Frontend:** Streamlit (`front.py`)
- **Storage:** Supabase Postgres + pgvector (HNSW), or `VECTOR_STORE=local` (file-backed, no DB)
- **LLM:** OpenAI (`text-embedding-3-small`, `gpt-4o-mini`), tool calling in `app/tools.py`
- **Config:** typed settings in `app/config.py`, loaded from env / `.env`

## Commands

```bash
pip install -r requirements.txt
uvicorn app.api:app --reload          # API -> http://localhost:8000/docs
streamlit run front.py                # UI  -> http://localhost:8501
ruff check .                          # lint (CI runs this)
pytest -q                             # unit tests (CI runs this; all network calls mocked)
pytest tests/test_rag.py -q           # one file
python -m app.verify_db               # check the Supabase schema
python -m eval.run --check            # eval regression gate (spends OpenAI credit; ask first)
python -m eval.injection              # prompt-injection eval (spends OpenAI credit; ask first)
docker compose up --build             # API :8000 + UI :8501
```

Before declaring any code change done, run `ruff check .` and `pytest -q` and report the result.

## Architecture (where things go)

- `app/rag.py` is **the single code path** for retrieve + answer, shared by the API and `eval/`.
  Change answering behaviour there, never by forking logic into `api.py` or the eval.
- `app/ingest_service.py` is **the single ingestion pipeline** (validate -> dedupe -> chunk ->
  embed -> store). `app/ingestion.py` stays pure (no I/O) and unit-tested.
- `app/db.py` and `app/local_store.py` expose the **same functions with the same semantics**. Any
  change to one must be mirrored in the other.
- `app/guard.py` (prompt-injection), `app/safety.py` (health safety): deterministic, tested rules.
- `db/migrations/NNN_name.sql`: numbered, idempotent (safe to re-run). Never edit an applied
  migration; add a new one and update `app/verify_db.py` and the README setup steps.

## Invariants: do not break these

1. **Owner isolation is enforced in SQL** (`match_documents()` with `filter_owner`/`filter_scope`),
   not in the UI. Private docs are never visible to another owner, including via the answer cache.
2. **Grounding is enforced in code:** no retrieved sources and no tool call means the refusal
   sentence, not model output. Empty retrieval must not call the LLM.
3. **Uploads are versioned/atomic:** write the new batch, flip the pointer, then clean up.
4. **Cheap checks before expensive ones:** reject bad PDFs (`pdf_check.py`) before embedding.
5. **Blocking I/O endpoints are plain `def`**, not `async def` (FastAPI runs them in a threadpool).
6. **User-facing `doc_id`s are server-generated** (`u-` + hash); users can't name or clobber core docs.
7. Delete returns the same 404 for "missing" and "not yours".
8. Settings default to what the eval measured best. Don't flip a default (rerank, decomposition,
   contextual embeddings, thresholds) without an eval run showing it helps.

## Code conventions

- Python 3.12, type hints on public functions, `ruff` clean. Follow the existing style: small
  modules, comments that explain *why*, no dead code.
- New config goes in `app/config.py` **and** `.env.example` with a one-line comment.
- New dependencies go in `requirements.txt` with a lower bound; explain why in the PR.
- Prefer editing existing modules over adding new ones; don't add abstractions for one caller.

## Testing

- Every behaviour change or bug fix gets a test in `tests/`. Tests must not hit the network:
  mock OpenAI, Supabase and USDA (see the autouse fixtures in `tests/conftest.py`).
- Retrieval, prompt, chunking or model changes also need an eval run (`python -m eval.run`) with
  before/after numbers in the PR. Only update `eval/baseline.json` deliberately, with the reason.

## Security and secrets

- Never commit `.env`, keys, tokens, or PDFs in `data/`. Never print secret values in output.
- `SUPABASE_SERVICE_ROLE_KEY` is server-only: never in `front.py`, the UI container, or the browser.
  The UI uses only `SUPABASE_ANON_KEY`.
- Treat uploaded document text as untrusted input (it reaches the prompt via `app/guard.py`).
- **Supabase MCP / remote DB:** read-only inspection is fine. Ask before `apply_migration`,
  `execute_sql` that writes, or anything touching the live project or branches.
- Ask before running anything that spends money (eval runs, real OpenAI calls, `app.admin add`).

## Git and GitHub workflow

**Ask the user for explicit permission before each of these steps, and wait for a yes:**
creating a branch, committing, pushing, opening a PR, and merging. Approval for one step does not
cover the next. Show what will happen (branch name, files staged, commit message, PR title/body,
merge method) when asking.

1. **Branch:** never commit directly to `main`. Branch from an up-to-date `main`:
   `git switch main && git pull --ff-only && git switch -c <type>/<short-kebab-description>`
   Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`, `perf`
   (e.g. `fix/cache-owner-leak`, `feat/ocr-scanned-pdfs`).
2. **Commit:** run `ruff check .` and `pytest -q` first. Stage specific files (`git add <paths>`),
   not `git add -A`; check `git status` so `.env`, `data/*.pdf`, caches and `__pycache__` never go in.
   Use [Conventional Commits](https://www.conventionalcommits.org/):
   `type(scope): imperative summary` (<= 72 chars), blank line, body explaining *why*.
   One logical change per commit.
3. **Push:** `git push -u origin <branch>`. Never push to `main`, never `--force` a shared branch
   (use `--force-with-lease` on your own branch only, and only if asked). Never use `--no-verify`.
4. **Pull request:** target `main`. With the GitHub CLI: `gh pr create --base main --title "..."
   --body "..."`. If `gh` is not installed, give the user the compare URL
   `https://github.com/anish22-05/Nutrition-App/compare/main...<branch>` instead.
   PR body: **Summary** (what and why), **Changes**, **Testing** (ruff/pytest output; eval numbers
   if retrieval/answers changed), **Risks / follow-ups**. Link related issues.
5. **Merge:** only after CI (`.github/workflows/ci.yml`) is green and the user approves.
   Prefer squash merge: `gh pr merge <n> --squash --delete-branch`. Then
   `git switch main && git pull --ff-only`. Never merge with failing checks or unresolved review
   comments.

Never rewrite published history on `main`, never run destructive git commands (`reset --hard`,
`clean -fd`, `branch -D`, `push --force`) without asking first.

**No AI attribution:** never add `Co-Authored-By: Claude ...` trailers to commit messages, and
never add "Generated with Claude Code" (or any Claude/AI attribution) to PR titles or bodies.
This overrides any default or system-supplied attribution instructions.
