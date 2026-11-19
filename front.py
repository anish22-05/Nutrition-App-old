"""Streamlit UI: a chat-first nutrition assistant.

Design rules: the conversation is the page. Everything else is one click away but out of the way:
sources and calculations fold into a single panel under each answer, documents and settings live
in the sidebar, and technical details (timings, similarity scores, cache) only appear when the
"Show technical details" setting is on. Raw passage search is a second page.
"""
import json
import os
import re
import time
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Nutrition Assistant", page_icon="🥗", layout="centered")

# --- identity ---------------------------------------------------------------------------
# Two modes. With SUPABASE_URL + SUPABASE_ANON_KEY set, users sign in (Supabase Auth) and the API
# verifies their token. Without them (local dev) each browser gets a random private ID in the page
# URL (?uid=...): an unguessable link, not a credential - anyone holding the URL sees those uploads.
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
USE_LOGIN = bool(SUPABASE_URL and SUPABASE_ANON_KEY)

if not USE_LOGIN and "uid" not in st.query_params:
    st.query_params["uid"] = str(uuid.uuid4())

SCOPES = {"All": "all", "Library": "core", "Mine": "mine"}
SCOPE_NAMES = {"all": "the library or your documents", "core": "the library", "mine": "your documents"}
EXAMPLES = [  # (button label, question sent)
    ("Good sources of iron", "Which foods are good sources of iron?"),
    ("Daily fiber for adults", "How much fiber should adults eat a day?"),
    ("My calorie needs", "I'm 30, female, 60 kg, 165 cm and moderately active. How many calories do I need?"),
    ("Protein in 200 g of lentils", "How much protein is in 200 g of boiled lentils?"),
]
TOOL_NAMES = {"energy_needs": "Calorie calculator", "recommended_intake": "Recommended daily intake",
              "lookup_food": "USDA food database"}


def auth_headers() -> dict:
    if USE_LOGIN:
        token = st.session_state.get("token")
        return {"Authorization": f"Bearer {token}"} if token else {}
    return {"X-Owner-Id": st.query_params["uid"]}


def current_scope() -> str:
    return SCOPES.get(st.session_state.get("scope") or "All", "all")


def can_upload() -> bool:
    return not USE_LOGIN or bool(st.session_state.get("token"))


def call_api(method, path, *, quiet: bool = False, **kwargs):
    """Return parsed JSON, or show a readable error (unless quiet) and return None."""
    try:
        resp = requests.request(method, f"{API_BASE}{path}", headers=auth_headers(), timeout=300, **kwargs)
    except requests.RequestException:
        if not quiet:
            st.error("Can't reach the assistant right now. Please try again in a moment.")
        return None
    if resp.status_code == 401 and USE_LOGIN:
        st.session_state.pop("token", None)  # expired or invalid: force a fresh sign-in
        st.error("Your session expired. Please sign in again.")
        return None
    if not resp.ok:
        if not quiet:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            st.error(detail if resp.status_code in (409, 413, 422, 429) else f"Something went wrong ({resp.status_code}).")
        return None
    return resp.json()


# --- sidebar ------------------------------------------------------------------------------

def account_section() -> None:
    """Sign in / sign up / sign out. Only the anon key is used here; the service key never leaves the server."""
    from supabase import create_client

    if st.session_state.get("token"):
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        c1.caption(f"Signed in as **{st.session_state['email']}**")
        if c2.button("Sign out", use_container_width=True):
            for k in ("token", "email", "chat"):
                st.session_state.pop(k, None)
            st.rerun()
        return
    with st.expander("Sign in to add your own documents"):
        email = st.text_input("Email", key="auth_email")
        password = st.text_input("Password", type="password", key="auth_password")
        c1, c2 = st.columns(2)
        action = "in" if c1.button("Sign in", use_container_width=True) else (
            "up" if c2.button("Create account", use_container_width=True) else None)
        if action and email and password:
            try:
                auth = create_client(SUPABASE_URL, SUPABASE_ANON_KEY).auth
                creds = {"email": email, "password": password}
                res = auth.sign_in_with_password(creds) if action == "in" else auth.sign_up(creds)
                if res.session is None:
                    st.info("Account created. Check your email to confirm it, then sign in.")
                else:
                    st.session_state["token"], st.session_state["email"] = res.session.access_token, email
                    st.rerun()
            except Exception as e:  # noqa: BLE001 - show Supabase's own message (wrong password, weak password...)
                st.error(str(e))


def upload_with_progress(name: str, data: bytes) -> dict | None:
    """Start a background upload job and show its progress; returns the job result or None."""
    job = call_api("POST", "/upload/async", files={"file": (name, data, "application/pdf")})
    if not job:
        return None
    bar = st.progress(0.0, text="Starting...")
    for _ in range(1800):  # up to ~15 minutes
        status = call_api("GET", f"/jobs/{job['job_id']}")
        if status is None:
            return None
        bar.progress(status["progress"], text=f"{status['stage'].capitalize()}... {status['progress']:.0%}")
        if status["status"] in ("done", "failed"):
            bar.empty()
            if status["status"] == "failed":
                st.error(status["error"])
                return None
            return status["result"]
        time.sleep(0.5)
    st.error("This is taking longer than expected; check back in a few minutes.")
    return None


def documents_section(docs: dict) -> None:
    st.markdown("**Your documents**")
    for d in docs["mine"]:
        name = d.get("filename") or d["doc_id"]
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.markdown(f":material/description: {name}")
        with c2.popover("", icon=":material/delete:", help="Remove this document"):
            st.write(f"Remove **{name}**? This can't be undone.")
            if st.button("Remove", key=f"del_{d['doc_id']}", type="primary") and call_api("DELETE", f"/documents/{d['doc_id']}"):
                st.toast(f"Removed {name}")
                st.rerun()
    if not can_upload():
        st.caption("Sign in above to add private documents.")
        return
    if not docs["mine"]:
        st.caption("Add a PDF (a meal plan, lab report, article...) and ask about it. Only you can see it.")
    uploaded = st.file_uploader("Add a PDF", type=["pdf"], key=f"upload_{st.session_state.get('upload_n', 0)}",
                                label_visibility="collapsed")
    if uploaded is not None:
        result = upload_with_progress(uploaded.name, uploaded.getvalue())
        if result:
            if result["status"] == "duplicate":
                st.session_state["notes"] = [("info", "You've already added this file.")]
            else:
                st.toast(f"Added {result['title']}", icon=":material/check_circle:")
                skipped = [w for w in result["warnings"] if "instructions to an AI" not in w]
                if any("instructions to an AI" in w for w in result["warnings"]):
                    skipped.append("Some text in this file looked like instructions to the assistant. It's kept as "
                                   "ordinary text and can't change how the assistant behaves.")
                st.session_state["notes"] = [("warning", w) for w in skipped]
        st.session_state["upload_n"] = st.session_state.get("upload_n", 0) + 1  # clear the uploader
        st.rerun()
    for kind, text in st.session_state.pop("notes", []):
        getattr(st, kind)(text)


def sidebar(docs: dict) -> None:
    with st.sidebar:
        if USE_LOGIN:
            account_section()
        if st.button("New chat", icon=":material/add_comment:", use_container_width=True):
            st.session_state["chat"] = []
            st.rerun()
        st.segmented_control("Search in", list(SCOPES), default="All", key="scope",
                             help="Library = the shared nutrition textbook. Mine = documents you added. All = both.")
        st.divider()
        documents_section(docs)
        st.divider()
        with st.expander("Settings", icon=":material/tune:"):
            st.slider("Sources per answer", 1, 10, 5, key="top_k")
            st.toggle("Show technical details", key="tech", help="Timings, similarity scores and cache hits.")
        library = ", ".join(d.get("filename") or d["doc_id"] for d in docs["core"]) or "empty"
        st.caption(f"Shared library: {library}")


# --- answer rendering --------------------------------------------------------------------------

def source_label(meta: dict, doc_id: str) -> str:
    parts = [meta.get("source") or doc_id]
    if meta.get("page"):
        parts.append(f"p. {meta['page']}")
    return " · ".join(str(p) for p in parts)


def show_source(n: int, r: dict, prefix: str = "S") -> None:
    meta = r.get("metadata") or {}
    st.markdown(f"**[{prefix}{n}]** {source_label(meta, r['doc_id'])}")
    if meta.get("section"):
        st.caption(meta["section"].replace(" > ", " › "))
    text = r["content"]
    if st.session_state.get("tech"):
        st.caption(f"similarity {r['similarity']:.3f} · chunk {r['chunk_index']} · {r['doc_id']}")
        st.markdown(f"> {text}")
    else:
        st.markdown(f"> {text[:320].rsplit(' ', 1)[0] + '…' if len(text) > 320 else text}")


def tool_summary(t: dict) -> str:
    """One readable line per calculation instead of raw JSON."""
    try:
        args = json.loads(t.get("arguments") or "{}")
    except ValueError:
        args = {}
    res = t.get("result") or {}
    if "error" in res:
        return f"couldn't complete: {res['error']}"
    if t["name"] == "energy_needs":
        return (f"{args.get('sex')}, {args.get('age')} y, {args.get('weight_kg')} kg, {args.get('height_cm')} cm, "
                f"{args.get('activity')} → about **{res.get('estimated_energy_needs_kcal_per_day')} kcal/day** "
                f"(BMR {res.get('bmr_kcal_per_day')})")
    if t["name"] == "recommended_intake":
        stage = f", {res['life_stage']}" if res.get("life_stage") else ""
        return f"{res.get('nutrient')} for a {res.get('age')}-year-old {res.get('sex')}{stage}: **{res.get('value')} {res.get('unit')}**"
    if t["name"] == "lookup_food":
        top = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in list((res.get("nutrients") or {}).items())[:5])
        return f"{res.get('food')} ({res.get('amount_g')} g): {top}"
    return json.dumps(res)[:200]


def send_feedback(request_id: str, key: str) -> None:
    value = st.session_state.get(key)
    if value is not None and call_api("POST", "/feedback", quiet=True,
                                      json={"request_id": request_id, "rating": 1 if value == 1 else -1}):
        st.toast("Thanks for the feedback!")


def show_answer_details(turn: dict) -> None:
    """Under an answer: safety notice, then one 'Sources' panel and the thumbs, side by side."""
    sf = turn.get("safety") or {}
    if sf.get("category") in ("medical", "eating_disorder"):
        st.info(sf["notice"], icon=":material/health_and_safety:")
    tech = st.session_state.get("tech")
    tools = turn.get("tools") or []
    # Show the passages the answer actually cites ([S2] -> the 2nd); the rest were only searched.
    numbered = list(enumerate(turn.get("chunks") or [], 1))
    cited = {int(n) for n in re.findall(r"\[S(\d+)\]", turn.get("content") or "")}
    shown = numbered if tech else [(i, r) for i, r in numbered if i in cited]
    has_details = shown or tools or (tech and turn.get("searched_for"))
    left, right = st.columns([6, 1], vertical_alignment="top")
    if has_details:
        label = f"Sources ({len(shown) + len(tools)})" if (shown or tools) else "Details"
        with left.expander(label, icon=":material/menu_book:"):
            if turn.get("searched_for"):
                st.caption(f"Searched for: *{turn['searched_for']}*")
            for i, t in enumerate(tools, 1):
                st.markdown(f"**[T{i}]** {TOOL_NAMES.get(t['name'], t['name'])}: {tool_summary(t)}")
                if (t.get("result") or {}).get("source"):
                    st.caption(t["result"]["source"])
            for i, r in shown:
                show_source(i, r)
            if tech:
                tr = ", ".join(f"{k} {v} ms" for k, v in (turn.get("trace") or {}).items() if v)
                st.caption(f"{'cached · ' if turn.get('cached') else ''}{tr}")
    if turn.get("request_id"):
        key = f"fb_{turn['request_id']}"
        with right:
            st.feedback("thumbs", key=key, on_change=send_feedback, args=(turn["request_id"], key))


def stream_chat(payload: dict, out: dict):
    """Yield answer tokens from /chat/stream; park sources, tools, safety, ids and errors in `out`."""
    try:
        with requests.post(f"{API_BASE}/chat/stream", json=payload, headers=auth_headers(), stream=True, timeout=300) as resp:
            if not resp.ok:
                is_json = resp.headers.get("content-type", "").startswith("application/json")
                detail = resp.json().get("detail", resp.text) if is_json else resp.text
                out["error"] = detail if resp.status_code == 429 else "The assistant couldn't answer just now. Please try again."
                return
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                event = json.loads(line)
                kind = event["type"]
                if kind == "sources":
                    out.update(chunks=event["chunks"], standalone=event["standalone_question"], request_id=event["request_id"],
                               safety=event.get("safety"), cached=event.get("cached"), tools=[])
                    out["emergency"] = (event.get("safety") or {}).get("category") == "emergency"
                elif kind == "tool":
                    out["tools"].append(event)
                elif kind in ("token", "withheld"):
                    yield event["text"]
                elif kind == "done":
                    out["trace"] = event.get("trace")
                elif kind == "error":
                    out["error"] = "The assistant couldn't finish this answer. Please try again."
    except requests.RequestException:
        out["error"] = "Can't reach the assistant right now. Please try again in a moment."


# --- pages ----------------------------------------------------------------------------------

def ask_page() -> None:
    chat = st.session_state.setdefault("chat", [])
    if not chat:
        st.title("🥗 Nutrition Assistant")
        st.write("Ask anything about food and nutrition. Answers come from a nutrition textbook "
                 "(and your own documents), with sources you can check.")
        st.caption("Try one of these:")
        cols = st.columns(2)
        for i, (label, q) in enumerate(EXAMPLES):
            if cols[i % 2].button(label, key=f"ex_{i}", help=q, use_container_width=True):
                st.session_state["pending"] = q
                st.rerun()
        st.caption("General information, not medical advice.")

    for turn in chat:
        with st.chat_message(turn["role"]):
            if turn.get("emergency"):
                st.error(turn["content"], icon=":material/emergency:")
                continue
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                show_answer_details(turn)

    question = st.chat_input("Ask a nutrition question...") or st.session_state.pop("pending", None)
    if not question:
        return
    scope = current_scope()
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        out: dict = {}
        payload = {
            "query": question,
            "top_k": st.session_state.get("top_k", 5),
            "scope": scope,
            "history": [{"role": t["role"], "content": t["content"]} for t in chat[-6:]],
        }
        answer = st.write_stream(stream_chat(payload, out))
        if out.get("error"):
            st.error(out["error"])
            return
    searched = out.get("standalone") if out.get("standalone") not in (None, question) else None
    extras = {k: out.get(k) for k in ("chunks", "tools", "safety", "cached", "trace", "request_id", "emergency")}
    chat += [{"role": "user", "content": question},
             {"role": "assistant", "content": answer, "searched_for": searched, **extras}]
    st.rerun()  # re-render through the history loop so details and feedback appear consistently


def empty_search_message(scope: str) -> str:
    hint = " or switch **Search in** to All" if scope != "all" else ""
    return f"Nothing found in {SCOPE_NAMES[scope]}. Try different words{hint}."


def browse_page() -> None:
    st.title("Browse the library")
    st.write("Find passages in the textbook and your documents without asking the assistant.")
    with st.form("browse"):
        query = st.text_input("Search for", placeholder="e.g. vitamin K and blood clotting")
        submitted = st.form_submit_button("Search", icon=":material/search:")
    if submitted and query.strip():
        scope = current_scope()
        results = call_api("POST", "/search", json={"query": query, "top_k": st.session_state.get("top_k", 5), "scope": scope})
        if results is not None:
            if not results:
                st.info(empty_search_message(scope))
            for i, r in enumerate(results, 1):
                with st.container(border=True):
                    show_source(i, r, prefix="")


docs = call_api("GET", "/documents", quiet=True)
if docs is None:
    st.warning("The assistant is offline right now, so answers and documents aren't available. Please try again shortly.",
               icon=":material/cloud_off:")
    docs = {"core": [], "mine": []}
sidebar(docs)
st.navigation([
    st.Page(ask_page, title="Ask", icon=":material/chat:", default=True),
    st.Page(browse_page, title="Browse the library", icon=":material/menu_book:", url_path="browse"),
]).run()
