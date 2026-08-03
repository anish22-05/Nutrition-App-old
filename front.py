import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Nutrition Chat App", layout="wide")
st.title("🥗 Nutrition Chat App")

tab1, tab2, tab3 = st.tabs(["Upload PDF", "Search", "Ask AI"])

with tab1:
    st.subheader("Upload a PDF to Supabase")
    doc_id = st.text_input("Document ID", value="nutrition-v1", key="upload_doc_id")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"], key="upload_pdf")

    if st.button("Upload", key="upload_button") and uploaded is not None:
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
        data = {"doc_id": doc_id}
        resp = requests.post(f"{API_BASE}/upload", files=files, data=data, timeout=300)
        if resp.ok:
            st.success(resp.json())
        else:
            st.error(resp.text)

with tab2:
    st.subheader("Semantic Search")
    query = st.text_input("Ask a question", key="search_query")
    filter_doc_id = st.text_input("Filter by doc_id (optional)", value="", key="search_doc_id")
    top_k = st.slider("Top K", 1, 20, 5, key="search_top_k")

    if st.button("Search", key="search_button"):
        payload = {
            "query": query,
            "top_k": top_k,
            "doc_id": filter_doc_id or None,
        }
        resp = requests.post(f"{API_BASE}/search", json=payload, timeout=120)
        if resp.ok:
            results = resp.json()
            if not results:
                st.info("No relevant chunks found.")
            else:
                st.subheader("Most relevant chunks")
                for r in results:
                    st.markdown(f"**Similarity:** {r['similarity']:.4f}")
                    st.write(r["content"])
                    st.caption(f"doc_id={r['doc_id']} | chunk={r['chunk_index']}")
                    st.divider()
        else:
            st.error(resp.text)

with tab3:
    st.subheader("Ask the assistant")
    query = st.text_input("Ask a question", key="chat_query")
    filter_doc_id = st.text_input("Filter by doc_id (optional)", value="", key="chat_doc_id")
    top_k = st.slider("Top K", 1, 20, 5, key="chat_top_k")

    if st.button("Ask", key="ask_button"):
        payload = {
            "query": query,
            "top_k": top_k,
            "doc_id": filter_doc_id or None,
        }
        resp = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)

        if resp.ok:
            data = resp.json()

            st.subheader("Answer")
            st.write(data["answer"])

            st.subheader("Relevant chunks")
            for r in data["chunks"]:
                st.markdown(f"**Similarity:** {r['similarity']:.4f}")
                st.write(r["content"])
                st.caption(f"doc_id={r['doc_id']} | chunk={r['chunk_index']}")
                st.divider()
        else:
            st.error(resp.text)
