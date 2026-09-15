"""
Streamlit manual-testing UI for the Document Q&A API.

This is a testing convenience, NOT part of the core FastAPI deliverable —
it talks to the API purely over HTTP, exactly like curl or a real client
would, and has no import dependency on anything in app/. That separation is
deliberate: the project spec is explicit about not adding unnecessary
frameworks to the core architecture, so this lives in tools/ with its own
requirements.txt and is never pulled into the Docker image.

Run it with the API already running (locally or via `docker compose up`):

    pip install -r tools/requirements.txt
    streamlit run tools/streamlit_app.py

Then open the URL Streamlit prints (usually http://localhost:8501).
"""

import requests
import streamlit as st

st.set_page_config(page_title="Document Q&A — Manual Test UI", layout="wide")


def _error_message(resp: requests.Response) -> str:
    """
    Extracts the clean {"error": "..."} body our API's exception handlers
    always return (see app/main.py) — falls back to raw text for anything
    that isn't JSON, e.g. if the API isn't actually reachable at all.
    """
    try:
        return f"{resp.status_code}: {resp.json().get('error', resp.text)}"
    except ValueError:
        return f"{resp.status_code}: {resp.text}"


if "documents" not in st.session_state:
    st.session_state.documents = []  # list of {"document_id", "filename", "chunks_created", "status"}

with st.sidebar:
    st.header("Connection")
    api_base_url = st.text_input("API base URL", value="http://localhost:8000")
    session_id = st.text_input("Session ID", value="streamlit-session-1")
    st.caption(
        "The session ID groups your queries into one conversation thread — "
        "used by /query, /conversations/{session_id}, and /stats."
    )

    st.divider()
    st.header("Stats")
    if st.button("Refresh stats"):
        try:
            resp = requests.get(f"{api_base_url}/stats", timeout=10)
            resp.raise_for_status()
            stats = resp.json()
            st.metric("Total documents", stats["total_documents"])
            st.metric("Total queries", stats["total_queries"])
            st.metric("Total conversations", stats["total_conversations"])
        except Exception as exc:
            st.error(f"Could not reach /stats: {exc}")

st.title("📄 Document Q&A — Manual Test UI")
st.caption(
    "A thin HTTP client for validating the API by hand — uploads, streaming "
    "queries, citations, conversation history, and deletion, all in one place."
)

upload_tab, query_tab, history_tab, manage_tab = st.tabs(
    ["Upload", "Ask a question", "Conversation history", "Manage documents"]
)

# --- Upload -----------------------------------------------------------------
with upload_tab:
    st.subheader("Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF or .txt file", type=["pdf", "txt"])

    if st.button("Upload", disabled=uploaded_file is None):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        }
        try:
            resp = requests.post(f"{api_base_url}/upload", files=files, timeout=60)
        except Exception as exc:
            st.error(f"Request failed: {exc}")
        else:
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.documents.append(data)
                st.success(
                    f"Uploaded **{data['filename']}** — "
                    f"{data['chunks_created']} chunks, status: `{data['status']}`, "
                    f"document_id: `{data['document_id']}`"
                )
            else:
                st.error(_error_message(resp))

    if st.session_state.documents:
        st.divider()
        st.write("Uploaded this session:")
        st.dataframe(
            [
                {
                    "filename": d["filename"],
                    "document_id": d["document_id"],
                    "chunks_created": d["chunks_created"],
                    "status": d["status"],
                }
                for d in st.session_state.documents
            ],
            use_container_width=True,
        )

# --- Ask a question -----------------------------------------------------------
with query_tab:
    st.subheader("Ask a question about an uploaded document")

    if not st.session_state.documents:
        st.info("Upload a document first (see the Upload tab).")
    else:
        options = {
            f"{d['filename']}  ({d['document_id']})": d["document_id"]
            for d in st.session_state.documents
        }
        selected_label = st.selectbox("Document", list(options.keys()))
        document_id = options[selected_label]
        question = st.text_area("Question", placeholder="What is this document about?")

        if st.button("Ask", disabled=not question.strip()):
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                try:
                    resp = requests.post(
                        f"{api_base_url}/query",
                        json={
                            "document_id": document_id,
                            "question": question,
                            "session_id": session_id,
                        },
                        stream=True,
                        timeout=120,
                    )
                except Exception as exc:
                    st.error(f"Request failed: {exc}")
                else:
                    if resp.status_code != 200:
                        st.error(_error_message(resp))
                        resp.close()
                    else:
                        def _tokens():
                            try:
                                for chunk in resp.iter_content(chunk_size=None):
                                    if chunk:
                                        yield chunk.decode("utf-8", errors="ignore")
                            finally:
                                resp.close()

                        # st.write_stream renders tokens live as they arrive
                        # and returns the fully concatenated text once the
                        # stream ends — same content the API's Sources: block
                        # appends at the end.
                        st.write_stream(_tokens())

# --- Conversation history ----------------------------------------------------
with history_tab:
    st.subheader(f"History for session: `{session_id}`")

    if st.button("Load history"):
        try:
            resp = requests.get(f"{api_base_url}/conversations/{session_id}", timeout=10)
        except Exception as exc:
            st.error(f"Request failed: {exc}")
        else:
            if resp.status_code != 200:
                st.error(_error_message(resp))
            else:
                conversations = resp.json()["conversations"]
                if not conversations:
                    st.info("No conversation history yet for this session.")
                for entry in conversations:
                    with st.chat_message("user"):
                        st.write(entry["question"])
                    with st.chat_message("assistant"):
                        st.write(entry["answer"])
                        if entry["sources"]:
                            st.caption("Sources: " + "; ".join(entry["sources"]))
                        st.caption(
                            f"model: {entry['model']} · "
                            f"latency: {entry['latency_ms']:.0f} ms · "
                            f"{entry['timestamp']}"
                        )

# --- Manage documents ---------------------------------------------------------
with manage_tab:
    st.subheader("Delete a document")

    if not st.session_state.documents:
        st.info("Nothing uploaded this session yet.")
    else:
        options = {
            f"{d['filename']}  ({d['document_id']})": d["document_id"]
            for d in st.session_state.documents
        }
        selected_label = st.selectbox("Document to delete", list(options.keys()), key="delete_select")
        document_id = options[selected_label]

        if st.button("Delete", type="primary"):
            try:
                resp = requests.delete(f"{api_base_url}/document/{document_id}", timeout=30)
            except Exception as exc:
                st.error(f"Request failed: {exc}")
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Deleted — {data['chunks_deleted']} chunks removed.")
                    st.session_state.documents = [
                        d for d in st.session_state.documents if d["document_id"] != document_id
                    ]
                else:
                    st.error(_error_message(resp))
