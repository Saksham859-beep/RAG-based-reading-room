"""
The Reading Room — RAG PDF Reader
A Streamlit front-end for a Mistral + Chroma RAG pipeline.
Same retrieval/generation logic as the original CLI script, wrapped in a
book-themed UI with PDF / TXT upload support.
"""

import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PERSIST_DIR = "chroma-data-Hub-book"
HERO_IMG = "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?fm=jpg&q=80&w=1000&auto=format&fit=crop"
SPINE_COLORS = ["#1F6F6B", "#7B3F61", "#C9A227", "#3C5A8A", "#A15843", "#4B7A5A"]

st.set_page_config(
    page_title="The Reading Room · RAG PDF Reader",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# THEME
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,700;1,600&family=Source+Sans+3:wght@400;600;700&family=Courier+Prime&display=swap');

:root{
  --bg-1:#FBF8F2; --bg-2:#EAF0E6; --ink:#1B2A4A; --muted:#6B5D54;
  --teal:#1F6F6B; --plum:#7B3F61; --gold:#C9A227; --card:#FFFEFA;
}

html, body, [class*="css"]{ font-family:'Source Sans 3', sans-serif; color:var(--ink); }

.stApp{
  background: radial-gradient(1200px 600px at 10% -10%, #F3EEE0 0%, transparent 60%),
              linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 100%);
}

/* Sidebar */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #F3F0E6 0%, #E7EEE3 100%);
  border-right: 1px solid #D9CFB8;
}
[data-testid="stSidebar"] .hero-frame{
  border-radius: 14px; overflow:hidden; margin-bottom: 14px;
  box-shadow: 0 8px 22px rgba(27,42,74,0.18);
  border: 3px solid var(--card);
}
[data-testid="stSidebar"] .hero-frame img{ display:block; width:100%; }
[data-testid="stSidebar"] h2{
  font-family:'Playfair Display', serif; font-style: italic; color: var(--plum);
  margin-top: 0;
}

/* Headline */
.app-title{
  font-family:'Playfair Display', serif; font-weight:700; font-size: 2.6rem;
  background: linear-gradient(90deg, var(--teal), var(--plum));
  -webkit-background-clip:text; -webkit-text-fill-color: transparent;
  margin-bottom: 0;
}
.app-subtitle{
  font-family:'Courier Prime', monospace; color: var(--muted);
  letter-spacing: 0.5px; margin-top: 2px; font-size: 0.95rem;
}
hr.divider{ border: none; border-top: 1px dashed #C9BFA6; margin: 14px 0 20px 0; }

/* Book spine cards for uploaded docs */
.book-spine{
  display:flex; align-items:center; gap:10px; padding:9px 12px;
  border-radius:8px; background: var(--card); box-shadow: 0 2px 6px rgba(0,0,0,0.08);
  border-left: 8px solid var(--spine-color, var(--teal));
  font-family:'Source Sans 3', sans-serif; font-size: 0.88rem; color: var(--ink);
  height: 42px;
}
.book-spine .icon{ font-size:1.1rem; }
.book-spine .name{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

/* Remove button (small, subtle, sits next to the spine) */
div[data-testid="stButton"] button[kind="secondary"].remove-btn,
.remove-btn-wrap .stButton>button{
  background: #FBEAEA !important; color:#8A2E2E !important;
  border: 1px solid #E5C6C6 !important; box-shadow:none !important;
  font-weight:600 !important; padding: 0.35rem 0.6rem !important;
  min-height: 42px;
}
.remove-btn-wrap .stButton>button:hover{
  background:#F6D3D3 !important; transform:none !important;
}

/* Chat bubbles */
.msg-row{ display:flex; margin: 14px 0; }
.msg-row.user{ justify-content:flex-end; }
.msg-row.ai{ justify-content:flex-start; }
.bubble{
  max-width: 72%; padding: 14px 18px; border-radius: 16px; line-height:1.55;
  box-shadow: 0 4px 14px rgba(27,42,74,0.10); font-size: 0.98rem;
}
.bubble.user{
  background: linear-gradient(135deg, var(--teal), #2C8B85); color:#F7FBF9;
  border-bottom-right-radius: 4px;
}
.bubble.ai{
  background: var(--card); color: var(--ink); border: 1px solid #EADFC5;
  border-bottom-left-radius: 4px;
}
.bubble.ai .label{
  font-family:'Playfair Display', serif; font-style: italic; color: var(--plum);
  font-weight:600; display:block; margin-bottom: 4px; font-size: 0.85rem;
}
.bubble.user .label{
  font-family:'Playfair Display', serif; font-style: italic; color: #E6FFF9;
  font-weight:600; display:block; margin-bottom: 4px; font-size: 0.85rem;
}
.sources{
  font-family:'Courier Prime', monospace; font-size: 0.72rem; color: var(--muted);
  margin-top: 8px; padding-top:6px; border-top: 1px dotted #D8CFB4;
}

/* Buttons */
.stButton>button{
  background: linear-gradient(135deg, var(--gold), #B98A1D); color:#2A1F04;
  border: none; border-radius: 8px; font-weight:700; padding: 0.5rem 1rem;
  box-shadow: 0 3px 8px rgba(201,162,39,0.35); transition: transform 0.15s ease;
}
.stButton>button:hover{ transform: translateY(-1px); filter: brightness(1.04); }

/* File uploader */
[data-testid="stFileUploaderDropzone"]{
  background: var(--card); border: 2px dashed #C9BFA6; border-radius: 12px;
}

/* Chat input */
[data-testid="stChatInput"] textarea{
  font-family:'Source Sans 3', sans-serif;
}

/* Empty state */
.empty-state{
  text-align:center; padding: 60px 20px; color: var(--muted);
  font-family:'Playfair Display', serif; font-style: italic; font-size:1.2rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# CACHED RESOURCES
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return MistralAIEmbeddings()


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatMistralAI(model="mistral-small-2506")


@st.cache_resource(show_spinner=False)
def load_vectorstore(_emb):
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=_emb)


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}
Question:
{question}
""",
        ),
    ]
)

embeddings = load_embeddings()
llm = load_llm()
vectorstore = load_vectorstore(embeddings)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "library" not in st.session_state:
    st.session_state.library = []  # list of file names already ingested


def sync_library_from_store():
    """Rebuild the sidebar's file list from what's actually stored in Chroma,
    so the UI never lies about what's really searchable (fixes the stale-shelf
    bug that happens if the DB folder is ever deleted/changed outside the app)."""
    try:
        raw = vectorstore.get(include=["metadatas"])
        sources = {
            m.get("source")
            for m in (raw.get("metadatas") or [])
            if m and m.get("source")
        }
        st.session_state.library = sorted(sources)
    except Exception:
        # Vectorstore empty or unreachable — leave library as-is rather than crash.
        pass


def remove_book(file_name: str):
    """Delete every chunk belonging to `file_name` from Chroma, then drop it
    from the sidebar list. This removes its embeddings entirely — the model
    can no longer retrieve anything from that file afterward."""
    try:
        vectorstore.delete(where={"source": file_name})
    except Exception as e:
        st.error(f"Couldn't remove '{file_name}' from the vector store: {e}")
        return
    if file_name in st.session_state.library:
        st.session_state.library.remove(file_name)
    st.success(f"Removed '{file_name}' and its embeddings.")


# Keep the sidebar list honest with what's actually in the DB on first load.
if "library_synced" not in st.session_state:
    sync_library_from_store()
    st.session_state.library_synced = True

# --------------------------------------------------------------------------
# SIDEBAR — LIBRARY (uploads)
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<div class="hero-frame"><img src="{HERO_IMG}"/></div>', unsafe_allow_html=True)
    st.markdown("## Your Library")
    st.caption("Add PDFs or text files, then ask questions grounded in what you upload.")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("📥 Add to Library", use_container_width=True):
        new_files = [f for f in (uploaded_files or []) if f.name not in st.session_state.library]
        if not new_files:
            st.info("Nothing new to add — upload a file first.")
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            with st.spinner("📖 Binding new pages into your library..."):
                for f in new_files:
                    ext = Path(f.name).suffix.lower()
                    if ext == ".pdf":
                        reader = PdfReader(f)
                        raw_text = "\n".join((p.extract_text() or "") for p in reader.pages)
                    elif ext == ".txt":
                        raw_text = f.read().decode("utf-8", errors="ignore")
                    else:
                        continue

                    chunks = splitter.split_text(raw_text)
                    if chunks:
                        vectorstore.add_texts(
                            chunks, metadatas=[{"source": f.name}] * len(chunks)
                        )
                        st.session_state.library.append(f.name)
            st.success(f"Added {len(new_files)} file(s) to the library.")
            st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if st.session_state.library:
        st.markdown("**On the shelf**")
        for i, name in enumerate(st.session_state.library):
            color = SPINE_COLORS[i % len(SPINE_COLORS)]
            icon = "📕" if name.lower().endswith(".pdf") else "📄"

            spine_col, btn_col = st.columns([5, 1])
            with spine_col:
                st.markdown(
                    f'<div class="book-spine" style="--spine-color:{color}">'
                    f'<span class="icon">{icon}</span><span class="name">{name}</span></div>',
                    unsafe_allow_html=True,
                )
            with btn_col:
                st.markdown('<div class="remove-btn-wrap">', unsafe_allow_html=True)
                if st.button("🗑️", key=f"remove_{name}", help=f"Remove '{name}' and its embeddings"):
                    remove_book(name)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.caption("Your shelf is empty — add a book to begin.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# --------------------------------------------------------------------------
# MAIN — HEADER
# --------------------------------------------------------------------------
st.markdown('<div class="app-title">📖 The Reading Room- Created By Saksham Jain😍</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">a RAG-powered reader for your PDFs &amp; notes</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------------------------------
chat_container = st.container()

def render_message(role, content, sources=None):
    label = "You" if role == "user" else "The Reading Room"
    bubble_class = "user" if role == "user" else "ai"
    sources_html = ""
    if sources:
        sources_html = f'<div class="sources">📎 Sources: {", ".join(sources)}</div>'
    chat_container.markdown(
        f'<div class="msg-row {bubble_class}">'
        f'<div class="bubble {bubble_class}"><span class="label">{label}</span>'
        f'{content}{sources_html}</div></div>',
        unsafe_allow_html=True,
    )

with chat_container:
    if not st.session_state.chat_history:
        st.markdown(
            '<div class="empty-state">"A room without books is like a body without a soul."<br>'
            "Add a document and ask your first question.</div>",
            unsafe_allow_html=True,
        )
    for msg in st.session_state.chat_history:
        render_message(msg["role"], msg["content"], msg.get("sources"))

# --------------------------------------------------------------------------
# CHAT INPUT
# --------------------------------------------------------------------------
query = st.chat_input("Ask something about your library…")

if query:
    if not st.session_state.library:
        st.session_state.chat_history.append({"role": "user", "content": query})
        render_message("user", query)
        with chat_container:
            st.warning("Your shelf is empty — add a document before asking questions.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        render_message("user", query)

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
        )

        with chat_container:
            with st.spinner("📖 Flipping through pages..."):
                docs = retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in docs)
                final_prompt = PROMPT.invoke({"context": context, "question": query})
                response = llm.invoke(final_prompt)

            source_names = sorted({d.metadata.get("source", "unknown") for d in docs}) if docs else []

            # Typewriter reveal, styled like the rest of the AI bubble
            placeholder = st.empty()
            full_text = response.content
            shown = ""
            for ch in full_text:
                shown += ch
                placeholder.markdown(
                    f'<div class="msg-row ai"><div class="bubble ai">'
                    f'<span class="label">The Reading Room</span>{shown}▌</div></div>',
                    unsafe_allow_html=True,
                )
                time.sleep(0.006)
            sources_html = (
                f'<div class="sources">📎 Sources: {", ".join(source_names)}</div>'
                if source_names
                else ""
            )
            placeholder.markdown(
                f'<div class="msg-row ai"><div class="bubble ai">'
                f'<span class="label">The Reading Room</span>{full_text}{sources_html}</div></div>',
                unsafe_allow_html=True,
            )

        st.session_state.chat_history.append(
            {"role": "ai", "content": full_text, "sources": source_names}
        )