"""
College Assistant - Streamlit App
----------------------------------
A RAG-powered college assistant built with LangGraph + Groq + FAISS.

This is a Streamlit conversion of an original CLI script. The core
LangGraph logic (classifier -> academic/fee/general -> response) is
preserved unchanged; only the I/O layer (input()/print() -> Streamlit
widgets) and resource construction (PDF loading, retriever/LLM/graph
building) have been adapted to Streamlit's execution model.

Required secret / environment variable:
    GROQ_API_KEY  -  API key for ChatGroq (https://console.groq.com)

You can supply it via (checked in this order):
    1. st.secrets["GROQ_API_KEY"]        (.streamlit/secrets.toml)
    2. environment variable GROQ_API_KEY (.env file, loaded via dotenv)
    3. a text box in the sidebar (session-only, not persisted)

Run locally:
    streamlit run app.py
"""

import os
import tempfile
from typing import TypedDict, Annotated

import streamlit as st
from dotenv import load_dotenv

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="wide",
)

# ============================================================
# Theme — "Registrar's Ledger"
# Deep ink navy + parchment + brass, with color-coded tabs that
# encode the routing decision (academic / fee / general) the way
# a library index card's colored tab encodes its category.
# ============================================================
LEDGER_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#16233F;
  --ink-2:#1E2F52;
  --paper:#FBF7EF;
  --paper-2:#F3ECDD;
  --brass:#B08D57;
  --brass-light:#D9BD8C;
  --hairline:#E4D9C2;
  --academic:#2F5233;
  --academic-bg:#E7EEE3;
  --fee:#7A2E2E;
  --fee-bg:#F3E4E0;
  --general:#4B5566;
  --general-bg:#E9EAEC;
  --text:#1B2A4A;
  --text-muted:#5B6472;
}
html, body, [class*="css"]{ font-family:'Inter', sans-serif; color:var(--text); }
h1, h2, h3 { color:var(--text); font-family:'Source Serif 4', serif; }
.ledger-title{ font-family:'Source Serif 4', serif; }
.stApp{ color:var(--text); }
.stApp p, .stApp li, .stApp span, .stApp label,
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li{
  color:var(--text) !important;
}
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span{
  color:var(--text) !important;
}
.stCaption, [data-testid="stCaptionContainer"]{ color:var(--text-muted) !important; }
[data-testid="stAlert"] p, [data-testid="stAlert"] span{ color:var(--text) !important; }
[data-testid="stChatInput"] textarea{ color:var(--text) !important; }
[data-testid="stChatInput"] textarea::placeholder{ color:var(--text-muted) !important; opacity:1; }
[data-testid="stSelectbox"] *{ color:var(--text) !important; }
div.stButton > button{ color:var(--text) !important; }
code, .ledger-tag, .stCaption, small { font-family:'IBM Plex Mono', monospace; }
.stApp{ background:var(--paper); }
.main .block-container{ padding-top:1.5rem; max-width:900px; }
/* ---- Masthead ---- */
.ledger-header{
  border-bottom:2px solid var(--ink);
  padding-bottom:14px;
  margin-bottom:6px;
}
.ledger-eyebrow{
  font-family:'IBM Plex Mono', monospace;
  font-size:0.72rem;
  letter-spacing:.14em;
  text-transform:uppercase;
  color:var(--brass);
  font-weight:600;
  margin-bottom:2px;
}
.ledger-title{
  font-size:2.1rem;
  font-weight:700;
  color:var(--ink);
  margin:0;
  letter-spacing:-.01em;
}
.ledger-sub{
  color:var(--text-muted);
  font-size:0.95rem;
  margin-top:6px;
  line-height:1.5;
}
.ledger-rule{
  height:1px;
  background:repeating-linear-gradient(90deg, var(--brass) 0 6px, transparent 6px 11px);
  margin:14px 0 22px 0;
}
/* ---- Sidebar ---- */
section[data-testid="stSidebar"]{
  background:var(--ink);
  border-right:1px solid var(--ink-2);
}
section[data-testid="stSidebar"] *{ color:var(--paper) !important; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3{
  font-family:'Source Serif 4', serif !important;
  color:var(--brass-light) !important;
}
section[data-testid="stSidebar"] hr{ border-color:var(--ink-2) !important; }
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small{
  color:#A9B4CC !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] [data-testid="stTextInput"] input,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div{
  background:var(--ink-2) !important;
  border:1px solid #33456E !important;
  border-radius:6px !important;
}
section[data-testid="stSidebar"] div.stButton > button{
  background:transparent !important;
  border:1px solid var(--brass) !important;
  color:var(--brass-light) !important;
  border-radius:6px;
  width:100%;
  font-size:0.85rem;
  letter-spacing:.03em;
}
section[data-testid="stSidebar"] div.stButton > button:hover{
  background:var(--brass) !important;
  color:var(--ink) !important;
}
/* ---- Buttons (main area) ---- */
div.stButton > button, [data-testid="stChatInput"] button{
  border-radius:6px;
}
/* ---- Alerts ---- */
[data-testid="stAlert"]{ border-radius:6px; border-left:4px solid var(--brass); }
/* ---- Chat messages: index-card look ---- */
[data-testid="stChatMessage"]{
  background:#FFFFFF;
  border:1px solid var(--hairline);
  border-left:5px solid var(--general);
  border-radius:8px;
  padding:4px 6px;
  margin-bottom:12px;
  box-shadow:0 1px 2px rgba(22,35,63,0.04);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  border-left-color:var(--brass);
  background:var(--paper-2);
}
[data-testid="stChatMessageAvatarUser"]{
  background:var(--ink) !important;
}
/* ---- Routing tag pill ---- */
.ledger-tag{
  display:inline-block;
  font-size:0.68rem;
  font-weight:600;
  letter-spacing:.08em;
  text-transform:uppercase;
  padding:2px 8px;
  border-radius:3px;
  margin-top:6px;
}
.ledger-tag.academic{ background:var(--academic-bg); color:var(--academic); }
.ledger-tag.fee{ background:var(--fee-bg); color:var(--fee); }
.ledger-tag.general{ background:var(--general-bg); color:var(--general); }
/* ---- Chat input ---- */
[data-testid="stChatInput"]{
  border:1px solid var(--brass) !important;
  border-radius:8px !important;
}
/* ---- Divider ---- */
hr{ border-color:var(--hairline); }
</style>
"""
st.markdown(LEDGER_CSS, unsafe_allow_html=True)

ROUTE_STYLE = {
    "academic": {"label": "Academic · Handbook", "avatar": "📘"},
    "fee": {"label": "Fees · Ledger", "avatar": "💳"},
    "general": {"label": "General", "avatar": "💬"},
}


def render_route_tag(query_type: str):
    """Renders a small color-coded pill that encodes the routing decision."""
    style = ROUTE_STYLE.get(query_type)
    if not style:
        return
    st.markdown(
        f'<span class="ledger-tag {query_type}">{style["label"]}</span>',
        unsafe_allow_html=True,
    )


PROGRAMME_OPTIONS = {
    "BCA": "BCA",
    "BBA": "BBA",
    "B.Com (H)": "B.Com (H)",
}

DEFAULT_ACADEMIC_PDF = "academics_handbook.pdf"
DEFAULT_FEE_PDF = "fee_structure.pdf"


# ============================================================
# State schema (unchanged from original script)
# ============================================================
class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]  # history (per-turn use, see note below)
    query_type: str
    retrieved_context: str


# ============================================================
# Resource builders (cached so they don't rebuild on every rerun)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Loads the sentence-transformers embedding model once per session."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def build_retriever_from_bytes(pdf_bytes: bytes, cache_key: str):
    """
    Builds a FAISS retriever from PDF bytes.
    `cache_key` (e.g. filename + size) is part of the cache signature so that
    uploading a different file correctly triggers a rebuild.
    """
    embeddings = get_embeddings()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        document = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(document)

        vectorstore = FAISS.from_documents(chunks, embeddings)
    finally:
        os.remove(tmp_path)

    return vectorstore.as_retriever(search_kwargs={"k": 4})


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4, groq_api_key=api_key)


# ============================================================
# Node factories (same logic as the original script, parameterized
# on llm / retrievers instead of relying on module-level globals)
# ============================================================
def make_classifier_node(llm):
    def classifier_node(state: State) -> dict:
        """Look at the latest user message and decide which path to take."""
        last_message = state["messages"][-1].content

        prompt = (
            "Classify the following student query into exactly one category: "
            "'academic', 'fee', or 'general'.\n\n"
            "Use 'academic' for questions about attendance, exams, grading, credits, "
            "promotion, course structure, summer training, or degree requirements.\n"
            "Use 'fee' for questions about tuition, payment, refund, late charges, "
            "scholarships, or any money-related topic.\n"
            "Use 'general' for greetings, casual talk, or anything not related to "
            "the college rules or fee.\n\n"
            f"Query: {last_message}\n\n"
            "Return only one word: academic, fee, or general."
        )

        response = llm.invoke(prompt)
        category = response.content.strip().lower()

        if "academic" in category:
            category = "academic"
        elif "fee" in category:
            category = "fee"
        else:
            category = "general"

        return {"query_type": category}

    return classifier_node


def make_academic_rag_node(academic_retriever):
    def academic_rag_node(state: State) -> dict:
        """Retrieves relevant chunks from the academics handbook."""
        query = state["messages"][-1].content
        docs = academic_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    return academic_rag_node


def make_fee_rag_node(fee_retriever):
    def fee_rag_node(state: State) -> dict:
        """Retrieves relevant chunks from the fee structure PDF."""
        query = state["messages"][-1].content
        docs = fee_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    return fee_rag_node


def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def make_response_node(llm):
    def response_node(state: State) -> dict:
        """Generates the final answer, personalized using the student's programme."""
        query = state["messages"][-1].content
        programme = state.get("programme", "Unknown")
        context = state["retrieved_context"]

        if context == "NO_RETRIEVAL_NEEDED":
            prompt = (
                f"You are a friendly college assistant talking to a {programme} student. "
                f"Answer this question using your own general knowledge:\n\n{query}"
            )
        else:
            prompt = (
                f"You are a college assistant helping a {programme} student. "
                f"Use the following context from the official college documents to answer "
                f"the question accurately. If the context mentions specific figures for "
                f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Give a clear, friendly, and precise answer."
            )

        response = llm.invoke(prompt)
        return {"messages": [("ai", response.content.strip())]}

    return response_node


def route_query(state: State):
    if state["query_type"] == "academic":
        return "academic_rag"
    elif state["query_type"] == "fee":
        return "fee_rag"
    else:
        return "general"


# ============================================================
# Graph builder (cached; underscore-prefixed args are excluded
# from Streamlit's hash-based cache key since they're unhashable)
# ============================================================
@st.cache_resource(show_spinner=False)
def build_graph(_llm, _academic_retriever, _fee_retriever):
    graph = StateGraph(State)

    graph.add_node("classifier", make_classifier_node(_llm))
    graph.add_node("academic_rag", make_academic_rag_node(_academic_retriever))
    graph.add_node("fee_rag", make_fee_rag_node(_fee_retriever))
    graph.add_node("general", general_node)
    graph.add_node("response", make_response_node(_llm))

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", route_query)
    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")
    graph.add_edge("response", END)

    return graph.compile()


# ============================================================
# Helpers
# ============================================================
def resolve_api_key() -> str:
    """Checks st.secrets, then env var, then returns empty string."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


def load_pdf_bytes(uploaded_file, default_path: str):
    """
    Returns (bytes, cache_key) for a PDF, preferring an uploaded file.
    Falls back to a local file with `default_path` if it exists on disk
    (mirrors the original script's hardcoded local-file behavior).
    """
    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        cache_key = f"{uploaded_file.name}-{len(data)}"
        return data, cache_key

    if os.path.exists(default_path):
        with open(default_path, "rb") as f:
            data = f.read()
        cache_key = f"{default_path}-{len(data)}"
        return data, cache_key

    return None, None


# ============================================================
# Sidebar - configuration
# ============================================================
with st.sidebar:
    st.markdown(
        '<div class="ledger-eyebrow" style="color:var(--brass-light);">Registrar\'s Desk</div>'
        '<h2 style="margin-top:0;">Configuration</h2>',
        unsafe_allow_html=True,
    )

    st.subheader("1 · Groq API Key")
    stored_key = resolve_api_key()
    if stored_key:
        st.success("API key loaded from secrets/environment.")
        groq_api_key = stored_key
    else:
        groq_api_key = st.text_input(
            "Enter your GROQ_API_KEY",
            type="password",
            help="Not found in st.secrets or environment variables. "
                 "You can enter it here for this session, or set it up "
                 "properly via .streamlit/secrets.toml or a .env file.",
        )

    st.divider()

    st.subheader("2 · Knowledge Base Documents")
    academic_upload = st.file_uploader(
        "Academics handbook (PDF)", type=["pdf"], key="academic_upload"
    )
    fee_upload = st.file_uploader(
        "Fee structure (PDF)", type=["pdf"], key="fee_upload"
    )
    st.caption(
        f"If left empty, the app will look for local files named "
        f"`{DEFAULT_ACADEMIC_PDF}` and `{DEFAULT_FEE_PDF}` next to `app.py`."
    )

    st.divider()

    st.subheader("3 · Student Programme")
    student_programme = st.selectbox(
        "Which programme are you in?",
        options=list(PROGRAMME_OPTIONS.keys()),
        index=0,
    )

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.pop("chat_history", None)
        st.rerun()


# ============================================================
# Main area
# ============================================================
st.markdown(
    """
    <div class="ledger-header">
        <div class="ledger-eyebrow">Student Records &amp; Enquiries</div>
        <p class="ledger-title">🎓 College Assistant</p>
        <p class="ledger-sub">
            Ask about <b>academics</b> — attendance, exams, credits, promotion — or
            <b>fees</b> — tuition, refunds, scholarships. General questions are
            answered directly, without a document lookup.
        </p>
    </div>
    <div class="ledger-rule"></div>
    """,
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ..., "query_type": ...}

# --- Guard: API key ---
if not groq_api_key:
    st.warning("Please provide a `GROQ_API_KEY` in the sidebar to continue.")
    st.stop()

# --- Guard: PDFs ---
academic_bytes, academic_key = load_pdf_bytes(academic_upload, DEFAULT_ACADEMIC_PDF)
fee_bytes, fee_key = load_pdf_bytes(fee_upload, DEFAULT_FEE_PDF)

missing = []
if academic_bytes is None:
    missing.append("academics handbook")
if fee_bytes is None:
    missing.append("fee structure")

if missing:
    st.info(
        "Please upload the following document(s) in the sidebar to build the "
        "knowledge base: " + ", ".join(missing) + "."
    )
    st.stop()

# --- Build resources (cached) ---
try:
    with st.spinner("Preparing knowledge base and model (first run may take a while)..."):
        academic_retriever = build_retriever_from_bytes(academic_bytes, academic_key)
        fee_retriever = build_retriever_from_bytes(fee_bytes, fee_key)
        llm = get_llm(groq_api_key)
        app_graph = build_graph(llm, academic_retriever, fee_retriever)
except Exception as e:
    st.error(f"Failed to initialize the assistant: {e}")
    st.stop()

st.success(f"Ready! You're set as a **{student_programme}** student.")

# --- Render existing chat history ---
for msg in st.session_state.chat_history:
    if msg["role"] == "assistant":
        avatar = ROUTE_STYLE.get(msg.get("query_type"), {}).get("avatar", "🎓")
        with st.chat_message("assistant", avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("query_type"):
                render_route_tag(msg["query_type"])
    else:
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(msg["content"])

# --- Chat input ---
user_query = st.chat_input("Ask your question...")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Consulting the records..."):
            try:
                # NOTE: mirrors the original script's behavior of invoking the
                # graph with only the current turn's message (no accumulated
                # multi-turn memory sent to the LLM).
                result = app_graph.invoke(
                    {
                        "programme": PROGRAMME_OPTIONS[student_programme],
                        "messages": [("human", user_query)],
                    }
                )
                answer = result["messages"][-1].content
                query_type = result.get("query_type", "")
                st.markdown(answer)
                if query_type:
                    render_route_tag(query_type)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "query_type": query_type}
                )
            except Exception as e:
                error_msg = f"Something went wrong while generating a response: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg, "query_type": ""}
                )