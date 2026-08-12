<div align="center">

# 🎓 College Assistant

### An AI-powered RAG chatbot for student queries — academics, fees & general help

Built with **LangGraph** · **Groq (Llama 3.3 70B)** · **FAISS** · **Streamlit**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-F55036?style=for-the-badge&logo=lightning&logoColor=white)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](#-license)
[![Made with Love](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red?style=flat-square)](#)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](#)
[![Status](https://img.shields.io/badge/status-active-success.svg?style=flat-square)](#)

</div>

---

## 📌 Overview

**College Assistant** is a Retrieval-Augmented Generation (RAG) chatbot that answers student
questions by intelligently routing them through a **LangGraph** state machine:

```
                         ┌───────────────┐
                         │   classifier   │
                         │ (LLM routing)  │
                         └───────┬────────┘
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
      ┌───────────────┐  ┌───────────────┐   ┌───────────────┐
      │ academic_rag   │  │   fee_rag      │   │   general      │
      │ (handbook PDF) │  │ (fee PDF)      │   │ (no retrieval) │
      └───────┬────────┘  └───────┬────────┘   └───────┬────────┘
              └──────────────────┴──────────────────────┘
                                  ▼
                         ┌───────────────┐
                         │   response     │
                         │ (personalized) │
                         └───────────────┘
```

1. 🧭 **Classify** — the LLM tags each question as `academic`, `fee`, or `general`.
2. 📚 **Retrieve** — `academic` and `fee` queries pull top-`k` chunks from the relevant PDF via FAISS.
3. ✍️ **Respond** — a final answer is generated, personalized to the student's programme (BCA / BBA / B.Com (H)).

This app was converted from a CLI script into a full Streamlit web interface, with the original
LangGraph logic left fully intact.

---

## ✨ Features

| | |
|---|---|
| 💬 | Chat-style interface with persistent conversation history |
| 🧠 | LLM-based query routing (`academic` / `fee` / `general`) |
| 📄 | Upload your own handbook & fee-structure PDFs on the fly |
| 🎓 | Programme-personalized answers (BCA, BBA, B.Com (H)) |
| ⚡ | Cached embeddings, retrievers, LLM & graph — fast reruns |
| 🔐 | Secrets handled via `st.secrets` / `.env`, never hardcoded |
| 🏷️ | Each answer is tagged with how it was routed, for transparency |
| 🛡️ | Graceful error handling for missing keys, PDFs, or API failures |

---

## 🖥️ Preview

```
🎓 College Assistant
──────────────────────────────────────────
You:      What's the attendance requirement for BCA?
Assistant: You need a minimum of 75% attendance to be
           eligible to sit for exams...
           Routed as: academic
```

---

## 🔑 Required Secret

| Name            | Description                              |
|-----------------|-------------------------------------------|
| `GROQ_API_KEY`  | API key for ChatGroq — [get one here](https://console.groq.com) |

Provide it via **one** of the following (checked in this order):

**Option A — `.streamlit/secrets.toml`**
```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

**Option B — `.env` file** (auto-loaded via `python-dotenv`)
```env
GROQ_API_KEY=your-groq-api-key-here
```

**Option C — Sidebar input**
Paste it directly into the sidebar at runtime (session-only, never saved).

> 📁 Templates included: `.streamlit/secrets.toml.example` and `.env.example`

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

---

## 🧭 How to Use

1. 🔑 Enter your Groq API key in the sidebar (skip if already set via secrets/`.env`).
2. 📄 Upload the **academics handbook** PDF and **fee structure** PDF in the sidebar
   — or place files named `academics_handbook.pdf` and `fee_structure.pdf` next to
   `app.py` to use as defaults.
3. 🎓 Select your programme: **BCA**, **BBA**, or **B.Com (H)**.
4. 💬 Ask your question in the chat box at the bottom of the page.

Each response includes a small caption showing how it was routed
(`academic`, `fee`, or `general`) for transparency.

---

## ⚙️ Configuration Reference

| Setting | Value |
|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (via `langchain_huggingface`) |
| LLM | `llama-3.3-70b-versatile` via Groq |
| Temperature | `0.4` |
| Chunk size / overlap | `800` / `100` |
| Retrieval top-k | `4` |

---

## 🧩 Project Structure

```
college_assistant_app/
├── app.py                          # Streamlit app (UI + LangGraph logic)
├── requirements.txt                # Python dependencies
├── README.md                       # You are here
├── .env.example                    # Template for local env vars
└── .streamlit/
    └── secrets.toml.example        # Template for Streamlit secrets
```

---

## 📝 Notes on Behavior Preserved from the Original Script

- Each chat turn is sent to the LangGraph app **independently** — the graph itself is not
  given prior conversation turns as context, matching the original CLI script's behavior
  exactly. The Streamlit UI still displays the full visible chat history for readability.
- No functional logic was changed — only the I/O layer (`input()`/`print()` → Streamlit
  widgets) and resource construction (cached instead of built at import time).

---

## ⚡ Performance Notes

- The embedding model, FAISS retrievers, LLM client, and compiled LangGraph app are all
  cached with `st.cache_resource`, so they build once per session (or once per unique
  uploaded file) instead of on every rerun.
- First run downloads the `all-MiniLM-L6-v2` model and may take a little longer.

---

## 🛠️ Tech Stack

<div align="center">

`Streamlit` · `LangGraph` · `LangChain` · `Groq` · `FAISS` · `HuggingFace Embeddings` · `PyPDF`

</div>

---

## 📄 License

This project structure is provided as-is for educational and internal use. Add your
preferred license here.

<div align="center">

Made with 💙 for students who just want a straight answer about attendance and fees.

</div>