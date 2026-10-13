import os
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

APP_TITLE = "Chat with Your PDFs"
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

st.markdown("""
<style>
.main .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.25); border-radius: 18px; margin-bottom: 1rem;}
.hero h1 {margin-bottom: .25rem;}
.muted {color: #777;}
.source-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: .8rem 1rem; margin: .5rem 0;}
</style>
""", unsafe_allow_html=True)

@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str
    embedding: List[float]


def get_api_key() -> str:
    try:
        secret = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret = ""
    return secret or os.getenv("OPENAI_API_KEY", "")


def extract_pdf(uploaded_file) -> List[Dict[str, Any]]:
    """Extract text page-by-page so citations can include the original page number."""
    reader = PdfReader(uploaded_file)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = " ".join(text.split())
        if text:
            pages.append({"source": uploaded_file.name, "page": page_number, "text": text})
    return pages


def split_into_chunks(page_records: List[Dict[str, Any]], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    chunks = []
    for record in page_records:
        text = record["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            snippet = text[start:end].strip()
            if snippet:
                raw_id = f"{record['source']}|{record['page']}|{start}|{snippet[:50]}"
                chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:12]
                chunks.append({
                    "text": snippet,
                    "source": record["source"],
                    "page": record["page"],
                    "chunk_id": chunk_id,
                })
            if end >= len(text):
                break
            start = max(0, end - overlap)
    return chunks


def batched(items, batch_size=64):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def create_embeddings(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    vectors = []
    for batch in batched(texts, 64):
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend([item.embedding for item in response.data])
    return vectors


def cosine_scores(matrix: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    matrix_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query_vector)
    denominator = matrix_norms * query_norm
    denominator[denominator == 0] = 1e-12
    return (matrix @ query_vector) / denominator


def retrieve(question: str, chunks: List[Chunk], matrix: np.ndarray, client: OpenAI, embed_model: str, top_k: int):
    response = client.embeddings.create(model=embed_model, input=[question])
    query_vector = np.asarray(response.data[0].embedding, dtype=np.float32)
    scores = cosine_scores(matrix, query_vector)
    ranked = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in ranked:
        chunk = chunks[int(idx)]
        results.append({
            "text": chunk.text,
            "source": chunk.source,
            "page": chunk.page,
            "score": float(scores[int(idx)]),
            "chunk_id": chunk.chunk_id,
        })
    return results


def answer_question(client: OpenAI, question: str, results: List[Dict[str, Any]], model: str, min_score: float):
    if not results or results[0]["score"] < min_score:
        return ("I couldn't find sufficiently relevant evidence in the uploaded documents to answer that reliably. "
                "Try rephrasing the question or uploading a document that covers this topic."), []

    context_blocks = []
    allowed_sources = []
    for i, item in enumerate(results, start=1):
        label = f"[S{i}] {item['source']}, page {item['page']}"
        context_blocks.append(f"{label}\n{item['text']}")
        allowed_sources.append({"label": f"S{i}", "source": item["source"], "page": item["page"]})

    system_prompt = """You are a careful document question-answering assistant. Answer using only the supplied document excerpts. If the excerpts do not contain enough evidence, say that the documents do not provide enough information. Do not use outside knowledge to fill gaps. Cite every material factual claim with the source labels exactly as provided, for example [S1]. Never invent source labels, page numbers, quotations, or facts. Distinguish direct evidence from reasonable interpretation."""
    user_prompt = f"QUESTION:\n{question}\n\nDOCUMENT EXCERPTS:\n" + "\n\n---\n\n".join(context_blocks)
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )
    return response.choices[0].message.content or "I couldn't generate an answer from the retrieved excerpts.", allowed_sources


def reset_index():
    for key in ["chunks", "embedding_matrix", "indexed_files", "chat_history", "last_retrieval", "last_answer"]:
        st.session_state.pop(key, None)

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embedding_matrix" not in st.session_state:
    st.session_state.embedding_matrix = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown(f"""
<div class="hero">
  <h1>📚 {APP_TITLE}</h1>
  <p>Upload PDFs, ask questions in plain English, and inspect the passages behind each answer.</p>
  <p class="muted">A portfolio demonstration of document ingestion, embeddings, semantic retrieval, source citations, and evidence-aware answering.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    api_key = get_api_key()
    if api_key:
        st.success("LLM API key detected")
    else:
        st.warning("Add OPENAI_API_KEY to enable embeddings and answers.")
    chat_model = st.text_input("Chat model", value=DEFAULT_CHAT_MODEL)
    embedding_model = st.text_input("Embedding model", value=DEFAULT_EMBED_MODEL)
    st.divider()
    chunk_size = st.slider("Chunk size (characters)", min_value=500, max_value=1800, value=1000, step=100)
    overlap = st.slider("Chunk overlap", min_value=0, max_value=300, value=150, step=25)
    top_k = st.slider("Retrieved passages", min_value=1, max_value=8, value=4)
    min_score = st.slider("Minimum similarity score", min_value=0.05, max_value=0.80, value=0.22, step=0.01,
                          help="A simple retrieval guardrail, not a calibrated confidence probability. Tune it on representative questions.")
    st.caption("API usage may incur charges. Uploaded files are processed in this app session and are not intentionally saved to disk.")

left, right = st.columns([1, 1.4], gap="large")
with left:
    st.subheader("1. Upload your PDFs")
    uploaded_files = st.file_uploader("Choose one or more text-based PDF files", type=["pdf"], accept_multiple_files=True,
                                      help="Scanned/image-only PDFs may require OCR before their text can be searched.")
    if uploaded_files:
        st.caption(f"Selected {len(uploaded_files)} file(s). Re-index after changing the selection.")
    index_button = st.button("Process documents", type="primary", use_container_width=True, disabled=not uploaded_files or not api_key)
    if index_button:
        try:
            client = OpenAI(api_key=api_key)
            all_pages, file_summaries = [], []
            with st.status("Reading and indexing documents…", expanded=True) as status:
                for uploaded in uploaded_files:
                    pages = extract_pdf(uploaded)
                    all_pages.extend(pages)
                    file_summaries.append({"name": uploaded.name, "pages_with_text": len(pages)})
                    st.write(f"Read **{uploaded.name}**: {len(pages)} page(s) containing extractable text")
                raw_chunks = split_into_chunks(all_pages, chunk_size, overlap)
                if not raw_chunks:
                    status.update(label="No extractable text found", state="error")
                    st.error("No text could be extracted. If these are scanned PDFs, run OCR first and upload searchable PDFs.")
                else:
                    st.write(f"Creating embeddings for {len(raw_chunks)} text chunks…")
                    vectors = create_embeddings(client, [c["text"] for c in raw_chunks], embedding_model)
                    chunks = [Chunk(text=c["text"], source=c["source"], page=c["page"], chunk_id=c["chunk_id"], embedding=v)
                              for c, v in zip(raw_chunks, vectors)]
                    st.session_state.chunks = chunks
                    st.session_state.embedding_matrix = np.asarray(vectors, dtype=np.float32)
                    st.session_state.indexed_files = file_summaries
                    st.session_state.chat_history = []
                    status.update(label="Documents indexed successfully", state="complete")
            if st.session_state.chunks:
                st.success(f"Indexed {len(st.session_state.indexed_files)} PDF(s) into {len(st.session_state.chunks)} searchable chunks.")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
            st.info("Check your API key, model names, network access, and whether the PDF contains selectable text.")
    if st.session_state.indexed_files:
        st.markdown("**Indexed files**")
        for item in st.session_state.indexed_files:
            st.write(f"📄 {item['name']} — {item['pages_with_text']} page(s) with text")
        if st.button("Clear indexed documents", use_container_width=True):
            reset_index()
            st.rerun()

with right:
    st.subheader("2. Ask a question")
    if not st.session_state.chunks:
        st.info("Upload your PDFs and select **Process documents** to begin.")
        st.markdown("**Try questions like these after indexing:**")
        st.markdown("- What is the main purpose of this document?\n- Summarize the key findings.\n- What dates, amounts, or requirements are mentioned?\n- Does the document explain [a specific topic]?")
    else:
        with st.form("question_form", clear_on_submit=True):
            question = st.text_input("Your question", placeholder="Ask something that the uploaded documents can answer…")
            submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)
        if submitted and question.strip():
            try:
                client = OpenAI(api_key=api_key)
                with st.spinner("Searching documents and preparing an evidence-based answer…"):
                    results = retrieve(question.strip(), st.session_state.chunks, st.session_state.embedding_matrix,
                                       client, embedding_model, top_k)
                    answer, cited_sources = answer_question(client, question.strip(), results, chat_model, min_score)
                st.session_state.chat_history.append({"question": question.strip(), "answer": answer,
                                                      "results": results, "cited_sources": cited_sources})
            except Exception as exc:
                st.error(f"Question failed: {exc}")
                st.info("Check your API key, model availability, and API account limits.")

        for turn in reversed(st.session_state.chat_history):
            st.markdown("---")
            st.markdown(f"**You asked:** {turn['question']}")
            st.markdown("**Answer**")
            st.markdown(turn["answer"])
            with st.expander("Inspect retrieved passages and similarity scores", expanded=True):
                st.caption("Similarity scores indicate vector similarity, not a probability that an answer is correct.")
                for rank, result in enumerate(turn["results"], start=1):
                    st.markdown(f"**[S{rank}] {result['source']} — page {result['page']}** · similarity `{result['score']:.3f}`")
                    st.write(result["text"])
                    st.divider()

st.markdown("---")
with st.expander("How this demo works"):
    st.markdown("""
    1. **Extract:** read selectable text from each PDF page.
    2. **Chunk:** split page text into overlapping passages while preserving filename and page number.
    3. **Embed:** convert passages into vectors using the configured embedding model.
    4. **Retrieve:** embed the question and rank passages by cosine similarity.
    5. **Generate:** ask the chat model to answer only from the retrieved passages and cite source labels.
    6. **Inspect:** show retrieved evidence so users can check the answer against the source.

    **Limitations:** this prototype uses an in-memory index, so reloading the app or starting a new session requires re-uploading and re-indexing files. Scanned PDFs need OCR. Similarity thresholds require evaluation on representative questions. The model can still make mistakes, so verify important answers against the source documents.
    """)
st.caption("Portfolio demo • Python • Streamlit • OpenAI embeddings and chat completions • NumPy cosine-similarity retrieval")
