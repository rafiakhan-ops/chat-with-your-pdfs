# Chat with Your PDFs — AI Document Assistant

## Features

- Upload multiple PDF files
- Extract page-level text and preserve filename/page metadata
- Chunk text with overlap
- Create embeddings with an OpenAI embedding model
- Retrieve relevant chunks using cosine similarity
- Generate answers grounded in retrieved excerpts
- Display retrieved passages and similarity scores
- Decline to answer when retrieval similarity is below a configurable threshold
- In-memory index: no persistent vector database setup required for the demo

## Architecture

See [`architecture.png`](architecture.png) or [`architecture.svg`](architecture.svg).

## Run locally

Requires Python 3.10 or later and an API key for an OpenAI-compatible OpenAI API account.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
# Copy .env.example to .env, then set OPENAI_API_KEY.
# Streamlit does not automatically load .env, so either export the environment
# variable in your shell or use .streamlit/secrets.toml as described below.

# PowerShell example:
$env:OPENAI_API_KEY="your-real-key"
# macOS/Linux example:
export OPENAI_API_KEY="your-real-key"

# 4. Start the app
streamlit run app.py
```

Alternatively, create `.streamlit/secrets.toml` locally:

```toml
OPENAI_API_KEY = "your-real-key"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
```

Do not commit `.streamlit/secrets.toml` or a real `.env` file.

## Deploy on Streamlit Community Cloud

1. Create a new **public GitHub repository** and upload these project files.
2. Open <https://share.streamlit.io/> and sign in with GitHub.
3. Create an app, select the repository and branch, and set the main file to `app.py`.
4. In the app's **Settings → Secrets**, add:

   ```toml
   OPENAI_API_KEY = "your-real-key"
   OPENAI_CHAT_MODEL = "gpt-4o-mini"
   OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
   ```

5. Deploy and test using a non-sensitive PDF.
6. Monitor API usage and configure spending limits/alerts with your provider where available.

Never put an API key directly in `app.py`, screenshots, a public repository, or the README.

## Suggested demo questions

Use a public, text-based PDF with enough content to answer the questions below:

- “What is the document's main purpose?”
- “Summarize the three most important points.”
- “What deadline or date does the document mention?”
- “Which section discusses [a specific topic]?”
- An unrelated question such as “What is the population of Mars?” to test whether the app declines when it retrieves no relevant evidence.

The unrelated question is a demonstration of the retrieval threshold, not a guarantee that every unsupported question will be detected. Tune the threshold against a small evaluation set.

## Suggested evaluation before showcasing

Create 10–20 questions from your test PDF. Record whether the expected passage appears in the retrieved top-k, whether the answer is supported by the passage, and whether citations point to the right file/page. Report actual results in your portfolio; do not claim unmeasured accuracy.

## Limitations and responsible use

- This is a prototype, not a production-secure multi-user document service.
- The vector index exists in the current Streamlit session and is not a persistent database.
- Scanned PDFs may require OCR before upload.
- PDF text extraction can miss tables, layout, footnotes, or complex formatting.
- Similarity scores are not calibrated confidence probabilities; a threshold alone cannot guarantee factual correctness.
- Verify important answers against the source documents. Do not upload confidential client or personal documents to a public demo.
- API usage may incur charges.

## Stack

Python · Streamlit · pypdf · OpenAI embeddings/chat API · NumPy cosine similarity

## Portfolio description

**Chat with Your PDFs — Source-Grounded AI Document Assistant**

Built a Streamlit application that lets users upload PDFs and ask natural-language questions about their contents. The app extracts text page by page, creates overlapping chunks, embeds them, retrieves semantically similar passages, and generates answers with source labels. It exposes retrieved passages and similarity scores for inspection and includes a retrieval guardrail for questions lacking relevant evidence.

**Skills demonstrated:** Python, Streamlit, LLM API integration, embeddings, semantic search, RAG architecture, PDF processing, source attribution, and evaluation-aware design.

Replace this description with what you personally implemented and tested, and include real evaluation results only after measuring them.
