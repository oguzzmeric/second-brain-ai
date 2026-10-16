# Second Brain AI V2: Agentic RAG Assistant

Upload your PDFs and chat with them. An LLM agent decides, for each question, whether to search your documents, list what is uploaded, or go to the web, and answers with the retrieved context.

**Live demo:** [second-brain-ai-i3.streamlit.app](https://second-brain-ai-i3.streamlit.app/)

## How it works

1. **Guard routing:** a small LLM call classifies the input as simple chat or a real question. Simple chat is answered directly (GPT-3.5-Turbo); questions go to the agent (GPT-4o-mini).
2. **Tool-calling agent (LangChain):** the agent has four tools and is instructed to check the user's documents first:
   - `search_local_pdf`: semantic search over the user's ChromaDB collection (top-k = 20, so answers can combine several documents)
   - `list_documents`: lists the uploaded files, so the agent never guesses what is in the database
   - `web_search` (DuckDuckGo) and `read_web_page`: used only for general questions the documents don't cover
3. **Per-session workspace:** each browser session gets its own temporary folder for uploaded PDFs and its own vector store. A "clear data" button deletes the session's files and vectors.
4. **Ingestion:** PDFs are split into chunks and embedded with OpenAI `text-embedding-3-small`.

## Tech stack

| Layer | Technology |
| :--- | :--- |
| LLMs | OpenAI GPT-4o-mini (agent), GPT-3.5-Turbo (routing, small talk) |
| Orchestration | LangChain tool-calling agent |
| Vector store | ChromaDB (per session) |
| Embeddings | OpenAI `text-embedding-3-small` |
| UI / hosting | Streamlit, Streamlit Community Cloud |

## Run locally

```bash
pip install -r requirements.txt
echo OPENAI_API_KEY=your_key_here > .env
streamlit run app2.py
```

## Project structure

- `app2.py`: Streamlit UI, session workspace, upload and clear-data logic
- `brain2.py`: guard routing, agent, tools
- `ingest2.py`: PDF loading, chunking, embedding, ChromaDB build
- `evaluate_rag.py`: RAGAS evaluation (context recall, answer relevancy) with a local Llama 3 judge via Ollama. Written for the earlier e5-embedding version; it needs to be updated to the current OpenAI embeddings before its scores are meaningful.

## Version notes

V1 used HuggingFace `intfloat/multilingual-e5-base` embeddings computed on the server. V2 moved embeddings to the OpenAI API, which made ingestion much faster on the small free-tier server.

---

**Author:** [Oğuz Umut Meriç](https://www.linkedin.com/in/o%C4%9Fuz-meri%C3%A7-983a6b218/)
