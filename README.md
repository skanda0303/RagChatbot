# RagChatbot

A high-performance agentic RAG (Retrieval-Augmented Generation) Chatbot engine. It utilizes advanced document retrieval techniques, agentic tool workflows, and a FastAPI-based server wrapper.

## Key Features

- **Hybrid Search & Retrieval**: Combines BM25 lexical search and Dense Vector Search (via `BGE-M3` embeddings and ChromaDB) using dynamic weighting parameters to retrieve highly relevant context.
- **Agentic Search Fallback**: Equipped with web search tools (Google Search and Tavily Web Search) allowing the LLM agent to fetch live, external information when internal documents don't satisfy the query.
- **FastAPI-powered Engine**: Exposes the chatbot conversation capabilities via a production-ready, lightweight HTTP server framework.

## Project Structure

- `ragbot/` - Core package directory containing:
  - `agent.py`: Initialises and configures the LangChain agent.
  - `api.py`: FastAPI endpoints and application setup.
  - `config.py`: Central configurations for model choice, weights, temp, and server options.
  - `ingestion.py`: Prepares, chunks, and indexes local files to ChromaDB.
  - `main.py`: Entry point server script.
  - `retriever.py`: Hybrid retriever configuration.
  - `tools.py`: Search and context retrieval tools.
