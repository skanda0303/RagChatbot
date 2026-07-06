# RagChatbot

A high-performance RAG (Retrieval-Augmented Generation) Chatbot leveraging FastAPI, Uvicorn, LangChain, ChromaDB, and Gemini models.

## Features

- **Hybrid Search**: Integrates BM25 and Vector Search (using BGE-M3 embeddings) with weighted results.
- **Agentic Search**: Equipped with Google Search and Tavily Web Search tools to fetch live information when local retrieval is insufficient.
- **FastAPI Endpoint**: Exposes the chatbot capabilities through a fast, lightweight HTTP server.

## Structure

- `ragbot/` - Core Python package containing agent, API, config, retriever, tools, and ingestion configurations.
- `.gitignore` - Standard rules to ignore virtual environments, cache files, database folders, and environment files.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/skanda0303/RagChatbot.git
   cd RagChatbot
   ```

2. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   python -m ragbot.main
   ```
