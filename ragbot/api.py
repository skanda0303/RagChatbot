"""
api.py — FastAPI web server.

Endpoints:
  POST /chat  — streams AI response word-by-word
  POST /clear — clears chat history for a session
  GET  /      — serves the frontend (index.html)

Session history is persisted per-session in SQLite (memory3.db).
"""

import asyncio
import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from ragbot.config import MEMORY_DB
from ragbot.agent import run_agent

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def create_app(chunks: list, retriever) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Agentic RAG Chatbot — Port 8003")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static file routes
    @app.get("/production_tables.css")
    def serve_css():
        return FileResponse(os.path.join(_PROJECT_ROOT, "production_tables.css"))

    @app.get("/favicon.ico")
    def serve_favicon():
        path = os.path.join(_PROJECT_ROOT, "favicon.ico")
        if os.path.exists(path):
            return FileResponse(path)
        from fastapi.responses import Response
        return Response(status_code=204)

    # Request model
    class ChatRequest(BaseModel):
        session_id: str
        message: str

    @app.post("/chat")
    async def chat(req: ChatRequest):
        async def response_generator():
            history = SQLChatMessageHistory(
                session_id=req.session_id, connection_string=MEMORY_DB,
            )

            t_start = time.perf_counter()
            try:
                final_answer = await run_agent(
                    query=req.message,
                    history_messages=history.messages,
                    chunks=chunks,
                    retriever=retriever,
                )
            except Exception as e:
                print(f"[ERROR] run_agent: {e}")
                final_answer = "Sorry, I encountered an internal error. Please try again."

            print(f"[TIMER] Agent time: {time.perf_counter() - t_start:.3f}s")

            if not final_answer or not final_answer.strip():
                final_answer = "I'm sorry, I wasn't able to generate a response. Please try rephrasing."

            print(f"[STREAM] Streaming {len(final_answer)} chars.")

            words = final_answer.split(" ")
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                await asyncio.sleep(0)

            history.add_message(HumanMessage(content=req.message))
            history.add_message(AIMessage(content=final_answer))

        return StreamingResponse(response_generator(), media_type="text/plain; charset=utf-8")

    @app.post("/clear")
    async def clear_chat(req: ChatRequest):
        history = SQLChatMessageHistory(
            session_id=req.session_id, connection_string=MEMORY_DB,
        )
        history.clear()
        return {"status": "ok"}

    @app.get("/")
    def root():
        return FileResponse(os.path.join(_PROJECT_ROOT, "index.html"))

    return app
