"""
agent.py — LLM setup, agent assembly, and query pipeline.

Initializes the Gemini LLM, assembles the LangChain agent with its
system prompt and tools, and exposes run_agent() which:
  1. Pre-fetches relevant RAG context and injects it into the prompt
  2. Runs the agent (may call web search / fetch tools as needed)
  3. Returns the final answer as a plain string
"""

import time
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from ragbot.config import LLM_MODEL, GOOGLE_API_KEY, LLM_TEMPERATURE, MAX_HISTORY_MESSAGES
from ragbot.tools import web_search, fetch_webpage_content, get_datetime, register_rag_context
from ragbot.retriever import filter_redundant_docs

# LLM
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=LLM_TEMPERATURE,
)

# Agent — only web tools; RAG context is injected directly into the prompt
agent_tools = [web_search, fetch_webpage_content, get_datetime]

agent_executor = create_agent(
    model=llm,
    tools=agent_tools,
    system_prompt=(
        f"You are a precise, fact-grounded assistant. "
        f"Current date is {datetime.now().strftime('%A, %B %d, %Y')}.\n\n"
        "SYNTHESIS RULES:\n"
        "1. Answer directly from facts in tool results or pre-fetched RAG context "
        "— never tell the user to visit a site.\n"
        "2. Be comprehensive and detailed. Use bullet points or bold text where helpful.\n"
        "3. Trust tool outputs over your training memory.\n"
        "4. Do not repeat identical search queries. You are encouraged to do multi-step "
        "searching or call 'fetch_webpage_content' on multiple URLs for detailed info.\n"
        "5. If you need detailed info from a URL returned by web_search, "
        "call 'fetch_webpage_content' with that URL.\n"
        "6. CHRONOLOGICAL TIMELINES: Verify dates carefully and list oldest to newest.\n"
        "7. WEB FALLBACK: If 'Pre-fetched RAG context' doesn't contain the answer or is blank, "
        "you MUST call 'web_search'. Never just say the text doesn't contain the info."
    ),
)


def initialise_agent(retriever, chunks: list) -> None:
    """Wire retriever and chunks into the tools module at startup."""
    register_rag_context(retriever, chunks)


def sanitize_tool_output(text: str) -> str:
    """Normalize whitespace and cap at 25k chars to prevent context overflow."""
    text = " ".join(text.split())
    if len(text) > 25000:
        text = text[:25000] + "... [truncated]"
    return text


async def run_agent(query: str, history_messages: list, chunks: list, retriever) -> str:
    """Process a user query: pre-fetch RAG context → run LLM agent → return answer."""
    print(f"[AGENT] Query: '{query}'")

    # Step 1: Pre-fetch RAG context
    rag_context  = ""
    has_rag_docs = False

    if chunks:
        try:
            t_ret = time.perf_counter()
            retrieved_docs = retriever.invoke(query)
            print(f"[AGENT] Retrieval: {time.perf_counter() - t_ret:.3f}s — {len(retrieved_docs)} docs")

            if retrieved_docs:
                seen, unique = set(), []
                for doc in retrieved_docs:
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        unique.append(doc)

                filtered = filter_redundant_docs(unique)
                if filtered:
                    has_rag_docs = True
                    formatted_chunks = [
                        doc.page_content.replace("●", "\n- ") for doc in filtered[:3]
                    ]
                    rag_context = "\n\n---\n\n".join(formatted_chunks)
                    print(f"[AGENT] Injecting {len(filtered[:3])} RAG chunks.")
        except Exception as e:
            print(f"[AGENT] Retrieval error: {e}")

    # Step 2: Build user input
    user_input = query
    if has_rag_docs:
        sanitized_rag = sanitize_tool_output(rag_context)
        user_input = (
            f"Pre-fetched RAG context:\n{sanitized_rag}\n\n"
            f"Now answer the query: {query}"
        )

    # Step 3: Run the agent
    try:
        inputs = {
            "messages": list(history_messages[-MAX_HISTORY_MESSAGES:])
            + [HumanMessage(content=user_input)]
        }
        response = await agent_executor.ainvoke(inputs)
        final_message = response["messages"][-1]
        content = final_message.content

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
            return "".join(text_parts)

        return str(content)
    except Exception as e:
        print(f"[AGENT ERROR] {e}")
        return f"An error occurred while executing the agent: {e}"
