import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

_oai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
_tav = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

def chat_llm(messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
    if not _oai:
        return "OpenAI API key not set. Please configure OPENAI_API_KEY."
    resp = _oai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

def web_search_and_summarize(query: str) -> str:
    if not _tav:
        return "Web search is not configured. Please set TAVILY_API_KEY."
    # Search
    results = _tav.search(query=query, search_depth="advanced", max_results=5)
    # Prepare plain text without URLs
    docs_text = []
    for r in results.get("results", []):
        snippet = r.get("content") or r.get("snippet") or ""
        title = r.get("title") or ""
        pub = r.get("published_date") or r.get("date") or ""
        piece = f"Title: {title}\nDate: {pub}\nSnippet: {snippet}".strip()
        docs_text.append(piece)
    corpus = "\n\n---\n\n".join(docs_text) if docs_text else "No results."

    # Summarize (no links/citations), include dates/numbers if present
    summarizer_sys = open("prompts/web_summarizer.txt", "r", encoding="utf-8").read()
    messages = [
        {"role": "system", "content": summarizer_sys},
        {"role": "user", "content": f"Query: {query}\n\nMaterials:\n{corpus}"}
    ]
    return chat_llm(messages, temperature=0.2)
