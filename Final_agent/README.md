# PP Agent (Lite) — LangGraph + Streamlit

Minimal, working prototype:
- **LangGraph** router → NORMAL / WEB_SEARCH (Tavily) / PP (one-question-at-a-time).
- **LLM**: OpenAI (concise responses).
- **Web Search**: Tavily → LLM summary (no links/citations).
- **PP**: Knowledge-driven questioning (no prefilled data), PuLP optimization.
- **Memory/Storage**: Local JSONL per conversation; simple long-lived memory file.
- **UI**: Streamlit chat + sidebar (new/load/rename/reset) + filled-variables panel.

## Quickstart

1. **Install**
   ```bash
   python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Configure**
   ```bash
   cp config/.env.example .env
   # fill OPENAI_API_KEY and TAVILY_API_KEY
   ```

3. **Run**
   ```bash
   streamlit run app.py
   ```

## Usage Tips
- Ask anything normally → normal chat.
- Ask “latest price of …”, “today’s weather in …”, “news about …” → web search mode (summary only).
- Talk PP: e.g., “Let’s optimize a weekly bakery plan.”
  The agent will ask for missing items (one at a time), then summarize and wait for “optimize”.


