from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from modes import detect_mode
from services import chat_llm, web_search_and_summarize
from pp import handle_pp_turn

class GraphState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    pp_state: Dict[str, Any]
    mode: str
    reply: str
    engine: str  

def _last_user_text(state: GraphState) -> str:
    last_user = next((m for m in reversed(state.get("messages", [])) if m.get("role") == "user"), None)
    return last_user["content"] if last_user else ""

def node_router(state: GraphState) -> GraphState:
    """Sticky PP routing: if PP is in progress or has collected fields, stay in PP."""
    pp = state.get("pp_state") or {}
    if (
        pp.get("_awaiting") or
        any(k in pp for k in ["products", "periods", "resources", "profit", "usage", "capacity"])
    ):
        state["mode"] = "PP"
        return state

    text = _last_user_text(state)
    state["mode"] = detect_mode(text)
    return state

def node_normal_chat(state: GraphState) -> GraphState:
    sys = open("prompts/system.txt", "r", encoding="utf-8").read()
    msgs = [{"role": "system", "content": sys}] + state.get("messages", [])[-8:]
    answer = chat_llm(msgs)
    state["reply"] = answer
    state["engine"] = "llm-normal"
    return state

def node_web_search(state: GraphState) -> GraphState:
    query = _last_user_text(state)
    summary = web_search_and_summarize(query)
    state["reply"] = summary
    state["engine"] = "web+llm"
    return state

def node_pp(state: GraphState) -> GraphState:
    user_text = _last_user_text(state)
    reply, new_pp_state = handle_pp_turn(user_text, state.get("pp_state") or {})
    state["pp_state"] = new_pp_state
    state["reply"] = reply
    # If a solution exists, we know the solver ran
    state["engine"] = "pp-solver" if "solution" in (new_pp_state or {}) else "pp-collect"
    return state

def route_edges(state: GraphState) -> str:
    mode = state.get("mode", "NORMAL")
    if mode == "WEB_SEARCH":
        return "web"
    if mode == "PP":
        return "pp"
    return "normal"

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("router", node_router)
    g.add_node("normal", node_normal_chat)
    g.add_node("web", node_web_search)
    g.add_node("pp", node_pp)

    g.set_entry_point("router")
    g.add_conditional_edges("router", route_edges, {
        "normal": "normal",
        "web": "web",
        "pp": "pp",
    })
    g.add_edge("normal", END)
    g.add_edge("web", END)
    g.add_edge("pp", END)
    return g.compile()
