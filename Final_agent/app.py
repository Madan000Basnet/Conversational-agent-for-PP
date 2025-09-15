#app.py
import os
import json
import time
from pathlib import Path
import streamlit as st

# --- load .env BEFORE anything else that might read env vars ---
try:
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv(usecwd=True), override=True)
    _ = load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)
    _ = load_dotenv(dotenv_path=Path(__file__).parent / "config/.env", override=False)
except Exception:
    pass

from graph import build_graph, GraphState
from storage import (
    ensure_dirs, list_conversations, new_conversation, load_conversation,
    append_message, rename_conversation_file, conversation_title_from_file
)

# Optional delete helper (fallback to filesystem if storage doesn't provide it)
try:
    from storage import delete_conversation_file  # expected signature: (path: Path) -> None
except Exception:
    def delete_conversation_file(path: Path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

# Optional memory extraction helper (safe if missing)
try:
    from storage import update_memory_with_text
except Exception:
    def update_memory_with_text(_):  # no-op fallback
        return

APP_TITLE = "Production Planning Agent"
DATA_DIR = Path("data")
CONV_DIR = DATA_DIR / "conversations"


def init_session():
    ensure_dirs()
    if "convo_file" not in st.session_state:
        st.session_state.convo_file = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pp_state" not in st.session_state:
        st.session_state.pp_state = {}  # collected PP variables
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()
    if "selected_convo_label" not in st.session_state:
        st.session_state.selected_convo_label = None


def load_selected_convo(file_path: Path):
    st.session_state.convo_file = file_path
    st.session_state.messages = load_conversation(file_path)
    # restore pp_state from last assistant tool message if present
    st.session_state.pp_state = {}
    for m in reversed(st.session_state.messages):
        if m.get("role") == "assistant" and m.get("meta", {}).get("pp_state") is not None:
            st.session_state.pp_state = m["meta"]["pp_state"]
            break


def _list_convo_options():
    """Return (labels, mapping) where labels is a list[str] for selectbox, mapping label->Path."""
    files = list_conversations()  # list[Path]
    # Sort newest first (by mtime), then by name
    files = sorted(files, key=lambda p: (-(p.stat().st_mtime), p.name))
    labels = []
    mapping = {}
    for f in files:
        title = conversation_title_from_file(f)
        label = f"{title} · {f.name}"
        # guarantee uniqueness in case of same title/name collision
        i = 1
        base_label = label
        while label in mapping:
            i += 1
            label = f"{base_label} ({i})"
        labels.append(label)
        mapping[label] = f
    return labels, mapping


def sidebar():
    st.sidebar.header("Conversations")

    # Diagnostics for keys + engine
    ok_openai = bool(os.getenv("OPENAI_API_KEY"))
    ok_tavily = bool(os.getenv("TAVILY_API_KEY"))
    last_engine = None
    for m in reversed(st.session_state.messages):
        if m.get("role") == "assistant" and m.get("meta", {}).get("engine"):
            last_engine = m["meta"]["engine"]
            break


    # Create / Reset
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("New", use_container_width=True):
            file_path = new_conversation()
            load_selected_convo(file_path)
            # update the dropdown selection to the new file
            title = conversation_title_from_file(file_path)
            st.session_state.selected_convo_label = f"{title} · {file_path.name}"
            st.rerun()
    with col2:
        if st.button("Reset", use_container_width=True):
            file_path = new_conversation()
            load_selected_convo(file_path)
            title = conversation_title_from_file(file_path)
            st.session_state.selected_convo_label = f"{title} · {file_path.name}"
            st.rerun()

    st.sidebar.markdown("---")

    # Dropdown of all conversations
    labels, mapping = _list_convo_options()
    placeholder = "— Select a conversation —"
    options = [placeholder] + labels

    # Compute default index for selectbox
    default_idx = 0
    if st.session_state.selected_convo_label and st.session_state.selected_convo_label in labels:
        default_idx = 1 + labels.index(st.session_state.selected_convo_label)
    elif st.session_state.convo_file:
        # try to map current file back to a label
        current_lbl = f"{conversation_title_from_file(st.session_state.convo_file)} · {st.session_state.convo_file.name}"
        if current_lbl in labels:
            default_idx = 1 + labels.index(current_lbl)

    selected_label = st.sidebar.selectbox(
        "Open conversation",
        options=options,
        index=default_idx,
        key="selectbox_convo",
    )

    # Load when selection changes
    if selected_label and selected_label != placeholder:
        if selected_label != st.session_state.get("selected_convo_label"):
            st.session_state.selected_convo_label = selected_label
            load_selected_convo(mapping[selected_label])
            st.rerun()

    # Rename / Delete for current conversation
    st.sidebar.markdown("---")
    if st.session_state.convo_file:
        current_label = conversation_title_from_file(st.session_state.convo_file)
        new_name = st.sidebar.text_input("Rename current conversation", value=current_label, key="rename_input")
        colr1, colr2 = st.sidebar.columns([1, 1])

        with colr1:
            if st.button("Rename", use_container_width=True):
                old_path = st.session_state.convo_file
                # Prefer a return value (new Path) if storage provides one
                try:
                    new_path = rename_conversation_file(old_path, new_name)  # expected to return Path (preferred)
                    if new_path is None:
                        raise TypeError("rename returned None")
                except TypeError:
                    # Fallback: assume .jsonl naming convention
                    new_path = CONV_DIR / f"{new_name}.jsonl"
                    # If storage didn't actually move it, try renaming ourselves
                    if not new_path.exists():
                        try:
                            old_path.rename(new_path)
                        except Exception:
                            pass

                if new_path.exists():
                    load_selected_convo(new_path)
                    # update dropdown selection
                    st.session_state.selected_convo_label = f"{conversation_title_from_file(new_path)} · {new_path.name}"
                    st.success("Conversation renamed.")
                    st.rerun()
                else:
                    st.error("Rename failed: target file not found after rename.")

        with colr2:
            confirm = st.checkbox("Confirm delete", key="confirm_delete")
            if st.button("Delete", type="primary", use_container_width=True, disabled=not confirm):
                path_to_delete = st.session_state.convo_file
                delete_conversation_file(path_to_delete)
                # Clear current session state & refresh list
                st.session_state.convo_file = None
                st.session_state.messages = []
                st.session_state.pp_state = {}
                st.session_state.selected_convo_label = None
                st.success("Conversation deleted.")
                st.rerun()
    else:
        st.sidebar.info("Create or load a conversation.")


def render_messages():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


def render_filled_vars():
    st.markdown("#### Filled variables (PP)")
    if not st.session_state.pp_state:
        st.info("No PP variables collected yet.")
        return
    # Hide internal control keys like _awaiting/_enriching
    to_show = {k: v for k, v in st.session_state.pp_state.items() if not str(k).startswith("_")}
    st.json(to_show)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    init_session()
    sidebar()

    left, right = st.columns([2, 1])

    with left:
        render_messages()
        user_input = st.chat_input("Type your message and press Enter…")
        if user_input:
            # Ensure a conversation file exists
            if not st.session_state.convo_file:
                file_path = new_conversation()
                load_selected_convo(file_path)
                title = conversation_title_from_file(file_path)
                st.session_state.selected_convo_label = f"{title} · {file_path.name}"

            # Save user text + update memory (names, etc.)
            user_msg = {"role": "user", "content": user_input, "ts": time.time()}
            st.session_state.messages.append(user_msg)
            append_message(st.session_state.convo_file, user_msg)
            try:
                update_memory_with_text(user_input)
            except Exception:
                pass

            # Run graph
            state: GraphState = {
                "messages": st.session_state.messages,
                "pp_state": st.session_state.pp_state,
                "mode": None,
                "reply": None,
                # "engine" is optional; graph may or may not set it
            }
            result = st.session_state.graph.invoke(state)
            reply = result.get("reply") or "(No reply.)"
            st.session_state.pp_state = result.get("pp_state", st.session_state.pp_state)

            # Append assistant msg (save pp_state snapshot + engine label into meta)
            assistant_msg = {
                "role": "assistant",
                "content": reply,
                "ts": time.time(),
                "meta": {"pp_state": st.session_state.pp_state, "engine": result.get("engine")}
            }
            st.session_state.messages.append(assistant_msg)
            append_message(st.session_state.convo_file, assistant_msg)

            # Rerender
            st.rerun()

    with right:
        render_filled_vars()


if __name__ == "__main__":
    main()
