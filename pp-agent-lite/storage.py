# storage.py
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

DATA_DIR = Path("data")
CONV_DIR = DATA_DIR / "conversations"
MEM_FILE = DATA_DIR / "memory.json"


# -------------------------- filesystem helpers --------------------------

def ensure_dirs():
    """Ensure data folders & memory store exist."""
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    MEM_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not MEM_FILE.exists():
        MEM_FILE.write_text(
            json.dumps({"names": []}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

def _now_slug() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def _safe_title(title: str) -> str:
    """
    Sanitize a user-provided title for filesystem use.
    - Strip surrounding whitespace & dots
    - Replace disallowed chars with '_'
    - Keep length reasonable
    """
    t = (title or "").strip()
    if not t:
        return _now_slug()
    # Replace any non portable characters
    t = re.sub(r'[^A-Za-z0-9 _.\-]+', '_', t)
    t = t.strip(" .")
    # Avoid empty after cleanup
    if not t:
        t = _now_slug()
    # Limit to 80 chars
    return t[:80]

def _unique_path_for_title(title: str) -> Path:
    """
    Build a non-colliding path under CONV_DIR for the given title.
    If "<title>.jsonl" exists, append "-<timestamp>".
    """
    base = _safe_title(title)
    p = CONV_DIR / f"{base}.jsonl"
    if not p.exists():
        return p
    return CONV_DIR / f"{base}-{_now_slug()}.jsonl"


# -------------------------- conversation I/O --------------------------

def list_conversations() -> List[Path]:
    """
    Return all conversation files sorted by mtime (newest first), then name.
    """
    files = list(CONV_DIR.glob("*.jsonl"))
    files.sort(key=lambda fp: (-(fp.stat().st_mtime), fp.name))
    return files

def new_conversation() -> Path:
    """
    Create a new conversation file with a meta header line.
    Returns the file Path.
    """
    fname = _now_slug() + ".jsonl"
    fp = CONV_DIR / fname
    fp.touch()
    meta = {"role": "system", "content": "Conversation started.", "ts": time.time()}
    fp.write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
    return fp

def load_conversation(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load a conversation, skipping the first meta line.
    Returns a list of message dicts.
    """
    if not file_path.exists():
        return []
    msgs: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                msgs.append(json.loads(line))
            except Exception:
                # Skip malformed lines instead of crashing the UI
                pass
    return msgs[1:] if msgs else []

def append_message(file_path: Path, message: Dict[str, Any]) -> None:
    """
    Append a single message (dict) as JSONL to the conversation file.
    """
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")

def conversation_title_from_file(file_path: Path) -> str:
    """
    Title = filename stem (safe & stable).
    """
    return file_path.stem

def rename_conversation_file(file_path: Path, new_title: str) -> Path:
    """
    Rename the conversation to <new_title>.jsonl (sanitized).
    If a file with the target name exists, a timestamp suffix is added.
    Returns the new Path.
    """
    if not file_path.exists():
        # Nothing to rename; return original path
        return file_path

    target = _unique_path_for_title(new_title)
    try:
        file_path.rename(target)
    except Exception:
        # On some FS (or if moving across devices), do a copy+unlink fallback
        data = file_path.read_bytes()
        target.write_bytes(data)
        try:
            file_path.unlink()
        except Exception:
            pass
    return target

def delete_conversation_file(file_path: Path) -> None:
    """
    Delete the given conversation file. Silently ignore if missing.
    """
    try:
        file_path.unlink()
    except FileNotFoundError:
        pass


# ------------------------------ memory I/O ------------------------------

def _load_memory() -> Dict[str, Any]:
    ensure_dirs()
    try:
        return json.loads(MEM_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"names": []}

def _save_memory(mem: Dict[str, Any]) -> None:
    ensure_dirs()
    MEM_FILE.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")

def _extract_probable_names(text: str) -> List[str]:
    """
    Very lightweight name spotter:
    - Sequences of 2–3 capitalized words (John, Jane Doe, Acme Corp)
    - Filters very common words
    This is intentionally simple to avoid introducing LLM calls here.
    """
    COMMON = {
        "I", "We", "My", "Our", "You", "He", "She", "They",
        "The", "A", "An", "And", "Or", "For", "To", "Of",
        "Production", "Planning", "Agent", "Week", "Month",
        "Oven", "Labor", "Mixer", "Packaging", "Machine", "Line"
    }
    # Normalize fancy quotes/dashes that might confuse regex
    t = text.replace("’", "'").replace("“", '"').replace("”", '"')
    # Grab 2–3 consecutive Capitalized tokens
    cand = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", t)
    out: List[str] = []
    seen = set()
    for c in cand:
        tokens = c.split()
        # Drop single common words (e.g., "The")
        if len(tokens) == 1 and c in COMMON:
            continue
        # Keep short multi-word or brand-like singletons
        key = c.strip()
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out

def update_memory_with_text(text: str) -> None:
    """
    Append any newly observed names to memory.json.
    Used by the UI after each user turn.
    """
    mem = _load_memory()
    have = set(mem.get("names") or [])
    new_names = [n for n in _extract_probable_names(text) if n not in have]
    if not new_names:
        return
    # Keep memory bounded to avoid unbounded growth
    all_names = (mem.get("names") or []) + new_names
    # De-dup while preserving order (existing first)
    ordered: List[str] = []
    seen = set()
    for n in all_names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    # Cap at 300 names (oldest first)
    if len(ordered) > 300:
        ordered = ordered[-300:]
    mem["names"] = ordered
    _save_memory(mem)
