# pp.py
from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

from modes import needs_objective, parse_objective
from services import chat_llm
from solvers import optimize

# Optional LLM struct extractor (safe fallback if absent)
try:
    from structured_extract import extract_pp_fields
except Exception:  # pragma: no cover
    extract_pp_fields = None  # type: ignore

# =========================
# Constants & prompts
# =========================

PROMPTS_DIR = Path(__file__).parent / "prompts"
PP_QUESTIONER_PATH = PROMPTS_DIR / "pp_questioner.txt"
try:
    QUESTIONER_PROMPT = PP_QUESTIONER_PATH.read_text(encoding="utf-8")
except Exception:
    QUESTIONER_PROMPT = (
        "You are a warm, friendly production-planning assistant. "
        "Ask one concise, polite question at a time to collect the next needed variable. "
        "Use the provided 'hint'. Keep questions short and kind. Avoid multi-part questions."
    )

OBJ_QUESTION = (
    "Happy to help with production planning! 🌟\n"
    "To get us on the right track, is your main objective **cost minimization** or **profit maximization**?"
)

# Friendly bridges
ACK_OK = [
    "Got it!",
    "Perfect, thanks!",
    "Thanks — noted.",
    "Great, thanks!",
]

# Intent lexicon for robust user parsing
INTENT_LEXICON = {
    "optimize": {"optimize", "optimise", "run", "solve", "go", "execute"},
    "skip": {
        "skip", "pass", "idk", "i dont know", "i don't know", "dont know",
        "don’t know", "no idea", "unknown", "not sure", "leave it", "next",
        "i don’t have this", "i dont have this", "don’t have", "dont have",
        "no info", "no information"
    },
    "done": {"done", "finished", "that’s all", "thats all", "no more", "nope", "none", "nothing else"},
    "yes": {"yes", "yep", "yeah", "sure", "ok", "okay", "affirmative"},
    "no": {"no", "nope", "negative"},
    "change": {"change", "update", "edit", "modify", "correct", "fix", "set"},
    "summary": {"summary", "summarize", "recap"},
}

# =========================
# Small value helpers
# =========================

def _is_empty(v: Any) -> bool:
    """Treat None, empty strings, and empty containers (including {}) as empty."""
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, (list, tuple, set, dict)) and len(v) == 0:
        return True
    return False


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _has_intent(text: str, intent: str) -> bool:
    t = _norm(text)
    for phrase in INTENT_LEXICON.get(intent, set()):
        if phrase in t:
            return True
    return False


def _first_intent(text: str, intents: List[str]) -> Optional[str]:
    for key in intents:
        if _has_intent(text, key):
            return key
    return None

# =========================
# Schema helpers  (use "variables" & "constraints" to match your pp_schema.json)
# =========================

def load_schema() -> Dict[str, Any]:
    path = Path(__file__).parent / "kg" / "pp_schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_vars(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(schema.get("variables", []))


def _iter_constraints(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(schema.get("constraints", []))


def _schema_hint(schema: Dict[str, Any], name: str) -> str:
    for f in _iter_vars(schema):
        if f.get("name") == name:
            return f.get("hint") or ""
    return ""


def _schema_type(schema: Dict[str, Any], name: str) -> str:
    for f in _iter_vars(schema):
        if f.get("name") == name:
            return f.get("type") or "string"
    return "string"


def _schema_fields_for_objective(schema: Dict[str, Any], objective: str) -> List[Dict[str, Any]]:
    """
    Select and prioritize fields to ask based on the chosen objective,
    always including core/minimal fields, and adding profit/cost coefficients.
    """
    fields = list(_iter_vars(schema))
    objective = (objective or "cost").lower()

    def wants(f: Dict[str, Any]) -> bool:
        name = (f.get("name") or "").lower()
        tier = (f.get("tier") or "minimal").lower()
        mod = (f.get("module") or "").lower()

        if tier == "minimal" or mod == "core":
            return True
        if objective == "profit" and name in {"profit", "revenue", "price"}:
            return True
        if objective == "cost" and ("cost" in name or name in {"setup_cost", "regular_cost", "holding_cost", "shortage_cost"}):
            return True
        return False

    keep = [f for f in fields if wants(f)]

    # rank so required/core/minimal beat plain 'order'
    def _rank_for(f: Dict[str, Any]) -> tuple:
        required_rank = 0 if f.get("required") else 1          # required first
        module_rank   = 0 if (f.get("module", "").lower() == "core") else 1
        tier_rank     = 0 if (f.get("tier", "minimal").lower() == "minimal") else 1
        order_rank    = int(f.get("order", 999))               # then by order
        return (required_rank, module_rank, tier_rank, order_rank)

    keep.sort(key=_rank_for)
    return keep


def _is_includable_field(f: Dict[str, Any]) -> bool:
    """
    Include fields when explicitly required/optional OR when flags are missing.
    This makes the queue robust if the schema omits those flags.
    """
    if f.get("required") is True:
        return True
    if f.get("optional") is True:
        return True
    if "required" not in f and "optional" not in f:
        return True
    return False

# =========================
# Question & ingestion helpers
# =========================

def _ask_one_question(field: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Use a short, friendly tone. One field at a time."""
    name = field.get("name")
    hint = (field.get("hint") or "").strip()
    type_str = field.get("type") or "string"

    sys_msg = QUESTIONER_PROMPT
    usr_msg = (
        f"Field name: {name}\n"
        f"Type: {type_str}\n"
        f"Hint: {hint}\n\n"
        "Write a one-sentence, kind, concrete question to collect this value. "
        "Avoid multi-part questions. Keep it short and helpful."
    )
    try:
        q = chat_llm(sys_msg, usr_msg)
        q = (q or "").strip()
    except Exception:
        q = ""

    if not q:
        # Friendly fallback
        q = f"Could you share **{name}**? {('Hint: ' + hint) if hint else ''}".strip()

    # Add a visible “skip” affordance
    q += "  \n(You can say **skip** if you don’t have this.)"
    return q


def _ingest_kv_from_text(text: str) -> Dict[str, Any]:
    """
    Very light extraction if structured_extract is not available.
    Recognizes patterns like:
      - name=value
      - set name to value
      - change name to value
    """
    out: Dict[str, Any] = {}
    t = text.strip()
    # name = value
    m = re.match(r"\s*([A-Za-z0-9_.\[\],\- ]+)\s*=\s*(.+)$", t)
    if m:
        out[m.group(1).strip()] = m.group(2).strip()
        return out

    # change/set <name> to <value>
    m2 = re.match(r"\s*(?:change|set|update|edit|modify)\s+(.+?)\s+(?:to|=)\s+(.+)$", t, flags=re.I)
    if m2:
        out[m2.group(1).strip()] = m2.group(2).strip()
        return out

    return out


def _apply_updates(state: Dict[str, Any], updates: Dict[str, Any]) -> None:
    """
    Apply {field: value} updates into pp_state.
    Simple top-level keys. If keys contain dots or bracket-like tokens,
    keep as strings (your solver can parse later if needed).
    """
    for k, v in updates.items():
        state[k] = v


def _friendly_summary(pp_state: Dict[str, Any]) -> str:
    """Summarize collected values (excluding internals)."""
    data = {k: v for k, v in pp_state.items() if not str(k).startswith("_")}
    if not data:
        return "I don’t have any details yet."
    lines = []
    lines.append("Here’s a quick summary of what I’ve collected so far:\n")
    for k, v in sorted(data.items()):
        lines.append(f"- **{k}**: {v}")
    lines.append("\nWould you like to **change** anything, or should I **optimize** now?")
    return "\n".join(lines)

# =========================
# Main PP handler (state machine)
# Phases:
#   1. objective
#   2. variables interview
#   3. constraints interview
#   4. finalize (summary → change or optimize)
# =========================

def handle_pp_turn(user_text: str, pp_state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Main PP turn handler implementing:
      - Warm objective prompt
      - Variable interview (schema-driven, one-by-one, with skip)
      - Constraint interview (optional, with skip)
      - “Any more info?” → summary → change/optimize choice
      - Change loop and optimize execution

    The state uses:
      _phase: "objective" | "variables" | "constraints" | "finalize"
      _awaiting: {"name": <field>, "type": <schema_type>}
      _enriching: {"queue": [field...], "idx": int}
      _constraints: {"queue": [name...], "idx": int}
    """
    text = (user_text or "").strip()

    # Global quick intents
    if _has_intent(text, "optimize"):
        try:
            msg, result = optimize(pp_state)
            pp_state["last_solution"] = result
            return msg, pp_state
        except Exception as e:
            return f"Optimization failed: {e}", pp_state

    # Parse objective if we’re currently asking it
    if (pp_state.get("_awaiting") or {}).get("name") == "objective":
        parsed = parse_objective(text)
        if parsed:
            pp_state["objective"] = "profit" if parsed.upper() == "PROFIT" else "cost"
            pp_state["_awaiting"] = {}
            # register modules (used by some setups)
            active = set(pp_state.get("_active_modules") or [])
            active.add("core")
            if pp_state["objective"] == "profit":
                active.add("profit")
            else:
                active.add("cost")
            pp_state["_active_modules"] = sorted(active)
            # move to variable phase after objective lock-in
            pp_state["_phase"] = "variables"
        else:
            # re-ask in a friendly way
            return OBJ_QUESTION, pp_state

    # Ensure phase
    phase = pp_state.get("_phase")
    if not phase:
        # If we already have an objective, go to variables; else ask objective
        if needs_objective(pp_state):
            pp_state["_phase"] = "objective"
            pp_state["_awaiting"] = {"name": "objective", "type": "enum[profit,cost]"}
            return OBJ_QUESTION, pp_state
        else:
            pp_state["_phase"] = "variables"

    # ==========
    # VARIABLES
    # ==========
    if pp_state["_phase"] == "variables":
        # Build queue once
        if not pp_state.get("_enriching"):
            schema = load_schema()
            fields = _schema_fields_for_objective(schema, pp_state.get("objective", "cost"))
            queue = [f["name"] for f in fields if _is_includable_field(f)]
            pp_state["_enriching"] = {"queue": queue, "idx": 0}

        enrich = pp_state["_enriching"]
        queue: List[str] = enrich.get("queue", [])
        idx = int(enrich.get("idx", 0))

        # Ingest current reply (if we were awaiting a field)
        awaiting = pp_state.get("_awaiting") or {}
        if awaiting.get("name"):
            # If user said skip-like, advance without saving
            if _has_intent(text, "skip"):
                pp_state["_awaiting"] = {}
            else:
                # Try structured extract first
                if extract_pp_fields:
                    try:
                        extracted = extract_pp_fields(text)
                        if isinstance(extracted, dict) and extracted:
                            for k, v in extracted.items():
                                pp_state[k] = v
                    except Exception:
                        pass
                # Fallback to naive kv
                updates = _ingest_kv_from_text(text)
                if updates:
                    _apply_updates(pp_state, updates)
                # As an ultra-simple fallback, if no kv recognized, store raw answer
                if not updates and not _has_intent(text, "done") and not _has_intent(text, "change"):
                    pp_state[awaiting["name"]] = text
            pp_state["_awaiting"] = {}

        # Ask next variable
        while idx < len(queue):
            nxt_name = queue[idx]
            # if already present, skip
            if not _is_empty(pp_state.get(nxt_name)):
                idx += 1
                continue

            schema = load_schema()
            field = {
                "name": nxt_name,
                "type": _schema_type(schema, nxt_name),
                "hint": _schema_hint(schema, nxt_name),
            }
            pp_state["_awaiting"] = {"name": nxt_name, "type": field["type"]}
            question = _ask_one_question(field, pp_state)
            pp_state["_enriching"]["idx"] = idx + 1
            return question, pp_state

        # Queue exhausted → move to constraints
        pp_state["_phase"] = "constraints"

    # =============
    # CONSTRAINTS
    # =============
    if pp_state["_phase"] == "constraints":
        # Build constraints queue once
        if not pp_state.get("_constraints"):
            schema = load_schema()
            cons = [c for c in _iter_constraints(schema)]
            pp_state["_constraints"] = {"queue": [c["name"] for c in cons], "idx": 0}

        cons = pp_state["_constraints"]
        cqueue: List[str] = cons.get("queue", [])
        cidx = int(cons.get("idx", 0))

        # If we were awaiting a specific constraint value, ingest or skip
        awaiting = pp_state.get("_awaiting") or {}
        if awaiting.get("name") and awaiting.get("name").startswith("constraint:"):
            if _has_intent(text, "skip"):
                pp_state["_awaiting"] = {}
            else:
                updates = _ingest_kv_from_text(text)
                if updates:
                    _apply_updates(pp_state, updates)
                else:
                    # store raw if user gave a sentence
                    pp_state[awaiting["name"]] = text
            pp_state["_awaiting"] = {}

        # Ask next constraint (politely optional)
        while cidx < len(cqueue):
            cname = cqueue[cidx]
            # If user already provided something for this constraint, skip
            if not _is_empty(pp_state.get(f"constraint:{cname}")):
                cidx += 1
                continue

            # Build a soft hint using 'applies' from schema if present
            schema = load_schema()
            hint = ""
            for c in _iter_constraints(schema):
                if c.get("name") == cname:
                    applies = c.get("applies") or []
                    if applies:
                        hint = f"This often uses: {', '.join(applies)}."
                    break

            q = (
                f"If you have details for **{cname}** (constraint), please share them. "
                f"{hint}  \n"
                f"(Say **skip** if you’d like to leave it blank.)"
            )
            pp_state["_awaiting"] = {"name": f"constraint:{cname}", "type": "string"}
            pp_state["_constraints"]["idx"] = cidx + 1
            return q, pp_state

        # Constraints exhausted → finalize
        pp_state["_phase"] = "finalize"

    # =========
    # FINALIZE
    # =========
    if pp_state["_phase"] == "finalize":
        # Handle change requests
        if _has_intent(text, "change"):
            updates = _ingest_kv_from_text(text)
            if updates:
                _apply_updates(pp_state, updates)
                return "Updated! Would you like any more **changes**, or should I **optimize** now?", pp_state
            # If user just said "change" without details
            return "Sure — tell me what to change (e.g., `change demand = {...}` or `set price to {...}`).", pp_state

        # If user says done/no more → provide summary and offer optimize
        if _has_intent(text, "done") or _has_intent(text, "summary"):
            return _friendly_summary(pp_state), pp_state

        # If no prior summary given, proactively show summary and ask next step
        if not pp_state.get("_finalized_once"):
            pp_state["_finalized_once"] = True
            return (
                _friendly_summary(pp_state),
                pp_state,
            )

        # Default gentle nudge
        return "Would you like to **change** anything, or should I **optimize** now?", pp_state

    # Fallback (shouldn’t hit)
    return "Let’s continue. What would you like to do next — share more details, **change** a value, or **optimize**?", pp_state
