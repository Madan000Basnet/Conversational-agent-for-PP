# structured_extract.py
from __future__ import annotations
import os, json
from typing import Dict, Any, List, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _jsonschema_for_type(t: str) -> Dict[str, Any]:
    # Map schema types to JSON Schema fragments
    if t == "list[string]":
        return {"type":"array","items":{"type":"string"}}
    if t == "boolean":
        return {"type":"boolean"}
    if t == "map[string, number]":
        return {"type":"object","additionalProperties":{"type":"number"}}
    if t == "map[string, map[string, number]]":
        return {"type":"object","additionalProperties":{"type":"object","additionalProperties":{"type":"number"}}}
    # default fallback
    return {"type":"string"}

def _build_extract_tool_from_schema() -> Dict[str, Any]:
    schema_path = os.path.join(os.path.dirname(__file__), "kg", "pp_schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    props: Dict[str, Any] = {}
    for v in s.get("variables", []):
        name = v["name"]
        vtype = v.get("type", "string")
        props[name] = _jsonschema_for_type(vtype)
    return {
        "type": "function",
        "function": {
            "name": "extract_pp",
            "description": "Extract production planning fields from arbitrary user text.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": props
            }
        }
    }

EXTRACT_TOOL = _build_extract_tool_from_schema()

SYSTEM_PROMPT = """You extract structured data for production planning.
- Read arbitrary user text (bullets, paragraphs, tables).
- If a field is UNKNOWN, omit it; do not guess.
- Normalize periods: keep tokens like W1, W2, M1, D1; if the text says "one week"/"2 weeks", map to ["W1"] / ["W1","W2"].
- Normalize resource nouns to short tokens (e.g., oven, labor, mixer). Do not use 'ingredients' as a capacity resource.
- Keep product names as given, short and human-friendly.
- Use numeric values (floats); parse currency symbols and units.
- Call the tool with only the fields you can extract. No prose."""

def _normalize_periods(periods: List[str]) -> List[str]:
    out = []
    for p in periods or []:
        s = str(p).strip()
        if not s: 
            continue
        sU = s.upper()
        if sU.endswith(" WEEKS"):
            try:
                n = int(sU.split()[0])
                out.extend([f"W{i+1}" for i in range(n)])
                continue
            except: 
                pass
        if sU in {"ONE WEEK","1 WEEK"}:
            out.append("W1")
            continue
        out.append(sU)
    # dedupe, keep order
    seen=set(); norm=[]
    for t in out:
        if t not in seen:
            seen.add(t); norm.append(t)
    return norm

def _num(v):
    try:
        if isinstance(v, (int,float)): return float(v)
        s = str(v).replace("$","").replace(",","").strip()
        return float(s)
    except:
        return None

def _num_map(d: Dict[str, Any]) -> Dict[str, float]:
    out={}
    for k,v in (d or {}).items():
        n=_num(v)
        if n is not None: out[str(k)] = n
    return out

def _nested_num_map(d: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out={}
    for k, row in (d or {}).items():
        inner={}
        for kk, vv in (row or {}).items():
            n=_num(vv)
            if n is not None: inner[str(kk)] = n
        if inner: out[str(k)] = inner
    return out

def _postprocess(obj: Dict[str, Any]) -> Dict[str, Any]:
    res: Dict[str, Any] = {}

    # products
    if isinstance(obj.get("products"), list):
        prods=[]; seen=set()
        for p in obj["products"]:
            s=str(p).strip(" .;:,")
            if s and 1 <= len(s.split()) <= 4:
                sl=s.lower()
                if sl not in seen:
                    seen.add(sl); prods.append(s)
        if prods: res["products"]=prods

    # periods
    if isinstance(obj.get("periods"), list):
        per=_normalize_periods(obj["periods"])
        if per: res["periods"]=per

    # resources (whitelist)
    if isinstance(obj.get("resources"), list):
        vocab = ["oven","labor","mixer","packaging","machine","line","worker","staff"]
        found=[]; seen=set()
        for r in obj["resources"]:
            tok=str(r).strip().lower()
            if tok.endswith("s") and tok[:-1] in vocab: tok=tok[:-1]
            if tok in vocab and tok not in seen:
                seen.add(tok); found.append(tok)
        if found: res["resources"]=found

    # numeric maps
    if "profit"   in obj: res["profit"]   = _num_map(obj["profit"])
    if "usage"    in obj: res["usage"]    = _nested_num_map(obj["usage"])
    if "capacity" in obj: res["capacity"] = _nested_num_map(obj["capacity"])
    if "demand"   in obj: res["demand"]   = _nested_num_map(obj["demand"])
    if "max_batches" in obj: res["max_batches"] = _num_map(obj["max_batches"])
    if "setup_cost"  in obj: res["setup_cost"]  = _num_map(obj["setup_cost"])
    if "min_batch"   in obj: res["min_batch"]   = _num_map(obj["min_batch"])

    if isinstance(obj.get("allow_backlog"), bool):
        res["allow_backlog"]=obj["allow_backlog"]

    # drop empties
    return {k:v for k,v in res.items() if v not in (None, "", [], {}, )}

def extract_pp_fields(text: str, hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    LLM-based extractor. Returns a partial dict with any fields it confidently found.
    Gracefully returns {} if OpenAI lib or key is missing.
    """
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return {}

    client = OpenAI()
    messages = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content": text}
    ]
    if hints:
        messages.append({"role":"user","content": json.dumps({"hints":hints}, ensure_ascii=False)})

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=[EXTRACT_TOOL],
            tool_choice={"type":"function","function":{"name":"extract_pp"}}
        )
        tcalls = resp.choices[0].message.tool_calls or []
        if not tcalls:
            return {}
        args = tcalls[0].function.arguments or "{}"
        raw = json.loads(args)
    except Exception:
        return {}

    return _postprocess(raw)
