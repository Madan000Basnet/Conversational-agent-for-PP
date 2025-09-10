# extract.py
import json
import re
from typing import Dict, List, Optional, Tuple

# ---------------------- small utilities ----------------------

_PERIOD_WORD2NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12
}

def _strip_quotes(s: str) -> str:
    return s.strip().strip("\"'“”‘’").strip()

def _to_float(tok: str) -> Optional[float]:
    """
    Parse a numeric token that may include currency symbols ($), thousands separators,
    and simple units (hours, h, hr, units, batches).
    """
    if tok is None:
        return None
    t = tok.strip().lower()
    # remove currency and percent signs, and trivial unit words
    t = re.sub(r"[$,%]", "", t)
    t = re.sub(r"\b(hours?|hrs?|h|units?|batches?)\b", "", t, flags=re.IGNORECASE)
    t = t.replace(",", " ")
    # keep digits, dot, minus
    m = re.findall(r"-?\d+(?:\.\d+)?", t)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None

def _split_candidates(segment: str) -> List[str]:
    parts = re.split(r"\s*(?:,| and | & |/|\|)\s*", segment, flags=re.IGNORECASE)
    items = []
    for p in parts:
        p = p.strip(" .;:,")
        if not p:
            continue
        if len(p) > 60 or len(p.split()) > 10:
            continue
        items.append(p)
    seen = set(); out = []
    for x in items:
        xl = x.lower()
        if xl not in seen:
            seen.add(xl); out.append(x)
    return out

def _extract_list_from_narrative(text: str, keyword: Optional[str] = None) -> List[str]:
    t = text.strip()
    patterns = []
    if keyword:
        patterns += [
            rf"{keyword}s?\s*(?:are|include|:)\s*(.+?)(?:[.;\n]|$)",
            rf"produces?.*?{keyword}s?.*?:\s*(.+?)(?:[.;\n]|$)",
        ]
    patterns += [
        r"produces(?:\s+\w+)*\s*(?:two|three|four|\d+)?\s*products?\s*:\s*(.+?)(?:[.;\n]|$)",
        r"products?\s*(?:are|include|:)\s*(.+?)(?:[.;\n]|$)",
    ]
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m:
            seg = m.group(1)
            items = _split_candidates(seg)
            if items:
                return items
    title_chunks = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", t)
    blacklist = {"A Local Bakery", "The Owner", "Production Plan", "Next Week",
                 "Total Profit", "Setup Costs"}
    title_items = [c.strip(" .,:;") for c in title_chunks if c not in blacklist]
    if len(title_items) >= 2:
        return title_items
    return []

# ---------------------- public parsers ----------------------

def try_parse_list(text: str, keyword: Optional[str] = None) -> List[str]:
    """
    Accepts JSON arrays, comma/and/ampersand separated lists, bullet lists,
    and simple narrative patterns. Optional `keyword` biases extraction
    (e.g., keyword='product' or 'resource').
    """
    t = text.strip()
    if not t:
        return []

    # JSON array first
    if t.startswith("["):
        try:
            arr = json.loads(t)
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass

    # Markdown/bullets/lines, possibly with "label: value"
    def _items_from_segment(seg: str) -> List[str]:
        seg = seg.replace("•", "\n").replace("–", "\n").replace("-", "\n")
        lines = re.split(r"[;\n]+", seg)
        raw = []
        for line in lines:
            line = line.strip(" .;:,")
            if not line:
                continue
            if ":" in line:
                left, _right = line.split(":", 1)
                line = left.strip()
            parts = re.split(r"\s*(?:,| and | & |/|\|)\s*", line, flags=re.IGNORECASE)
            for p in parts:
                p = p.strip(" .;:,")
                if p:
                    raw.append(p)
        out, seen = [], set()
        for x in raw:
            if len(x) > 50 or len(x.split()) > 8:
                continue
            xl = x.lower()
            if xl not in seen:
                seen.add(xl); out.append(x)
        return out

    # Keyword-anchored narrative blocks
    if keyword:
        patts = [
            rf"{keyword}s?\s*(?:are|include|:)\s*(.+?)(?:\n\s*\n|^Constraints?:|\Z)",
            rf"produces?.*?{keyword}s?.*?:\s*(.+?)(?:\n\s*\n|^Constraints?:|\Z)",
        ]
        for p in patts:
            m = re.search(p, t, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
            if m:
                seg = m.group(1).strip()
                items = _items_from_segment(seg)
                if items:
                    return items

    # Plain inline list
    parts = re.split(r"\s*(?:,| and | & |/|\|)\s*", t, flags=re.IGNORECASE)
    items = [p.strip(" .;:,") for p in parts if p.strip(" .;:,")]
    items = [i for i in items if len(i) <= 50 and len(i.split()) <= 8]
    out, seen = [], set()
    for x in items:
        xl = x.lower()
        if xl not in seen:
            seen.add(xl); out.append(x)

    if out:
        return out

    # Last-chance narrative scrape (Products: …)
    if keyword in (None, "product", "resource"):
        return _extract_list_from_narrative(t, keyword=keyword)

    return out



def try_parse_bool(text: str) -> Optional[bool]:
    t = text.strip().lower().strip("\"'“”‘’")
    if t in ["true","yes","y","1","enable","enabled","allow","allowed"]:
        return True
    if t in ["false","no","n","0","disable","disabled","disallow","not allowed"]:
        return False
    if t in ["skip","none","n/a","na","nope","not available","nothing else","that's all","that is all"]:
        return None
    return None


def try_parse_number_map(text: str) -> Dict[str, float]:
    """
    Parse maps like:
      - JSON: {"W1":100,"W2":80}
      - CSV/lines: W1=100, W2=80  OR  W1: 100
      - Bullets/lines with units: oven: 100 hours; labor=80h
    Returns {key: float}
    """
    t = text.strip()
    # JSON first
    if t.startswith("{"):
        try:
            obj = json.loads(t)
            out: Dict[str, float] = {}
            for k, v in obj.items():
                f = _to_float(str(v))
                if f is not None:
                    out[str(k)] = f
            return out
        except Exception:
            pass

    # key=value or key:value separated by commas/semicolons/newlines
    pairs = re.split(r"[;,\n]+", t)
    out: Dict[str, float] = {}
    for p in pairs:
        if "=" in p:
            k, v = p.split("=", 1)
        elif ":" in p:
            k, v = p.split(":", 1)
        else:
            continue
        k = _strip_quotes(k)
        f = _to_float(v)
        if k and f is not None:
            out[k] = f
    return out

def _transpose_if_period_first(raw: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    If the dict looks like {period: {product: val}}, transpose to {product: {period: val}}.
    Heuristic: detect period-like keys such as W1/M2/2025-09-10.
    """
    def _is_period_token(s: str) -> bool:
        s = str(s).strip()
        if re.fullmatch(r"[WMD]\d+", s, flags=re.IGNORECASE):
            return True
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return True
        return False

    # Count period-like keys at top level
    top_keys = list(raw.keys())
    periodish = sum(1 for k in top_keys if _is_period_token(k))
    # Count period-like keys inside (product-first) to decide orientation
    inner_keys = set()
    for v in raw.values():
        if isinstance(v, dict):
            inner_keys |= set(v.keys())
    inner_periodish = sum(1 for k in inner_keys if _is_period_token(k))

    # If top-level looks like periods and inner looks like products, transpose
    if periodish > inner_periodish:
        trans: Dict[str, Dict[str, float]] = {}
        for t, row in raw.items():
            if not isinstance(row, dict):
                continue
            for p, val in row.items():
                trans.setdefault(str(p), {})[str(t)] = float(val)
        return trans
    return raw

def try_parse_nested_number_map(text: str) -> Dict[str, Dict[str, float]]:
    """
    Parse nested maps like:
      - JSON: {"Cookies":{"W1":10,"W2":12},"Brownies":{"W1":8}}
      - Product-first: Classic: W1=10, W2=12; Brownies: W1=8
      - Period-first:  W1: Classic=10, Brownies=8; W2: Classic=12
    Returns {outer: {inner: float}} and auto-transposes to {product: {period: value}}
    when the input is period-first.
    """
    t = text.strip()
    # JSON first
    if t.startswith("{"):
        try:
            obj = json.loads(t)
            clean: Dict[str, Dict[str, float]] = {}
            for outer, inner in obj.items():
                if isinstance(inner, dict):
                    row: Dict[str, float] = {}
                    for k, v in inner.items():
                        f = _to_float(str(v))
                        if f is not None:
                            row[str(k)] = f
                    if row:
                        clean[str(outer)] = row
            return _transpose_if_period_first(clean)
        except Exception:
            pass

    # Semi-structured lines: "Outer: k=v, k=v; Outer2: k=v"
    out: Dict[str, Dict[str, float]] = {}
    entries = re.split(r";|\n", t)
    for e in entries:
        if ":" not in e:
            continue
        left, right = e.split(":", 1)
        outer = _strip_quotes(left)
        pairs = re.split(r"[;,]", right)
        row: Dict[str, float] = {}
        for p in pairs:
            if "=" in p:
                k, v = p.split("=", 1)
            elif ":" in p:
                k, v = p.split(":", 1)
            else:
                continue
            key = _strip_quotes(k)
            f = _to_float(v)
            if key and f is not None:
                row[key] = f
        if outer and row:
            out[outer] = row

    if out:
        return _transpose_if_period_first(out)

    return {}

# ---------------------- optional helpers (if you want later) ----------------------

def try_parse_period_tokens(text: str) -> List[str]:
    """
    Extract period tokens from free text:
      - Explicit: W1, M2, D3, 2025-09-10
      - Quantities: "2 weeks", "one month" -> ["W1","W2"] / ["M1"]
      - Bare number "2" -> ["W1","W2"]
    """
    t = text.strip()
    toks = re.findall(r"\b(?:W\d+|M\d+|D\d+|\d{4}-\d{2}-\d{2})\b", t, flags=re.IGNORECASE)
    if toks:
        return [tok.upper() for tok in toks]
    m = re.search(r"\b(\d+)\s*(week|month|day)s?\b", t, flags=re.IGNORECASE)
    if m:
        n = int(m.group(1)); unit = m.group(2).lower()
        prefix = {"week":"W","month":"M","day":"D"}[unit]
        return [f"{prefix}{i+1}" for i in range(n)]
    m2 = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(week|month|day)s?\b", t, flags=re.IGNORECASE)
    if m2:
        n = _PERIOD_WORD2NUM[m2.group(1).lower()]
        unit = m2.group(2).lower()
        prefix = {"week":"W","month":"M","day":"D"}[unit]
        return [f"{prefix}{i+1}" for i in range(n)]
    m3 = re.fullmatch(r"\s*(\d+)\s*", t)
    if m3:
        n = int(m3.group(1))
        return [f"W{i+1}" for i in range(n)]
    if re.search(r"\bnext\s+week\b", t, flags=re.IGNORECASE):
        return ["W1"]
    return []

# --- relaxed per-key number extractor (natural language friendly) ---

def try_parse_number_map_relaxed(text: str,
                                 keys: Optional[List[str]] = None,
                                 keyword: Optional[str] = None) -> Dict[str, float]:
    """
    Natural-language tolerant mapping parser.
    Tries JSON / key=value first via try_parse_number_map, and if that fails:
      - If keys are provided (e.g., product names), find a number near each key.
      - Accepts currency symbols, units, and words between key and number.
      - Optionally bias search using `keyword` (e.g., "profit", "cost", "hours").

    Returns {key: float} for any keys it can confidently extract.
    """
    # 1) First, reuse strict parser (handles JSON, k=v, k: v)
    strict = try_parse_number_map(text)
    if strict:
        return strict

    out: Dict[str, float] = {}
    if not keys:
        return out

    t = text

    # helper: pull a numeric value within a small window after the match
    def _find_number_after(span_start: int, window: int = 180) -> Optional[float]:
        window_text = t[span_start: span_start + window]
        # prefer "... profit $5" if keyword present
        if keyword:
            m_kw = re.search(
                rf"{re.escape(keyword)}[^0-9$-]{{0,40}}\$?\s*(-?\d+(?:\.\d+)?)",
                window_text, flags=re.IGNORECASE)
            if m_kw:
                val = _to_float(m_kw.group(1))
                if val is not None:
                    return val
        # generic number in window
        m = re.search(r"\$?\s*(-?\d+(?:\.\d+)?)", window_text)
        if m:
            return _to_float(m.group(1))
        return None

    # For each expected key (e.g., each product), look for a number nearby
    for k in keys:
        # match key in a tolerant way (ignore multiple spaces)
        patt = re.compile(re.escape(k), flags=re.IGNORECASE)
        best_val: Optional[float] = None
        for m in patt.finditer(t):
            val = _find_number_after(m.end())
            if val is not None:
                best_val = val
                break
        if best_val is not None:
            out[k] = best_val

    return out

