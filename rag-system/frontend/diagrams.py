"""
Robust diagram rendering with multi-tier fallback.

Tier 1: Sanitize syntax issues (cheap)
Tier 2: Native render (Mermaid via streamlit-mermaid, or Graphviz native)
Tier 3: LLM-based repair (~1s, free on Groq)
Tier 4: Cross-format conversion (Mermaid -> Graphviz)
Tier 5: Show source code in expander as graceful fallback
"""
from __future__ import annotations
import re
import streamlit as st


MERMAID_RE = re.compile(r"```mermaid\s*\n(.*?)\n```", re.DOTALL)
DOT_RE = re.compile(r"```dot\s*\n(.*?)\n```", re.DOTALL)


# ---------- Sanitizers ----------
def sanitize_mermaid(code: str) -> str:
    """Clean common Mermaid syntax issues from LLM output."""
    code = (code.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'"))

    def clean_label(match):
        label = match.group(1)
        label = (label.replace("<", "lt").replace(">", "gt")
                      .replace("(", "").replace(")", "")
                      .replace("&", "and")
                      .replace('"', "").replace("'", ""))
        return f"[{label}]"

    code = re.sub(r"\[([^\[\]]+)\]", clean_label, code)
    return "\n".join(ln for ln in code.splitlines() if ln.strip())


def sanitize_dot(code: str) -> str:
    """Clean common DOT syntax issues."""
    code = (code.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'"))
    return code.strip()


# ---------- Render attempts ----------
def _try_render_mermaid(code: str) -> bool:
    """Try to render Mermaid. Returns True if successful."""
    try:
        from streamlit_mermaid import st_mermaid
        st_mermaid(code, height="auto")
        return True
    except Exception:
        return False


def _try_render_dot(code: str) -> bool:
    """Try to render Graphviz. Returns True if successful."""
    try:
        st.graphviz_chart(code, use_container_width=True)
        return True
    except Exception:
        return False


# ---------- LLM repair ----------
@st.cache_data(ttl=600, show_spinner=False)
def llm_repair(code: str, kind: str) -> str:
    """Use the small LLM to fix syntax errors. Cached so retries are instant."""
    try:
        from backend.llm import get_rewrite_llm
        llm = get_rewrite_llm()
        prompt = f"""Fix any syntax errors in this {kind} diagram code. Return ONLY the
corrected code, with no explanations, no markdown fences, no commentary.

Common issues to fix:
- Replace problematic characters (<, >, &, parentheses) in node labels with safe alternatives
- Ensure all nodes have proper bracket/quote syntax
- Remove duplicate, empty, or malformed lines

Broken {kind} code:
{code}

Fixed {kind} code:"""
        response = llm.invoke(prompt).content.strip()
        # Strip code fences if model wrapped output
        response = re.sub(r"^```(?:\w+)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)
        return response.strip()
    except Exception:
        return code


# ---------- Mermaid to DOT conversion ----------
def mermaid_to_dot(mermaid: str) -> str:
    """Convert simple Mermaid flowcharts to Graphviz DOT (best-effort).
    Handles flowchart LR/TD with --> edges. Falls back to a simple wrapper otherwise.
    """
    lines = [ln.strip() for ln in mermaid.splitlines() if ln.strip()]
    if not lines:
        return ""

    # Detect direction
    first = lines[0].lower()
    rankdir = "LR"
    if "td" in first or "tb" in first:
        rankdir = "TB"
    elif "lr" in first or "flowchart" in first or "graph" in first:
        rankdir = "LR"

    # Strip the header line (flowchart LR / graph TD / etc)
    if any(first.startswith(k) for k in ("flowchart", "graph")):
        lines = lines[1:]

    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []

    edge_re = re.compile(r"(\w+)(?:\[([^\]]*)\])?\s*-->(?:\|[^|]*\|)?\s*(\w+)(?:\[([^\]]*)\])?")
    node_re = re.compile(r"^(\w+)\[([^\]]*)\]$")

    for ln in lines:
        m = edge_re.search(ln)
        if m:
            a, alabel, b, blabel = m.groups()
            if a and a not in nodes:
                nodes[a] = alabel or a
            if b and b not in nodes:
                nodes[b] = blabel or b
            edges.append((a, b))
            continue
        m = node_re.match(ln)
        if m:
            nid, label = m.groups()
            nodes[nid] = label

    if not nodes:
        return ""

    out = [f"digraph G {{\n  rankdir={rankdir};",
           '  node [shape=box, style="rounded,filled", fillcolor="#1a1f2e", fontcolor="#e6e8ee", color="#3a4256"];',
           '  edge [color="#6b7488", fontcolor="#98a2b8"];']
    for nid, label in nodes.items():
        clean = label.replace('"', "").replace("\n", " ")
        out.append(f'  {nid} [label="{clean}"];')
    for a, b in edges:
        out.append(f"  {a} -> {b};")
    out.append("}")
    return "\n".join(out)


# ---------- Public renderers ----------
def render_diagrams(content: str) -> str:
    """Find all diagrams in the text, render each with full fallback chain,
    and return the content with diagram code blocks removed."""
    mermaid_blocks = MERMAID_RE.findall(content)
    dot_blocks = DOT_RE.findall(content)

    # Render Mermaid blocks
    for raw in mermaid_blocks:
        _render_with_fallback(raw, kind="mermaid")

    # Render DOT blocks
    for raw in dot_blocks:
        _render_with_fallback(raw, kind="dot")

    # Strip both block types from content
    text = MERMAID_RE.sub("", content)
    text = DOT_RE.sub("", text)
    return text.strip()


def _render_with_fallback(raw: str, kind: str):
    """Try increasingly aggressive strategies to render a diagram."""
    raw = raw.strip()
    if not raw:
        return

    if kind == "mermaid":
        sanitized = sanitize_mermaid(raw)
        # Tier 1: native render — clean success, no source noise
        if _try_render_mermaid(sanitized):
            return
        # Tier 2: LLM repair (hint that something was patched up)
        repaired = llm_repair(sanitized, "Mermaid")
        if repaired != sanitized and _try_render_mermaid(repaired):
            _show_source(repaired, kind, label="📋 Diagram source (auto-repaired)")
            return
        # Tier 3: convert to DOT and try Graphviz
        as_dot = mermaid_to_dot(sanitized)
        if as_dot and _try_render_dot(as_dot):
            _show_source(as_dot, "dot", label="📋 Diagram source (converted to Graphviz)")
            return
        # Tier 4: give up gracefully
        st.info("📊 Diagram couldn't be rendered. View the source below.")
        _show_source(sanitized, "mermaid", label="📋 Diagram source", expanded=True)

    elif kind == "dot":
        sanitized = sanitize_dot(raw)
        if _try_render_dot(sanitized):
            return
        repaired = llm_repair(sanitized, "Graphviz DOT")
        if repaired != sanitized and _try_render_dot(repaired):
            _show_source(repaired, kind, label="📋 Diagram source (auto-repaired)")
            return
        st.info("📊 Diagram couldn't be rendered. View the source below.")
        _show_source(sanitized, "dot", label="📋 Diagram source", expanded=True)


def _show_source(code: str, kind: str, label: str = "📋 Diagram source", expanded: bool = False):
    """Always show the source code in a collapsible expander."""
    with st.expander(label, expanded=expanded):
        st.code(code, language=kind)