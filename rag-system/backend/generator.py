"""
LLM synthesis.

The model produces a structured answer that includes:
  - Answer        : a 2-5 sentence direct response
  - Key points    : 2-5 bullets summarizing the most important takeaways
  - Citations     : a list of [Sx] markers backed by source docs

Streaming is supported via stream_answer().
"""
from __future__ import annotations

from typing import Iterator
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .config import SETTINGS
from .llm import get_chat_llm


def _get_llm(streaming: bool = False):
    return get_chat_llm()


SYSTEM_PROMPT = """You are a knowledgeable, patient teacher. Answer the user's
question based ONLY on the provided source material — but write naturally, like
explaining to a curious learner. Do NOT include citation markers, source numbers,
or reference tags in your output.

Use chat history only to resolve coreferences (pronouns, "it", "that").

If the sources don't contain enough information, say so clearly — don't invent facts.

# Visual diagrams (when helpful)
If a diagram would help, choose ONE format and use it consistently:

PREFERRED: Graphviz DOT (more reliable, especially for processes)
```dot
digraph G {{
    rankdir=LR;
    A [label="Input data"];
    B [label="Process step"];
    C [label="Output"];
    A -> B -> C;
}}
```

ALTERNATIVE: Mermaid (use only if Graphviz doesn't fit)
```mermaid
flowchart LR
    A[Input data] --> B[Process step] --> C[Output]
```

Diagram rules:
- Diagrams MUST be grounded in the sources. Don't invent components.
- Keep diagrams compact: 4 to 10 nodes is ideal.
- Use simple labels in quotes (Graphviz) or square brackets (Mermaid).
- Avoid special characters: no <, >, &, parentheses, or curly braces inside labels.
- For Mermaid, use only square brackets: A[Label], not A(Label) or A{{Label}}.
- Place the diagram between the Answer and Key Points sections.
- Skip diagrams for definition/lookup questions.

# Required output format

### TL;DR
<A single crisp sentence (15-30 words) that captures the core answer. No fluff, no preamble.>

### Answer
<A thorough, educational explanation in 4-8 sentences. Define key terms, give
context, and walk the reader through the concept clearly. Write naturally — no
inline citation tags or reference numbers.>

<optional Graphviz DOT or Mermaid diagram>

### Key Points
- <substantive takeaway, 1-2 full sentences>
- <substantive takeaway, 1-2 full sentences>
- <substantive takeaway, 1-2 full sentences>
- <optional 4th bullet if useful>
- <optional 5th bullet if useful>

### Real-world Example
<One short, concrete example or analogy (2-4 sentences) that grounds the idea
in something tangible. Skip this section ONLY if the question is a simple
definition or lookup that doesn't benefit from an example.>

### Want to learn more?
- <a related follow-up question the learner might ask>
- <another related follow-up question>
- <one more related follow-up question>
"""


HUMAN_PROMPT = """Question: {question}

{history_block}Sources:
{sources}

Answer the question following the required format."""


def format_sources(docs: list[Document]) -> tuple[str, list[dict]]:
    lines = []
    meta = []
    for i, d in enumerate(docs, start=1):
        tag = f"S{i}"
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        lines.append(f"[{tag}] (source: {src}, page: {page})\n{d.page_content}\n")
        meta.append({
            "tag": tag,
            "source": src,
            "page": page,
            "snippet": d.page_content[:500],
            "chunk_id": d.metadata.get("chunk_id"),
        })
    return "\n".join(lines), meta


def format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["Recent chat history (for coreference only):"]
    for turn in history[-SETTINGS.memory_window:]:
        role = turn.get("role", "user")
        content = turn.get("content", "").strip()
        if not content:
            continue
        lines.append(f"{role.capitalize()}: {content[:300]}")
    return "\n".join(lines) + "\n\n"


def synthesize(question: str, docs: list[Document], history: list[dict] | None = None) -> tuple[str, list[dict]]:
    """Non-streaming synthesis. Returns (answer_text, citations_meta)."""
    sources_block, citations = format_sources(docs)
    if not docs:
        return (
            "### Answer\nI couldn't find relevant information in the knowledge base to answer this question.\n\n"
            "### Key Points\n- No matching documents were retrieved.\n\n"
            "### Citations Used\n_(none)_",
            [],
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])
    msgs = prompt.format_messages(
        question=question,
        sources=sources_block,
        history_block=format_history(history or []),
    )
    out = _get_llm().invoke(msgs).content
    return out, citations


def stream_answer(question: str, docs: list[Document], history: list[dict] | None = None) -> Iterator[tuple[str, list[dict]]]:
    """Streaming synthesis: yields (token, citations_meta) tuples."""
    sources_block, citations = format_sources(docs)
    if not docs:
        yield ("### Answer\nI couldn't find relevant information in the knowledge base to answer this question.", [])
        return

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])
    msgs = prompt.format_messages(
        question=question,
        sources=sources_block,
        history_block=format_history(history or []),
    )

    llm = _get_llm()
    for chunk in llm.stream(msgs):
        token = getattr(chunk, "content", "") or ""
        if token:
            yield (token, citations)