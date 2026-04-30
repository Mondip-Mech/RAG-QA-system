"""
Contextual compression — extracts only the relevant sentences from each chunk.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .config import SETTINGS
from .llm import get_chat_llm


_llm = None
def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_chat_llm(temperature=0.0)
    return _llm


COMPRESS_PROMPT = ChatPromptTemplate.from_template(
    """Extract from the passage below ONLY the sentences that are directly
relevant to answering the question. Preserve original wording. If nothing in
the passage is relevant, output exactly: NO_RELEVANT_CONTENT

Question: {question}

Passage:
\"\"\"
{passage}
\"\"\"

Relevant sentences:"""
)


def compress_documents(question: str, docs: list[Document]) -> list[Document]:
    if not SETTINGS.use_compression or not docs:
        return docs

    llm = _get_llm()
    out: list[Document] = []
    total_chars = 0

    for d in docs:
        if total_chars >= SETTINGS.max_context_chars:
            break
        try:
            msg = COMPRESS_PROMPT.format_messages(question=question, passage=d.page_content)
            extracted = llm.invoke(msg).content.strip()
        except Exception:
            extracted = d.page_content

        if not extracted or "NO_RELEVANT_CONTENT" in extracted:
            continue

        budget = SETTINGS.max_context_chars - total_chars
        extracted = extracted[:budget]
        total_chars += len(extracted)

        out.append(Document(page_content=extracted, metadata=dict(d.metadata)))

    if not out:
        return [
            Document(
                page_content=d.page_content[: SETTINGS.max_context_chars // max(len(docs), 1)],
                metadata=dict(d.metadata),
            )
            for d in docs
        ]
    return out