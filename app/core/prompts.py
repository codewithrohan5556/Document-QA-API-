"""
Centralized prompt templates.

Kept separate from rag_service.py so the prompt can be read, reviewed, and
tuned on its own — and so no other module ever embeds a large prompt string
inline. If we add more prompts later (query rewriting, HyDE, etc.), they
go here too.
"""

from langsmith import traceable

RAG_PROMPT_TEMPLATE = """You are a helpful assistant answering questions about an uploaded document, using ONLY the context provided below.

Rules:
- Answer using only the information in the context below. Do not use outside knowledge.
- Do not invent or assume any information that isn't present in the context.
- If the answer cannot be found in the context, say so clearly instead of guessing.
- When you use information from a specific part of the context, reference it
  using the bracketed label shown, e.g. [page 2] or [chunk 2].

Context:
{context}

Question: {question}

Answer:"""


# Langsmith Tracing
@traceable(name="build_context", run_type="prompt")
def build_context(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into the context block the template expects.
    Takes plain dicts (not RetrievedChunk directly) to keep this module
    free of a dependency on retrieval_service.

    Labels each chunk [page N] when it has a real page number (PDFs), or
    [chunk N] when it doesn't (.txt, where page_number is the -1 sentinel) —
    matching the same page-vs-chunk fallback used for source citations in
    rag_service.py, so what the LLM sees and what the user sees agree.
    """
    parts = []
    for chunk in chunks:
        page_number = chunk.get("page_number", -1)
        label = f"page {page_number}" if page_number and page_number > 0 else f"chunk {chunk['chunk_id']}"
        parts.append(f"[{label}] (source: {chunk['filename']})\n{chunk['text']}")
    return "\n\n".join(parts)
