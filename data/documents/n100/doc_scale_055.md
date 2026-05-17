DOCUMENT_ID: doc_scale_055
TITLE: Structure-Aware Chunking for Markdown and HTML
CATEGORY: Chunking
CORPUS_SIZES: n100
VARIANT: v5

CONTENT:
Many document corpora have rich structural metadata in the form of Markdown
headers, HTML section tags, or LaTeX sections that can guide chunking.
Structure-aware chunking respects these boundaries by treating top-level
sections as the primary chunk unit and applying further splitting only when a
section exceeds the maximum chunk size. This preserves the logical structure
of the document within each chunk: a chunk beginning under a specific header
stays topically focused on the content of that section. Metadata from the
structural hierarchy (the heading text and nesting level) can be prepended
to each chunk to provide contextual framing for the embedding model. For
Markdown documents parsed into an AST, extracting chunks at the H2 or H3
heading level is a practical default that balances granularity and coherence.
