DOCUMENT_ID: doc_scale_080
TITLE: Token Budget and Generation Length Control
CATEGORY: Language Models
CORPUS_SIZES: n100
VARIANT: v10

CONTENT:
Controlling the length of generated answers is important for both quality and
cost in RAG systems. The max_tokens parameter sets a hard cap on the number
of tokens generated per inference call, preventing runaway generation that
consumes excessive compute and produces overly verbose answers. For factoid
question answering, answers of 50-100 tokens are typically sufficient.
For summarisation or explanation tasks, 200-500 tokens may be needed.
Setting max_tokens too low truncates answers mid-sentence, producing poor
user experience. Setting it too high wastes inference compute for short
answers. A calibration step that measures the length distribution of answers
on a representative query set and sets max_tokens at the 95th percentile of
that distribution provides a data-driven approach to token budget selection.
The stop parameter allows explicit stop sequences to terminate generation
cleanly at natural sentence boundaries.
