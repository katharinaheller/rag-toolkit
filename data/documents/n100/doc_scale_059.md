DOCUMENT_ID: doc_scale_059
TITLE: Chunking for Multi-Modal Documents
CATEGORY: Chunking
CORPUS_SIZES: n100
VARIANT: v9

CONTENT:
Documents containing a mix of text, tables, figures, and code require
specialised chunking strategies for each content type. Tables should be
extracted and serialised as Markdown or CSV text before embedding to preserve
their relational structure. Code blocks should be separated from surrounding
prose and chunked at function or class boundaries rather than at character
boundaries, with language-specific parsers identifying these units. Figures
require a multimodal embedding model or a caption extraction step to produce
a text description suitable for dense retrieval. The main challenge in multi-
modal chunking is maintaining alignment between the text surrounding a figure
and the figure content itself, so that a query about the figure's topic
retrieves the associated text passage. Captioning models and layout parsers
such as Nougat or PaddleOCR are commonly used to extract structured text
from complex document layouts.
