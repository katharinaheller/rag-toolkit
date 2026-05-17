DOCUMENT_ID: doc_scale_058
TITLE: Chunk Metadata and Provenance
CATEGORY: Chunking
CORPUS_SIZES: n100
VARIANT: v8

CONTENT:
Each chunk should carry metadata that enables provenance tracking, access
control enforcement, and result presentation. Standard metadata fields include
the source document path, the document title, the section heading under which
the chunk appears, the character offset within the document, the chunk index
within the document, and any document-level access control labels. This
metadata is stored alongside the chunk embedding in the document store and
is included in retrieval results, allowing the generator's prompt to include
source attribution and allowing the application layer to filter results by
access permissions before returning them to the user. Metadata should be
indexed separately from the chunk content in the document store to allow
efficient filtering by metadata fields without requiring re-embedding when
metadata values change.
