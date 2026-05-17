DOCUMENT_ID: doc_scale_056
TITLE: Chunk Overlap: Trade-offs and Best Practices
CATEGORY: Chunking
CORPUS_SIZES: n100
VARIANT: v6

CONTENT:
Chunk overlap is a mechanism that ensures relevant content near chunk
boundaries is not lost when a sentence or concept spans two adjacent chunks.
Without overlap, information at the boundary of a chunk may be partially
retrieved by one chunk and partially by the next, with neither chunk providing
enough context to answer a query. With overlap, each chunk contains a prefix
from the previous chunk and a suffix from the next, ensuring that boundary-
spanning content appears fully in at least one chunk. The optimal overlap
fraction depends on the typical length of boundary-spanning information:
a 10-20% overlap of the chunk size is a common starting point. Higher overlap
increases the total number of chunks and the storage and embedding cost,
while lower overlap risks missing boundary content.
