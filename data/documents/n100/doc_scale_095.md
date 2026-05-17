DOCUMENT_ID: doc_scale_095
TITLE: Retry and Timeout Configuration for Ollama Clients
CATEGORY: Infrastructure
CORPUS_SIZES: n100
VARIANT: v5

CONTENT:
HTTP clients communicating with Ollama must be configured with appropriate
timeouts and retry logic to handle transient failures gracefully. Generation
requests may take 30-180 seconds on CPU hardware, requiring a long request
timeout (at least 180 seconds). Connection timeouts should be kept short
(5-10 seconds) to detect unresponsive servers quickly. Transient errors such
as HTTP 503 (server busy) during model loading warrant automatic retries
with exponential backoff. A maximum of 3 retries with delays of 2, 4, and 8
seconds provides a reasonable balance between fault tolerance and failing fast
on persistent errors. The Ollama server's model loading phase (when a new
model is called for the first time after restart) can take 10-30 seconds,
during which requests receive 503 responses; retry logic must handle this
cold-start period gracefully.
