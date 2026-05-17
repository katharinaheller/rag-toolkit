DOCUMENT_ID: doc_scale_092
TITLE: Ollama REST API Reference
CATEGORY: Infrastructure
CORPUS_SIZES: n100
VARIANT: v2

CONTENT:
The Ollama REST API provides endpoints for model management and text
generation. The POST /api/generate endpoint is the primary inference
endpoint, accepting a JSON body with fields: model (model tag string),
prompt (input text), stream (boolean, default true), temperature (float),
top_p (float), seed (integer), and repeat_penalty (float). When stream is
true, the API returns newline-delimited JSON objects with a 'response' field
containing incrementally generated text. When stream is false, it returns a
single JSON object with the complete generated text. The POST /api/chat
endpoint supports multi-turn conversation using a 'messages' array in OpenAI
format. The GET /api/tags endpoint lists available models. The DELETE
/api/delete endpoint removes a model from the local registry. All endpoints
use JSON for request and response bodies and return HTTP 200 for success.
