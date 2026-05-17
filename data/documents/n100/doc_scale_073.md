DOCUMENT_ID: doc_scale_073
TITLE: Mistral Instruction Format and System Prompts
CATEGORY: Language Models
CORPUS_SIZES: n100
VARIANT: v3

CONTENT:
Mistral 7B Instruct uses a specific conversation template based on the
[INST] and [/INST] delimiters to structure multi-turn conversations. A single-
turn instruction is formatted as: [INST] <user message> [/INST], with the
model generating its response after the closing delimiter. System prompts
(instructions provided before the first user turn) are prepended before the
first [INST] token. For RAG applications, the system prompt typically contains
the instruction "Answer only from the provided context. If the context does
not contain the answer, say so." followed by the retrieved context passages,
and the first user message contains the query. This prompt structure ensures
the model treats the retrieved context as authoritative and avoids confabulating
information from its parametric knowledge.
