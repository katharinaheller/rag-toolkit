DOCUMENT_ID: doc_scale_084
TITLE: Exact Match and Token F1 for Generation Evaluation
CATEGORY: Evaluation
CORPUS_SIZES: n100
VARIANT: v4

CONTENT:
Exact Match (EM) is a binary generation metric that equals 1 only if the
generated answer string, after normalisation, is identical to the reference
answer string. Normalisation typically involves lowercasing, removing
punctuation, and stripping articles. EM is strict and does not award partial
credit, making it appropriate only when the expected answer is short and
unambiguous, such as a named entity or a number. Token F1 provides a more
lenient alternative by computing the token-level precision and recall between
the generated answer and the reference. Tokens in both the generated answer
and the reference are treated as a bag, and F1 is the harmonic mean of token
precision (fraction of generated tokens in the reference) and token recall
(fraction of reference tokens in the generated answer). Token F1 handles
paraphrases and word order variation better than EM but still requires
normalisation of the reference answer.
