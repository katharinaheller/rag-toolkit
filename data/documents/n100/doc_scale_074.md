DOCUMENT_ID: doc_scale_074
TITLE: Deterministic Generation: Temperature and Seed
CATEGORY: Language Models
CORPUS_SIZES: n100
VARIANT: v4

CONTENT:
Achieving reproducible, deterministic generation from a large language model
requires controlling both the temperature sampling parameter and the random
number generator seed. Temperature controls the sharpness of the next-token
probability distribution: at temperature 0.0 the model always selects the
highest-probability token (greedy decoding), eliminating randomness entirely.
At higher temperatures (0.5 to 1.0) the model samples from a smoothed
distribution, producing varied outputs across runs. For RAG evaluation, setting
temperature to 0.0 is essential to ensure that running the same query twice
produces the same answer, enabling fair comparison between system configurations.
The seed parameter initialises the random number generator and is relevant
when temperature > 0.0; at temperature 0.0 the seed has no effect since
greedy decoding is deterministic regardless of RNG state.
