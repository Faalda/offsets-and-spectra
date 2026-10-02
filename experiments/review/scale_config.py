"""
Frozen configuration for the scaled source-family batch (batch 4).

The encoder list, pooling, projection rule and hypotheses below are fixed BEFORE any
new embedding is computed or any result is looked at (see results/review/prereg_families.json,
which records the SHA-256 of this file and of the embedding and analysis scripts).

Design: 20 NEW English source encoders (not among the six of Appendix A.2), each aligned to
the SAME 17 monolingual BERT targets used throughout (benchmark.XL); FLORES dev fits every map,
devtest evaluates it.  Source embeddings use tasks.crosslingual.embed_sentences semantics:
mean-pooled last hidden state, L2-normalised, max_length 128, e5 models get the 'query: ' prefix.
The only change is a pad-token fix for decoder models (gpt2, pythia), which have none.

Projection rule (declared in advance, no post hoc choice): d = 768 native; d > 768 mean-preserving
projection to 768 (top-768 right singular vectors of the UNCENTRED training matrix); d < 768
native dimension (semi-orthogonal maps, CCA truncated to min(dx, dy)).

Source mean dominance is computed on the native English dev embeddings, before any projection.
"""

NEW_SOURCES = [
    # (huggingface id, note)
    ("roberta-base", "encoder; used in dev cross-architecture pairs, not with the 17 targets"),
    ("google/electra-base-discriminator", "encoder; as above"),
    ("microsoft/deberta-base", "encoder; as above"),
    ("distilbert-base-uncased", "distilled encoder"),
    ("bert-base-cased", "encoder"),
    ("bert-large-uncased", "encoder, d=1024"),
    ("roberta-large", "encoder, d=1024"),
    ("albert-base-v2", "encoder, shared-parameter"),
    ("xlm-roberta-base", "multilingual encoder"),
    ("xlm-roberta-large", "multilingual encoder, d=1024"),
    ("sentence-transformers/all-mpnet-base-v2", "sentence embedding"),
    ("sentence-transformers/all-MiniLM-L12-v2", "sentence embedding, d=384"),
    ("BAAI/bge-base-en-v1.5", "retrieval embedding"),
    ("BAAI/bge-large-en-v1.5", "retrieval embedding, d=1024"),
    ("thenlper/gte-base", "retrieval embedding"),
    ("thenlper/gte-large", "retrieval embedding, d=1024"),
    ("intfloat/e5-base-v2", "retrieval embedding, 'query: ' prefix"),
    ("facebook/contriever", "unsupervised retrieval encoder"),
    ("gpt2", "decoder LM"),
    ("EleutherAI/pythia-160m", "decoder LM"),
]

# sources already analysed in Appendix A.2 (pooled only as a secondary analysis)
OLD_SOURCES = ["mpnet", "gte-qwen2", "bert-unc", "modernbert-large", "e5-base", "e5-large"]
OLD_NATIVE_MD = {"mpnet": 0.084, "gte-qwen2": 0.233, "bert-unc": 0.578,
                 "modernbert-large": 0.866, "e5-base": 0.741, "e5-large": 0.734}

N_PERM = 200_000       # Monte Carlo permutations for the source-level test
N_BOOT = 4000          # pigeonhole bootstrap replicates
SEED = 0
