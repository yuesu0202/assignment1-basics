# scripts/compression_ratio.py
import random
from cs336_basics.tokenizer import Tokenizer      # adjust if tokenizer.py lives elsewhere

EOT = "<|endoftext|>"
TS_PKL  = "tinystories_token/TinyStoriesV2-GPT4-train_bpe.pkl"
OWT_PKL = "owt_token/owt_train_bpe.pkl"
TS_TXT  = "data/TinyStoriesV2-GPT4-valid.txt"
OWT_TXT = "data/owt_valid.txt"

def sample_docs(path, n=10, read_chars=50_000_000, seed=0):
    """Sample n documents from the first ~50M characters (avoids loading the whole file)."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read(read_chars)
    docs = [d for d in text.split(EOT) if d.strip()][:-1]   # drop last: may be cut off
    return random.Random(seed).sample(docs, n)

def compression(tok, docs, label):
    nbytes = ntoks = 0
    for d in docs:
        ids = tok.encode(d)
        assert tok.decode(ids) == d, "round-trip failed"     # sanity check
        nbytes += len(d.encode("utf-8"))
        ntoks += len(ids)
    print(f"{label}: {nbytes:,} bytes / {ntoks:,} tokens = {nbytes / ntoks:.2f} bytes/token")
    return nbytes / ntoks

if __name__ == "__main__":
    ts_tok  = Tokenizer.from_files(TS_PKL, TS_PKL, [EOT])
    owt_tok = Tokenizer.from_files(OWT_PKL, OWT_PKL, [EOT])
    ts_docs  = sample_docs(TS_TXT)
    owt_docs = sample_docs(OWT_TXT)

    compression(ts_tok, ts_docs, "TinyStories tokenizer (10K) on TinyStories")
    compression(owt_tok, owt_docs, "OWT tokenizer (32K) on OWT")