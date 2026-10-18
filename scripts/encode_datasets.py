# scripts/encode_datasets.py
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import find_chunk_boundaries

EOT = "<|endoftext|>"
TS_PKL = "tinystories_token/TinyStoriesV2-GPT4-train_bpe.pkl"
OWT_PKL = "owt_token/owt_train_bpe.pkl"

JOBS = [  # (input text, tokenizer, output)
    ("data/TinyStoriesV2-GPT4-train.txt", TS_PKL,  "tinystories_token/ts_train.bin"),
    ("data/TinyStoriesV2-GPT4-valid.txt", TS_PKL,  "tinystories_token/ts_valid.bin"),
    ("data/owt_train.txt",                OWT_PKL, "owt_token/owt_train.bin"),
    ("data/owt_valid.txt",                OWT_PKL, "owt_token/owt_valid.bin"),
]

_tok = None

def _init(pkl):
    global _tok
    _tok = Tokenizer.from_files(pkl, pkl, [EOT])          # one tokenizer per worker

def _encode_range(args):
    path, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    return np.array(_tok.encode(text), dtype=np.uint16)   # 2 bytes/token when sent back

def encode_file(path, pkl, out_path, workers):
    tok = Tokenizer.from_files(pkl, pkl, [EOT])
    assert max(tok.vocab) < 2**16, "vocab too large for uint16"

    with open(path, "rb") as f:
        bounds = find_chunk_boundaries(f, workers * 32, EOT.encode("utf-8"))
    ranges = [(path, s, e) for s, e in zip(bounds[:-1], bounds[1:])]
    nbytes = os.path.getsize(path)

    print(f"{path} -> {out_path}  ({nbytes / 1e9:.2f} GB, {len(ranges)} chunks)", flush=True)
    t0, ntok = time.perf_counter(), 0
    with Pool(workers, initializer=_init, initargs=(pkl,)) as pool, open(out_path, "wb") as out:
        # imap (not imap_unordered) keeps chunks in file order
        for i, arr in enumerate(pool.imap(_encode_range, ranges), 1):
            arr.tofile(out)
            ntok += len(arr)
            if i % 50 == 0 or i == len(ranges):
                print(f"  {i}/{len(ranges)} chunks, {ntok:,} tokens", flush=True)
    dt = time.perf_counter() - t0

    # Sanity check: decoding the first tokens reproduces the start of the file
    ids = np.memmap(out_path, dtype=np.uint16, mode="r")
    with open(path, encoding="utf-8") as f:
        head = f.read(5000)
    decoded = tok.decode(ids[:200].tolist()).rstrip("\ufffd")   # drop a possibly cut-off char
    assert head.startswith(decoded), "decoded tokens don't match the source text"

    print(f"  done: {ntok:,} tokens, {nbytes / ntok:.2f} bytes/token, "
          f"{dt:.0f}s ({nbytes / dt / 1e6:.1f} MB/s with {workers} workers), "
          f"{os.path.getsize(out_path) / 1e9:.2f} GB on disk\n")

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for path, pkl, out_path in JOBS:
        if os.path.exists(out_path):
            print(f"skip {out_path} (already exists)")
            continue
        encode_file(path, pkl, out_path, workers)