# scripts/train_tinystories.py
import time, pickle, resource, cProfile, pstats
from cs336_basics.train_bpe import pretokenize_file, bpe_merges   # adjust to your module path

def main(path, vocab_size=10_000, special_tokens=("<|endoftext|>",), out="tinystories_bpe"):
    special_tokens = list(special_tokens)
    t0 = time.perf_counter()
    total = pretokenize_file(path, special_tokens)
    t1 = time.perf_counter()
    vocab, merges = bpe_merges(total, vocab_size, [t.encode() for t in special_tokens])
    t2 = time.perf_counter()

    # ru_maxrss is in KB on Linux; CHILDREN = largest single worker, not the sum
    self_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    child_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    print(f"pretokenize {t1-t0:.1f}s | merges {t2-t1:.1f}s | total {t2-t0:.1f}s")
    print(f"peak RSS: main {self_mb:.0f} MB, largest worker {child_mb:.0f} MB")

    # Lossless binary for reuse
    with open(f"{out}.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
    # Human-readable for inspection (repr keeps it lossless)
    with open(f"{out}_vocab.txt", "w") as f:
        for i, tok in vocab.items():
            f.write(f"{i}\t{tok!r}\n")
    with open(f"{out}_merges.txt", "w") as f:
        for a, b in merges:
            f.write(f"{a!r} {b!r}\n")

    longest = max(vocab.values(), key=len)
    print(f"longest token: {longest!r} ({len(longest)} bytes) -> {longest.decode('utf-8', errors='replace')!r}")

if __name__ == "__main__":
    import sys, os
    path = sys.argv[1]
    out_dir = "tinystories_token"
    os.makedirs(out_dir, exist_ok=True)                      # creates the folder if missing
    name = os.path.basename(path).replace(".txt", "")        # e.g. TinyStoriesV2-GPT4-train
    out = os.path.join(out_dir, f"{name}_bpe")
    if "--profile" in sys.argv:
        prof = os.path.join(out_dir, "bpe.prof")
        cProfile.run("main(path, out=out)", prof)
        pstats.Stats(prof).sort_stats("cumtime").print_stats(15)
    else:
        main(path, out=out)