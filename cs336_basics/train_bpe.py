import os
from typing import BinaryIO, Counter
from collections import defaultdict, Counter
from numpy import byte
import regex as re
import time

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# def pre_tokenize_chunk(chunk: bytes, special_tokens: list[str]) -> dict[byte, int]:
#     """
#     Pre-tokenize a chunk of text.
#     """
#     special_tokens.sort(key = lambda x: len(x), reverse = True)
#     PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
#     special_tokens_str = "|".join(re.escape(special_token) for special_token in special_tokens)
#     split_chunks = re.split(special_tokens_str, chunk.decode("utf-8")) if special_tokens_str else list(chunk.decode("utf-8"))
#     d = defaultdict(int)
#     if len(split_chunks) == 1:
#         token = (re.findall(PAT, str(split_chunks)))
#         for item in token:
#             d[item.encode("utf-8")] += 1
#     else:
#         for split_chunk in split_chunks:
#             token = (re.findall(PAT, split_chunk))
#             for item in token:
#                 d[item.encode("utf-8")] += 1
#     return d

def pre_tokenize_chunk(chunk: bytes, special_tokens: list[str]) -> dict[bytes, int]:
    """Pre-tokenize a chunk of text, never merging across special tokens."""
    text = chunk.decode("utf-8")
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    if special_tokens:
        # Longest first so overlapping tokens like "<|eot|><|eot|>" match greedily
        toks = sorted(special_tokens, key=len, reverse=True)
        split_pat = "|".join(re.escape(t) for t in toks)
        segments = re.split(split_pat, text)
    else:
        segments = [text]          # whole text is one segment

    d = defaultdict(int)
    for seg in segments:
        for m in PAT.finditer(seg):  # finditer avoids building a big list
            d[m.group().encode("utf-8")] += 1
    return d

def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes], new_tok: bytes) -> tuple[bytes, ...]:
    """Replace every non-overlapping occurrence of `pair`, scanning left to right."""
    out, i = [], 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_tok)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)

# def bpe_merges(total: Counter, vocab_size: int, special_tokens_bytes: list[str]):
#     # Initial vocab: 256 single bytes + special tokens
#     vocab = {i: bytes([i]) for i in range(256)}
#     for tok in special_tokens_bytes:
#         vocab[len(vocab)] = tok

#     # b"caf\xc3\xa9" -> (b'c', b'a', b'f', b'\xc3', b'\xa9')
#     words = Counter({tuple(bytes([b]) for b in w): c for w, c in total.items()})
#     merges = []

#     while len(vocab) < vocab_size:
#         pair_counts = Counter()
#         for word, c in words.items():
#             for a, b in zip(word, word[1:]):
#                 pair_counts[(a, b)] += c
#         if not pair_counts:
#             break  # nothing left to merge

#         # Highest count; ties broken by lexicographically greater pair
#         best = max(pair_counts, key=lambda p: (pair_counts[p], p))
#         new_tok = best[0] + best[1]
#         merges.append(best)
#         vocab[len(vocab)] = new_tok

#         new_words = Counter()
#         for word, c in words.items():
#             new_words[merge_word(word, best, new_tok)] += c
#         words = new_words

#     return vocab, merges

# def bpe_merges(total, vocab_size, special_tokens_bytes: list[bytes]):
#     vocab = {i: bytes([i]) for i in range(256)}
#     for tok in special_tokens_bytes:
#         vocab[len(vocab)] = tok

#     # Words stored by integer id so the index can point at them
#     words = [tuple(bytes([b]) for b in w) for w in total]
#     counts = list(total.values())

#     pair_counts = defaultdict(int)          # pair -> weighted count
#     pair_index = defaultdict(set)           # pair -> ids of words containing it
#     for wid, word in enumerate(words):
#         c = counts[wid]
#         for p in zip(word, word[1:]):
#             pair_counts[p] += c
#             pair_index[p].add(wid)

#     merges = []
#     while len(vocab) < vocab_size and pair_counts:
#         # Same tie-break as before: highest count, then lexicographically greatest pair
#         best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
#         new_tok = best[0] + best[1]
#         merges.append(best)
#         vocab[len(vocab)] = new_tok

#         for wid in list(pair_index[best]):  # copy: the loop modifies this set
#             old, c = words[wid], counts[wid]

#             # Remove the old word's contribution
#             for p in zip(old, old[1:]):
#                 pair_counts[p] -= c
#                 if pair_counts[p] == 0:
#                     del pair_counts[p]
#                 pair_index[p].discard(wid)

#             # Add the new word's contribution
#             new = merge_word(old, best, new_tok)
#             words[wid] = new
#             for p in zip(new, new[1:]):
#                 pair_counts[p] += c
#                 pair_index[p].add(wid)

#         pair_index.pop(best, None)

#     return vocab, merges
   
# def train_bpe(
#     input_path: str | os.PathLike,
#     vocab_size: int,
#     special_tokens: list[str],
#     **kwargs,
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#     """Given the path to an input corpus, run train a BPE tokenizer and
#     output its vocabulary and merges.

#     Args:
#         input_path (str | os.PathLike): Path to BPE tokenizer training data.
#         vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
#         special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
#             These strings will never be split into multiple tokens, and will always be
#             kept as a single token. If these special tokens occur in the `input_path`,
#             they are treated as any other string.

#     Returns:
#         tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#             vocab:
#                 The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
#                 to bytes (token bytes)
#             merges:
#                 BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
#                 representing that <token1> was merged with <token2>.
#                 Merges are ordered by order of creation.
#     """
#     t0 = time.perf_counter()
#     special_tokens_bytes = [t.encode("utf-8") for t in special_tokens]
#     split_token = special_tokens_bytes[0] if special_tokens_bytes else b"<|endoftext|>"

#     total = Counter()
#     with open(input_path, "rb") as f:
#         f.seek(0, os.SEEK_END)
#         file_size = f.tell()
#         num_chunks = max(1, min(os.cpu_count() or 1, file_size // (1 << 20)))  # ~1 chunk per MB, capped
#         boundaries = find_chunk_boundaries(f, num_chunks, split_token)
#         t1 = time.perf_counter()
#         for start, end in zip(boundaries[:-1], boundaries[1:]):
#             f.seek(start)
#             chunk = f.read(end - start)
#             total.update(pre_tokenize_chunk(chunk, special_tokens))
#     t2 = time.perf_counter()    
#     vocab, merges = bpe_merges(total= total, vocab_size=vocab_size, special_tokens_bytes= special_tokens_bytes)
#     t3 = time.perf_counter()
#     print(f"chunking {t1-t0:.3f}s | pretokenize {t2-t1:.3f}s | merges {t3-t2:.3f}s")
#     return vocab, merges


import os
from multiprocessing import Pool
from collections import Counter

def _count_range(args):
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
    return pre_tokenize_chunk(chunk, special_tokens)

def pretokenize_file(input_path, special_tokens, num_workers=None):
    num_workers = num_workers or os.cpu_count() or 1
    split_token = (special_tokens[0] if special_tokens else "<|endoftext|>").encode("utf-8")

    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        # Small files (the speed test): 1 chunk, no pool startup cost
        # Large files: ~4 chunks per worker for load balancing
        num_chunks = 1 if file_size < (1 << 20) else num_workers * 4
        boundaries = find_chunk_boundaries(f, num_chunks, split_token)

    jobs = [(input_path, s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]
    total = Counter()
    if len(jobs) == 1:
        total.update(_count_range(jobs[0]))
    else:
        with Pool(num_workers) as pool:
            for d in pool.imap_unordered(_count_range, jobs):
                total.update(d)
    return total


import heapq
from collections import defaultdict

class _Rev:
    """Inverts comparison so the min-heap pops the lexicographically largest pair on ties."""
    __slots__ = ("p",)
    def __init__(self, p): self.p = p
    def __lt__(self, other): return self.p > other.p
    def __eq__(self, other): return self.p == other.p

def bpe_merges(total, vocab_size, special_tokens_bytes: list[bytes]):
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens_bytes:
        vocab[len(vocab)] = tok

    words = [tuple(bytes([b]) for b in w) for w in total]
    counts = list(total.values())

    pair_counts = defaultdict(int)
    pair_index = defaultdict(set)
    for wid, word in enumerate(words):
        for p in zip(word, word[1:]):
            pair_counts[p] += counts[wid]
            pair_index[p].add(wid)

    heap = [(-c, _Rev(p)) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges = []
    while len(vocab) < vocab_size:
        # Pop until we find an entry whose count is still current (lazy deletion)
        best = None
        while heap:
            negc, rp = heapq.heappop(heap)
            if pair_counts.get(rp.p) == -negc:
                best = rp.p
                break
        if best is None:
            break

        new_tok = best[0] + best[1]
        merges.append(best)
        vocab[len(vocab)] = new_tok

        touched = set()
        for wid in list(pair_index[best]):
            old, c = words[wid], counts[wid]
            for p in zip(old, old[1:]):
                pair_counts[p] -= c
                if pair_counts[p] == 0:
                    del pair_counts[p]
                pair_index[p].discard(wid)
                touched.add(p)
            new = merge_word(old, best, new_tok)
            words[wid] = new
            for p in zip(new, new[1:]):
                pair_counts[p] += c
                pair_index[p].add(wid)
                touched.add(p)
        pair_index.pop(best, None)

        # Push fresh entries for every pair whose count changed
        for p in touched:
            if p in pair_counts:
                heapq.heappush(heap, (-pair_counts[p], _Rev(p)))

    return vocab, merges

def train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    total = pretokenize_file(input_path, special_tokens)
    return bpe_merges(total, vocab_size, [t.encode("utf-8") for t in special_tokens])

