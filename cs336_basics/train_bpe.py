import os
from typing import BinaryIO, Counter
from collections import defaultdict, Counter
from numpy import byte
import regex as re

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

def bpe_merges(total: Counter, vocab_size: int, special_tokens_bytes: list[str]):
    # Initial vocab: 256 single bytes + special tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens_bytes:
        vocab[len(vocab)] = tok

    # b"caf\xc3\xa9" -> (b'c', b'a', b'f', b'\xc3', b'\xa9')
    words = Counter({tuple(bytes([b]) for b in w): c for w, c in total.items()})
    merges = []

    while len(vocab) < vocab_size:
        pair_counts = Counter()
        for word, c in words.items():
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] += c
        if not pair_counts:
            break  # nothing left to merge

        # Highest count; ties broken by lexicographically greater pair
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        new_tok = best[0] + best[1]
        merges.append(best)
        vocab[len(vocab)] = new_tok

        new_words = Counter()
        for word, c in words.items():
            new_words[merge_word(word, best, new_tok)] += c
        words = new_words

    return vocab, merges
   
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    
    file = open(input_path, "rb")
    num_chunks = 128
    special_token_bytes = special_tokens.encode("utf-8")
    boundaries = find_chunk_boundaries(file, num_chunks, special_token_bytes)
    
    vocab = {}
    merges = []
    total = Counter()
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        chunk = file.read(end - start)
        d = pre_tokenize_chunk(chunk, special_tokens)
        total.update(pre_tokenize_chunk(chunk, special_tokens))
    
    
    return boundaries