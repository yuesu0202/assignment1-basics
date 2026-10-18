import ast
import pickle
from typing import Iterable, Iterator
import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
_BYTES_LIT = re.compile(r"""b'(?:[^'\\]|\\.)*'|b"(?:[^"\\]|\\.)*\"""")

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.byte_to_id = {b: i for i , b in self.vocab.items()}
        
        self.special_tokens = sorted(special_tokens or [], key = len, reverse = True)
        self.special_set = set(self.special_tokens)
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in self.byte_to_id:
                new_id = max(self.vocab, default=-1) + 1
                self.vocab[new_id] = b
                self.byte_to_id[b] = new_id
        
        self.special_re = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
            if self.special_tokens else None
        )
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.cache: dict[bytes, list[int]] = {}
        
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                   special_tokens: list[str] | None = None) -> "Tokenizer":
        return cls(cls._load_vocab(vocab_filepath), cls._load_merges(merges_filepath), special_tokens)

    @staticmethod
    def _load_vocab(path):
        if path.endswith(".pkl"):
            obj = pickle.load(open(path, "rb"))
            return obj["vocab"] if isinstance(obj, dict) and "vocab" in obj else obj
        vocab = {}                                            # "id\tb'...'" lines
        with open(path) as f:
            for line in f:
                i, r = line.rstrip("\n").split("\t", 1)
                vocab[int(i)] = ast.literal_eval(r)
        return vocab

    @staticmethod
    def _load_merges(path):
        if path.endswith(".pkl"):
            obj = pickle.load(open(path, "rb"))
            return obj["merges"] if isinstance(obj, dict) and "merges" in obj else obj
        merges = []                                           # "b'a' b'b'" lines
        with open(path) as f:
            for line in f:
                lits = _BYTES_LIT.findall(line)
                if len(lits) == 2:
                    merges.append((ast.literal_eval(lits[0]), ast.literal_eval(lits[1])))
        return merges
    
    def _bpe(self, word:bytes) -> list[int]:
        """Apply merges to one pre-token, lowest rank first."""
        if word in self.cache:
            return self.cache[word]
        parts = [bytes([b]) for b in word]
        while len(parts) > 1:
            best, best_rank = None, None
            for pair in zip(parts, parts[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break                                         # no applicable merge left
            merged, out, i = best[0] + best[1], [], 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == best[0] and parts[i + 1] == best[1]:
                    out.append(merged)
                    i += 2
                else:
                    out.append(parts[i])
                    i += 1
            parts = out
        ids = [self.byte_to_id[p] for p in parts]
        self.cache[word] = ids
        return ids
    
    def _encode_plain(self, text: str) -> list[int]:
        ids = []
        for m in PAT.finditer(text):
            ids.extend(self._bpe(m.group().encode("utf-8")))
        return ids
    
    def encode(self, text: str) -> list[int]:
        if self.special_re is None:
            return self._encode_plain(text)
        ids = []
        for seg in self.special_re.split(text):              # specials kept via capture group
            if not seg:
                continue
            if seg in self.special_set:
                ids.append(self.byte_to_id[seg.encode("utf-8")])
            else:
                ids.extend(self._encode_plain(seg))
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buf = ""
        for piece in iterable:
            buf += piece
            cut = self._safe_cut(buf)
            if cut > 0:
                yield from self.encode(buf[:cut])
                buf = buf[cut:]
        if buf:
            yield from self.encode(buf)
            
    def _safe_cut(self, buf: str) -> int:
        """Largest prefix length of buf whose encoding can't change when more text arrives."""
        # 1) Don't cut into a possible special token still arriving, e.g. buf ends with "<|endof"
        limit = len(buf)
        for tok in self.special_tokens:
            for k in range(min(len(tok) - 1, len(buf)), 0, -1):
                if buf.endswith(tok[:k]):
                    limit = min(limit, len(buf) - k)
                    break
        head = buf[:limit]
        # 2) Text right after a complete special token starts fresh
        start = 0
        if self.special_re is not None:
            for m in self.special_re.finditer(head):
                start = m.end()
        # 3) The last pre-token might still grow (" hel" + "lo"), so hold it back
        starts = [m.start() for m in PAT.finditer(head, start)]
        return start + starts[-1] if starts else start

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

