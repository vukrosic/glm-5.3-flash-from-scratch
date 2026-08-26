"""A deterministic byte tokenizer trained from no external vocabulary."""
from __future__ import annotations


class ByteTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    sep_id = 3
    byte_offset = 4
    vocab_size = 260

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = [self.byte_offset + value for value in text.encode("utf-8")]
        if bos:
            ids.insert(0, self.bos_id)
        if eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        values = bytearray()
        for token in ids:
            if token == self.eos_id:
                break
            if token >= self.byte_offset:
                values.append(token - self.byte_offset)
        return values.decode("utf-8", errors="replace")
