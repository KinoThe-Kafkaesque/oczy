"""Byte-level vocabulary for R24 tiny decoder.

Vocab 260 = 256 byte values + 4 specials.
Matches memory oczy-tiny-decoder-feasibility (v1): covers entire task charset (28 chars)
which is strict subset of byte range, so no OOV even for unseen strings.
"""
from __future__ import annotations

VOCAB_SIZE = 260
PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
UNK_ID = 259  # reserved, not used for byte decoding

# printable ASCII subset matters only for debugging
CHARSET_SIZE = 28  # live catalog charset, but vocab is full byte

def encode_bytes(text: str) -> list[int]:
    """Encode str -> list of byte ids (0-255)."""
    return list(text.encode("utf-8"))

def decode_bytes(ids: list[int]) -> str:
    """Decode byte ids 0-255 -> str. Skips specials >=256. EOS terminates."""
    filtered = []
    for i in ids:
        if i == EOS_ID:
            break
        if 0 <= i < 256:
            filtered.append(i)
    return bytes(filtered).decode("utf-8", errors="replace")

def encode_with_eos(text: str) -> list[int]:
    return encode_bytes(text) + [EOS_ID]

def encode_with_bos_eos(text: str) -> list[int]:
    return [BOS_ID] + encode_bytes(text) + [EOS_ID]

def max_answer_len() -> int:
    # live catalog max 23, pad to 32 for batching
    return 32
