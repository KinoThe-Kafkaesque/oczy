"""R24 tiny shared frozen decoder package."""
from .decoder import TinyDecoderConfig, TinySharedDecoder
from .vocab import VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID, encode_bytes, decode_bytes
from .oracle import PerRuleOracleEncoder, TextOracleEncoder
__all__ = ["TinyDecoderConfig","TinySharedDecoder","PerRuleOracleEncoder","TextOracleEncoder","VOCAB_SIZE","PAD_ID","BOS_ID","EOS_ID"]
