"""Exercise the production decoding boundary with controlled generated token IDs."""

from types import SimpleNamespace

import pytest
import torch

from oczy.experiments.meta_cortex.calibration import FrozenScorer
from oczy.experiments.meta_cortex.contracts import DialogueMessage
from oczy.experiments.meta_cortex.organ import QwenFrozenOrgan


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 3

    def apply_chat_template(self, *args, **kwargs):
        return "prompt"

    def encode(self, text, **kwargs):
        return [0]

    def decode(self, ids, *, skip_special_tokens=False):
        vocabulary = {0: "", 1: "left", 2: "<|im_end|>", 3: "<|endoftext|>", 4: "."}
        return "".join(vocabulary[i] for i in ids if not skip_special_tokens or i not in (2, 3))


class TokenModel(torch.nn.Module):
    def __init__(self, sequence):
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 4)
        self.sequence = sequence
        self.position = 0

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, **kwargs):
        logits = torch.full((1, 1, 5), -100.0)
        logits[0, 0, self.sequence[self.position]] = 100.0
        self.position += 1
        return SimpleNamespace(logits=logits, past_key_values=None)

    def generate(self, inputs_embeds, **kwargs):
        return torch.tensor([self.sequence] * inputs_embeds.shape[0])


def adapter(sequence, *, same_pad_eos):
    organ = QwenFrozenOrgan.__new__(QwenFrozenOrgan)
    organ._closed = False
    organ.feature_dim = 4
    organ._model = TokenModel(sequence).eval()
    organ._tokenizer = Tokenizer()
    if same_pad_eos:
        organ._tokenizer.pad_token_id = organ._tokenizer.eos_token_id
    return organ


@pytest.mark.parametrize("same_pad_eos", [False, True])
@pytest.mark.parametrize("sequence,expected", [([1, 2], "left"), ([2], ""), ([1, 4, 2], "left.")])
def test_scalar_and_batch_return_answer_text_without_transport_tokens(same_pad_eos, sequence, expected):
    organ = adapter(sequence, same_pad_eos=same_pad_eos)
    messages = (DialogueMessage("user", "Which direction?"),)
    bank = torch.zeros(1, 0, 4)
    scalar = organ.generate(messages, bank, max_new_tokens=4)
    batch = organ.generate_batch([messages], bank, max_new_tokens=4)
    assert scalar == expected
    assert batch == [expected]
    # Punctuation remains meaningful and EOS-only output is still wrong.
    assert FrozenScorer().score_response("left", scalar) == (expected == "left")
