"""Versioned greedy batch candidate; historical organ execution stays frozen.

Transformers 5.0.0 extends the attention mask during generation, but does not
extend caller-supplied position IDs. Let it derive positions afresh on each
step. Explicit GenerationConfig also avoids inherited repetition penalties.
This helper is qualified separately before adoption by an experiment.
"""

import torch
from transformers import GenerationConfig

from .organ import render_chat


@torch.inference_mode()
def generate_batch(organ, messages_batch, soft_banks, max_new_tokens=32, *, repetition_penalty=1.0):
    organ._check_open()
    batch = len(messages_batch)
    if batch == 0 or soft_banks.ndim != 3 or soft_banks.shape[0] != batch or soft_banks.shape[2] != organ.feature_dim:
        raise ValueError("Expected nonempty prompts and matching [batch, width, feature_dim] banks")
    if not torch.isfinite(soft_banks).all():
        raise ValueError("Nonfinite bank")
    tokenizer = organ._tokenizer
    embed = organ._model.get_input_embeddings()
    device = embed.weight.device
    tokens = [tokenizer.encode(render_chat(prompt, tokenizer), add_special_tokens=True) for prompt in messages_batch]
    width, length = soft_banks.shape[1], max(map(len, tokens))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.full((batch, length), pad, dtype=torch.long, device=device)
    mask = torch.zeros((batch, width + length), dtype=torch.long, device=device)
    mask[:, :width] = 1
    for i, row in enumerate(tokens):
        ids[i, -len(row):] = torch.tensor(row, dtype=torch.long, device=device)
        mask[i, width + length - len(row):] = 1
    embedded = embed(ids)
    inputs = torch.cat([soft_banks.to(device=device, dtype=embedded.dtype), embedded], dim=1)
    config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                              repetition_penalty=repetition_penalty, pad_token_id=pad, eos_token_id=tokenizer.eos_token_id)
    sequences = organ._model.generate(inputs_embeds=inputs, attention_mask=mask, generation_config=config)
    answers = []
    for row in sequences.tolist():
        if tokenizer.eos_token_id in row:
            row = row[:row.index(tokenizer.eos_token_id) + 1]
        answers.append(tokenizer.decode(row, skip_special_tokens=True))
    return answers
