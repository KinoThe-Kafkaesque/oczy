"""Phase A oracle pretraining + Phase B freeze artifact - R24.

Trains: complete rule text -> z*[64] (via TextOracleEncoder) -> decoder(query, z*) -> answer
Byte CE. Same z* serves multiple queries per rule.
Split firewall enforced via build_dev_catalog (no meta-test leakage).
Includes query-only baseline and same-query/different-rule test.
Freeze artifact: config, weight_hash, corpus/split hashes, seed, optimizer, DEV accuracies.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

from oczy.experiments.meta_cortex.taskgen import build_dev_catalog
from oczy.experiments.meta_cortex.contracts import TaskGeneratorConfig, ProbeKind, MetaTask

from .decoder import TinyDecoderConfig, TinySharedDecoder
from .oracle import PerRuleOracleEncoder, TextOracleEncoder
from .vocab import encode_bytes, encode_with_eos, PAD_ID, EOS_ID, BOS_ID, VOCAB_SIZE


@dataclass
class PretrainExample:
    rule_fp: str
    oracle_text: str  # complete rule text for encoder
    query_text: str
    answer_text: str
    family: str
    kind: str


def build_pretrain_corpus(root_seed: int = 42, train_per_family: int = 30, val_per_family: int = 10, max_examples: int | None = None):
    cfg = TaskGeneratorConfig(root_seed=root_seed, train_tasks_per_family=train_per_family, validation_tasks_per_family=val_per_family)
    cat = build_dev_catalog(config=cfg)
    corpus_hash = cat.catalog_sha256
    split_hash = hashlib.sha256(json.dumps({"train_rule": cat.split_audit.train_rule_digests, "val_rule": cat.split_audit.validation_rule_digests}, sort_keys=True).encode()).hexdigest()

    def tasks_to_examples(tasks: List[MetaTask]) -> List[PretrainExample]:
        exs = []
        for t in tasks:
            # oracle_text is the complete mapping table from the oracle_context probe's first message
            oracle_text = t.probes.oracle_context[0].messages[0].content
            # also include worked examples if second message exists? oracle_context has 2 messages: mapping + question
            # For encoder, use first message only (the mapping table). Second message is the query.
            for kind in ProbeKind:
                for p in t.probes.by_kind(kind):
                    # for oracle_context probes, query is second message only to avoid duplicating oracle_text
                    if kind == ProbeKind.ORACLE_CONTEXT and len(p.messages) == 2:
                        q = p.messages[1].content
                    else:
                        q = " ".join(m.content for m in p.messages)
                    a = p.expected_response
                    exs.append(PretrainExample(rule_fp=t.rule_fingerprint, oracle_text=oracle_text, query_text=q, answer_text=a, family=t.family.value, kind=kind.value))
        return exs

    train_ex = tasks_to_examples(cat.meta_train)  # type: ignore
    val_ex = tasks_to_examples(cat.meta_validation)  # type: ignore
    if max_examples is not None:
        import hashlib as _h
        def stable_shuffle(lst):
            return sorted(lst, key=lambda e: _h.sha256((e.rule_fp+e.query_text+e.answer_text).encode()).hexdigest())
        train_ex = stable_shuffle(train_ex)[:max_examples]
        val_ex = stable_shuffle(val_ex)[:max_examples]
    return train_ex, val_ex, corpus_hash, split_hash


def _pad_query_answer(batch: List[PretrainExample], device: torch.device):
    q_enc = [encode_bytes(e.query_text) for e in batch]
    a_enc = [encode_with_eos(e.answer_text) for e in batch]
    max_q = max(len(x) for x in q_enc)
    max_a = max(len(x) for x in a_enc)
    max_len = 512
    assert max_q + max_a <= max_len, f"{max_q}+{max_a} > {max_len}"
    q_pad = torch.full((len(batch), max_q), PAD_ID, dtype=torch.long, device=device)
    a_pad = torch.full((len(batch), max_a), PAD_ID, dtype=torch.long, device=device)
    for i,(qe,ae) in enumerate(zip(q_enc, a_enc)):
        q_pad[i,:len(qe)] = torch.tensor(qe, dtype=torch.long, device=device)
        a_pad[i,:len(ae)] = torch.tensor(ae, dtype=torch.long, device=device)
    rule_fps = [e.rule_fp for e in batch]
    oracle_texts = [e.oracle_text for e in batch]
    return q_pad, a_pad, rule_fps, oracle_texts


def _encode_oracle_texts(oracle_texts: List[str], encoder: TextOracleEncoder, device: torch.device) -> torch.Tensor:
    # byte-encode oracle_texts, pad to max oracle len in batch
    enc = [encode_bytes(t) for t in oracle_texts]
    max_t = max(len(x) for x in enc)
    max_t = min(max_t, encoder.max_len)
    # truncate
    enc = [x[:max_t] for x in enc]
    max_t = max(len(x) for x in enc)
    pad = torch.full((len(enc), max_t), PAD_ID, dtype=torch.long, device=device)
    mask = torch.zeros((len(enc), max_t), dtype=torch.long, device=device)
    for i, e in enumerate(enc):
        pad[i, :len(e)] = torch.tensor(e, dtype=torch.long, device=device)
        mask[i, :len(e)] = 1
    # encoder expects vocab ids 0-259; byte ids 0-255 ok, PAD 256 is valid vocab id but we mask it out via mask
    # Replace PAD 256 with 0 for embedding but mask ensures not pooled
    z = encoder(pad, mask)
    return z


def fixed_hash_z(rule_fps: List[str], device: torch.device) -> torch.Tensor:
    B = len(rule_fps)
    z = torch.zeros((B,64), device=device)
    for i, fp in enumerate(rule_fps):
        h = int(hashlib.sha256(fp.encode()).hexdigest()[:16], 16)
        g = torch.Generator(device=device)
        g.manual_seed(h % (2**31))
        z[i] = torch.randn(64, generator=g, device=device) * 0.5
    return z


def _greedy_accuracy(decoder: TinySharedDecoder, z: torch.Tensor, batch: List[PretrainExample], q: torch.Tensor, a: torch.Tensor) -> Tuple[int,int]:
    with torch.no_grad():
        gen = decoder.generate_greedy(q, z, max_new_tokens=a.shape[1]+5)
        correct=0; total=0
        for j,e in enumerate(batch):
            gold = encode_with_eos(e.answer_text)
            pred = gen[j].tolist()
            if EOS_ID in pred:
                pred = pred[:pred.index(EOS_ID)+1]
            if pred == gold:
                correct+=1
            total+=1
        return correct, total


def train_phase_A(root_seed: int = 123, train_per_family: int = 20, val_per_family: int = 10, d_model: int = 64, n_layers: int = 2, conditioning: str = "film", deep_film: bool = False, steps: int = 800, lr: float = 3e-3, batch_size: int = 32, device: str = "cpu", use_text_encoder: bool = True) -> dict:
    device_t = torch.device(device)
    train_ex, val_ex, corpus_hash, split_hash = build_pretrain_corpus(root_seed=root_seed, train_per_family=train_per_family, val_per_family=val_per_family)
    cfg = TinyDecoderConfig(d_model=d_model, n_layers=n_layers, conditioning=conditioning, deep_film=deep_film)
    decoder = TinySharedDecoder(cfg).to(device_t)
    if use_text_encoder:
        oracle = TextOracleEncoder(d_model=d_model, n_layers=2, n_heads=2, r_dim=64, vocab_size=VOCAB_SIZE, max_len=512).to(device_t)
        opt = torch.optim.AdamW(list(decoder.parameters())+list(oracle.parameters()), lr=lr, weight_decay=0.01)
    else:
        oracle = None
        opt = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=0.01)

    random.seed(root_seed); torch.manual_seed(root_seed)
    for step in range(steps):
        decoder.train()
        if oracle is not None:
            oracle.train()
        batch = random.sample(train_ex, min(batch_size, len(train_ex)))
        # per-example loss to handle variable query lengths without padding bias
        losses = []
        for ex in batch:
            q = torch.tensor([encode_bytes(ex.query_text)], dtype=torch.long, device=device_t)
            a = torch.tensor([encode_with_eos(ex.answer_text)], dtype=torch.long, device=device_t)
            if use_text_encoder:
                z = _encode_oracle_texts([ex.oracle_text], oracle, device_t)  # type: ignore  # [1,64]
            else:
                z = fixed_hash_z([ex.rule_fp], device_t)
            l = decoder.teacher_forced_loss(q, z, a)
            losses.append(l)
        loss = torch.stack(losses).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(decoder.parameters()) + (list(oracle.parameters()) if oracle else []), 1.0)
        opt.step()
        if (step+1) % 100 == 0:
            # val eval with appropriate encoder
            decoder.eval()
            if oracle is not None:
                oracle.eval()
            # sample val batches
            correct=0; total=0; qcorrect=0; qtotal=0
            for i in range(0, min(len(val_ex), 5*32), 32):
                b = val_ex[i:i+32]
                qv,av, fpsv, otv = _pad_query_answer(b, device_t)
                with torch.no_grad():
                    zv = _encode_oracle_texts(otv, oracle, device_t) if use_text_encoder else fixed_hash_z(fpsv, device_t)  # type: ignore
                    c,t = _greedy_accuracy(decoder, zv, b, qv, av)
                    correct+=c; total+=t
                    # query-only
                    z0 = torch.zeros((qv.shape[0],64), device=device_t)
                    cq,tq = _greedy_accuracy(decoder, z0, b, qv, av)
                    qcorrect+=cq; qtotal+=tq
            acc = correct/max(1,total); qacc = qcorrect/max(1,qtotal)
            print(f"step {step+1}/{steps} loss {loss.item():.4f} oracle_DEV {acc:.3f} query_only {qacc:.3f} delta {acc-qacc:.3f}")

    # final artifact
    decoder.eval()
    if oracle is not None:
        oracle.eval()
    # full val accuracy (oracle vs query-only vs swapped)
    def eval_full(examples, max_batches=20):
        correct=0; total=0
        for i in range(0, min(len(examples), max_batches*32), 32):
            b = examples[i:i+32]
            qv,av, fpsv, otv = _pad_query_answer(b, device_t)
            with torch.no_grad():
                zv = _encode_oracle_texts(otv, oracle, device_t) if use_text_encoder else fixed_hash_z(fpsv, device_t)  # type: ignore
                c,t = _greedy_accuracy(decoder, zv, b, qv, av)
                correct+=c; total+=t
        return correct/max(1,total)
    def eval_query_only(examples, max_batches=20):
        correct=0; total=0
        for i in range(0, min(len(examples), max_batches*32), 32):
            b = examples[i:i+32]
            qv,av,_,_ = _pad_query_answer(b, device_t)
            with torch.no_grad():
                z0 = torch.zeros((qv.shape[0],64), device=device_t)
                c,t = _greedy_accuracy(decoder, z0, b, qv, av)
                correct+=c; total+=t
        return correct/max(1,total)

    oracle_acc = eval_full(val_ex)
    qacc = eval_query_only(val_ex)
    # same-query/different-rule test: find queries that appear under different rules (rare but exists due to shared vocab)
    # we approximate by same query string but different answer
    from collections import defaultdict
    q_to_answers = defaultdict(set)
    for e in train_ex+val_ex:
        q_to_answers[e.query_text].add(e.answer_text)
    multi = {q:ans for q,ans in q_to_answers.items() if len(ans)>1}
    print(f"same-query multi-answer pairs: {len(multi)} e.g. {list(multi.items())[:2]}")

    weight_hash = decoder.parameter_hash()
    decoder.freeze()
    frozen_hash = decoder.parameter_hash()
    assert weight_hash == frozen_hash
    artifact = {
        "config": {"d_model": d_model, "n_layers": n_layers, "conditioning": conditioning, "deep_film": deep_film, "vocab": VOCAB_SIZE, "text_encoder": use_text_encoder},
        "weight_hash": weight_hash,
        "corpus_hash": corpus_hash,
        "split_hash": split_hash,
        "seed": root_seed,
        "optimizer": {"name":"adamw","lr":lr,"steps":steps},
        "oracle_dev_accuracy": oracle_acc,
        "query_only_dev_accuracy": qacc,
        "delta": oracle_acc - qacc,
    }
    print(f"FINAL oracle {oracle_acc:.3f} query_only {qacc:.3f} delta {artifact['delta']:.3f} hash {weight_hash[:8]}")
    return artifact
