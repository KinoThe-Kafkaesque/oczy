"""Lane 06: Bounded-Growth Consolidation.

Measures the combined serialized footprint of the ExperienceAutoencoder and
IdentityHypernetwork organs after a small fixed curriculum of 4 corrections.
Lower is better: the lane wants the pair to remain <= ~1/10 of A0 (~229KB),
i.e. <= ~23KB.

The two organs are pure-numpy (no LM backend needed) and are constructed at
default config. A fixed deterministic curriculum runs 4 episodes through
``ExperienceAutoencoder.encode`` and the matching lessons through
``IdentityHypernetwork.update_identity``, so both organs grow from their
empty initial state to a non-trivial size. The measurement is the sum of
``status(include_size=True)['serialized_bytes']`` from the two organs.

Wrapped in try/except returning float('nan'); never raises.
Deterministic: fixed seed, fixed correction text.
"""

from __future__ import annotations


def name() -> str:
    return "lane_06_combined_footprint_bytes"


def measure() -> float:
    try:
        from experience_autoencoder import ExperienceAutoencoder
        from identity_hypernetwork import IdentityHypernetwork

        autoencoder = ExperienceAutoencoder()  # default seed=42
        hypernet = IdentityHypernetwork()  # default latent_dim=8, seed=0

        # Fixed 4-episode curriculum: each episode encodes into the
        # autoencoder's latent dz and, in parallel, registers the corrected
        # label as a hypernetwork identity-bump lesson. Both calls mutate the
        # organ's vocab / projection state to non-trivial size.
        episodes = [
            {
                "situation": "what color is the sky on a clear day",
                "model_answer": "green",
                "correction": "no the sky is blue not green",
                "revised_answer": "blue",
                "outcome": "corrected",
            },
            {
                "situation": "capital of france",
                "model_answer": "lyon",
                "correction": "incorrect the capital is paris",
                "revised_answer": "paris",
                "outcome": "corrected",
            },
            {
                "situation": "two plus two",
                "model_answer": "five",
                "correction": "wrong two plus two is four",
                "revised_answer": "four",
                "outcome": "corrected",
            },
            {
                "situation": "largest planet",
                "model_answer": "mars",
                "correction": "no jupiter is the largest planet",
                "revised_answer": "jupiter",
                "outcome": "corrected",
            },
        ]

        for ep in episodes:
            autoencoder.encode(ep)
            autoencoder.train_step(ep)  # hebbian update to the sensing matrix
            hypernet.update_identity(
                {
                    "source": "user_correction",
                    "correct_label": ep["revised_answer"],
                    "token": ep["revised_answer"],
                }
            )

        ae_status = autoencoder.status(include_size=True)
        hn_status = hypernet.status(include_size=True)
        ae_bytes = int(ae_status.get("serialized_bytes", 0))
        hn_bytes = int(hn_status.get("serialized_bytes", 0))
        return float(ae_bytes + hn_bytes)
    except Exception:
        return float("nan")