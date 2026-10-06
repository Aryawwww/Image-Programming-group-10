"""The 16 candidate rounds. C and HSV alpha only; HOG cell parameters stay fixed."""

from __future__ import annotations

C_VALUES = (0.01, 0.1, 1.0, 10.0)
HSV_ALPHAS = (0.0, 0.25, 0.5, 1.0)
TRAIN_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
SEED = 0


def iter_rounds() -> list[dict[str, float | int]]:
    rounds: list[dict[str, float | int]] = []
    number = 1
    for c_value in C_VALUES:
        for alpha in HSV_ALPHAS:
            rounds.append({"round": number, "C": float(c_value), "alpha": float(alpha)})
            number += 1
    if len(rounds) != 16:
        raise RuntimeError(f"expected 16 rounds, got {len(rounds)}")
    return rounds


def sgd_alpha(c_value: float, n_samples: int) -> float:
    """Map a linear-SVM C to sklearn SGD alpha: alpha = 1 / (C * n)."""
    return 1.0 / (float(c_value) * max(int(n_samples), 1))
