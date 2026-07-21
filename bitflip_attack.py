"""
Attribution-guided bit-flip attack (reviewer R7.9 — hardware faults / tampering, TH-3).

Grounded in the established bit-flip / hardware-fault literature:
  - Hong et al., "Terminal Brain Damage: Exposing the Graceless Degradation in Deep
    Neural Networks Under Hardware Fault Attacks", USENIX Security 2019 — a SINGLE flip
    of the most-significant exponent bit of a float32 parameter causes catastrophic
    accuracy degradation. This motivates flipping the exponent MSB.
  - Rakin, Chen & Fan, "Bit-Flip Attack: Crushing Neural Network with Progressive Bit
    Search (BFA)", ICCV 2019 — the BFA framework. We deliberately DO NOT use their
    gradient-based progressive bit search (out of scope); instead the target parameters
    are chosen by SYNAPSE's probe-derived neuron ranking (attribution-guided targeting).
  - Rowhammer (Kim et al., ISCA 2014; "OneFlip") — the physical mechanism that makes
    single-bit flips in DRAM realizable.

Honest framing: attribution-guided fault injection, NOT a SOTA optimal BFA. The flip is
reversible (view the float32 storage as int32, XOR the exponent-MSB bit, view back).

IEEE-754 float32 layout:  [ sign:1 | exponent:8 | mantissa:23 ]
                            bit 31    bits 30..23   bits 22..0
The most-significant exponent bit is bit 30 -> XOR mask 0x40000000.
"""

from __future__ import annotations
import numpy as np

# MSB of the 8-bit exponent of an IEEE-754 float32 (bit 30).
EXP_MSB_MASK = np.int32(0x40000000)


def flip_exponent_msb(values):
    """Return a copy of `values` (float32) with the exponent-MSB flipped on every element.
    Pure numpy; used by the self-test and usable on any array."""
    a = np.array(values, dtype=np.float32)          # own copy
    a.view(np.int32)[...] ^= EXP_MSB_MASK            # in-place XOR on the int view
    return a


def flip_exponent_msb_columns_(weight, cols) -> int:
    """
    In-place: flip the exponent-MSB bit of every float32 weight in the given COLUMNS of a
    2-D weight tensor (shape [out_features, in_features]). Columns index the input
    dimension (= hidden units), matching the neuron->column mapping used by the existing
    weight-space attacks (idx = neuron % hidden_size).

    `weight` is a torch.Tensor (float32). Returns the number of bits flipped.
    Reversible: call again on the same columns to restore.
    """
    import torch
    if weight.dtype != torch.float32:
        raise TypeError(f"bit-flip defined for float32 weights; got {weight.dtype}")
    cols = torch.as_tensor(sorted(set(int(c) for c in cols)), dtype=torch.long, device=weight.device)
    iv = weight.view(torch.int32)                    # shares storage with `weight`
    iv[:, cols] = iv[:, cols] ^ torch.tensor(int(EXP_MSB_MASK), dtype=torch.int32, device=weight.device)
    return int(weight.shape[0] * cols.numel())


# ---------------------------------------------------------------------------
# Self-test (local): correctness + reversibility of the exponent-MSB flip.
# ---------------------------------------------------------------------------
def _selftest():
    # 1) Known IEEE-754 values (numpy core).
    #    1.0f = 0x3F800000 ; flipping bit 30 -> 0x7F800000 = +inf.
    assert np.isposinf(flip_exponent_msb(np.float32(1.0))), "1.0 exponent-MSB flip should give +inf"
    #    2.0f = 0x40000000 ; flipping bit 30 -> 0x00000000 = 0.0.
    assert flip_exponent_msb(np.float32(2.0)) == np.float32(0.0), "2.0 flip should give 0.0"
    # 2) Reversibility: flipping twice restores the original bit pattern.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000).astype(np.float32)
    back = flip_exponent_msb(flip_exponent_msb(x))
    assert np.array_equal(back.view(np.int32), x.view(np.int32)), "double flip must be identity"
    # 3) A single flip changes (almost) every finite non-zero value.
    changed = np.sum(x.view(np.int32) != flip_exponent_msb(x).view(np.int32))
    assert changed == x.size, "every element's bit pattern should change"
    print(f"[selftest] numpy core OK (reversible; {changed}/{x.size} changed)")

    # 4) Torch column flip on a weight matrix: reversible and restricted to `cols`.
    try:
        import torch
    except Exception:
        print("[selftest] torch not available, skipping tensor test"); print("ALL SELFTESTS PASSED"); return
    W = torch.randn(5, 8, dtype=torch.float32)
    W0 = W.clone()
    cols = [1, 4, 7]
    n = flip_exponent_msb_columns_(W, cols)
    assert n == 5 * 3
    # untouched columns identical
    keep = [c for c in range(8) if c not in cols]
    assert torch.equal(W[:, keep], W0[:, keep]), "non-target columns must be unchanged"
    # target columns changed
    assert not torch.equal(W[:, cols], W0[:, cols]), "target columns must change"
    # restore
    flip_exponent_msb_columns_(W, cols)
    assert torch.equal(W.view(torch.int32), W0.view(torch.int32)), "double flip must restore weights"
    print("[selftest] torch column flip OK (reversible, column-restricted)")
    print("ALL SELFTESTS PASSED")


if __name__ == "__main__":
    _selftest()
