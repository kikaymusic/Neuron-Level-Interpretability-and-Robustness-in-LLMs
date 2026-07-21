"""
Alternative neuron-importance rankings for reviewers R1.1 / R3.3 / R7.5 (group #8).

The decision (option c) is to compute two neuron-level attribution methods and compare
their rankings against SYNAPSE's linear-probe ranking (rank correlation + top-k overlap,
via ranking_agreement.py). The probe stays the operational method; these only VALIDATE it.

Methods (both defined at the [CLS] position, over the L*hidden neuron space, matching the
probe):
  - activation x gradient  : |a_i * d(logit_target)/d a_i|, averaged over examples. The
    cheap, standard neuron-importance proxy (Shrikumar et al., "Computationally Efficient
    Measures of Internal Neuron Importance").
  - conductance            : the integrated-gradients-based measure of neuron importance
    (Dhamdhere et al., "How Important Is a Neuron?", ICLR 2019), integrated over the layer
    activation path (pure torch; works with integer input ids, no external dependency).

Note: standard SHAP / integrated gradients are FEATURE-attribution methods; the reviewer
asked for neuron-level analysis, so we use their neuron-level analogues (as argued in the
rebuttal). Both methods are pure torch — no captum / new dependency.
"""

from __future__ import annotations
import numpy as np
import torch


def _target_idx(logits, target):
    return int(logits.argmax(dim=-1)) if target == "pred" else int(target)


def _agg_positions(t, agg, cls_pos):
    """Aggregate a [1, seq, hidden] tensor over positions -> [hidden]. agg: 'cls' or 'mean'."""
    if agg == "mean":
        return t.mean(dim=1).squeeze(0)          # mean over tokens (for [CLS]-less models, e.g. GPT-2)
    return t[:, cls_pos, :].squeeze(0)           # the [CLS] position


def activation_times_gradient(forward_fn, samples, layer_modules, target="pred",
                              cls_pos=0, agg="cls"):
    """
    importance[L*hidden] = mean over samples of |act * grad|, aggregated over positions.

    forward_fn(sample) -> logits [1, C]   (sample is whatever the closure needs)
    layer_modules      : list of L modules whose output is (or starts with) a
                         [1, seq, hidden] tensor.
    agg                : 'cls' (position cls_pos, encoders) or 'mean' (over tokens, GPT-2).
    """
    acc, n = None, 0
    for sample in samples:
        captured = {}
        handles = []
        for li, m in enumerate(layer_modules):
            def mk(li):
                def hook(mod, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    o.retain_grad()
                    captured[li] = o
                return hook
            handles.append(m.register_forward_hook(mk(li)))
        try:
            logits = forward_fn(sample)
            tgt = _target_idx(logits, target)
            logits[0, tgt].backward()
            vecs = []
            for li in range(len(layer_modules)):
                o = captured[li]
                imp = (_agg_positions(o, agg, cls_pos) *
                       _agg_positions(o.grad, agg, cls_pos)).abs().detach().cpu().numpy()
                vecs.append(imp)
        finally:
            for h in handles:
                h.remove()
        v = np.concatenate(vecs)
        acc = v if acc is None else acc + v
        n += 1
    return acc / max(1, n)


def conductance_importance(forward_fn, samples, layer_modules, target="pred",
                           cls_pos=0, n_steps=20, agg="cls"):
    """
    Neuron conductance (integrated-gradients-based importance; Dhamdhere et al., "How
    Important Is a Neuron?", ICLR 2019), implemented in PURE TORCH by integrating the
    gradient of the target logit w.r.t. each layer's activation along a straight path from
    a zero baseline to the actual activation:

        conductance_i = a_i * (1/m) * sum_alpha  d(logit_target)/d a_i |_{alpha * a}

    Interpolating the LAYER activation (not the input) means it works with integer input
    ids (BERT/GPT-2) without captum. No external dependency.

    forward_fn(sample) -> logits [1, C]; layer_modules: list of L modules with a
    [1, seq, hidden] (or tuple-first) output. agg: 'cls' or 'mean'. n_steps: Riemann steps.
    """
    acc, n = None, 0
    for sample in samples:
        # 1) capture the actual activations (detached constants for the path)
        actual = {}
        handles = []
        for li, m in enumerate(layer_modules):
            def mk(li):
                def hook(mod, inp, out):
                    actual[li] = (out[0] if isinstance(out, tuple) else out).detach()
                return hook
            handles.append(m.register_forward_hook(mk(li)))
        forward_fn(sample)
        for h in handles:
            h.remove()

        # 2) per layer, integrate gradients over the interpolation path
        vecs = []
        alphas = torch.linspace(0.0, 1.0, n_steps)
        for li, m in enumerate(layer_modules):
            a = actual[li]
            grad_sum = torch.zeros_like(a)
            for alpha in alphas:
                scaled = (float(alpha) * a).clone().requires_grad_(True)
                def repl(mod, inp, out, s=scaled):
                    return ((s,) + tuple(out[1:])) if isinstance(out, tuple) else s
                h = m.register_forward_hook(repl)
                try:
                    logits = forward_fn(sample)
                    tgt = _target_idx(logits, target)
                    g = torch.autograd.grad(logits[0, tgt], scaled)[0]
                finally:
                    h.remove()
                grad_sum += g
            cond = a * (grad_sum / n_steps)
            vecs.append(_agg_positions(cond, agg, cls_pos).abs().detach().cpu().numpy())
        v = np.concatenate(vecs)
        acc = v if acc is None else acc + v
        n += 1
    return acc / max(1, n)


# ---------------------------------------------------------------------------
# Self-test (local): a toy model where one neuron of layer-2's CLS output drives
# the target class. Both methods must rank that neuron at the top.
# ---------------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)
    H, C, SEQ = 6, 3, 4

    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(H, H)
        def forward(self, x):
            return torch.relu(self.lin(x))            # [1, seq, H] tensor output

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1, self.l2 = ToyLayer(), ToyLayer()
            self.head = torch.nn.Linear(H, C)
            with torch.no_grad():                     # make neuron 3 (layer-2 CLS) drive class 1
                self.head.weight.zero_(); self.head.weight[1, 3] = 5.0
        def forward(self, x):
            return self.head(self.l2(self.l1(x))[:, 0, :])   # classify CLS

    model = Toy().eval()
    layers = [model.l1, model.l2]
    xs = [torch.randn(1, SEQ, H) for _ in range(8)]

    imp = activation_times_gradient(lambda x: model(x), xs, layers, target=1, cls_pos=0)
    assert imp.shape == (2 * H,), imp.shape
    # the driving neuron is index H+3 (layer-2, neuron 3); must be the max
    assert int(imp.argmax()) == H + 3, (imp.argmax(), imp)
    print(f"[selftest] activation x gradient OK (top neuron = {imp.argmax()} == {H+3}); shape {imp.shape}")

    impc = conductance_importance(lambda x: model(x), xs, layers, target=1, cls_pos=0, n_steps=20)
    assert impc.shape == (2 * H,), impc.shape
    assert int(impc.argmax()) == H + 3, (impc.argmax(), impc)
    print(f"[selftest] conductance (pure torch) OK (top neuron = {impc.argmax()} == {H+3}); shape {impc.shape}")

    # mean-aggregation path (for [CLS]-less models like GPT-2) must also run and be sane
    impm = activation_times_gradient(lambda x: model(x), xs, layers, target=1, agg="mean")
    assert impm.shape == (2 * H,)
    print(f"[selftest] agg='mean' OK; shape {impm.shape}")
    print("ALL SELFTESTS PASSED")


if __name__ == "__main__":
    _selftest()
