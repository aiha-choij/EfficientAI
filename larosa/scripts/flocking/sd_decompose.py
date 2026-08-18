"""s/d decomposition -- variance-component measurement of grouping cost
(topic: sd-decomposition, request 20260818-223947-sd-decomp). Diagnostic
only, no training -- hooks and forward passes over LLaMA2-7B.

Answers three questions (see the request doc for full derivation):
  Q1 (GSLR post-hoc check): did GSLR's l2,1 penalty raise static importance
      variance (sigma_mu^2) rather than lower token-unique variance (d)?
  Q2 (G0 headroom): which variance component dominates grouping cost Delta(g)
      in the ORIGINAL model, per layer, and how does Delta(g) scale with g?
  Q3 (attribution): does token-unique variance d come from the gate branch
      or the up branch of the LLaMA FFN?

Theory summary (contribution mass and 3-component variance model):
  FFN: u = W_up x, a = act_fn(W_gate x), i = u * a, y = W_down i.
  Contribution mass of neuron j at token t: w_tj = i_tj^2 * c_j^2, where
  c_j = ||W_down0[:, j]||_2 is ALWAYS the ORIGINAL (untuned) down_proj column
  norm -- frozen gauge, never recomputed from a tuned weight.
  3-component model per block T of g consecutive tokens (never crossing a
  2048-token window boundary):
      w_tj = mu_j + c_ctx_j(T) + eps_tj
      s := sigma_mu^2 + rho*sigma^2  (block-shared variance)
      d := (1-rho)*sigma^2           (token-unique variance)
  Grouping cost: Delta(g) = k*lambda(p) * (sqrt(s+d) - sqrt(s+d/g)); the only
  thing that makes Delta > 0 is d. Scale-free form:
      Delta_norm(g) = 1 - sqrt(rho* + (1-rho*)/g),  rho* = s/(s+d)

Estimators (see request doc section 2 for the full derivation; implemented
exactly, fp64 running accumulators per pitfall #2):
  d_hat       = mean over blocks T, neurons j of SS_j(T)/(g-1) / D
  V_A_hat     = mean over blocks T of population-variance-across-j of A_j(T)
  s_hat       = V_A_hat - d_hat/g               (clipped >= 0 for reporting)
  sigma_mu2_hat = bias-corrected variance-across-j of block-mean mu_hat_j
  rho_sigma2_hat = s_hat - sigma_mu2_hat

Reuses (per request instructions, from auto/gslr-stage1):
  - gslr_group_ppl.swap_layer / should_use_retuned for GSLR/B1d weight
    loading (layer 31 always original). We call swap_layer with
    condition="gslr_dense" everywhere -- this installs retuned weights for
    layers 0..30 but MaskedMLP.apply_mask=False, i.e. no masking, exactly
    what this diagnostic needs (statistics on unmasked i).
  - analyze_topk_overlap.py's forward-hook-on-activation pattern, adapted:
    since we never mask, we hook gate_proj/up_proj (not down_proj) to
    recover gate and up separately (needed for the Q3 attribution variants).
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import gslr_layer_tune as glt  # noqa: E402  (same dir as this script)
from gslr_group_ppl import swap_layer, should_use_retuned  # noqa: E402

CONTEXT = 2048
D_FF = 11008
N_LAYERS = 32
G_FULL_SWEEP = [1, 8, 16, 32, 64, 128]
G_EDGE = 128  # FastForward-style first/last block exclusion, only meaningful at g=128


# --------------------------------------------------------------------------
# Variance-component accumulator (streaming, fp64 running totals per
# pitfall #2 -- big per-block tensors stay fp32, only the long-horizon
# scalar/vector totals are fp64, since the per-block mean-then-deviation
# reduction is itself numerically stable in fp32 for g <= 128 terms).
# --------------------------------------------------------------------------

class Accum:
    def __init__(self, D, device):
        self.D = D
        self.sum_A = torch.zeros(D, dtype=torch.float64, device=device)
        self.total_SS = 0.0
        self.total_VA = 0.0
        self.total_w = 0.0
        self.n_T = 0
        self.n_tokens = 0

    def add_window(self, w32, g):
        seqlen, D = w32.shape
        assert D == self.D
        assert seqlen % g == 0, f"seqlen {seqlen} not divisible by g={g}"
        nb = seqlen // g
        wv = w32.contiguous().view(nb, g, D)
        A = wv.mean(dim=1)  # (nb, D) fp32 -- block means
        self.sum_A += A.sum(dim=0).double()
        if g > 1:
            dev = wv - A.unsqueeze(1)
            SS = (dev * dev).sum(dim=1)  # (nb, D), two-pass (stable) not naive one-pass
            self.total_SS += SS.sum().double().item()
        Abar = A.mean(dim=1, keepdim=True)
        VA_per_block = ((A - Abar) ** 2).mean(dim=1)  # (nb,) population variance across D
        self.total_VA += VA_per_block.sum().double().item()
        self.n_T += nb
        self.total_w += wv.sum().double().item()
        self.n_tokens += seqlen


def finalize_accum(acc, g):
    D = acc.D
    n_T = acc.n_T
    d_hat = (acc.total_SS / (g - 1)) / (n_T * D) if g > 1 else None
    V_A_hat = acc.total_VA / n_T
    s_hat_raw = V_A_hat - ((d_hat / g) if d_hat is not None else 0.0)
    s_hat = max(s_hat_raw, 0.0)

    mu_hat = acc.sum_A / n_T  # (D,) fp64
    mu_mean = mu_hat.mean()
    var_mu_hat = ((mu_hat - mu_mean) ** 2).mean().item()  # population var across D
    if n_T > 1:
        sigma_mu2_raw = (var_mu_hat - V_A_hat / n_T) / (1.0 - 1.0 / n_T)
    else:
        sigma_mu2_raw = float("nan")
    sigma_mu2_hat = max(sigma_mu2_raw, 0.0) if not math.isnan(sigma_mu2_raw) else float("nan")
    rho_sigma2_hat = (s_hat - sigma_mu2_hat) if not math.isnan(sigma_mu2_hat) else float("nan")

    sigma_tot2 = s_hat + (d_hat if d_hat is not None else 0.0)
    rho_star = (s_hat / sigma_tot2) if sigma_tot2 > 0 else float("nan")

    if g > 1 and sigma_tot2 > 0:
        delta_hat = math.sqrt(sigma_tot2) - math.sqrt(max(s_hat + d_hat / g, 0.0))
        delta_norm = 1.0 - math.sqrt(max(rho_star + (1.0 - rho_star) / g, 0.0))
    else:
        delta_hat = 0.0
        delta_norm = 0.0

    mu_w = acc.total_w / (acc.n_tokens * D)
    return {
        "g": g, "n_T": n_T, "D": D,
        "d_hat": d_hat, "V_A_hat": V_A_hat,
        "s_hat_raw": s_hat_raw, "s_hat": s_hat,
        "sigma_mu2_raw": sigma_mu2_raw, "sigma_mu2_hat": sigma_mu2_hat,
        "rho_sigma2_hat": rho_sigma2_hat,
        "sigma_tot2": sigma_tot2, "rho_star": rho_star,
        "delta_hat": delta_hat, "delta_norm": delta_norm,
        "mu_w": mu_w,
    }


# --------------------------------------------------------------------------
# Model / data
# --------------------------------------------------------------------------

def load_windows(tok, n_windows, context_size=CONTEXT):
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = ""
    for sample in dataset:
        text += sample["text"] + "\n\n"
    testenc = tok(text, return_tensors="pt").input_ids
    n_total = testenc.numel() // context_size
    n = min(n_windows, n_total)
    windows = testenc[:, : n * context_size].reshape(n, context_size)
    return windows, n_total


def load_model(model_path, attn):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation=attn)
    model.to("cuda").eval()
    return model


def compute_gauge(model, device):
    """c_j per layer from the model's CURRENT down_proj -- caller must call
    this before any swap_layer call, so it reflects the original weights."""
    c = {}
    for li, layer in enumerate(model.model.layers):
        Wd0 = layer.mlp.down_proj.weight.detach().float()  # (hidden, D)
        c[li] = Wd0.pow(2).sum(dim=0).sqrt().to(device)
    return c


def run_windows(model, windows, device):
    with torch.no_grad():
        for wi in range(windows.shape[0]):
            inp = windows[wi:wi + 1].to(device)
            model(inp)


def register_gu_hooks(model, callback):
    """callback(layer_idx, gate, up) with gate/up = (seqlen, D) bf16, gate
    already through act_fn. Relies on gate_proj being evaluated before
    up_proj in `act_fn(gate_proj(x)) * up_proj(x)` (true for both LlamaMLP
    and MaskedMLP with apply_mask=False -- same forward expression)."""
    handles = []
    gate_cache = {}
    for li, layer in enumerate(model.model.layers):
        act_fn = layer.mlp.act_fn

        def gate_hook(module, inp, out, li=li):
            gate_cache[li] = out

        def up_hook(module, inp, out, li=li, act_fn=act_fn):
            gate_pre = gate_cache.pop(li)
            gate = act_fn(gate_pre)
            callback(li, gate.squeeze(0), out.squeeze(0))

        handles.append(layer.mlp.gate_proj.register_forward_hook(gate_hook))
        handles.append(layer.mlp.up_proj.register_forward_hook(up_hook))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# --------------------------------------------------------------------------
# Part A passes (original weights only)
# --------------------------------------------------------------------------

def run_full_sweep(model, windows, c_by_layer, g_values, device):
    """A-orig: full variant, all g in g_values, all layers, plus a
    first/last-128-block-excluded variant at g=128 (pitfall #4)."""
    n_layers = len(model.model.layers)
    accums = {li: {g: Accum(D_FF, device) for g in g_values} for li in range(n_layers)}
    edge_accums = {li: Accum(D_FF, device) for li in range(n_layers)}
    has_edge = G_EDGE in g_values

    def callback(li, gate, up):
        i = gate.float() * up.float()
        c2 = (c_by_layer[li] ** 2)[None, :]
        w = (i * i) * c2
        for g in g_values:
            accums[li][g].add_window(w, g)
        if has_edge:
            w_trim = w[G_EDGE:-G_EDGE]
            if w_trim.shape[0] > 0:
                edge_accums[li].add_window(w_trim, G_EDGE)

    handles = register_gu_hooks(model, callback)
    run_windows(model, windows, device)
    remove_hooks(handles)

    results = {}
    for li in range(n_layers):
        results[li] = {str(g): finalize_accum(accums[li][g], g) for g in g_values}
        if has_edge:
            results[li]["128_no_edge"] = finalize_accum(edge_accums[li], G_EDGE)
    return results


def run_attribution(model, windows, c_by_layer, g_values, device):
    """A-attr (Q3): gateFrozen (i = u * a_bar_T) and upFrozen (i = u_bar_T *
    a) variants, for g in g_values, all layers."""
    n_layers = len(model.model.layers)
    accums = {
        li: {g: {"gateFrozen": Accum(D_FF, device), "upFrozen": Accum(D_FF, device)}
             for g in g_values}
        for li in range(n_layers)
    }

    def callback(li, gate, up):
        c2 = (c_by_layer[li] ** 2)[None, :]
        gate32 = gate.float()
        up32 = up.float()
        seqlen, D = gate32.shape
        for g in g_values:
            nb = seqlen // g
            gate_b = gate32.contiguous().view(nb, g, D)
            up_b = up32.contiguous().view(nb, g, D)
            gate_bar = gate_b.mean(dim=1, keepdim=True)
            up_bar = up_b.mean(dim=1, keepdim=True)
            i_gf = (gate_bar * up_b).reshape(seqlen, D)   # gate frozen -> up's own variance remains
            i_uf = (gate_b * up_bar).reshape(seqlen, D)   # up frozen -> gate's own variance remains
            w_gf = (i_gf * i_gf) * c2
            w_uf = (i_uf * i_uf) * c2
            accums[li][g]["gateFrozen"].add_window(w_gf, g)
            accums[li][g]["upFrozen"].add_window(w_uf, g)

    handles = register_gu_hooks(model, callback)
    run_windows(model, windows, device)
    remove_hooks(handles)

    results = {}
    for li in range(n_layers):
        results[li] = {}
        for g in g_values:
            results[li][str(g)] = {
                "gateFrozen": finalize_accum(accums[li][g]["gateFrozen"], g),
                "upFrozen": finalize_accum(accums[li][g]["upFrozen"], g),
            }
    return results


def run_rule_and_overlap(model, windows, c_by_layer, g_values, K, device):
    """A-rule (section 2.2) + self-check #4 (adjacent-token top-K overlap,
    reference target ~0.316 from prior C(1) measurement). Rule 'mass' uses
    sum_t w (equiv. to ranking by the block MEAN A_j(T)); rule 'l1' uses
    sum_t sqrt(w) = sum_t |i|*c, GSLR's historical group-score convention."""
    n_layers = len(model.model.layers)
    stats = {li: {g: {"num_mass": 0.0, "num_l1": 0.0, "num_pertoken": 0.0, "den": 0.0}
                  for g in g_values}
              for li in range(n_layers)}
    overlap_stats = {li: {"inter_d1": 0.0, "pairs_d1": 0, "k_eff_sum": 0.0, "n_tok": 0}
                      for li in range(n_layers)}

    def callback(li, gate, up):
        i = gate.float() * up.float()
        c = c_by_layer[li]
        absval = i.abs() * c[None, :]  # sqrt(w) = |i|*c
        w = absval * absval
        seqlen, D = w.shape

        idx_tok = w.topk(K, dim=-1).indices
        mask_tok = torch.zeros_like(w, dtype=torch.bool).scatter_(1, idx_tok, True)
        num_pertoken = (w * mask_tok).sum().item()
        den = w.sum().item()

        inter = (mask_tok[1:] & mask_tok[:-1]).sum().item()
        ov = overlap_stats[li]
        ov["inter_d1"] += inter
        ov["pairs_d1"] += seqlen - 1
        ov["k_eff_sum"] += mask_tok.sum(dim=-1).double().sum().item()
        ov["n_tok"] += seqlen

        for g in g_values:
            nb = seqlen // g
            wv = w.contiguous().view(nb, g, D)
            absv = absval.contiguous().view(nb, g, D)
            score_mass = wv.sum(dim=1)
            score_l1 = absv.sum(dim=1)
            idx_mass = score_mass.topk(K, dim=-1).indices
            idx_l1 = score_l1.topk(K, dim=-1).indices
            mask_mass = torch.zeros_like(score_mass, dtype=torch.bool).scatter_(1, idx_mass, True)
            mask_l1 = torch.zeros_like(score_l1, dtype=torch.bool).scatter_(1, idx_l1, True)
            mask_mass_tok = mask_mass.repeat_interleave(g, dim=0)
            mask_l1_tok = mask_l1.repeat_interleave(g, dim=0)
            s = stats[li][g]
            s["num_mass"] += (w * mask_mass_tok).sum().item()
            s["num_l1"] += (w * mask_l1_tok).sum().item()
            s["num_pertoken"] += num_pertoken
            s["den"] += den

    handles = register_gu_hooks(model, callback)
    run_windows(model, windows, device)
    remove_hooks(handles)

    results = {}
    for li in range(n_layers):
        results[li] = {}
        for g in g_values:
            s = stats[li][g]
            results[li][str(g)] = {
                "R_mass": s["num_mass"] / s["den"] if s["den"] > 0 else None,
                "R_l1": s["num_l1"] / s["den"] if s["den"] > 0 else None,
                "R_pertoken": s["num_pertoken"] / s["den"] if s["den"] > 0 else None,
            }
        ov = overlap_stats[li]
        k_eff = ov["k_eff_sum"] / ov["n_tok"] if ov["n_tok"] > 0 else float("nan")
        results[li]["overlap_d1"] = (
            ov["inter_d1"] / (ov["pairs_d1"] * k_eff) if ov["pairs_d1"] > 0 and k_eff > 0 else None)
        results[li]["k_eff"] = k_eff
    return results


# --------------------------------------------------------------------------
# Part B (weight-swap arms)
# --------------------------------------------------------------------------

def run_own_stats(model, windows, c_by_layer, g, device):
    """The arm's own (really-propagated) forward pass -- section 2 stats
    under whatever weights are CURRENTLY installed in `model`."""
    n_layers = len(model.model.layers)
    accums = {li: Accum(D_FF, device) for li in range(n_layers)}

    def callback(li, gate, up):
        i = gate.float() * up.float()
        c2 = (c_by_layer[li] ** 2)[None, :]
        w = (i * i) * c2
        accums[li].add_window(w, g)

    handles = register_gu_hooks(model, callback)
    run_windows(model, windows, device)
    remove_hooks(handles)
    return {li: finalize_accum(accums[li], g) for li in range(n_layers)}


def load_arm_weights(gslr_dir, n_layers, device, dtype=torch.bfloat16):
    """Per layer: (Wg, Wu, Wd) for layers should_use_retuned() says yes,
    None otherwise (layer 31 -- drift is trivially 0 there, no compute)."""
    out = {}
    for li in range(n_layers):
        if should_use_retuned("gslr_dense", li):
            sd = torch.load(os.path.join(gslr_dir, f"layer_{li}.pt"), map_location=device)
            out[li] = (sd["wg"].to(device=device, dtype=dtype),
                       sd["wu"].to(device=device, dtype=dtype),
                       sd["wd"].to(device=device, dtype=dtype))
        else:
            out[li] = None
    return out


def compute_counterfactual_drift(model, weight_fn, windows, device):
    """weight_fn(li) -> (Wg, Wu, Wd) or None (drift defined 0). `model` MUST
    be running with ORIGINAL weights when this is called -- x and y_orig
    come from its real (propagating) forward; y_arm is a side computation
    on the SAME x, never fed back (no propagation, per section 1.5/3)."""
    n_layers = len(model.model.layers)
    num = {li: 0.0 for li in range(n_layers)}
    den = {li: 0.0 for li in range(n_layers)}
    skip = set()
    handles = []
    for li, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        wgt = weight_fn(li)
        if wgt is None:
            skip.add(li)
            continue
        Wg, Wu, Wd = wgt

        def hook(module, inp, out, li=li, Wg=Wg, Wu=Wu, Wd=Wd):
            x = inp[0]
            gate = module.act_fn(F.linear(x, Wg))
            up = F.linear(x, Wu)
            y_arm = F.linear(gate * up, Wd)
            diff = out.float() - y_arm.float()
            num[li] += diff.pow(2).sum().item()
            den[li] += out.float().pow(2).sum().item()

        handles.append(mlp.register_forward_hook(hook))

    run_windows(model, windows, device)
    remove_hooks(handles)

    result = {}
    for li in range(n_layers):
        if li in skip:
            result[li] = 0.0
        else:
            result[li] = math.sqrt(num[li] / den[li]) if den[li] > 0 else None
    return result


def rel_diff(a, b):
    if a is None or b is None:
        return None
    denom = abs(a) if abs(a) > 1e-12 else 1e-12
    return abs(a - b) / denom


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def dump(out_dir, name, payload):
    path = os.path.join(out_dir, f"decomp_{name}.json")
    with open(path, "w") as f:
        json.dump(sanitize(payload), f, indent=2)
    glt.log(f"wrote {path}")


# --------------------------------------------------------------------------
# Part A / Part B orchestration
# --------------------------------------------------------------------------

def run_part_a(model, windows, c_by_layer, K, args, device, git_commit):
    glt.log("Part A: full sweep (A-orig) ...")
    t0 = time.time()
    sweep = run_full_sweep(model, windows, c_by_layer, G_FULL_SWEEP, device)
    glt.log(f"  done in {time.time() - t0:.1f}s")
    dump(args.out_dir, "A-orig",
         {"config": vars(args), "git_commit": git_commit, "layers": sweep})

    glt.log("Part A: attribution (A-attr) ...")
    t0 = time.time()
    attr = run_attribution(model, windows, c_by_layer, [16, 128], device)
    for li in attr:
        for g in list(attr[li].keys()):
            attr[li][g]["full"] = sweep[li][g]
    glt.log(f"  done in {time.time() - t0:.1f}s")
    dump(args.out_dir, "A-attr",
         {"config": vars(args), "git_commit": git_commit, "layers": attr})

    glt.log("Part A: rule comparison + overlap self-check (A-rule) ...")
    t0 = time.time()
    rule = run_rule_and_overlap(model, windows, c_by_layer, [16, 64, 128], K, device)
    glt.log(f"  done in {time.time() - t0:.1f}s")
    dump(args.out_dir, "A-rule",
         {"config": vars(args), "git_commit": git_commit, "K": K, "layers": rule})


def run_part_b(model, windows, c_by_layer, K, args, device, git_commit):
    n_layers = len(model.model.layers)
    orig_mlps = [layer.mlp for layer in model.model.layers]

    def restore():
        for li, layer in enumerate(model.model.layers):
            layer.mlp = orig_mlps[li]

    def install(gslr_dir, g):
        a = argparse.Namespace(sparsity=args.sparsity, g=g)
        for li, layer in enumerate(model.model.layers):
            swap_layer(layer, li, "gslr_dense", a, gslr_dir)

    self_check = {}

    glt.log("self-check: drift-zero sanity (own weights vs own weights) ...")
    dz = compute_counterfactual_drift(
        model,
        lambda li: (model.model.layers[li].mlp.gate_proj.weight,
                    model.model.layers[li].mlp.up_proj.weight,
                    model.model.layers[li].mlp.down_proj.weight),
        windows, device)
    self_check["drift_zero_sanity"] = dz
    max_dz = max(v for v in dz.values() if v is not None)
    glt.log(f"  max drift (should be ~0): {max_dz:.3e}")

    arms = [
        ("B-orig", None, [16, 64]),
        ("B-gslr-16-0.3", os.path.join(args.gslr_root, "16_0.3"), [16]),
        ("B-gslr-16-1.0", os.path.join(args.gslr_root, "16_1.0"), [16]),
        ("B-gslr-64-0.3", os.path.join(args.gslr_root, "64_0.3"), [64]),
        ("B-gslr-64-1.0", os.path.join(args.gslr_root, "64_1.0"), [64]),
        ("B-b1d-16", os.path.join(args.b1d_root, "16"), [16]),
    ]

    b_orig_own = {}
    b1d_own_16 = None

    for name, gdir, gs in arms:
        for g in gs:
            glt.log(f"Part B: {name} g={g} own-stats ...")
            restore()
            if gdir is not None:
                install(gdir, g)
            own = run_own_stats(model, windows, c_by_layer, g, device)

            if gdir is None:
                b_orig_own[g] = own
                drift = {li: 0.0 for li in range(n_layers)}
            else:
                restore()
                t0 = time.time()
                arm_w = load_arm_weights(gdir, n_layers, device)
                drift = compute_counterfactual_drift(model, lambda li: arm_w[li], windows, device)
                del arm_w
                torch.cuda.empty_cache()
                glt.log(f"  drift pass done in {time.time() - t0:.1f}s")

            if name == "B-b1d-16":
                b1d_own_16 = own

            payload = {
                "config": vars(args), "git_commit": git_commit,
                "arm": name, "g": g, "gslr_dir": gdir,
                "layers": {li: {**own[li], "drift": drift[li]} for li in range(n_layers)},
            }
            fname = name if len(gs) == 1 else f"{name}-g{g}"
            dump(args.out_dir, fname, payload)

    glt.log("self-check: B1d invariance vs B-orig(g=16) (i must be bit-identical -- "
             "B1d changes only W_down, and c_j is frozen to the ORIGINAL W_down "
             "everywhere, so i itself never changes) ...")
    diffs = []
    for li in range(n_layers):
        o = b_orig_own[16][li]
        b = b1d_own_16[li]
        diffs.append({
            "layer": li,
            "s_hat_rel_diff": rel_diff(o["s_hat"], b["s_hat"]),
            "d_hat_rel_diff": rel_diff(o["d_hat"], b["d_hat"]),
            "rho_star_diff": (abs(o["rho_star"] - b["rho_star"])
                               if o["rho_star"] is not None and b["rho_star"] is not None
                               else None),
        })
    self_check["b1d_invariance"] = diffs

    glt.log("self-check: restoration verify (fresh B-orig g=16, must match first measurement) ...")
    restore()
    verify = run_own_stats(model, windows, c_by_layer, 16, device)
    vdiffs = []
    for li in range(n_layers):
        o = b_orig_own[16][li]
        v = verify[li]
        vdiffs.append({
            "layer": li,
            "s_hat_rel_diff": rel_diff(o["s_hat"], v["s_hat"]),
            "d_hat_rel_diff": rel_diff(o["d_hat"], v["d_hat"]),
        })
    self_check["restoration_verify"] = vdiffs

    dump(args.out_dir, "B-selfcheck",
         {"config": vars(args), "git_commit": git_commit, "self_check": self_check})


# --------------------------------------------------------------------------
# Self-test (section 4.1 synthetic recovery, 4.2 g=1 identity) -- pure CPU,
# no model/GPU needed.
# --------------------------------------------------------------------------

def selftest():
    torch.manual_seed(0)

    print("=== self-check 1: synthetic recovery ===")
    D = 3000
    sigma_mu2_true = 0.30
    rho_sigma2_true = 0.20
    d_true = 0.50
    all_ok = True
    for g in [16, 128]:
        for n_T in [100, 1000]:
            mu_j = torch.randn(D, dtype=torch.float64) * math.sqrt(sigma_mu2_true)
            acc = Accum(D, torch.device("cpu"))
            for _ in range(n_T):
                ctx = torch.randn(D, dtype=torch.float64) * math.sqrt(rho_sigma2_true)
                noise = torch.randn(g, D, dtype=torch.float64) * math.sqrt(d_true)
                block = (mu_j + ctx).unsqueeze(0) + noise
                acc.add_window(block.float(), g)
            res = finalize_accum(acc, g)

            def relerr(est, true):
                return abs(est - true) / abs(true)

            d_err = relerr(res["d_hat"], d_true)
            s_err = relerr(res["s_hat"], sigma_mu2_true + rho_sigma2_true)
            mu_err = relerr(res["sigma_mu2_hat"], sigma_mu2_true)
            ok = d_err < 0.05 and s_err < 0.05 and mu_err < 0.10
            all_ok = all_ok and ok
            print(f"  g={g:4d} n_T={n_T:5d}  d_hat={res['d_hat']:.4f} (true {d_true}, "
                  f"err {d_err:.1%})  s_hat={res['s_hat']:.4f} "
                  f"(true {sigma_mu2_true + rho_sigma2_true:.4f}, err {s_err:.1%})  "
                  f"sigma_mu2_hat={res['sigma_mu2_hat']:.4f} (true {sigma_mu2_true}, "
                  f"err {mu_err:.1%})  {'OK' if ok else 'FAIL'}")
    assert all_ok, "synthetic recovery self-check failed -- see per-case errors above"
    print("self-check 1 OK")

    print("=== self-check 2: g=1 identity ===")
    D2, seqlen2 = 64, 500
    w = torch.rand(seqlen2, D2, dtype=torch.float32)
    acc = Accum(D2, torch.device("cpu"))
    acc.add_window(w, 1)
    res = finalize_accum(acc, 1)
    direct_VA = w.double().var(dim=1, unbiased=False).mean().item()
    assert abs(res["V_A_hat"] - direct_VA) < 1e-6, f"{res['V_A_hat']} vs {direct_VA}"
    assert res["d_hat"] is None
    assert res["delta_hat"] == 0.0 and res["delta_norm"] == 0.0
    print(f"  V_A_hat={res['V_A_hat']:.6f} direct={direct_VA:.6f}  OK")
    print("self-check 2 OK")

    print("ALL SELFTESTS OK")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["selftest", "partA", "partB"])
    ap.add_argument("--model", default="/raid/LLM/llama2-7b")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--n_windows", type=int, default=64)
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--out_dir", default=os.path.expanduser("~/workspace/results/sd"))
    ap.add_argument("--gslr_root", default=os.path.expanduser("~/workspace/gslr/llama2-7b-full"))
    ap.add_argument("--b1d_root", default=os.path.expanduser("~/workspace/gslr25/llama2-7b/b1d"))
    args = ap.parse_args()

    if args.mode == "selftest":
        selftest()
        return

    assert CONTEXT % 128 == 0
    for g in G_FULL_SWEEP:
        assert CONTEXT % g == 0, f"context {CONTEXT} not divisible by g={g}"

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")
    K = int(math.floor((1 - args.sparsity) * D_FF))
    glt.log(f"sparsity={args.sparsity} -> K={K}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    windows, n_total = load_windows(tok, args.n_windows, CONTEXT)
    glt.log(f"loaded {windows.shape[0]} / {n_total} non-overlapping {CONTEXT}-token windows")

    glt.log(f"loading model {args.model} (attn={args.attn}) ...")
    model = load_model(args.model, args.attn)
    c_by_layer = compute_gauge(model, device)  # MUST be before any swap_layer call
    git_commit = glt.git_hash()

    if args.mode == "partA":
        run_part_a(model, windows, c_by_layer, K, args, device, git_commit)
    else:
        run_part_b(model, windows, c_by_layer, K, args, device, git_commit)

    glt.log("done.")


if __name__ == "__main__":
    main()
