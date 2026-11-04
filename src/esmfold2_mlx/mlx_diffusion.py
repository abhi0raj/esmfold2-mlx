"""The ESMFold2 diffusion sampler (DiffusionStructureHead.sample) ported to MLX,
using the PyTorch checkpoint's weights.

Per fold, everything that does not depend on the noise level is computed once:
the pair conditioning z and, from it, the 12 token-attention pair biases
(PyTorch recomputes LayerNorm + projection of the [L, L, 256] grid in every
block of every step), the atom conditioning c and the six adaLN modulations
derived from it, the 3D RoPE tables and the sliding-window mask. Each step then
runs one compiled MLX function: denoiser + Kabsch alignment + Euler update.
The noise-level-only terms (s and the token blocks' adaLN / output gates, ~1/3
of the token transformer's FLOPs) are computed for several steps per matmul.

Noise is drawn with torch on the model's device in exactly the upstream order,
so seed-matched folds see the same random numbers as PyTorch. The atom
attention keeps upstream's bf16 q/k/v and bf16-rounded RoPE tables.

Usage:
    from esmfold2_mlx.mlx_diffusion import MlxDiffusion
    mlx_head = MlxDiffusion(torch_model.structure_head)
    out = mlx_head.sample(**the_kwargs_forward_passes_to_structure_head_sample)
"""

import math

import mlx.core as mx
import numpy as np
import torch
from esm.models.esmfold2.layers import CHAR_VOCAB_SIZE, MAX_CHARS

LN_EPS = 1e-5
RMS_EPS = float(torch.finfo(torch.float32).eps)  # F.rms_norm(eps=None) on fp32
COND_BUDGET = 512 * 2**20  # bytes of precomputed per-step noise conditioning


def _mx(t):
    return mx.array(t.detach().float().cpu().numpy())


def _lin(m):
    return _mx(m.weight).T  # transposed once so every call is a plain x @ W


def _ln(m):
    return (_mx(m.weight), _mx(m.bias))


# --------------------------------------------------------------------------
# weight copy
# --------------------------------------------------------------------------

def _transition_layer(m):  # TransitionLayer: LN -> silu(a) * b -> out
    return {"ln": _ln(m.norm), "a": _lin(m.a_proj), "b": _lin(m.b_proj), "out": _lin(m.out_proj)}


def _swa_block(blk):
    at, ffn = blk.attn, blk.ffn
    return {"Wqkv": _lin(at.Wqkv), "gate": _lin(at.gate_proj), "out": _lin(at.out_proj),
            "up": _lin(ffn.w_up), "down": _lin(ffn.w_down)}


def diffusion_params(head):
    """Copy every weight the sampler needs out of a PyTorch DiffusionStructureHead."""
    dm = head.diffusion_module
    cond, tt, enc, dec = dm.conditioning, dm.token_transformer, dm.atom_encoder, dm.atom_decoder
    # Everything in the token transformer that depends only on s (adaLN gate/shift,
    # output gates) is batched into two wide matmuls per step instead of 72 small
    # ones. adaLN's s_scale is folded into the weights: LN(s; w) @ W = LN(s) @ (w * W).
    ada_w, ada_b, og_w, og_b = [], [], [], []
    for blk in [m for pair in zip(tt.attn_blocks, tt.transition_blocks) for m in pair]:
        ln_w = blk.adaln.s_scale.detach().float()[:, None]
        ada_w += [ln_w * blk.adaln.s_gate.weight.detach().float().T, ln_w * blk.adaln.s_shift.weight.detach().float().T]
        ada_b += [blk.adaln.s_gate.bias.detach().float(), torch.zeros_like(blk.adaln.s_gate.bias).float()]
        gate = blk.out_gate if hasattr(blk, "out_gate") else blk.output_gate
        og_w.append(gate.weight.detach().float().T)
        og_b.append(gate.bias.detach().float())
    cat = lambda xs: _mx(torch.cat(xs, -1))  # noqa: E731
    token = {"ada_w": cat(ada_w), "ada_b": cat(ada_b), "og_w": cat(og_w), "og_b": cat(og_b), "attn": [], "trans": []}
    for a in tt.attn_blocks:
        # q | k | v | g in one matmul; only q has a bias
        w = torch.cat([a.q_proj.weight, a.kv_proj.weight, a.g_proj.weight], 0)
        b = torch.cat([a.q_proj.bias, torch.zeros(3 * a.d_model, device=w.device)])
        token["attn"].append({"qkvg": _mx(w).T, "qkvg_b": _mx(b), "o": _lin(a.out_proj)})
    for t in tt.transition_blocks:
        token["trans"].append({"swish": _lin(t.lin_swish), "out": _lin(t.lin_out)})
    # Each token block's pair bias is LN_b(z) @ P_b. All blocks share LN's
    # statistics, so the 12 affine LNs + projections fold into one [256, 12*H] matmul.
    blocks = tt.attn_blocks
    bias_w = torch.cat([b.pair_norm.weight[:, None] * b.pair_bias_proj.weight.T for b in blocks], -1)
    bias_b = torch.cat([b.pair_bias_proj(b.pair_norm.bias) for b in blocks], -1)
    pair = {"ln": _ln(cond.z_input_norm), "proj": _lin(cond.z_proj),
            "transitions": [_transition_layer(b) for b in cond.z_transitions],
            "bias_w": _mx(bias_w), "bias_b": _mx(bias_b)}
    atom_blocks = list(enc.atom_transformer.blocks) + list(dec.atom_transformer.blocks)
    atom = {"lin": _lin(enc.atom_linear), "ln": _ln(enc.atom_norm),
            # adaLN modulations of all 6 atom blocks depend only on c: one matmul
            "mods": _mx(torch.cat([b.adaln_modulation[1].weight for b in atom_blocks], 0)).T,
            "s_ln": _ln(cond.s_input_norm), "s_proj": _lin(cond.s_proj)}
    return {
        "pair": pair, "atom": atom,
        "s_transitions": [_transition_layer(b) for b in cond.s_transitions],
        "fourier_w": _mx(cond.fourier.w), "fourier_b": _mx(cond.fourier.b),
        "noise_ln": _ln(cond.noise_norm), "noise_proj": _lin(cond.noise_proj),
        # pred_r1 is always zero at inference, so only the first 3 input columns matter
        "coords": _mx(enc.coords_linear.weight[:, :3]).T,
        "enc_blocks": [_swa_block(b) for b in enc.atom_transformer.blocks],
        "a2t": _lin(enc.atom_to_token_linear),
        "s_step_ln": _ln(dm.s_step_norm), "s_to_token": _lin(dm.s_to_token),
        "token": token,
        "token_ln": _ln(dm.token_norm),
        "t2a": _lin(dec.token_to_atom_linear),
        "dec_blocks": [_swa_block(b) for b in dec.atom_transformer.blocks],
        "dec_ln": _ln(dec.norm), "dec_out": _lin(dec.output_linear),
    }


# --------------------------------------------------------------------------
# per-step network
# --------------------------------------------------------------------------

def _layer_norm(x, wb=(None, None)):
    return mx.fast.layer_norm(x, wb[0], wb[1], LN_EPS)


def _silu(x):
    return x * mx.sigmoid(x)


def _transition(p, x):
    h = _layer_norm(x, p["ln"])
    return (_silu(h @ p["a"]) * (h @ p["b"])) @ p["out"]


def _rope(x, cos, sin):
    # x: [B, N, H, D]; cos/sin: [B, N, 1, D] (already doubled to D)
    x1, x2 = mx.split(x, 2, axis=-1)
    return x * cos + mx.concatenate([-x2, x1], axis=-1) * sin


def _local_attention(q, k, v, mask, chunk, scale):
    """Sliding-window attention computed chunk by chunk: each block of ``chunk``
    queries sees only the 3 * chunk keys around it, instead of all N keys.
    q, k, v: [B, N, H, D]; mask: [nC, 1, chunk, 3 * chunk]."""
    B, N, H, D = q.shape
    nC = mask.shape[0]
    pad = nC * chunk - N
    q = mx.pad(q, [(0, 0), (0, pad), (0, 0), (0, 0)])
    q = q.reshape(B, nC, chunk, H, D).transpose(0, 1, 3, 2, 4).reshape(B * nC, H, chunk, D)

    def windows(t):
        t = mx.pad(t, [(0, 0), (chunk, pad + chunk), (0, 0), (0, 0)]).reshape(B, nC + 2, chunk, H, D)
        t = mx.concatenate([t[:, :nC], t[:, 1:nC + 1], t[:, 2:]], axis=2)  # [B, nC, 3*chunk, H, D]
        return t.transpose(0, 1, 3, 2, 4).reshape(B * nC, H, 3 * chunk, D)

    o = mx.fast.scaled_dot_product_attention(q, windows(k), windows(v), scale=scale, mask=mask)
    return o.reshape(B, nC, H, chunk, D).transpose(0, 1, 3, 2, 4).reshape(B, nC * chunk, H * D)[:, :N]


def _swa_block_apply(p, mod, x, ctx, n_heads):
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
    B, N, d = x.shape
    hd = d // n_heads
    h = mx.fast.rms_norm(x, None, RMS_EPS) * (1 + scale_a) + shift_a
    qkv = (h @ p["Wqkv"]).reshape(B, N, 3, n_heads, hd)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, N, H, hd]
    # upstream runs this attention in bf16 (after fp32 qk-norm and RoPE)
    q = _rope(mx.fast.rms_norm(q, None, RMS_EPS), ctx["cos"], ctx["sin"]).astype(mx.bfloat16)
    k = _rope(mx.fast.rms_norm(k, None, RMS_EPS), ctx["cos"], ctx["sin"]).astype(mx.bfloat16)
    v = v.astype(mx.bfloat16)
    if "swa_local_mask" in ctx:
        o = _local_attention(q, k, v, ctx["swa_local_mask"], ctx["swa_local_mask"].shape[2], hd ** -0.5)
    else:
        o = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3),
            scale=hd ** -0.5, mask=ctx["swa_mask"]).transpose(0, 2, 1, 3).reshape(B, N, d)
    o = o.astype(mx.float32) * ctx["valid"]
    o = (o * mx.sigmoid(h @ p["gate"])) @ p["out"]
    x = x + gate_a * o
    h = mx.fast.rms_norm(x, None, RMS_EPS) * (1 + scale_f) + shift_f
    x1, x2 = mx.split(h @ p["up"], 2, axis=-1)
    return x + gate_f * ((_silu(x1) * x2) @ p["down"])


def _attn_pair_bias(p, x, out_gate, bias, n_heads):
    """AttentionPairBias after adaLN; ``bias`` is the precomputed pair bias (+ key mask)."""
    B, L, d = x.shape
    hd = d // n_heads
    qkvg = x @ p["qkvg"] + p["qkvg_b"]
    q, k, v, g = mx.split(qkvg, 4, axis=-1)
    q, k, v = (t.reshape(B, L, n_heads, hd).transpose(0, 2, 1, 3) for t in (q, k, v))
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5, mask=bias)
    o = o.transpose(0, 2, 1, 3).reshape(B, L, d) * mx.sigmoid(g)
    return out_gate * (o @ p["o"])


def _cond_transition(p, x, out_gate):
    x1, x2 = mx.split(x @ p["swish"], 2, axis=-1)
    return out_gate * ((_silu(x1) * x2) @ p["out"])


def noise_conditioning(p, s0, t, sigma):
    """Everything that depends only on the noise level: s (DiffusionConditioning)
    and the token transformer's adaLN / output-gate terms. ``t``: [S] noise levels,
    computed for several steps at once (bigger, more efficient matmuls)."""
    t_noise = 0.25 * mx.log(mx.maximum(t / sigma, 1e-20))
    n = mx.cos(2.0 * math.pi * (t_noise[:, None] * p["fourier_w"] + p["fourier_b"]))
    s = s0 + (_layer_norm(n, p["noise_ln"]) @ p["noise_proj"])[:, None]
    for blk in p["s_transitions"]:
        s = s + _transition(blk, s)
    pt = p["token"]
    return {"s_tok": _layer_norm(s, p["s_step_ln"]) @ p["s_to_token"],
            "ada": _layer_norm(s) @ pt["ada_w"] + pt["ada_b"],
            "og": mx.sigmoid(s @ pt["og_w"] + pt["og_b"])}


def token_transformer(p, a, cond, pair_bias, n_heads):
    nb = len(p["attn"])
    ada = mx.split(cond["ada"], 4 * nb, axis=-1)  # (gate, shift) per sub-block
    og = mx.split(cond["og"], 2 * nb, axis=-1)
    for i in range(nb):
        x = mx.sigmoid(ada[4 * i]) * _layer_norm(a) + ada[4 * i + 1]
        a = a + _attn_pair_bias(p["attn"][i], x, og[2 * i], pair_bias[i], n_heads)
        x = mx.sigmoid(ada[4 * i + 2]) * _layer_norm(a) + ada[4 * i + 3]
        a = a + _cond_transition(p["trans"][i], x, og[2 * i + 1])
    return a


def denoise(p, ctx, cond, x_noisy, t, sigma, n_atom_heads, n_token_heads):
    """DiffusionModule.forward for one noise level ``t`` (shape [B]); ``cond`` from noise_conditioning."""
    # atom encoder
    r = x_noisy / mx.sqrt(t * t + sigma * sigma)[:, None, None]
    q = ctx["c"] + r @ p["coords"]
    for blk, mod in zip(p["enc_blocks"], ctx["enc_mods"]):
        q = _swa_block_apply(blk, mod, q, ctx, n_atom_heads)
    q_skip = q
    a = ctx["atom_to_token_mean"] @ mx.maximum(q @ p["a2t"], 0.0)  # scatter-mean as a matmul
    a = a + cond["s_tok"]

    # token transformer
    a = _layer_norm(token_transformer(p["token"], a, cond, ctx["pair_bias"], n_token_heads), p["token_ln"])

    # atom decoder
    q = q_skip + mx.take(a @ p["t2a"], ctx["tok_idx"], axis=1)
    for blk, mod in zip(p["dec_blocks"], ctx["dec_mods"]):
        q = _swa_block_apply(blk, mod, q, ctx, n_atom_heads)
    r_update = _layer_norm(q, p["dec_ln"]) @ p["dec_out"]

    s2, t2 = sigma * sigma, t * t
    out = (s2 / (s2 + t2))[:, None, None] * x_noisy
    return out + ((sigma * t) / mx.sqrt(s2 + t2))[:, None, None] * r_update, a


def pair_biases(p, z_trunk, rel_pos, n_heads):
    """DiffusionConditioning's pair z, then every token block's pair bias: [nb, B, H, L, L]."""
    z = _layer_norm(mx.concatenate([z_trunk, rel_pos], axis=-1), p["ln"]) @ p["proj"]
    for blk in p["transitions"]:
        z = z + _transition(blk, z)
    B, L = z.shape[:2]
    bias = _layer_norm(z) @ p["bias_w"] + p["bias_b"]
    return bias.reshape(B, L, L, -1, n_heads).transpose(3, 0, 4, 1, 2)


def _det3(m):
    return (m[..., 0, 0] * (m[..., 1, 1] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 1])
            - m[..., 0, 1] * (m[..., 1, 0] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 0])
            + m[..., 0, 2] * (m[..., 1, 0] * m[..., 2, 1] - m[..., 1, 1] * m[..., 2, 0]))


def rigid_align(x, x_gt, w):
    """Weighted Kabsch: align x onto x_gt (DiffusionStructureHead._weighted_rigid_align)."""
    w = w[..., None]
    denom = mx.maximum(w.sum(axis=-2, keepdims=True), 1e-8)
    mu = (x * w).sum(axis=-2, keepdims=True) / denom
    mu_gt = (x_gt * w).sum(axis=-2, keepdims=True) / denom
    x_c, xgt_c = x - mu, x_gt - mu_gt
    H = (w * xgt_c).swapaxes(-1, -2) @ x_c
    U, _, Vh = mx.linalg.svd(H, stream=mx.cpu)  # 3x3: no GPU kernel, and the CPU is instant
    d = _det3(U @ Vh)
    D = mx.stack([mx.ones_like(d), mx.ones_like(d), d], axis=-1)
    R = (U * D[..., None, :]) @ Vh
    return x_c @ R.swapaxes(-1, -2) + mu_gt


# --------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------

class MlxDiffusion:
    def __init__(self, head, compile=True):
        self.head = head
        dm = head.diffusion_module
        self.p = diffusion_params(head)
        self.sigma = float(dm.sigma_data)
        self.n_atom_heads = dm.atom_encoder.atom_transformer.blocks[0].attn.n_heads
        self.n_token_heads = dm.token_transformer.attn_blocks[0].num_heads
        self.half_window = dm.atom_encoder.atom_transformer.blocks[0].attn.half_window
        self._step = mx.compile(self._step_fn) if compile else self._step_fn
        pt = self.p["token"]
        self._cond_width = pt["og_w"].shape[0] + pt["ada_w"].shape[1] + pt["og_w"].shape[1]  # s_tok, ada, og

    def _step_fn(self, p, ctx, cond, x, rot, shift, eps, coef):
        """One sampling step: augment, add noise, denoise, align, Euler update.

        ``coef`` = [t_hat, eps_std, eta * (sigma_t - t_hat) / t_hat], so one
        compiled graph serves every step.
        """
        t_hat, eps_std, k = coef[0:1], coef[1], coef[2]
        m = ctx["atom_mask"][..., None]
        mean = (x * m).sum(axis=1, keepdims=True) / mx.maximum(m.sum(axis=1, keepdims=True), 1.0)
        x = (x - mean) @ rot + shift
        x_noisy = x + eps_std * eps
        x_den, a = denoise(p, ctx, cond, x_noisy, t_hat, self.sigma, self.n_atom_heads, self.n_token_heads)
        x_noisy = rigid_align(x_noisy, x_den, ctx["atom_mask"])
        return x_noisy + k * (x_noisy - x_den), a

    @torch.no_grad()
    def _context(self, kw):
        """Step-invariant tensors, computed once per fold."""
        enc = self.head.diffusion_module.atom_encoder
        B, N = kw["ref_pos"].shape[:2]
        L = kw["s_inputs"].shape[1]

        # pair conditioning and token pair biases, in MLX (the [L, L, 512] work dominates the per-fold cost)
        bias = pair_biases(self.p["pair"], _to_mx(kw["z_trunk"]), _to_mx(kw["relative_position_encoding"]),
                           self.n_token_heads)
        tok_mask = kw.get("token_attention_mask")
        if tok_mask is not None and not bool(tok_mask.all()):
            bias = bias + mx.where(_to_mx(tok_mask)[:, None, None, :] > 0, 0.0, float(np.finfo(np.float32).min))

        # one device->host copy of the small inputs; the rest is built on the CPU or in MLX
        kw = {k: (v.detach().cpu() if torch.is_tensor(v) and k not in ("z_trunk", "relative_position_encoding") else v)
              for k, v in kw.items()}
        ref_pos, ref_mask, tok_idx = kw["ref_pos"], kw["ref_mask"], kw["tok_idx"]
        pa = self.p["atom"]
        s0 = _layer_norm(_to_mx(kw["s_inputs"]), pa["s_ln"]) @ pa["s_proj"]
        feats = torch.cat([ref_pos, kw["ref_charge"].unsqueeze(-1), ref_mask.unsqueeze(-1), kw["ref_element"],
                           kw["ref_atom_name_chars"].reshape(B, N, MAX_CHARS * CHAR_VOCAB_SIZE)], dim=-1)
        c = _layer_norm(_to_mx(feats) @ pa["lin"], pa["ln"])
        mods = mx.split(_silu(c) @ pa["mods"], 36, axis=-1)
        mods = [mods[6 * i:6 * i + 6] for i in range(6)]
        cos, sin = enc.atom_transformer._build_3d_rope(ref_pos, kw["ref_space_uid"])  # bf16, as upstream

        # sliding-window mask over valid atoms (rank distance <= half_window)
        valid = ref_mask.bool()
        w = self.half_window
        n_valid = int(valid.sum())
        if B == 1 and bool(valid[0, :n_valid].all()):
            # padding atoms only at the end (the usual case): rank == position for
            # valid atoms, so a banded, chunked attention gives the same mask
            nC = -(-N // w)
            qpos = torch.arange(nC * w).reshape(nC, 1, w, 1)
            kpos = (torch.arange(nC)[:, None] * w - w + torch.arange(3 * w)).reshape(nC, 1, 1, 3 * w)
            band = ((qpos - kpos).abs() <= w) & (kpos >= 0) & (kpos < n_valid) & (qpos < n_valid)
            local = band | (qpos == kpos)
            swa = {"swa_local_mask": mx.array(local.numpy())}
        else:
            rank = torch.cumsum(valid, dim=1) - 1
            allowed = ((rank[:, :, None] - rank[:, None, :]).abs() <= w) & valid[:, None] & valid[:, :, None]
            allowed |= torch.eye(N, dtype=torch.bool, device=valid.device)
            swa = {"swa_mask": mx.array(allowed.cpu().numpy())[:, None]}

        # scatter-mean (atoms -> tokens) as a fixed [L, N] averaging matrix
        onehot = torch.nn.functional.one_hot(tok_idx.long(), L).float() * valid[..., None].float()  # [B, N, L]
        avg = (onehot / onehot.sum(1, keepdim=True).clamp(min=1)).transpose(1, 2)

        ctx = {"s0": s0, "c": c, "enc_mods": mods[:3], "dec_mods": mods[3:], **swa,
               "atom_to_token_mean": _to_mx(avg), "tok_idx": mx.array(tok_idx[0].numpy().astype(np.int32)),
               "atom_mask": _to_mx(ref_mask), "valid": _to_mx(valid)[..., None],
               "cos": _to_mx(torch.cat([cos, cos], -1))[:, :, None], "sin": _to_mx(torch.cat([sin, sin], -1))[:, :, None],
               "pair_bias": [mx.contiguous(bias[i]) for i in range(bias.shape[0])]}
        mx.eval(ctx)
        return ctx

    @torch.no_grad()
    def sample(self, **kw):
        head = self.head
        s_inputs = kw["s_inputs"]
        assert s_inputs.shape[0] == 1 and kw.get("num_diffusion_samples", 1) == 1 and not kw.get("return_atom_repr")
        device, n_atoms = s_inputs.device, kw["tok_idx"].shape[1]

        steps = kw.get("num_sampling_steps") or head.inference_num_steps
        schedule = head.inference_noise_schedule(int(steps), device)
        max_sigma = kw.get("max_inference_sigma", 256.0)
        if max_sigma is not None:
            schedule = schedule[schedule <= float(max_sigma)]
            schedule = torch.nn.functional.pad(schedule, (1, 0), value=float(max_sigma))
        lam = head.noise_scale if kw.get("noise_scale") is None else float(kw["noise_scale"])
        eta = head.step_scale if kw.get("step_scale") is None else float(kw["step_scale"])
        gammas = torch.where(schedule > head.gamma_min, torch.full_like(schedule, head.gamma_0),
                             torch.zeros_like(schedule))

        # every random draw of upstream sample(), in upstream order, on the same generator
        x0 = schedule[0] * torch.randn(1, n_atoms, 3, device=device, dtype=torch.float32)
        rots, shifts, epss = [], [], []
        for _ in range(len(schedule) - 1):
            rots.append(head._random_rotations(1, torch.float32, device))
            shifts.append(torch.randn(1, 1, 3, device=device))
            epss.append(torch.randn(1, n_atoms, 3, device=device))

        sched, gam = schedule.tolist(), gammas.tolist()
        coefs = []
        for sigma_tm, sigma_t, gamma in zip(sched[:-1], sched[1:], gam[1:]):
            t_hat = sigma_tm * (1.0 + gamma)
            eps_std = lam * max(t_hat**2 - sigma_tm**2, 0.0) ** 0.5
            coefs.append([t_hat, eps_std, eta * (sigma_t - t_hat) / t_hat])

        ctx = self._context(kw)
        to_mx = lambda t: mx.array(t.cpu().numpy())  # noqa: E731
        x = to_mx(x0)
        rots, shifts, epss = to_mx(torch.stack(rots)), to_mx(torch.stack(shifts)), to_mx(torch.stack(epss))
        coefs = mx.array(np.array(coefs, dtype=np.float32))
        n_steps = coefs.shape[0]
        # noise-level terms for as many steps at once as fit in COND_BUDGET bytes
        per_step = 4 * s_inputs.shape[1] * self._cond_width
        group = max(1, min(n_steps, COND_BUDGET // per_step))
        for i in range(n_steps):
            if i % group == 0:
                conds = noise_conditioning(self.p, ctx["s0"], coefs[i:i + group, 0], self.sigma)
            j = i % group
            cond = {k: v[j:j + 1] for k, v in conds.items()}
            x, a = self._step(self.p, ctx, cond, x, rots[i], shifts[i], epss[i], coefs[i])
            mx.async_eval(x)  # bounds the lazy graph; the GPU never waits on Python
        mx.eval(x, a)
        return {"sample_atom_coords": torch.from_numpy(np.array(x)).to(device),
                "diff_token_repr": torch.from_numpy(np.array(a)).to(device)}


def _to_mx(t):
    return mx.array(t.detach().float().cpu().numpy())
