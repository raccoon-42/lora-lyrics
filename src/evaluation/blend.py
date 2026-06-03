"""Adapter blending: interpolate two LoRA adapters in delta space.

For a LoRA the update is dW = (alpha/r) * B @ A. To get
    dW_blend(a) = a * dW_x + (1 - a) * dW_y
the exact low-rank form is *rank concatenation* into a rank-2r adapter:
    A_blend = [ a*A_x ; (1-a)*A_y ]   (2r, in)
    B_blend = [ B_x , B_y ]           (out, 2r)
    config: r -> 2r, lora_alpha -> 2*lora_alpha   (keeps scale alpha/r constant)
Averaging A and B separately is wrong (it adds cross-terms B_x @ A_y). Endpoints
(a=0/1) recover the pure adapters exactly -- the built-in sanity check.

Building/validating blends is CPU-only; only generating from them needs the GPU.
"""

import json

import torch
from safetensors.torch import load_file, save_file

from config import ADAPTERS_DIR, blend_pair_key


def load_lora(name):
    """Load (state_dict, config) for an adapter under ADAPTERS_DIR."""
    sd = load_file(ADAPTERS_DIR / name / "adapter_model.safetensors")
    cfg = json.load(open(ADAPTERS_DIR / name / "adapter_config.json"))
    return sd, cfg


def blend_adapters(name_a, name_b, alpha, out_name=None):
    """Delta-space interpolation alpha*A + (1-alpha)*B via rank concatenation.

    Both inputs must be plain LoRA with identical r / lora_alpha / key sets.
    Writes a new rank-2r adapter under ADAPTERS_DIR and returns its name.
    """
    sd_a, cfg_a = load_lora(name_a)
    sd_b, cfg_b = load_lora(name_b)

    assert cfg_a["r"] == cfg_b["r"] and cfg_a["lora_alpha"] == cfg_b["lora_alpha"]
    assert not cfg_a.get("use_dora") and not cfg_b.get("use_dora"), "DoRA not supported"
    assert set(sd_a) == set(sd_b), "adapter key sets differ"

    blended = {}
    for k in sd_a:
        ta, tb = sd_a[k].float(), sd_b[k].float()
        if k.endswith("lora_A.weight"):          # (r, in)  -> concat on rank dim 0
            t = torch.cat([alpha * ta, (1 - alpha) * tb], dim=0)
        elif k.endswith("lora_B.weight"):        # (out, r) -> concat on rank dim 1
            t = torch.cat([ta, tb], dim=1)
        else:
            raise ValueError(f"unexpected key {k}")
        blended[k] = t.to(sd_a[k].dtype)

    out_name = out_name or f"blend_{blend_pair_key(name_a, name_b)}_a{alpha:.2f}"
    out_dir = ADAPTERS_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(cfg_a)
    cfg["r"] = cfg_a["r"] * 2
    cfg["lora_alpha"] = cfg_a["lora_alpha"] * 2      # preserve scale = alpha / r
    cfg["inference_mode"] = True
    json.dump(cfg, open(out_dir / "adapter_config.json", "w"), indent=2)
    save_file(blended, out_dir / "adapter_model.safetensors")
    print(f"wrote {out_dir}  (r={cfg['r']}, alpha={cfg['lora_alpha']}, {len(blended)} tensors)")
    return out_name


def materialize_delta(name, prefix):
    """Reconstruct dW = (alpha/r) * B @ A for one layer `prefix` of adapter `name`.
    Used to numerically verify blends match delta-space interpolation."""
    sd = load_file(ADAPTERS_DIR / name / "adapter_model.safetensors")
    cfg = json.load(open(ADAPTERS_DIR / name / "adapter_config.json"))
    scale = cfg["lora_alpha"] / cfg["r"]
    A = sd[prefix + ".lora_A.weight"].float()
    B = sd[prefix + ".lora_B.weight"].float()
    return scale * (B @ A)
