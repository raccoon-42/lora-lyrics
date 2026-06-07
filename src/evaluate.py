"""Evaluate baselines + every trained adapter, caching results under results/.

Run before the figures notebook (06_evaluation):

    uv run python evaluate.py

Loads the base model + classifier once, then writes one JSON per result, grouped
by artist:
  - baselines: results/baselines/<artist>/{zero_shot,few_shot}.json
  - adapters:  results/adapters/<artist>/<variant>.json
  - blends:    results/blends/<pair>/a<alpha>.json

Baselines recompute only when their spec (n_samples + prompt) changes; adapters
recompute only when their weights are newer than the cached entry; blends
recompute only when their spec (n_samples + alpha + source adapter mtimes)
changes. Untouched results are skipped (no GPU generation).

Pass --force to ignore all caches and recompute everything, or one of
--force-baselines / --force-adapters / --force-blends to recompute just that group.
"""
import json
import random
from pathlib import Path

import pandas as pd
import torch

from config import ARTISTS, DATA_DIR, RESULTS_DIR, ADAPTERS_DIR as WEIGHTS_DIR, MODEL_PATH_IT, adapter_registry, blend_pair_key
from classifier.classify import classify
from classifier.model import load_classifier
from evaluation.blend import blend_adapters
from evaluation.metrics import evaluate_adapter
from generation.generate import generate_samples
from generation.model import load_base_model

N_SAMPLES = 10
FEWSHOT_EXAMPLES = 3
FEWSHOT_SEED = 42
GEN_SEED = 0          # reset before each adapter/blend eval -> paired sampling noise across adapters

# Gojira-anchored SW set: same Gojira-SW endpoint in every pair -> alpha = Gojira
# weight everywhere, so the 4 crossover curves overlay on one axis. SW adapters are
# plain LoRA r8 structurally (SW only changes training loss), so they blend the same way.
BLEND_PAIRS = [
    ("gojira_lora_r8", "tool_lora_r8"),          # original plain reference pair
    ("gojira_lora_r8_sw", "tool_lora_r8_sw"),
    ("gojira_lora_r8_sw", "death_lora_r8_sw"),
    ("gojira_lora_r8_sw", "opeth_lora_r8_sw"),
    ("gojira_lora_r8_sw", "mastodon_lora_r8_sw"),
]
BLEND_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]   # 1 = pure first source

BASELINES_DIR = RESULTS_DIR / "baselines"
ADAPTERS_DIR = RESULTS_DIR / "adapters"
BLENDS_DIR = RESULTS_DIR / "blends"


def _weights_mtime(path):
    # Newest file under the adapter dir -- bumps whenever the adapter is retrained.
    return max(f.stat().st_mtime for f in path.rglob("*") if f.is_file())


def _cache_baseline(model, tokenizer, clf, artist, method, prompt, force=False, **gen_kwargs):
    # Spec-based guard: baselines have no weight file, so recompute only when
    # n_samples or the (deterministic) prompt changes. force=True ignores the cache.
    cache_file = BASELINES_DIR / artist.lower().replace(" ", "_") / f"{method}.json"
    spec = {"n_samples": N_SAMPLES, "prompt": prompt}
    if not force and cache_file.exists() and json.load(open(cache_file)).get("spec") == spec:
        print(f"cached {artist} {method}: up to date")
        return

    print(f"\n=== {artist} {method} ===")
    samples = generate_samples(model, tokenizer, prompt, N_SAMPLES, **gen_kwargs)
    df = pd.DataFrame([classify(clf, text) for text in samples])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({"target": artist, "method": method, "samples": samples,
                   "df": df.to_dict(orient="list"), "spec": spec}, f, indent=2)
    print(f"  mean {df[artist].mean():.4f} +/- {df[artist].std():.4f}")


def run_baselines(model, tokenizer, clf, force=False):
    # B1 -- zero-shot: artist name in prompt, no examples.
    for artist in ARTISTS:
        prompt = f"Write song lyrics in the style of {artist}.\n\n"
        _cache_baseline(model, tokenizer, clf, artist, "zero_shot", prompt, force)

    # B2 -- few-shot: artist name + FEWSHOT_EXAMPLES in-context examples. Fixed
    # seed so the example selection (and thus the prompt) is stable across runs,
    # which keeps the spec cache reliable.
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    random.seed(FEWSHOT_SEED)
    for artist in ARTISTS:
        # Use the cleaned column so examples carry no [????] redactions or
        # [Verse]/[Chorus] headers -- matches the text the adapters train on.
        lyrics = train_df[train_df["artist"] == artist]["clean"].tolist()
        examples = random.sample(lyrics, FEWSHOT_EXAMPLES)
        prompt = f"Write song lyrics in the style of {artist}.\n\n"
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n{ex}\n\n"
        prompt += f"Now write new song lyrics in the style of {artist}:\n\n"
        _cache_baseline(model, tokenizer, clf, artist, "few_shot", prompt, force)


# B3/B4 use the instruction-tuned model with REALISTIC prompts: name the band and
# its genre (disambiguates "Tool"/"Gojira") and ask for lyrics only. Few-shot reuses
# B2's example selection (clean col, FEWSHOT_SEED) so the comparison is fair.
ARTIST_GENRE = {
    "Gojira": "the French progressive metal band Gojira",
    "Tool": "the American progressive metal band Tool",
    "Death": "the American death metal band Death",
    "Mastodon": "the American progressive sludge metal band Mastodon",
    "Opeth": "the Swedish progressive death metal band Opeth",
}
_LYRICS_ONLY = " Write only the lyrics, with no title, section labels, commentary, or explanation."


def _chat_prompt(tokenizer, content):
    # Render a single user turn through the model's chat template, with the
    # generation prompt appended so the model continues as the assistant.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )


def run_baselines_it(clf, force=False):
    # B3/B4 -- instruction-tuned baselines. Loads the -it model itself (the base
    # model used for B1/B2/adapters can't follow instructions), runs zero-/few-shot
    # with proper chat-formatted prompts, then frees it before adapters/blends.
    if not Path(MODEL_PATH_IT).exists():
        print(f"[skip] -it model not found at {MODEL_PATH_IT} -- run download_it.py for B3/B4")
        return
    model, tokenizer = load_base_model(MODEL_PATH_IT)

    # The chat template already emits BOS -> tell the tokenizer not to add a second
    # one (detected, not assumed, so it stays correct if the template changes).
    bos = tokenizer.bos_token
    add_special = not (bos and _chat_prompt(tokenizer, "x").startswith(bos))
    # Stop at end-of-turn so the instruct model doesn't ramble past the lyrics.
    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if eot is not None and eot != tokenizer.unk_token_id:
        eos_ids.append(eot)
    gen_kw = dict(add_special_tokens=add_special, eos_token_id=eos_ids)

    # B3 -- zero-shot (instruct): named band + genre, no examples.
    for artist in ARTISTS:
        content = f"Write original song lyrics in the style of {ARTIST_GENRE[artist]}.{_LYRICS_ONLY}"
        prompt = _chat_prompt(tokenizer, content)
        _cache_baseline(model, tokenizer, clf, artist, "zero_shot_it", prompt, force, **gen_kw)

    # B4 -- few-shot (instruct): same 3 examples as B2 (clean col, FEWSHOT_SEED).
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    random.seed(FEWSHOT_SEED)
    for artist in ARTISTS:
        lyrics = train_df[train_df["artist"] == artist]["clean"].tolist()
        examples = random.sample(lyrics, FEWSHOT_EXAMPLES)
        content = f"Here are example songs by {ARTIST_GENRE[artist]}:\n\n"
        for i, ex in enumerate(examples, 1):
            content += f"Example {i}:\n{ex}\n\n"
        content += f"Now write original new song lyrics in the same style.{_LYRICS_ONLY}"
        prompt = _chat_prompt(tokenizer, content)
        _cache_baseline(model, tokenizer, clf, artist, "few_shot_it", prompt, force, **gen_kw)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_adapters(base_model, tokenizer, clf, force=False):
    for a in adapter_registry():
        if not a.path.exists():
            print(f"skip {a.label}: {a.path} not found")
            continue

        mtime = _weights_mtime(a.path)
        cache_file = ADAPTERS_DIR / a.result_relpath
        if not force and cache_file.exists() and json.load(open(cache_file)).get("mtime") == mtime:
            print(f"cached {a.label}: up to date")
            continue

        print(f"\n=== {a.label} ===")
        torch.manual_seed(GEN_SEED)   # same RNG state per adapter -> paired A/B (e.g. plain vs SW)
        samples, df = evaluate_adapter(base_model, tokenizer, clf, a.path)
        # Store everything JSON-native; the Adapter object is re-attached from the
        # registry on load, so it never needs serializing.
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({"target": a.artist, "samples": samples,
                       "df": df.to_dict(orient="list"), "mtime": mtime}, f, indent=2)
        print(f"  Target-artist mean: {df[a.artist].mean():.4f} "
              f"+/- {df[a.artist].std():.4f}  (wrote {a.result_relpath})")


def run_blends(base_model, tokenizer, clf, force=False):
    # Blends have no weight file of their own -- they're rebuilt (CPU) from two
    # source adapters each time. Spec-based guard: recompute only when n_samples,
    # the alpha, the source names, or the SOURCE adapters' mtimes change. The
    # source adapters are never retrained here; only the derived blend is rebuilt.
    for src_a, src_b in BLEND_PAIRS:
        pa, pb = WEIGHTS_DIR / src_a, WEIGHTS_DIR / src_b
        if not (pa.exists() and pb.exists()):
            print(f"skip blend {src_a}+{src_b}: source adapter missing")
            continue

        pair = blend_pair_key(src_a, src_b)   # gojira_tool / gojira_sw_tool_sw
        src_mtimes = [_weights_mtime(pa), _weights_mtime(pb)]
        for alpha in BLEND_ALPHAS:
            cache_file = BLENDS_DIR / pair / f"a{alpha:.2f}.json"
            spec = {"n_samples": N_SAMPLES, "src_a": src_a, "src_b": src_b,
                    "alpha": alpha, "src_mtimes": src_mtimes}
            if not force and cache_file.exists() and json.load(open(cache_file)).get("spec") == spec:
                print(f"cached blend {pair} a={alpha:.2f}: up to date")
                continue

            print(f"\n=== blend {pair} a={alpha:.2f} ===")
            blend_name = blend_adapters(src_a, src_b, alpha,   # CPU: (re)writes rank-2r adapter
                                        out_name=f"blend_{pair}_a{alpha:.2f}")
            torch.manual_seed(GEN_SEED)
            samples, df = evaluate_adapter(base_model, tokenizer, clf, WEIGHTS_DIR / blend_name)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump({"src_a": src_a, "src_b": src_b, "alpha": alpha,
                           "samples": samples, "df": df.to_dict(orient="list"),
                           "spec": spec}, f, indent=2)
            ea = next(a for a in ARTISTS if a.lower() == src_a.split('_')[0])  # alpha-weighted endpoint
            eb = next(a for a in ARTISTS if a.lower() == src_b.split('_')[0])  # (1-alpha)-weighted endpoint
            print(f"  {ea}={df[ea].mean():.4f}  {eb}={df[eb].mean():.4f}  "
                  f"(wrote {pair}/a{alpha:.2f}.json)")


def main(force_baselines=False, force_adapters=False, force_blends=False):
    base_model, tokenizer = load_base_model()
    clf = load_classifier()

    print("\n##### BASELINES (base model) #####")
    run_baselines(base_model, tokenizer, clf, force=force_baselines)

    print("\n##### BASELINES (instruct model) #####")
    run_baselines_it(clf, force=force_baselines)

    print("\n##### ADAPTERS #####")
    run_adapters(base_model, tokenizer, clf, force=force_adapters)

    print("\n##### BLENDS #####")
    run_blends(base_model, tokenizer, clf, force=force_blends)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Evaluate baselines + adapters + blends, caching under results/.")
    p.add_argument("--force", action="store_true",
                   help="ignore all caches and recompute everything")
    p.add_argument("--force-baselines", action="store_true", help="recompute baselines only")
    p.add_argument("--force-adapters", action="store_true", help="recompute adapters only")
    p.add_argument("--force-blends", action="store_true", help="recompute blends only")
    args = p.parse_args()

    main(force_baselines=args.force or args.force_baselines,
         force_adapters=args.force or args.force_adapters,
         force_blends=args.force or args.force_blends)
