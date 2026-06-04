"""Lyric sampling from the base model (optionally with an adapter attached)."""

from config import PROMPT

# Default sampling config for lyric generation (matches the original 06_evaluation).
GEN_KWARGS = dict(
    max_new_tokens=512,
    min_new_tokens=200,
    temperature=0.9,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
)


def generate_lyrics(model, tokenizer, prompt=PROMPT, add_special_tokens=True, **gen_kwargs):
    """Sample lyrics from `model`. `gen_kwargs` overrides the defaults in GEN_KWARGS.

    `add_special_tokens=False` for prompts that already carry their special tokens
    (e.g. a chat-template-rendered instruct prompt) -- avoids a doubled BOS."""
    kwargs = {**GEN_KWARGS, **gen_kwargs}
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096,
                       add_special_tokens=add_special_tokens).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, **kwargs)
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)


def generate_samples(model, tokenizer, prompt, n, **gen_kwargs):
    """`n` independent completions from `prompt`, each via generate_lyrics (so the
    same GEN_KWARGS). Used by the baselines notebook for zero-/few-shot prompts."""
    return [generate_lyrics(model, tokenizer, prompt=prompt, **gen_kwargs) for _ in range(n)]
