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


def generate_lyrics(model, tokenizer, prompt=PROMPT, **gen_kwargs):
    """Sample lyrics from `model`. `gen_kwargs` overrides the defaults in GEN_KWARGS."""
    kwargs = {**GEN_KWARGS, **gen_kwargs}
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, **kwargs)
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
