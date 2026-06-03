"""QLoRA/DoRA adapter training for artist-conditional generation.

`train_adapter` takes the (already kbit-prepared) base model, tokenizer, and the
training DataFrame explicitly -- no reliance on notebook globals -- and saves the
adapter under the path derived from `config.Adapter`, so naming lives in one place.
"""

from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from config import Adapter
from .data import artist_dataset
from .style_loss import make_style_loss_func

# Inject LoRA into every attention + MLP projection of the language-model layers.
TARGET_MODULES = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def train_adapter(
    model, tokenizer, train_df, artist,
    r=8, use_dora=False, epochs=3, lr=2e-4, style_weights=None,
    overwrite=False,
):
    """Train one LoRA/DoRA adapter for `artist` and save it under ADAPTERS_DIR.

    `model` must already be wrapped with `prepare_model_for_kbit_training`.
    Passing `style_weights` switches to the style-weighted loss and the `_sw`
    output suffix (so SW adapters don't clobber the plain ones).

    If the adapter already exists on disk it is left untouched and its path is
    returned, so re-running the training cell is a no-op. Pass `overwrite=True`
    to force a retrain.
    """
    spec = Adapter(artist, "dora" if use_dora else "lora", r, sw=style_weights is not None)
    output_dir = str(spec.path)

    # Key on the final saved weights, not mere dir existence -- the dir also holds
    # intermediate checkpoint-*/ subdirs, so a crashed run leaves a dir with no
    # adapter_model.safetensors at the root. (This is the file blend.py loads.)
    if (spec.path / "adapter_model.safetensors").exists() and not overwrite:
        print(f"[skip] adapter exists, not retraining: {output_dir} (overwrite=True to force)")
        return output_dir

    dataset = artist_dataset(train_df, artist)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=r * 2,
        target_modules=TARGET_MODULES,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=use_dora,
    )
    peft_model = get_peft_model(model, lora_config)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        max_length=512,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="none",
        weight_decay=0.05,
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        compute_loss_func=make_style_loss_func(style_weights) if style_weights is not None else None,
    )

    trainer.train()
    peft_model.save_pretrained(output_dir)
    peft_model.unload()
    print(f"Saved: {output_dir} ({len(dataset)} songs, {spec.kind}, r={r}{'_sw' if spec.sw else ''})")
    return output_dir
