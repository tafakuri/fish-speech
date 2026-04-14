"""Lightning callback that generates audio samples at regular training steps.

Saves a wav file per test sentence into ``output_dir/step_XXXXXXXX/``
so you can track audio quality progress alongside the loss curve.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import lightning as L
from lightning import Callback, Trainer

from fish_speech.utils.logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class AudioSampleCallback(Callback):
    """Generate audio samples from the training model at regular step intervals.

    At every ``sample_every_n_steps`` training steps this callback:
    1. Switches the model to eval mode (loralib auto-merges LoRA weights)
    2. Generates semantic tokens for each test sentence via ``generate_long()``
    3. Decodes the tokens to audio through the VQGAN codec
    4. Saves ``sample_NN.wav`` files under ``output_dir/step_XXXXXXXX/``
    5. Restores the model to train mode (loralib auto-un-merges)

    Args:
        codec_checkpoint_path: Path to ``codec.pth`` (usually inside the base
            model checkpoint directory, e.g. ``checkpoints/fish-speech-1.5/codec.pth``).
        sentences: List of text strings to synthesise at each checkpoint.
        prompt_text: Transcript of the reference speaker clip (used for voice
            conditioning).
        prompt_tokens_path: Path to the ``.npy`` file of VQ tokens extracted
            from the reference speaker clip (produced by ``extract_vq.py``).
        output_dir: Directory under which per-step subdirectories are created.
        sample_every_n_steps: Generate samples every this many training steps.
        temperature: Sampling temperature (0.5–0.9 recommended).
        top_p: Nucleus sampling cutoff.
        top_k: Top-k sampling cutoff.
        repetition_penalty: Penalise repeated tokens.
        max_new_tokens: Hard cap on generated token count (safety guard).
    """

    def __init__(
        self,
        codec_checkpoint_path: str,
        sentences: list,
        prompt_text: str,
        prompt_tokens_path: str,
        output_dir: str = "samples",
        sample_every_n_steps: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 30,
        repetition_penalty: float = 1.3,
        max_new_tokens: int = 500,
    ):
        super().__init__()
        self.codec_checkpoint_path = Path(codec_checkpoint_path)
        self.sentences = list(sentences)
        self.prompt_text = prompt_text
        self.prompt_tokens_path = Path(prompt_tokens_path)
        self.output_dir = Path(output_dir)
        self.sample_every_n_steps = sample_every_n_steps
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens

        self._codec = None
        self._prompt_tokens: Optional[torch.Tensor] = None

    def setup(self, trainer: Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if stage != "fit":
            return

        from fish_speech.models.text2semantic.inference import load_codec_model

        device = pl_module.device
        log.info(f"AudioSampleCallback: loading codec from {self.codec_checkpoint_path}")
        self._codec = load_codec_model(self.codec_checkpoint_path, device=device)

        self._prompt_tokens = torch.from_numpy(np.load(self.prompt_tokens_path))
        log.info(
            f"AudioSampleCallback: reference tokens shape {self._prompt_tokens.shape}, "
            f"saving samples to {self.output_dir}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        if step == 0 or step % self.sample_every_n_steps != 0:
            return
        # Only generate on rank 0 in multi-GPU setups
        if trainer.local_rank != 0:
            return
        self._generate_samples(trainer, pl_module)

    def _generate_samples(
        self, trainer: Trainer, pl_module: L.LightningModule
    ) -> None:
        from fish_speech.models.text2semantic.inference import (
            decode_one_token_ar,
            decode_to_audio,
            generate_long,
        )

        step = trainer.global_step
        step_dir = self.output_dir / f"step_{step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        model = pl_module.model
        device = pl_module.device
        was_training = model.training

        log.info(f"AudioSampleCallback: generating {len(self.sentences)} sample(s) at step {step}")

        try:
            model.eval()
            with torch.no_grad():
                model.setup_caches(
                    max_batch_size=1,
                    max_seq_len=model.config.max_seq_len,
                    dtype=next(model.parameters()).dtype,
                )
                prompt_tokens = self._prompt_tokens.to(device)

                for idx, sentence in enumerate(self.sentences):
                    out_path = step_dir / f"sample_{idx:02d}.wav"
                    try:
                        codes = None
                        for response in generate_long(
                            model=model,
                            device=device,
                            decode_one_token=decode_one_token_ar,
                            text=sentence,
                            num_samples=1,
                            max_new_tokens=self.max_new_tokens,
                            top_p=self.top_p,
                            top_k=self.top_k,
                            temperature=self.temperature,
                            repetition_penalty=self.repetition_penalty,
                            iterative_prompt=False,
                            prompt_text=self.prompt_text,
                            prompt_tokens=prompt_tokens,
                        ):
                            if response.action == "sample" and response.codes is not None:
                                codes = response.codes

                        if codes is None:
                            log.warning(
                                f"Step {step}: no codes generated for sentence {idx} — skipping"
                            )
                            continue

                        audio = decode_to_audio(codes.to(device), self._codec)
                        torchaudio.save(
                            str(out_path),
                            audio.unsqueeze(0).float().cpu(),
                            self._codec.sample_rate,
                        )
                        duration = audio.shape[-1] / self._codec.sample_rate
                        log.info(f"  [{idx}] {out_path.name}  ({duration:.1f}s)")

                    except Exception as exc:
                        log.warning(f"Step {step}: sample {idx} failed — {exc}")

        finally:
            if was_training:
                model.train()
