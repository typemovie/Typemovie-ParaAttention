import functools
from typing import List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline, QwenImageTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

import para_attn.primitives as DP
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.para_attn_interface import UnifiedAttnMode


def parallelize_transformer(transformer: QwenImageTransformer2DModel, *, mesh=None):
    """
    Parallelize QwenImageTransformer2DModel with batch-parallel (data parallel on dim=0)
    and sequence-parallel (token parallel on dim=-2) using para-attn.

    This wrapper mirrors the model's forward to:
      - shard inputs on batch and sequence dims before compute
      - compute RoPE freqs once, then shard them on sequence dim
      - run attention under UnifiedAttnMode(mesh)
      - gather outputs back on sequence and batch dims to keep API identical
    """
    if getattr(transformer, "_is_parallelized", False):
        return transformer

    mesh = init_context_parallel_mesh(transformer.device.type, mesh=mesh)
    batch_mesh = mesh["batch"]
    seq_mesh = mesh["ring", "ulysses"]._flatten()

    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self: QwenImageTransformer2DModel,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[dict] = None,
        return_dict: bool = True,
        *args,
        **kwargs,
    ):
        # 1) Pre-shard batch-wise inputs
        if isinstance(timestep, torch.Tensor) and timestep.ndim != 0 and timestep.shape[0] == hidden_states.shape[0]:
            timestep = DP.get_assigned_chunk(timestep, dim=0, group=batch_mesh)

        hidden_states = DP.get_assigned_chunk(hidden_states, dim=0, group=batch_mesh)
        if encoder_hidden_states is not None:
            encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=0, group=batch_mesh)
        if encoder_hidden_states_mask is not None:
            # Mask shape: (B, S_txt) -> split on batch then on seq-last dim
            encoder_hidden_states_mask = DP.get_assigned_chunk(encoder_hidden_states_mask, dim=0, group=batch_mesh)

        # 2) Sequence-parallel shard on token dim (-2)
        hidden_states = DP.get_assigned_chunk(hidden_states, dim=-2, group=seq_mesh)
        if encoder_hidden_states is not None:
            encoder_hidden_states = DP.get_assigned_chunk(encoder_hidden_states, dim=-2, group=seq_mesh)
        if encoder_hidden_states_mask is not None:
            # for 2D mask (B, S_txt) the sequence dimension is -1
            encoder_hidden_states_mask = DP.get_assigned_chunk(encoder_hidden_states_mask, dim=-1, group=seq_mesh)

        # 3) LoRA scaling passed via attention_kwargs (same semantics as original forward)
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        # 4) Compute the rest of forward largely as in the original, but under UnifiedAttnMode and with RoPE sharded
        with UnifiedAttnMode(mesh):
            # Embeddings and projections
            hidden_states = self.img_in(hidden_states)

            if timestep is not None:
                timestep = timestep.to(hidden_states.dtype)

            if encoder_hidden_states is not None:
                encoder_hidden_states = self.txt_norm(encoder_hidden_states)
                encoder_hidden_states = self.txt_in(encoder_hidden_states)

            if guidance is not None:
                guidance = guidance.to(hidden_states.dtype) * 1000

            # Note: QwenTimestepProjEmbeddings signature is (timestep, hidden_states)
            temb = (
                self.time_text_embed(timestep, hidden_states)
                if guidance is None
                else self.time_text_embed(timestep, guidance, hidden_states)
            )

            # Compute full RoPE freqs using provided shapes/lens then shard on seq-dim for local chunk
            # pos_embed returns (vid_freqs [S_img, D], txt_freqs [S_txt, D])
            image_rotary_emb_full = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
            vid_freqs_full, txt_freqs_full = image_rotary_emb_full

            vid_freqs_local = DP.get_assigned_chunk(vid_freqs_full, dim=-2, group=seq_mesh)
            txt_freqs_local = DP.get_assigned_chunk(txt_freqs_full, dim=-2, group=seq_mesh)
            image_rotary_emb_local = (vid_freqs_local, txt_freqs_local)

            # Transformer blocks
            for block in self.transformer_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        encoder_hidden_states_mask,
                        temb,
                        image_rotary_emb_local,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb_local,
                        joint_attention_kwargs=attention_kwargs,
                    )

            # Only image stream is output
            hidden_states = self.norm_out(hidden_states, temb)
            sample_local = self.proj_out(hidden_states)

        # 5) Unscale LoRA if needed
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        # 6) Gather outputs back on sequence and batch dims (keep API identical to original)
        sample = DP.get_complete_tensor(sample_local, dim=-2, group=seq_mesh)
        sample = DP.get_complete_tensor(sample, dim=0, group=batch_mesh)

        if return_dict:
            return Transformer2DModelOutput(sample=sample)
        return (sample,)

    transformer.forward = new_forward.__get__(transformer)
    transformer._is_parallelized = True

    return transformer


def parallelize_pipe(pipe: DiffusionPipeline, *, shallow_patch: bool = False, **kwargs):
    """
    Parallelize a DiffusionPipeline that contains a QwenImageTransformer2DModel as `pipe.transformer`.
    This also patches pipeline.__call__ to create a synchronized generator when none is provided.
    """
    if not getattr(pipe, "_is_parallelized", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, **kwargs):
            # Ensure all ranks use the same seed if user didn't pass a generator
            if generator is None and getattr(self, "_is_parallelized", False):
                seed_t = torch.randint(0, torch.iinfo(torch.int64).max, [1], dtype=torch.int64, device=self.device)
                seed_t = DP.get_complete_tensor(seed_t, dim=0)
                seed_t = DP.get_assigned_chunk(seed_t, dim=0, idx=0)
                seed = seed_t.item()
                seed -= torch.iinfo(torch.int64).min
                generator = torch.Generator(self.device).manual_seed(seed)
            return original_call(self, *args, generator=generator, **kwargs)

        new_call._is_parallelized = True
        pipe.__class__.__call__ = new_call
        pipe.__class__._is_parallelized = True

    if not shallow_patch and hasattr(pipe, "transformer") and isinstance(pipe.transformer, QwenImageTransformer2DModel):
        parallelize_qwen_image_transformer(pipe.transformer, **kwargs)

    return pipe