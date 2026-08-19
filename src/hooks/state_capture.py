"""Capture Mamba write/store/read tensors on the HF sequential (slow) path.

Works without ``selective_scan_cuda``. Monkeypatches ``MambaMixer.slow_forward``
so each forward pass records :math:`h_t`, :math:`B`, :math:`C`, :math:`\\Delta`,
conv output, residual, and mixer output for any selected layers.

Usage::

    with capture_mamba_states(model, layers=[17]) as cap:
        out = model(input_ids=ids, use_cache=False)
    print(cap.summary())
    h = cap.layers[17].h  # [B, T, E, N]
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Set

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class LayerStateTrace:
    """All tensors captured at one layer during a single forward pass.

    Shapes (batch first):
      residual:      [B, T, D]   - skip stream before the mixer add
      mixer_out:     [B, T, D]   - output of ``out_proj``
      residual_out:  [B, T, D]   - residual + mixer_out
      conv_out:      [B, T, E]   - after the depthwise conv + activation
      delta:         [B, T, E]   - Δ after softplus (discrete step size)
      B:             [B, T, N]   - input projection for the state write
      C:             [B, T, N]   - output projection for the state read
      h:             [B, T, E, N] - SSM state after each token
      gate:          [B, T, E]   - gating signal before the final gating multiply
    """

    layer_idx: int
    residual: Optional[Tensor] = None
    mixer_out: Optional[Tensor] = None
    residual_out: Optional[Tensor] = None
    conv_out: Optional[Tensor] = None
    delta: Optional[Tensor] = None
    B: Optional[Tensor] = None
    C: Optional[Tensor] = None
    h: Optional[Tensor] = None
    gate: Optional[Tensor] = None


@dataclass
class CaptureBundle:
    """Collection of per-layer traces for one forward pass."""

    layers: Dict[int, LayerStateTrace] = field(default_factory=dict)

    def summary(self) -> str:
        lines = []
        for idx in sorted(self.layers):
            tr = self.layers[idx]
            bits = [f"L{idx}"]
            for name in ("residual", "h", "delta", "B", "C", "conv_out"):
                t = getattr(tr, name)
                if t is not None:
                    bits.append(f"{name}={tuple(t.shape)}")
            lines.append(" ".join(bits))
        return "\n".join(lines)


def _normalize_layers(layers: Optional[Sequence[int]], n_layers: int) -> Set[int]:
    if layers is None:
        return set(range(n_layers))
    return {int(i) for i in layers}


def _instrumented_slow_forward(
    self,
    input_states: Tensor,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
    *,
    store: LayerStateTrace,
    keep_on_cpu: bool,
):
    """Drop-in replacement for ``MambaMixer.slow_forward`` that records activations.

    Only the full-sequence prefill path (``cache_params is None``) is used for
    interpretability dumps. The cache path is preserved for completeness but will
    raise if called outside of a full prefill.
    """
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype

    # HF sets use_cache=True by default, so cache_params is non-None even on full
    # prefill. The prefill marker from transformers: cache_position length == conv_kernel.
    if cache_params is not None:
        if cache_position is None or cache_position.shape[0] != self.conv_kernel_size:
            raise RuntimeError(
                "State capture only supports full-sequence prefill. "
                "Call model(..., use_cache=False)."
            )

    projected_states = self.in_proj(input_states).transpose(1, 2)
    hidden_states, gate = projected_states.chunk(2, dim=1)

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    if cache_params is not None:
        ssm_state = cache_params.ssm_states[self.layer_idx].clone().to(hidden_states.device)
        conv_state = nn.functional.pad(
            hidden_states,
            (self.conv_kernel_size - hidden_states.shape[-1], 0),
        )
        cache_params.update_conv_state(self.layer_idx, conv_state, cache_position)
    else:
        ssm_state = torch.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size),
            device=hidden_states.device,
            dtype=dtype,
        )
    hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters,
        [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
        dim=-1,
    )
    discrete_time_step = nn.functional.softplus(self.dt_proj(time_step)).transpose(1, 2)

    A = -torch.exp(self.A_log.float())
    discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
    discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

    h_steps: List[Tensor] = []
    scan_outputs: List[Tensor] = []
    for i in range(seq_len):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
        h_steps.append(ssm_state)
        scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))
        scan_outputs.append(scan_output[:, :, 0])
    scan_output = torch.stack(scan_outputs, dim=-1)
    scan_output = scan_output + (hidden_states * self.D[None, :, None])
    scan_output = scan_output * self.act(gate)

    if cache_params is not None:
        cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

    contextualized_states = self.out_proj(scan_output.transpose(1, 2))

    def park(t: Tensor) -> Tensor:
        t = t.detach()
        return t.cpu() if keep_on_cpu else t

    store.conv_out = park(hidden_states.transpose(1, 2).contiguous())
    store.delta    = park(discrete_time_step.transpose(1, 2).contiguous())
    store.B        = park(B)
    store.C        = park(C)
    store.gate     = park(gate.transpose(1, 2).contiguous())
    store.h        = park(torch.stack(h_steps, dim=1).contiguous())
    store.mixer_out = park(contextualized_states)

    return contextualized_states


@contextmanager
def capture_mamba_states(
    model: nn.Module,
    layers: Optional[Sequence[int]] = None,
    keep_on_cpu: bool = True,
) -> Iterator[CaptureBundle]:
    """Context manager that records state traces for selected layers.

    Example::

        with capture_mamba_states(model, layers=[0, 17]) as cap:
            out = model(input_ids=ids)
        print(cap.summary())
    """
    backbone = model.backbone if hasattr(model, "backbone") else model
    blocks = backbone.layers
    want = _normalize_layers(layers, len(blocks))
    bundle = CaptureBundle()
    originals = {}
    block_hooks = []

    for layer_idx, block in enumerate(blocks):
        if layer_idx not in want:
            continue
        mixer = block.mixer
        trace = LayerStateTrace(layer_idx=layer_idx)
        bundle.layers[layer_idx] = trace
        originals[layer_idx] = mixer.slow_forward

        def make_slow(m=mixer, tr=trace):
            def slow_forward(input_states, cache_params=None, cache_position=None, attention_mask=None):
                return _instrumented_slow_forward(
                    m, input_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                    store=tr,
                    keep_on_cpu=keep_on_cpu,
                )
            return slow_forward

        mixer.slow_forward = make_slow()

        def make_block_hook(tr=trace):
            def hook(module, inputs, output):
                residual_in = inputs[0]
                tr.residual = residual_in.detach().cpu() if keep_on_cpu else residual_in.detach()
                tr.residual_out = output.detach().cpu() if keep_on_cpu else output.detach()
            return hook

        block_hooks.append(block.register_forward_hook(make_block_hook()))

    try:
        yield bundle
    finally:
        for layer_idx, block in enumerate(blocks):
            if layer_idx in originals:
                block.mixer.slow_forward = originals[layer_idx]
        for h in block_hooks:
            h.remove()


def h_norm_by_time(h: Tensor) -> Tensor:
    """h [B, T, E, N] → [T] mean Frobenius norm over batch and channel dims."""
    return h.float().flatten(2).norm(dim=-1).mean(dim=0)


def delta_mean_by_time(delta: Tensor) -> Tensor:
    """delta [B, T, E] → [T] mean over batch and intermediate dim."""
    return delta.float().mean(dim=(0, 2))
