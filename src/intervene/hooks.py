"""Monkeypatch the HF Mamba slow path to apply causal interventions.

Supports editing ``h``, ``B``, ``C``, ``Δ``, ``residual``, and the SSM scan
(``local_only``) at nominated token positions and layers, all within a single
forward pass and without modifying model weights.

The two public entry points are:
  - ``apply_patches(model, state, layers)`` - context manager for one condition
  - ``PatchState`` - mutable per-example config passed into the context
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Set

import torch
import torch.nn as nn
from torch import Tensor

from .spec import Intervention


@dataclass
class PatchState:
    """Mutable bag of per-example config shared between the mixer and block hooks.

    Reset the per-example fields (``times``, ``donor_h``, etc.) for each example
    before calling the model; the ``spec`` and ``layer_set`` are shared across all
    examples in one condition.
    """

    spec: Optional[Intervention] = None
    times: List[int] = field(default_factory=list)
    layer_set: Set[int] = field(default_factory=set)
    # Per-layer donor tensors (for swap / restore / residual-restore / C-patch)
    donor_h: Dict[int, Tensor] = field(default_factory=dict)
    donor_residual: Dict[int, Tensor] = field(default_factory=dict)
    donor_C: Dict[int, Tensor] = field(default_factory=dict)
    # Captured outputs from a preceding clean run (for residual-noise budget matching)
    mixer_out: Dict[int, Tensor] = field(default_factory=dict)
    # Captured h and residual sequences (when capture=True)
    h_write_end: Dict[int, Tensor] = field(default_factory=dict)
    h_seq: Dict[int, Tensor] = field(default_factory=dict)
    residual_seq: Dict[int, Tensor] = field(default_factory=dict)
    # Per-layer [T] L2 of (mixer_intervened - mixer_clean); used to budget residual noise
    delta_mag: Dict[int, Tensor] = field(default_factory=dict)
    capture: bool = False
    noise_seed: int = 0
    # Token index at which to paste the donor h (defaults to last write token)
    restore_at: Optional[int] = None
    # If set, zero only these N-slot indices of h (channel ablation)
    channels_override: Optional[List[int]] = None


def _times_mask(times: Sequence[int], seq_len: int) -> List[int]:
    return [int(t) for t in times if 0 <= int(t) < seq_len]


def _instrumented_slow_forward(
    self,
    input_states: Tensor,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
    *,
    layer_idx: int,
    state: PatchState,
):
    """``MambaMixer.slow_forward`` extended with optional h/B/C/Δ/SSM edits.

    All edits are applied in-place on cloned tensors so the clean forward is not
    mutated. The ``local_only`` op zeroes the scan output at every step, leaving
    only the conv + D·u path - this is the "state off" control.
    """
    spec = state.spec
    residual_restore = (
        spec is not None and spec.target == "residual" and spec.op == "restore"
    )
    active = spec is not None and layer_idx in state.layer_set and (
        spec.target != "residual" or residual_restore
    )
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    times = set(_times_mask(state.times, seq_len))

    if cache_params is not None:
        if cache_position is None or cache_position.shape[0] != self.conv_kernel_size:
            raise RuntimeError(
                "Interventions only support full-sequence prefill. "
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
            device=hidden_states.device, dtype=dtype,
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

    # Apply gate-level ops (B-zero, C-zero, C-swap, Δ-clamp) before the scan.
    if active and times:
        B = B.clone()
        C = C.clone()
        discrete_time_step = discrete_time_step.clone()
        t_list = sorted(times)
        if spec.target == "B" and spec.op == "zero":
            B[:, t_list, :] = 0
        elif spec.target == "C" and spec.op == "zero":
            C[:, t_list, :] = 0
        elif spec.target == "C" and spec.op == "swap":
            donor_c = state.donor_C.get(layer_idx)
            if donor_c is not None:
                vec = donor_c.to(device=C.device, dtype=C.dtype)
                if vec.dim() == 1:
                    vec = vec.unsqueeze(0)
                for t in t_list:
                    C[:, t, :] = vec
        elif spec.target == "delta" and spec.op == "clamp":
            discrete_time_step[:, :, t_list] = float(spec.clamp)

    A = -torch.exp(self.A_log.float())
    discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
    discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

    local_only = active and spec.target == "ssm" and spec.op == "local"
    zero_h = active and (
        (spec.target == "h" and spec.op in ("zero", "restore")) or residual_restore
    )
    swap_h = active and spec.target == "h" and spec.op in ("swap", "restore")
    chans = list(state.channels_override) if state.channels_override is not None else (
        list(spec.channels) if spec is not None and spec.channels else None
    )
    write_end  = max(times) if times else max(0, seq_len - 2)
    restore_at = state.restore_at if state.restore_at is not None else write_end
    has_h_donor = swap_h and state.donor_h.get(layer_idx) is not None

    scan_outputs: List[Tensor] = []
    h_at_write_end = None
    h_frames: List[Tensor] = []

    for i in range(seq_len):
        if not local_only:
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]

        # Decide whether to wipe h at this position.
        # residual-restore: wipe every window token (including the restore site).
        # h-restore: wipe only positions before restore_at so later tokens evolve normally.
        if residual_restore:
            wipe_here = zero_h and i in times
        elif has_h_donor:
            wipe_here = zero_h and i in times and i < restore_at
        else:
            wipe_here = zero_h and i in times

        if wipe_here:
            if chans:
                ssm_state = ssm_state.clone()
                ssm_state[:, :, chans] = 0
            else:
                ssm_state = torch.zeros_like(ssm_state)

        if has_h_donor and i == restore_at:
            donor = state.donor_h[layer_idx]
            ssm_state = donor.to(device=ssm_state.device, dtype=ssm_state.dtype)
            if ssm_state.dim() == 2:
                ssm_state = ssm_state.unsqueeze(0)

        if i == write_end:
            h_at_write_end = ssm_state.detach()
        if state.capture:
            h_frames.append(ssm_state.detach()[0].contiguous().cpu())

        if local_only:
            scan_outputs.append(
                torch.zeros(batch_size, self.intermediate_size,
                            device=hidden_states.device, dtype=dtype)
            )
        else:
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))
            scan_outputs.append(scan_output[:, :, 0])

    scan_output = torch.stack(scan_outputs, dim=-1)
    scan_output = scan_output + (hidden_states * self.D[None, :, None])
    scan_output = scan_output * self.act(gate)

    if cache_params is not None:
        cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

    contextualized_states = self.out_proj(scan_output.transpose(1, 2))

    if state.capture and layer_idx in state.layer_set:
        state.mixer_out[layer_idx] = contextualized_states.detach()[0].contiguous().cpu()
        if h_at_write_end is not None:
            state.h_write_end[layer_idx] = h_at_write_end.detach()[0].contiguous().cpu()
        if h_frames:
            state.h_seq[layer_idx] = torch.stack(h_frames, dim=0)

    return contextualized_states


def _block_forward_with_residual(
    block: nn.Module,
    hidden_states: Tensor,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
    *,
    layer_idx: int,
    state: PatchState,
):
    """Block forward that handles residual-stream edits and captures the skip vector.

    The mixer still runs on the (already wiped) h, so the C3 residual-restore
    control is: wipe h during write, then paste this example's clean skip vector
    back at restore_at. The mixer output is unchanged; only the residual path differs.
    """
    residual = hidden_states
    spec = state.spec
    times = _times_mask(state.times, residual.shape[1])
    seq_len = residual.shape[1]
    write_end  = max(times) if times else max(0, seq_len - 2)
    restore_at = state.restore_at if state.restore_at is not None else write_end

    normed = block.norm(hidden_states.to(dtype=block.norm.weight.dtype))
    if block.residual_in_fp32:
        residual = residual.to(torch.float32)
    mixer_out = block.mixer(
        normed,
        cache_params=cache_params,
        cache_position=cache_position,
        attention_mask=attention_mask,
    )

    # Residual restore: paste the clean skip vector at restore_at so the LM head
    # sees the same residual as a clean run. The mixer already ran on wiped h.
    if (
        spec is not None
        and spec.target == "residual"
        and spec.op == "restore"
        and restore_at is not None
        and 0 <= restore_at < seq_len
    ):
        donor = state.donor_residual.get(layer_idx)
        if donor is not None:
            residual = residual.clone()
            vec = donor.to(device=residual.device, dtype=residual.dtype)
            residual[:, restore_at, :] = vec if vec.dim() == 2 else vec.unsqueeze(0)

    # Residual noise: add a random vector at each window position, L2-budgeted to
    # match the mixer-output change from zero_h_write on the same example.
    if spec is not None and spec.op == "noise" and times:
        residual = residual.clone()
        mag_map = state.delta_mag.get(layer_idx)
        d = residual.shape[-1]
        g = torch.Generator(device="cpu")
        g.manual_seed(state.noise_seed + 1009 * layer_idx)
        for t in times:
            mag = 0.0
            if mag_map is not None and t < mag_map.numel():
                mag = float(mag_map[t].item())
            if mag <= 0.0:
                continue
            noise = torch.randn(d, generator=g, dtype=torch.float32)
            nrm = float(noise.norm().clamp_min(1e-8).item())
            residual[:, t, :] = residual[:, t, :] + (noise * (mag / nrm)).to(
                device=residual.device, dtype=residual.dtype
            )

    if state.capture and layer_idx in state.layer_set:
        # Capture the skip stream (before the mixer add) for the store-vs-readout probe.
        skip = hidden_states.detach()[0].contiguous().cpu()
        state.residual_seq[layer_idx] = skip.float() if skip.dtype != torch.float32 else skip

    return residual + mixer_out


@contextmanager
def apply_patches(
    model: nn.Module,
    state: PatchState,
    layers: Sequence[int],
) -> Iterator[PatchState]:
    """Patch selected mixers (and the residual path) for one causal condition.

    Mutate ``state`` fields per example inside the context; ``spec`` and
    ``layer_set`` stay fixed for the whole condition.
    """
    backbone = model.backbone if hasattr(model, "backbone") else model
    blocks = backbone.layers
    want = {int(i) for i in layers}
    state.layer_set = want
    orig_slow: Dict[int, object] = {}
    orig_block: Dict[int, object] = {}

    for layer_idx, block in enumerate(blocks):
        if layer_idx not in want:
            continue
        mixer = block.mixer
        orig_slow[layer_idx] = mixer.slow_forward
        orig_block[layer_idx] = block.forward

        def make_slow(m=mixer, idx=layer_idx):
            def slow_forward(input_states, cache_params=None, cache_position=None, attention_mask=None):
                return _instrumented_slow_forward(
                    m, input_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                    layer_idx=idx,
                    state=state,
                )
            return slow_forward

        mixer.slow_forward = make_slow()

        def make_block(b=block, idx=layer_idx):
            def forward(hidden_states, cache_params=None, cache_position=None, attention_mask=None):
                # Use the wrapped block only when we need residual edits or capture.
                spec = state.spec
                use_wrapped = state.capture or (spec is not None and spec.target == "residual")
                if use_wrapped:
                    return _block_forward_with_residual(
                        b, hidden_states,
                        cache_params=cache_params,
                        cache_position=cache_position,
                        attention_mask=attention_mask,
                        layer_idx=idx,
                        state=state,
                    )
                return orig_block[idx](
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )
            return forward

        block.forward = make_block()

    try:
        yield state
    finally:
        for layer_idx, block in enumerate(blocks):
            if layer_idx in orig_slow:
                block.mixer.slow_forward = orig_slow[layer_idx]
            if layer_idx in orig_block:
                block.forward = orig_block[layer_idx]
