# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def emulate_int(w, bits, method, scale=None, zero_point=None):
    q = globals()[f"emulate_int{bits}_{method}"]
    return q(w, scale=scale, zero_point=zero_point)


def quantize(w, scale, zero_point):
    return (torch.clamp(torch.round(w / scale + zero_point), 0, 255) - zero_point) * scale


def emulate_int8_histogram(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.HistogramObserver()
        _ = obs(w.float())
        scale, zero_point = obs.calculate_qparams()
        device = w.device
        scale = scale.to(device).type_as(w)
        zero_point = zero_point.to(device).type_as(w)
    return quantize(w, scale, zero_point), scale, zero_point


def emulate_int8_channel(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.PerChannelMinMaxObserver(
            ch_axis=-1, qscheme=torch.per_channel_symmetric
        )
        _ = obs(w)
        scale, zero_point, ch_axis = obs.get_qparams()
        device = w.device
        scale = scale.to(device).type_as(w)
        zero_point = zero_point.to(device).type_as(w)
    return quantize(w, scale, zero_point), scale, zero_point


def emulate_int8_tensor(w, scale=None, zero_point=None):
    if scale is None:
        obs = torch.quantization.observer.MinMaxObserver()
        _ = obs(w)
        scale, zero_point = obs.calculate_qparams()
        device = w.device
        scale = scale.to(device).type_as(w)
        zero_point = zero_point.to(device).type_as(w)
    return quantize(w, scale, zero_point), scale, zero_point


# -------------------------------
# Generic helpers for n-bit quantization (used for 4-bit support)
# -------------------------------


def _calc_scale_zero_point(w: torch.Tensor, bits: int):
    """Calculate scale and zero_point for given tensor *w* and bit-width."""
    w_min = w.min()
    w_max = w.max()

    # Handle edge case where all values are the same
    if (w_max - w_min).abs() < 1e-8:
        scale = torch.tensor(1.0, device=w.device, dtype=w.dtype)
        zero_point = torch.tensor(0.0, device=w.device, dtype=w.dtype)
    else:
        scale = (w_max - w_min) / (2 ** bits - 1)
        zero_point = torch.round(-w_min / scale)
        zero_point.clamp_(0, 2 ** bits - 1)

    return scale, zero_point


def _quantize_nbits(w: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, bits: int):
    """Quantize tensor *w* to *bits* and dequantize back to float."""
    q_w = torch.clamp(torch.round(w / scale + zero_point), 0, 2 ** bits - 1)
    return (q_w - zero_point) * scale


# -------------------------------
# 4-bit (INT4) tensor quantization
# -------------------------------


def emulate_int4_tensor(w, scale=None, zero_point=None):
    """Per-tensor 4-bit quantization using simple min-max scaling."""
    if scale is None or zero_point is None:
        scale, zero_point = _calc_scale_zero_point(w.float(), bits=4)
        scale = scale.to(device=w.device).type_as(w)
        zero_point = zero_point.to(device=w.device).type_as(w)

    return _quantize_nbits(w, scale, zero_point, bits=4), scale, zero_point


# Histogram and per-channel observers are not available for 4-bit in the
# standard PyTorch quantization API. For now, fall back to the simple tensor
# method so that the rest of the code works seamlessly.


def emulate_int4_histogram(w, scale=None, zero_point=None):
    return emulate_int4_tensor(w, scale=scale, zero_point=zero_point)


def emulate_int4_channel(w, scale=None, zero_point=None):
    # For channels, compute scale/zero_point per last dimension channel
    if scale is None or zero_point is None:
        # Compute per-channel min/max
        w_min = w.min(dim=-1, keepdim=True)[0]
        w_max = w.max(dim=-1, keepdim=True)[0]
        # Avoid degenerate ranges
        scale = (w_max - w_min) / (2 ** 4 - 1)
        scale[scale == 0] = 1.0
        zero_point = torch.round(-w_min / scale)
        zero_point = zero_point.clamp(0, 2 ** 4 - 1)
        scale = scale.to(device=w.device).type_as(w)
        zero_point = zero_point.to(device=w.device).type_as(w)

    return _quantize_nbits(w, scale, zero_point, bits=4), scale, zero_point
