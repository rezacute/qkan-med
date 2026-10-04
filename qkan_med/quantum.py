"""Exact batched state-vector simulation with first-order adjoint gradients."""
import math
import torch
from torch import nn
from torch.autograd.function import once_differentiable


def rotate(state, theta, wire, kind):
    blocks = state.reshape(state.shape[0], 2**wire, 2, -1)
    a, b = blocks[:, :, 0], blocks[:, :, 1]
    theta = theta.reshape(-1, 1, 1) / 2
    if kind == "RY":
        c, s = theta.cos(), theta.sin()
        result = torch.stack((c * a - s * b, s * a + c * b), 2)
    else:
        phase = torch.polar(torch.ones_like(theta), theta)
        result = torch.stack((phase.conj() * a, phase * b), 2)
    return result.reshape_as(state)


def pauli(state, wire, kind):
    blocks = state.reshape(state.shape[0], 2**wire, 2, -1)
    a, b = blocks[:, :, 0], blocks[:, :, 1]
    return torch.stack((-1j * b, 1j * a) if kind == "RY" else (a, -b), 2).reshape_as(state)


def operations(n):
    ops = [("RY", wire, -1, 0) for wire in range(n)]
    for layer in range(2):
        for wire in range(n):
            ops.extend((kind, wire, layer, axis)
                       for axis, kind in enumerate(("RZ", "RY", "RZ")))
        edges = ([(i, i + 1) for i in range(n - 1)] if layer == 0 else
                 [(i, i + 1) for parity in (0, 1) for i in range(parity, n - 1, 2)])
        ops.extend(("CX", control, target, 0) for control, target in edges)
    return ops


class StateVectorAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, angles, signs, permutations, ops):
        dtype = torch.complex128 if inputs.dtype == torch.float64 else torch.complex64
        state = torch.zeros((len(inputs), signs.shape[1]), device=inputs.device, dtype=dtype)
        state[:, 0] = 1
        for kind, wire, layer, axis in ops:
            if kind == "CX":
                state = state.index_select(1, permutations[(wire, layer)])
            else:
                theta = inputs[:, wire] * math.pi if layer == -1 else angles[layer, wire, axis]
                state = rotate(state, theta, wire, kind)
        ctx.save_for_backward(inputs, angles, state, signs)
        ctx.permutations, ctx.ops = permutations, ops
        return (state.real.square() + state.imag.square()) @ signs.T

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        inputs, angles, state, signs = ctx.saved_tensors
        # dL = 2 Re <adjoint | d(state)> for real weighted Z expectations.
        adjoint = (grad_output @ signs) * state
        dx, dw = torch.zeros_like(inputs), torch.zeros_like(angles)
        for kind, wire, layer, axis in reversed(ctx.ops):
            if kind == "CX":
                permutation = ctx.permutations[(wire, layer)]
                state, adjoint = state.index_select(1, permutation), adjoint.index_select(1, permutation)
            else:
                derivative = (adjoint.conj() * pauli(state, wire, kind)).sum(1).imag
                if layer == -1:
                    dx[:, wire] = derivative * math.pi
                    theta = inputs[:, wire] * math.pi
                else:
                    dw[layer, wire, axis] = derivative.sum()
                    theta = angles[layer, wire, axis]
                state = rotate(state, -theta, wire, kind)
                adjoint = rotate(adjoint, -theta, wire, kind)
        return dx, dw, None, None, None


class QuantumKAN(nn.Module):
    def __init__(self):
        n, precision = 16, "single"
        super().__init__()
        self.n, self.precision = n, precision
        self.angles = nn.Parameter(0.1 * torch.randn(2, n, 3))
        self.ops = operations(n)
        self._cache = None

    def forward(self, x):
        dtype = torch.float64 if self.precision == "double" else torch.float32
        key = (x.device, dtype)
        if self._cache is None or self._cache[0] != key:
            indices = torch.arange(2**self.n, device=x.device)
            signs = torch.stack([1 - 2 * ((indices >> (self.n - wire - 1)) & 1)
                                 for wire in range(self.n)]).to(dtype)
            permutations = {(wire, target): indices ^ (((indices >> (self.n - wire - 1)) & 1)
                            << (self.n - target - 1)) for kind, wire, target, _ in self.ops if kind == "CX"}
            self._cache = (key, signs, permutations)
        _, signs, permutations = self._cache
        # Disable autocast here: quantum arithmetic must retain chosen precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            result = StateVectorAdjoint.apply(x.to(dtype), self.angles.to(dtype), signs, permutations, self.ops)
        return result.to(x.dtype)
