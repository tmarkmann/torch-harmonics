# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2026 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


import math

import torch
import torch.nn as nn

from torch_harmonics.fft import _pad_dim_right, irfft, rfft
from torch_harmonics.quadrature import precompute_radii
from torch_harmonics.utils import check

# real dtypes with a complex counterpart; bfloat16 deliberately absent
_COMPLEX_FOR_REAL = {torch.float16: torch.complex32, torch.float32: torch.complex64, torch.float64: torch.complex128}


def _keep_complex(fn):
    """
    Wrap an ``_apply`` function so complex tensors land on a complex dtype.

    ``Module.to(dtype=<real dtype>)`` casts a complex tensor to that real dtype, discarding
    the imaginary part. For complex spectral weights this makes the contraction fail on a
    dtype mismatch. Requesting a real dtype is read here as a request for the matching
    precision, so complex128 becomes complex64 under ``.to(torch.float32)``. A real dtype
    with no complex counterpart leaves the dtype alone rather than corrupting it.
    """

    def apply(t):
        if not t.is_complex():
            return fn(t)

        # a zero-element probe reveals the target device and dtype without a full copy
        probe = fn(t.new_empty(0))
        if probe.is_complex():
            return fn(t)

        target = _COMPLEX_FOR_REAL.get(probe.dtype)
        if target is None:
            return t.to(device=probe.device)

        return t.to(device=probe.device, dtype=target)

    return apply


class _MellinBase(nn.Module):
    r"""
    Shared setup of the Mellin transforms: geometric radial grid, Mellin frequencies and
    origin phase.

    Parameters
    ----------
    nr : int
        Number of radial points
    rmin, rmax : float
        Bounds on r, or on rho = (r - inner_radius) / inner_radius for the exterior domain
    wmax : int, optional
        Number of retained non-negative Mellin modes, by default the full bandwidth ``ntot // 2 + 1``
    domain : str, optional
        Either ``"half-line"`` or ``"exterior"``, by default ``"half-line"``
    inner_radius : float, optional
        Inner radius, required for ``domain="exterior"``, by default None
    dim : int, optional
        Radial axis, by default -1
    npad : int, optional
        Number of zero-padded nodes appended in log space, by default 0
    c : float, optional
        Real part of the Mellin variable s = c - i omega. By default 0
    """

    def __init__(self, nr, rmin, rmax, wmax=None, domain="half-line", inner_radius=None, dim=-1, npad=0, c=0.0):

        super().__init__()

        self.nr = nr
        self.domain = domain
        self.inner_radius = inner_radius
        self.dim = dim
        self.npad = npad
        self.c = c

        # geometric radial grid
        x, r, w = precompute_radii(nr, rmin, rmax, domain=domain, inner_radius=inner_radius)

        # log-grid spacing dx = dr/r
        self.dx = (x[-1] - x[0]).item() / (nr - 1)
        self.ntot = nr + npad
        self.length = self.ntot * self.dx
        self.wmax = self.ntot // 2 + 1 if wmax is None else wmax

        # Mellin frequencies keeping |k| < wmax
        modes = self._modes()
        index = torch.nonzero(modes.abs() < self.wmax).squeeze(-1)
        omega = 2.0 * math.pi * modes[index].to(torch.float64) / self.length
        self.nw = omega.shape[0]

        # precompute origin phase angle omega * x0 to stay accurate in single precision
        origin_angle = torch.remainder(omega * x[0], 2.0 * math.pi)

        # Mellin weight r^c in exponential form e^{c x}
        rc = torch.exp(c * x)

        self.register_buffer("x", x, persistent=False)
        self.register_buffer("r", r, persistent=False)
        self.register_buffer("w", w, persistent=False)
        self.register_buffer("omega", omega, persistent=False)
        self.register_buffer("origin_angle", origin_angle, persistent=False)
        self.register_buffer("index", index, persistent=False)
        self.register_buffer("rc", rc, persistent=False)

    def _modes(self) -> torch.Tensor:
        """Integer mode numbers k of the full FFT in torch.fft order."""
        raise NotImplementedError

    def _origin_phase(self, sign: float, dx: float, dtype: torch.dtype) -> torch.Tensor:
        """``dx * exp(sign * i * omega * x0)`` as a complex tensor of the given dtype."""

        angle = sign * self.origin_angle
        return torch.complex(dx * torch.cos(angle), dx * torch.sin(angle)).to(dtype)

    def extra_repr(self):
        return f"nr={self.nr}, wmax={self.wmax}, c={self.c},\n domain={self.domain}, inner_radius={self.inner_radius},\n dim={self.dim}, npad={self.npad}"

    def _check_dim(self, x: torch.Tensor, size: int, what: str):
        check(-x.dim() <= self.dim < x.dim(), lambda: f"Expected tensor with a dim={self.dim} axis but got {x.dim()} dimensions instead")
        check(x.shape[self.dim] == size, lambda: f"Expected {what} shape[{self.dim}]=={size}, got {x.shape[self.dim]}")


class RealMellinTransform(_MellinBase):
    r"""
    Defines a module for computing the forward (real-valued) Mellin transform.
    Only the non-negative frequencies are returned, since the input is real.

    Parameters
    ----------
    Identical to :class:`_MellinBase`.
    """

    def _modes(self):
        return torch.round(torch.fft.rfftfreq(self.ntot, dtype=torch.float64) * self.ntot).long()

    def forward(self, x: torch.Tensor):
        """
        Compute the forward (real) Mellin transform.

        Parameters
        ----------
        x : torch.Tensor
            Real-valued signal with ``nr`` points along ``dim``.

        Returns
        -------
        torch.Tensor
            Complex Mellin coefficients with ``wmax`` points along ``dim``.
        """

        self._check_dim(x, self.nr, "radius")

        # transpose to put the radial dim on the fast axis
        x = x.movedim(self.dim, -1)

        # Mellin weight r^c
        x = x * self.rc.to(x.real.dtype)

        # zero-pad the log box so that the wrap-around of the periodic FFT does not couple both ends
        x = _pad_dim_right(x, -1, self.ntot)

        x = rfft(x, nmodes=self.wmax, dim=-1, norm="backward")

        # geometric quadrature and origin phase
        return (self._origin_phase(-1.0, self.dx, x.dtype) * x).movedim(-1, self.dim)


class InverseRealMellinTransform(_MellinBase):
    r"""
    Defines a module for computing the inverse (real-valued) Mellin transform.

    Parameters
    ----------
    Identical to :class:`RealMellinTransform`.
    """

    def _modes(self):
        return torch.round(torch.fft.rfftfreq(self.ntot, dtype=torch.float64) * self.ntot).long()

    def forward(self, x: torch.Tensor):
        """
        Compute the inverse (real) Mellin transform.

        Parameters
        ----------
        x : torch.Tensor
            Complex Mellin coefficients with ``wmax`` points along ``dim``.

        Returns
        -------
        torch.Tensor
            Real-valued signal with ``nr`` points along ``dim``.
        """

        self._check_dim(x, self.wmax, "modes")

        # transpose to put the radial dim on the fast axis
        x = x.movedim(self.dim, -1)

        # apply inverse FFT
        x = irfft(x * self._origin_phase(1.0, 1.0 / self.length, x.dtype), n=self.ntot, dim=-1, norm="forward")

        # drop the padded tail and undo the Mellin weight
        x = x.narrow(-1, 0, self.nr) / self.rc.to(x.real.dtype)

        return x.movedim(-1, self.dim)


class MellinTransform(_MellinBase):
    r"""
    Defines a module for computing the forward (complex-valued) Mellin transform.
    Returns the frequencies ``|k| < wmax`` in torch.fft order.

    Parameters
    ----------
    Identical to :class:`_MellinBase`, with ``wmax`` the largest retained ``|k|`` plus one.
    """

    def _modes(self):
        return torch.round(torch.fft.fftfreq(self.ntot, dtype=torch.float64) * self.ntot).long()

    def forward(self, x: torch.Tensor):
        """
        Compute the forward (complex) Mellin transform.

        Parameters
        ----------
        x : torch.Tensor
            Signal with ``nr`` points along ``dim``, real or complex.

        Returns
        -------
        torch.Tensor
            Complex Mellin coefficients with ``nw = min(2 * wmax - 1, ntot)`` points along ``dim``.
        """

        self._check_dim(x, self.nr, "radius")

        # transpose to put the radial dim on the fast axis
        x = x.movedim(self.dim, -1)

        # Mellin weight r^c
        x = x * self.rc.to(x.real.dtype)

        # zero-pad the log box so that the wrap-around of the periodic FFT does not couple both ends
        x = _pad_dim_right(x, -1, self.ntot)

        # apply complex fft
        x = torch.fft.fft(x, dim=-1, norm="backward")

        # keep the lowest |k|, preserving torch.fft order
        if self.nw < self.ntot:
            x = x.index_select(-1, self.index)

        # geometric quadrature and origin phase
        return (self._origin_phase(-1.0, self.dx, x.dtype) * x).movedim(-1, self.dim)


class InverseMellinTransform(_MellinBase):
    r"""
    Defines a module for computing the inverse (complex-valued) Mellin transform.

    Parameters
    ----------
    Identical to :class:`MellinTransform`.
    """

    def _modes(self):
        return torch.round(torch.fft.fftfreq(self.ntot, dtype=torch.float64) * self.ntot).long()

    def forward(self, x: torch.Tensor):
        """
        Compute the inverse (complex) Mellin transform.

        Parameters
        ----------
        x : torch.Tensor
            Complex Mellin coefficients with ``nw`` points along ``dim``.

        Returns
        -------
        torch.Tensor
            Complex signal with ``nr`` points along ``dim``.
        """

        self._check_dim(x, self.nw, "modes")

        # transpose to put the radial dim on the fast axis
        x = x.movedim(self.dim, -1)

        # geometric quadrature and origin phase
        x = x * self._origin_phase(1.0, 1.0 / self.length, x.dtype)

        # zero-fill the discarded middle frequencies
        if self.nw < self.ntot:
            x = x.new_zeros(*x.shape[:-1], self.ntot).index_copy(-1, self.index, x)

        # apply the inverse FFT
        x = torch.fft.ifft(x, n=self.ntot, dim=-1, norm="forward")

        # drop the padded tail and undo the Mellin weight
        x = x.narrow(-1, 0, self.nr) / self.rc.to(x.real.dtype)

        return x.movedim(-1, self.dim)
