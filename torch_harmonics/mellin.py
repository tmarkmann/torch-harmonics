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
from typing import Optional, Tuple

import torch
import torch.nn as nn

from torch_harmonics.fft import _pad_dim_right, irfft, rfft
from torch_harmonics.quadrature import geometric_weights


def radial_grid(
    nr: int, vmin: float, vmax: float, domain: str = "half-line", R: Optional[float] = None, periodic: bool = True, dtype: torch.dtype = torch.float64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Radial grid and quadrature weights.

    Parameters
    -----------
    nr : int
        Number of radial nodes
    vmin : float
        Lower bound on r, or on rho / R = (r - R) / R for the exterior domain
    vmax : float
        Upper bound
    domain : str, optional
        Either "half-line" or "exterior", by default "half-line"
    R : float, optional
        Inner radius, required for domain="exterior", by default None
    periodic : bool, optional
        Whether the grid is periodic, by default True
    dtype : torch.dtype, optional
        Floating point type, by default torch.float64

    Returns
    -------
    x : torch.Tensor
        Reduced coordinate of the nodes
    r : torch.Tensor
        Radial nodes
    w : torch.Tensor
        Trapezoidal weights for the integral over dr
    """

    if domain == "half-line":
        r, w = geometric_weights(nr, vmin, vmax, periodic=periodic)
        x = torch.log(r)
    elif domain == "exterior":
        if R is None:
            raise ValueError("R must be given for domain='exterior'")
        rho, wrho = geometric_weights(nr, vmin, vmax, periodic=periodic)
        x, r, w = torch.log(rho), R + R * rho, R * wrho
    else:
        raise ValueError(f"unknown domain: {domain}")

    return x.to(dtype), r.to(dtype), w.to(dtype)


class RealMellinTransform(nn.Module):
    r"""
    Defines a module for computing the forward (real-valued) Mellin transform.
    Precomputes the geometric radial grid, the Mellin frequencies and the origin phase.

    Parameters
    ----------
    nr : int
        Number of radial points
    r_min, r_max : float
        Bounds on r, or on rho / R = (r - R) / R for the exterior domain
    nmodes : int, optional
        Number of retained Mellin modes, by default the full bandwidth ``ntot // 2 + 1``.
    domain : str, optional
        Either ``"half-line"`` or ``"exterior"``, by default ``"half-line"``
    R : float, optional
        Inner radius, required for ``domain="exterior"``, by default None
    dim : int, optional
        Radial axis, by default -1.
    npad : int, optional
        Number of zero-padded nodes appended in log space, by default 0
    """

    def __init__(self, nr, r_min, r_max, nmodes=None, domain="half-line", R=None, dim=-1, npad=0):

        super().__init__()

        self.nr = nr
        self.domain = domain
        self.R = R
        self.dim = dim
        self.npad = npad

        # geometric radial grid
        x, r, w = radial_grid(nr, r_min, r_max, domain=domain, R=R, periodic=True)

        # log-grid spacing dx = dr/r
        self.h = (x[-1] - x[0]).item() / (nr - 1)
        self.ntot = nr + npad
        self.length = self.ntot * self.h
        self.nmodes = self.ntot // 2 + 1 if nmodes is None else nmodes

        # Mellin frequencies representable on a periodic log box of length L
        omega = 2.0 * math.pi * torch.arange(self.nmodes, dtype=torch.float64) / self.length
        # Quadrature weights
        weights = self.h * torch.exp(torch.complex(torch.zeros_like(omega), -omega * x[0].item()))

        self.register_buffer("x", x, persistent=False)
        self.register_buffer("r", r, persistent=False)
        self.register_buffer("w", w, persistent=False)
        self.register_buffer("omega", omega, persistent=False)
        self.register_buffer("weights", weights, persistent=False)

    def extra_repr(self):
        return f"nr={self.nr}, nmodes={self.nmodes},\n domain={self.domain}, R={self.R},\n dim={self.dim}, npad={self.npad}"

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
            Complex Mellin coefficients with ``nmodes`` points along ``dim``.
        """

        torch._check(-x.dim() <= self.dim < x.dim(), lambda: f"Expected tensor with a dim={self.dim} axis but got {x.dim()} dimensions instead")
        torch._check(x.shape[self.dim] == self.nr, lambda: f"Expected radius shape[{self.dim}]=={self.nr}, got {x.shape[self.dim]}")

        # transpose to put the radial dim on the fast axis
        x = x.movedim(self.dim, -1)

        # the log grid needs to be periodic for FFT
        x = _pad_dim_right(x, -1, self.ntot)

        # apply real fft
        x = rfft(x, nmodes=self.nmodes, dim=-1, norm="backward")

        # geometric quadrature and shift
        return (self.weights.to(x.dtype) * x).movedim(-1, self.dim)


class InverseRealMellinTransform(nn.Module):
    r"""
    Defines a module for computing the inverse (real-valued) Mellin transform.

    Parameters
    ----------
    Identical to :class:`RealMellinTransform`.
    """

    def __init__(self, nr, r_min, r_max, nmodes=None, domain="half-line", R=None, dim=-1, npad=0):

        super().__init__()

        self.nr = nr
        self.domain = domain
        self.R = R
        self.dim = dim
        self.npad = npad

        x, r, w = radial_grid(nr, r_min, r_max, domain=domain, R=R, periodic=True)

        self.h = (x[-1] - x[0]).item() / (nr - 1)
        self.ntot = nr + npad
        self.length = self.ntot * self.h
        self.nmodes = self.ntot // 2 + 1 if nmodes is None else nmodes

        # reciprocal of the forward weights
        omega = 2.0 * math.pi * torch.arange(self.nmodes, dtype=torch.float64) / self.length
        weights = torch.exp(torch.complex(torch.zeros_like(omega), omega * x[0].item())) / self.length

        self.register_buffer("x", x, persistent=False)
        self.register_buffer("r", r, persistent=False)
        self.register_buffer("w", w, persistent=False)
        self.register_buffer("omega", omega, persistent=False)
        self.register_buffer("weights", weights, persistent=False)

    def extra_repr(self):
        return f"nr={self.nr}, nmodes={self.nmodes},\n domain={self.domain}, R={self.R},\n dim={self.dim}, npad={self.npad}"

    def forward(self, x: torch.Tensor):
        """
        Compute the inverse (real) Mellin transform.

        Parameters
        ----------
        x : torch.Tensor
            Complex Mellin coefficients with ``nmodes`` points along ``dim``.

        Returns
        -------
        torch.Tensor
            Real-valued signal with ``nr`` points along ``dim``.
        """

        torch._check(-x.dim() <= self.dim < x.dim(), lambda: f"Expected tensor with a dim={self.dim} axis but got {x.dim()} dimensions instead")
        torch._check(x.shape[self.dim] == self.nmodes, lambda: f"Expected modes shape[{self.dim}]=={self.nmodes}, got {x.shape[self.dim]}")

        # transpose to put the radial dim on the fast axis
        x = x.movedim(self.dim, -1)

        # apply inverse FFT
        x = irfft(x * self.weights.to(x.dtype), n=self.ntot, dim=-1, norm="forward")

        # drop the padded tail
        return x.narrow(-1, 0, self.nr).movedim(-1, self.dim)
