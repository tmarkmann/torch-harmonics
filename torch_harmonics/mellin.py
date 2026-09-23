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
from torch_harmonics.quadrature import QuadratureS2, precompute_radii
from torch_harmonics.sht import InverseRealSHT, RealSHT
from torch_harmonics.truncation import truncate_sht


# real dtypes with a complex counterpart; bfloat16 deliberately absent
_COMPLEX_FOR_REAL = {torch.float16: torch.complex32, torch.float32: torch.complex64, torch.float64: torch.complex128}


def _keep_complex(fn):
    """
    Wrap an ``_apply`` function so complex tensors land on a complex dtype.

    ``Module.to(dtype=<real dtype>)`` casts a complex tensor to that real dtype, discarding
    the imaginary part. For the Mellin log-origin phase that is silent: ``forward`` restores
    complexity with ``weights.to(x.dtype)``, so the unit-modulus phase simply reappears as
    ``cos(omega * x0)`` and colours the spectrum. For the spectral weights it is loud, and the
    contraction fails on a dtype mismatch. Requesting a real dtype is read here as a request
    for the matching precision, so complex128 becomes complex64 under ``.to(torch.float32)``.
    A real dtype with no complex counterpart leaves the dtype alone rather than corrupting it.
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
    wmax : int, optional
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

    def __init__(self, nr, r_min, r_max, wmax=None, domain="half-line", R=None, dim=-1, npad=0):

        super().__init__()

        self.nr = nr
        self.domain = domain
        self.R = R
        self.dim = dim
        self.npad = npad

        # geometric radial grid
        x, r, w = precompute_radii(nr, r_min, r_max, domain=domain, R=R, periodic=True)

        # log-grid spacing dx = dr/r
        self.h = (x[-1] - x[0]).item() / (nr - 1)
        self.ntot = nr + npad
        self.length = self.ntot * self.h
        self.wmax = self.ntot // 2 + 1 if wmax is None else wmax

        # Mellin frequencies representable on a periodic log box of length L
        omega = 2.0 * math.pi * torch.arange(self.wmax, dtype=torch.float64) / self.length
        # Quadrature weights
        weights = self.h * torch.exp(torch.complex(torch.zeros_like(omega), -omega * x[0].item()))

        self.register_buffer("x", x, persistent=False)
        self.register_buffer("r", r, persistent=False)
        self.register_buffer("w", w, persistent=False)
        self.register_buffer("omega", omega, persistent=False)
        self.register_buffer("weights", weights, persistent=False)

    def extra_repr(self):
        return f"nr={self.nr}, wmax={self.wmax},\n domain={self.domain}, R={self.R},\n dim={self.dim}, npad={self.npad}"

    def _apply(self, fn, *args, **kwargs):
        return super()._apply(_keep_complex(fn), *args, **kwargs)

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

        torch._check(-x.dim() <= self.dim < x.dim(), lambda: f"Expected tensor with a dim={self.dim} axis but got {x.dim()} dimensions instead")
        torch._check(x.shape[self.dim] == self.nr, lambda: f"Expected radius shape[{self.dim}]=={self.nr}, got {x.shape[self.dim]}")

        # transpose to put the radial dim on the fast axis
        x = x.movedim(self.dim, -1)

        # the log grid needs to be periodic for FFT
        x = _pad_dim_right(x, -1, self.ntot)

        # apply real fft
        x = rfft(x, nmodes=self.wmax, dim=-1, norm="backward")

        # geometric quadrature and shift
        return (self.weights.to(x.dtype) * x).movedim(-1, self.dim)


class InverseRealMellinTransform(nn.Module):
    r"""
    Defines a module for computing the inverse (real-valued) Mellin transform.

    Parameters
    ----------
    Identical to :class:`RealMellinTransform`.
    """

    def __init__(self, nr, r_min, r_max, wmax=None, domain="half-line", R=None, dim=-1, npad=0):

        super().__init__()

        self.nr = nr
        self.domain = domain
        self.R = R
        self.dim = dim
        self.npad = npad

        x, r, w = precompute_radii(nr, r_min, r_max, domain=domain, R=R, periodic=True)

        self.h = (x[-1] - x[0]).item() / (nr - 1)
        self.ntot = nr + npad
        self.length = self.ntot * self.h
        self.wmax = self.ntot // 2 + 1 if wmax is None else wmax

        # reciprocal of the forward weights
        omega = 2.0 * math.pi * torch.arange(self.wmax, dtype=torch.float64) / self.length
        weights = torch.exp(torch.complex(torch.zeros_like(omega), omega * x[0].item())) / self.length

        self.register_buffer("x", x, persistent=False)
        self.register_buffer("r", r, persistent=False)
        self.register_buffer("w", w, persistent=False)
        self.register_buffer("omega", omega, persistent=False)
        self.register_buffer("weights", weights, persistent=False)

    def extra_repr(self):
        return f"nr={self.nr}, wmax={self.wmax},\n domain={self.domain}, R={self.R},\n dim={self.dim}, npad={self.npad}"

    def _apply(self, fn, *args, **kwargs):
        return super()._apply(_keep_complex(fn), *args, **kwargs)

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

        torch._check(-x.dim() <= self.dim < x.dim(), lambda: f"Expected tensor with a dim={self.dim} axis but got {x.dim()} dimensions instead")
        torch._check(x.shape[self.dim] == self.wmax, lambda: f"Expected modes shape[{self.dim}]=={self.wmax}, got {x.shape[self.dim]}")

        # transpose to put the radial dim on the fast axis
        x = x.movedim(self.dim, -1)

        # apply inverse FFT
        x = irfft(x * self.weights.to(x.dtype), n=self.ntot, dim=-1, norm="forward")

        # drop the padded tail
        return x.narrow(-1, 0, self.nr).movedim(-1, self.dim)


def _wrapped_modes(wmax: int, ntot: int) -> torch.Tensor:
    """Signed mode numbers retained by the complex transform, in torch.fft order."""

    freqs = torch.cat([torch.arange((ntot + 1) // 2), torch.arange(-(ntot // 2), 0)])
    if 2 * wmax - 1 >= ntot:
        return freqs
    return torch.cat([freqs[:wmax], freqs[ntot - wmax + 1 :]])


class MellinTransform(nn.Module):
    r"""
    Defines a module for computing the forward (complex-valued) Mellin transform.
    Precomputes the geometric radial grid, the Mellin frequencies and the origin phase.

    Parameters
    ----------
    nr : int
        Number of radial points
    r_min, r_max : float
        Bounds on r, or on rho / R = (r - R) / R for the exterior domain
    wmax : int, optional
        Largest retained ``|w|`` plus one, by default the full bandwidth ``wtot // 2 + 1``
    domain : str, optional
        Either ``"half-line"`` or ``"exterior"``, by default ``"half-line"``
    R : float, optional
        Inner radius, required for ``domain="exterior"``, by default None
    dim : int, optional
        Radial axis, by default -1.
    npad : int, optional
        Number of zero-padded nodes appended in log space, by default 0
    """

    def __init__(self, nr, r_min, r_max, wmax=None, domain="half-line", R=None, dim=-1, npad=0):

        super().__init__()

        self.nr = nr
        self.domain = domain
        self.R = R
        self.dim = dim
        self.npad = npad

        # geometric radial grid
        x, r, w = precompute_radii(nr, r_min, r_max, domain=domain, R=R, periodic=True)

        # log-grid spacing dx = dr/r
        self.h = (x[-1] - x[0]).item() / (nr - 1)
        self.ntot = nr + npad
        self.length = self.ntot * self.h
        self.wmax = self.ntot // 2 + 1 if wmax is None else wmax

        # signed Mellin frequencies, in torch.fft order
        omega = 2.0 * math.pi * _wrapped_modes(self.wmax, self.ntot).to(torch.float64) / self.length
        self.nw = omega.shape[0]
        # Quadrature weights
        weights = self.h * torch.exp(torch.complex(torch.zeros_like(omega), -omega * x[0].item()))

        self.register_buffer("x", x, persistent=False)
        self.register_buffer("r", r, persistent=False)
        self.register_buffer("w", w, persistent=False)
        self.register_buffer("omega", omega, persistent=False)
        self.register_buffer("weights", weights, persistent=False)

    def extra_repr(self):
        return f"nr={self.nr}, wmax={self.wmax},\n domain={self.domain}, R={self.R},\n dim={self.dim}, npad={self.npad}"

    def _apply(self, fn, *args, **kwargs):
        return super()._apply(_keep_complex(fn), *args, **kwargs)

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
            Complex Mellin coefficients with ``2 * wmax - 1`` points along ``dim``.
        """

        torch._check(-x.dim() <= self.dim < x.dim(), lambda: f"Expected tensor with a dim={self.dim} axis but got {x.dim()} dimensions instead")
        torch._check(x.shape[self.dim] == self.nr, lambda: f"Expected radius shape[{self.dim}]=={self.nr}, got {x.shape[self.dim]}")

        # transpose to put the radial dim on the fast axis
        x = x.movedim(self.dim, -1)

        # the log grid needs to be periodic for FFT
        x = _pad_dim_right(x, -1, self.ntot)

        # apply complex fft
        x = torch.fft.fft(x, dim=-1, norm="backward")

        # keep the lowest |n|, preserving torch.fft order
        if self.nw < self.ntot:
            x = torch.cat([x.narrow(-1, 0, self.wmax), x.narrow(-1, self.ntot - self.wmax + 1, self.wmax - 1)], dim=-1)

        # geometric quadrature and shift
        return (self.weights.to(x.dtype) * x).movedim(-1, self.dim)


class InverseMellinTransform(nn.Module):
    r"""
    Defines a module for computing the inverse (complex-valued) Mellin transform.

    Parameters
    ----------
    Identical to :class:`MellinTransform`.
    """

    def __init__(self, nr, r_min, r_max, wmax=None, domain="half-line", R=None, dim=-1, npad=0):

        super().__init__()

        self.nr = nr
        self.domain = domain
        self.R = R
        self.dim = dim
        self.npad = npad

        x, r, w = precompute_radii(nr, r_min, r_max, domain=domain, R=R, periodic=True)

        self.h = (x[-1] - x[0]).item() / (nr - 1)
        self.ntot = nr + npad
        self.length = self.ntot * self.h
        self.wmax = self.ntot // 2 + 1 if wmax is None else wmax

        # reciprocal of the forward weights
        omega = 2.0 * math.pi * _wrapped_modes(self.wmax, self.ntot).to(torch.float64) / self.length
        self.nw = omega.shape[0]
        weights = torch.exp(torch.complex(torch.zeros_like(omega), omega * x[0].item())) / self.length

        self.register_buffer("x", x, persistent=False)
        self.register_buffer("r", r, persistent=False)
        self.register_buffer("w", w, persistent=False)
        self.register_buffer("omega", omega, persistent=False)
        self.register_buffer("weights", weights, persistent=False)

    def extra_repr(self):
        return f"nr={self.nr}, wmax={self.wmax},\n domain={self.domain}, R={self.R},\n dim={self.dim}, npad={self.npad}"

    def _apply(self, fn, *args, **kwargs):
        return super()._apply(_keep_complex(fn), *args, **kwargs)

    def forward(self, x: torch.Tensor):
        """
        Compute the inverse (complex) Mellin transform.

        Parameters
        ----------
        x : torch.Tensor
            Complex Mellin coefficients with ``2 * wmax - 1`` points along ``dim``.

        Returns
        -------
        torch.Tensor
            Complex signal with ``nr`` points along ``dim``.
        """

        torch._check(-x.dim() <= self.dim < x.dim(), lambda: f"Expected tensor with a dim={self.dim} axis but got {x.dim()} dimensions instead")
        torch._check(x.shape[self.dim] == self.nw, lambda: f"Expected modes shape[{self.dim}]=={self.nw}, got {x.shape[self.dim]}")

        # transpose to put the radial dim on the fast axis
        x = x.movedim(self.dim, -1)

        # geometric quadrature and shift
        x = x * self.weights.to(x.dtype)

        # zero-fill the discarded middle frequencies
        if self.nw < self.ntot:
            zeros = x.new_zeros(*x.shape[:-1], self.ntot - self.nw)
            x = torch.cat([x.narrow(-1, 0, self.wmax), zeros, x.narrow(-1, self.wmax, self.wmax - 1)], dim=-1)

        # apply the inverse FFT
        x = torch.fft.ifft(x, n=self.ntot, dim=-1, norm="forward")

        # drop the padded tail
        return x.narrow(-1, 0, self.nr).movedim(-1, self.dim)


class SpectralConvRadialS2(nn.Module):
    r"""
    Spectral convolution layer on :math:`(0, \infty) \times S^2` or
    :math:`[R, \infty) \times S^2`, combining the spherical harmonic transform on the
    angular axes with the Mellin transform on the radial axis.

    Parameters
    ----------
    in_shape : Tuple[int]
        Input grid shape ``(nr, nlat, nlon)``.
    out_shape : Tuple[int]
        Output grid shape ``(nr, nlat, nlon)``.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    r_min, r_max : float
        Bounds on r, or on ``rho / R = (r - R) / R`` for the exterior domain.
    num_groups : int, optional
        Number of channel groups for grouped spectral weights, by default 1.
    grid_in, grid_out : str, optional
        Angular quadrature grids for the forward and inverse SHT, by default
        ``"equiangular"``.
    domain : str, optional
        Either ``"half-line"`` or ``"exterior"`` for the radial dimension, by default ``"half-line"``.
    R : float, optional
        Inner radius, required for ``domain="exterior"``, by default None.
    lmax, mmax, wmax : int, optional
        Angular and radial truncation, by default inferred from the grids via
        :func:`torch_harmonics.truncate_sht` and from ``nr``.
    npad : int, optional
        Zero-padded radial nodes in log space, by default 0.
    bias : bool, optional
        If ``True``, adds a learnable spectral bias scaled by the volume integral, by
        default ``False``.
    """

    def __init__(
        self,
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        in_channels: int,
        out_channels: int,
        r_min: float,
        r_max: float,
        num_groups: Optional[int] = 1,
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        domain: Optional[str] = "half-line",
        R: Optional[float] = None,
        lmax: Optional[int] = None,
        mmax: Optional[int] = None,
        wmax: Optional[int] = None,
        npad: Optional[int] = 0,
        bias: Optional[bool] = False,
    ):
        super().__init__()

        if in_channels % num_groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by num_groups ({num_groups})")
        if out_channels % num_groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by num_groups ({num_groups})")

        # copy inputs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups

        nr_in, nlat_in, nlon_in = in_shape
        nr_out, nlat_out, nlon_out = out_shape

        # compute the angular truncation
        lmax_in, mmax_in = truncate_sht(nlat_in, nlon_in, lmax, mmax, grid=grid_in)
        lmax_out, mmax_out = truncate_sht(nlat_out, nlon_out, lmax, mmax, grid=grid_out)
        self.lmax = min(lmax_in, lmax_out, mmax_in, mmax_out)
        self.mmax = self.lmax

        # angular transforms
        self.sht = RealSHT(nlat_in, nlon_in, lmax=self.lmax, mmax=self.mmax, grid=grid_in)
        self.isht = InverseRealSHT(nlat_out, nlon_out, lmax=self.lmax, mmax=self.mmax, grid=grid_out)

        # compute the radial truncation; the largest bandwidth both grids can tile when they differ
        if wmax is None:
            wmax = (nr_in + npad) // 2 + 1 if nr_in == nr_out else min(nr_in + npad, nr_out + npad) // 2

        # radial transforms; the SHT owns (-2, -1), so the radial axis sits at -3
        self.mellin = MellinTransform(nr_in, r_min=r_min, r_max=r_max, wmax=wmax, domain=domain, R=R, dim=-3, npad=npad)
        self.imellin = InverseMellinTransform(nr_out, r_min=r_min, r_max=r_max, wmax=wmax, domain=domain, R=R, dim=-3, npad=npad)
        self.wmax = wmax
        self.nw = self.mellin.nw

        # weight shape, diagonal in (omega, l) and shared over m
        weight_shape = [num_groups, in_channels // num_groups, out_channels // num_groups, self.nw, self.lmax]

        # Compute scaling factor for correct initialization
        scale = math.sqrt(1.0 / (in_channels // num_groups)) * torch.ones(self.lmax, dtype=torch.complex64)
        # seemingly the first weight is not really complex, so we need to account for that
        scale[0] *= math.sqrt(2.0)
        self.weight = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.complex64))

        if bias:
            self.spectral_bias = nn.Parameter(torch.zeros(1, self.in_channels, self.nw, self.lmax, self.mmax, dtype=torch.complex64))
            self.quadrature = QuadratureS2(img_shape=(nlat_in, nlon_in), grid=grid_in, normalize=False)
            # radial half of the volume element, normalized so the bias stays trainable and scale covariant
            radial_weight = self.mellin.w * self.mellin.r**2
            self.register_buffer("radial_weight", (radial_weight / radial_weight.sum()).to(torch.float32), persistent=False)

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels},\n lmax={self.lmax}, mmax={self.mmax}, nw={self.nw},\n num_groups={self.num_groups}"

    def _apply(self, fn, *args, **kwargs):
        return super()._apply(_keep_complex(fn), *args, **kwargs)

    @torch.compile
    def _contract_diagonal(self, ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
        resc = torch.einsum("bgiwlm,giowl->bgowlm", ac, bc)
        return resc

    def forward(self, x):
        """
        Apply the joint angular-radial spectral convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape ``(batch, in_channels, nr_in, nlat_in, nlon_in)``.

        Returns
        -------
        torch.Tensor
            Convolved signal of shape ``(batch, out_channels, nr_out, nlat_out, nlon_out)``.
        """
        dtype = x.dtype

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.to(torch.float32)

            # compute the volume integral in case the bias is used
            if hasattr(self, "spectral_bias"):
                integral = torch.einsum("bcr,r->bc", self.quadrature(x), self.radial_weight)

            # transform the angular axes, then the radial one
            x = self.mellin(self.sht(x)).contiguous()

        # store the shapes
        B, C, W, L, M = x.shape

        # deal with bias
        if hasattr(self, "spectral_bias"):
            x = x + integral.reshape(B, C, 1, 1, 1) * self.spectral_bias

        # perform contraction
        x = x.reshape(B, self.num_groups, C // self.num_groups, W, L, M)
        xp = self._contract_diagonal(x, self.weight)
        x = xp.reshape(B, self.out_channels, W, L, M).contiguous()

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = self.isht(self.imellin(x))

        # convert datatype
        x = x.to(dtype=dtype)

        return x
