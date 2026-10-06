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

from torch_harmonics import InverseRealSHT, RealSHT
from torch_harmonics.mellin import InverseMellinTransform, InverseRealMellinTransform, MellinTransform, RealMellinTransform, _keep_complex
from torch_harmonics.quadrature import QuadratureRadialS2, QuadratureS2
from torch_harmonics.truncation import truncate_sht


class SpectralConvS2(nn.Module):
    r"""
    Spectral convolution layer on :math:`S^2` implemented via real SHT
    (Driscoll--Healy formulation, see https://api.semanticscholar.org/CorpusID:122817218).

    Given a multi-channel input signal :math:`u^{c_i}(\theta, \lambda)` on the
    sphere, the layer computes the output channels :math:`v^{c_o}(\theta, \lambda)`
    in three steps:

    1. **Forward SHT** (cf. :class:`~torch_harmonics.RealSHT`) -- transform
       each input channel to spectral space:

    .. math::

        \hat{u}_l^{m,\,c_i} = \text{SHT}\!\left[\, u^{c_i}(\theta, \lambda) \,\right]

    2. **Spectral contraction** -- mix channels with learnable weights
       :math:`K_l^{c_o,\,c_i}` that are diagonal in :math:`(l, m)` (i.e.\ the
       same weight is applied to every order :math:`m` at a given degree
       :math:`l`):

    .. math::

        \hat{v}_l^{m,\,c_o}
            = \sum_{c_i} K_l^{c_o,\,c_i}\; \hat{u}_l^{m,\,c_i}

    3. **Inverse SHT** (cf. :class:`~torch_harmonics.InverseRealSHT`) --
       transform back to the spatial domain:

    .. math::

        v^{c_o}(\theta, \lambda)
            = \text{ISHT}\!\left[\, \hat{v}_l^{m,\,c_o} \,\right]

    Because the spectral weights depend only on degree :math:`l` and not on order
    :math:`m`, this corresponds to an **isotropic** (azimuthally symmetric)
    convolution kernel on the sphere.  When ``num_groups > 1``, the channel
    contraction is performed independently within each group (grouped
    convolution).

    **Spectral bias.**
    When ``bias=True``, a learnable spectral bias :math:`b_l^{m,\,c_i}` is
    added to the SHT coefficients before the channel contraction.  The bias is
    modulated by the spatial integral (zeroth moment) of each input channel:

    .. math::

        I^{c_i} = \int_0^{2\pi}\!\int_0^{\pi}
            u^{c_i}(\theta,\lambda)\,\sin\theta\;d\theta\;d\lambda

    .. math::

        \hat{u}_l^{m,\,c_i} \;\leftarrow\;
            \hat{u}_l^{m,\,c_i} + I^{c_i}\, b_l^{m,\,c_i}

    This allows the layer to learn a spectral response that depends on the
    global mean of each input channel, effectively coupling the zero-frequency
    content into all spectral modes.

    Parameters
    ----------
    in_shape : Tuple[int]
        Spatial input grid shape ``(nlat, nlon)``.
    out_shape : Tuple[int]
        Spatial output grid shape ``(nlat, nlon)``.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    num_groups : int, optional
        Number of channel groups for grouped spectral weights, by default 1.
    grid_in : str, optional
        Grid used for the forward SHT (``"equiangular"``, ``"legendre-gauss"``,
        ``"lobatto"``, ``"equiangular-trapezoidal"``), by default ``"equiangular"``.
    grid_out : str, optional
        Grid used for the inverse SHT, same options as ``grid_in``.
    bias : bool, optional
        If ``True``, adds a learnable spectral bias computed from the spatial
        integral, by default ``False``.

    Examples
    --------
    >>> import torch
    >>> import torch_harmonics as th
    >>> conv = th.SpectralConvS2(
    ...     in_shape=(128, 256), out_shape=(128, 256),
    ...     in_channels=16, out_channels=32,
    ... )
    >>> x = torch.randn(4, 16, 128, 256)
    >>> y = conv(x)
    >>> y.shape
    torch.Size([4, 32, 128, 256])

    Raises
    ------
    AssertionError
        If ``in_channels`` or ``out_channels`` is not divisible by
        ``num_groups``.

    Notes
    -----
    The SHT truncation ``lmax``/``mmax`` is the minimum of the input and output
    truncations.
    """

    def __init__(
        self,
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        in_channels: int,
        out_channels: int,
        num_groups: Optional[int] = 1,
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
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

        # compute truncation
        lmax_in, mmax_in = truncate_sht(in_shape[0], in_shape[1], grid=grid_in)
        lmax_out, mmax_out = truncate_sht(out_shape[0], out_shape[1], grid=grid_out)

        # compute lmax and lmin
        lmax = min(lmax_in, lmax_out)
        mmax = min(mmax_in, mmax_out)
        self.lmax = min(lmax, mmax)
        self.mmax = self.lmax

        # set up sht layers
        self.sht = RealSHT(*in_shape, grid=grid_in, lmax=self.lmax, mmax=self.mmax)
        self.isht = InverseRealSHT(*out_shape, grid=grid_out, lmax=self.lmax, mmax=self.mmax)

        # weight shape
        weight_shape = [num_groups, in_channels // num_groups, out_channels // num_groups, self.lmax]

        # Compute scaling factor for correct initialization
        scale = math.sqrt(1.0 / (in_channels // num_groups)) * torch.ones(self.lmax, dtype=torch.complex64)
        # seemingly the first weight is not really complex, so we need to account for that
        scale[0] *= math.sqrt(2.0)
        self.weight = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.complex64))

        if bias:
            self.spectral_bias = nn.Parameter(torch.zeros(1, self.in_channels, self.lmax, self.mmax, dtype=torch.complex64))
            self.quadrature = QuadratureS2(img_shape=in_shape, grid=grid_in, normalize=False)

    @torch.compile
    def _contract_lwise(self, ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
        resc = torch.einsum("bgixy,giox->bgoxy", ac, bc)
        return resc

    def forward(self, x):
        """
        Apply the spectral convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape ``(batch, in_channels, nlat_in, nlon_in)``.

        Returns
        -------
        torch.Tensor
            Convolved signal of shape ``(batch, out_channels, nlat_out, nlon_out)``.
        """
        dtype = x.dtype

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.to(torch.float32)

            # compute integral in case if bias is used
            if hasattr(self, "spectral_bias"):
                integral = self.quadrature(x)

            # perform SHT. The result is contiguous by construction, so no .contiguous() here:
            # a copy over a complex buffer cannot be codegen'd by inductor, as triton has no
            # complex type and the kernel signature then fails with KeyError: 'complex64'.
            x = self.sht(x)

        # store the shapes
        B, C, H, W = x.shape

        # deal with bias. `integral` is computed from the real-valued input above, so it is a
        # real tensor and broadcasts over the re/im axis: applying the bias on the real view is
        # exact and keeps this pointwise op off the complex dtype (see above).
        if hasattr(self, "spectral_bias"):
            bias = integral.reshape(B, C, 1, 1, 1) * torch.view_as_real(self.spectral_bias)
            x = torch.view_as_complex(torch.view_as_real(x) + bias)

        # perform contraction
        x = x.reshape(B, self.num_groups, C // self.num_groups, H, W)
        xp = self._contract_lwise(x, self.weight)
        # merge the group axes on the real view: the einsum output can be non-contiguous, in
        # which case this reshape copies, and that copy must not be over a complex buffer
        x = torch.view_as_complex(torch.view_as_real(xp).reshape(B, self.out_channels, H, W, 2).contiguous())

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = self.isht(x)

        # convert datatype
        x = x.to(dtype=dtype)

        return x


class SpectralConvRadialS2(nn.Module):
    r"""
    Spectral convolution layer on :math:`(0, \infty) \times S^2` or
    :math:`[R, \infty) \times S^2` with :math:`R` the inner radius, combining the spherical harmonic transform on the
    angular axes with the Mellin transform on the radial axis. The weights are diagonal in :math:`(\omega, l)` and
    shared across the order :math:`m`.

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
    rmin, rmax : float
        Bounds on r, or on ``rho = (r - inner_radius) / inner_radius`` for the exterior domain.
    num_groups : int, optional
        Number of channel groups for grouped spectral weights, by default 1.
    grid_in, grid_out : str, optional
        Angular quadrature grids for the forward and inverse SHT, by default
        ``"equiangular"``.
    domain : str, optional
        Either ``"half-line"`` or ``"exterior"`` for the radial dimension, by default ``"half-line"``.
    inner_radius : float, optional
        Inner radius, required for ``domain="exterior"``, by default None.
    lmax, mmax, wmax : int, optional
        Angular and radial truncation, by default inferred from the grids via
        :func:`torch_harmonics.truncate_sht` and from ``nr``.
    npad : int, optional
        Zero-padded radial nodes in log space, by default 0.
    c : float, optional
        Real part of the Mellin variable s = c - i omega, by default 0.5.
    extension : str, optional
        Continuation of the radial signal, ``"circular"`` or ``"reflect"``
        (mirrored, i.e. a DCT-I), see :class:`MellinTransform`. By default ``"circular"``.
    real_kernel : bool, optional
        If ``True``, the radial kernel is real in log r and only the non-negative Mellin
        frequencies carry weights. If ``False``, all frequencies carry independent complex
        weights. By default ``True``.
    bias : bool, optional
        If ``True``, adds a learnable spectral bias scaled by the volume mean, by
        default ``False``.
    """

    def __init__(
        self,
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        in_channels: int,
        out_channels: int,
        rmin: float,
        rmax: float,
        num_groups: Optional[int] = 1,
        grid_in: Optional[str] = "equiangular",
        grid_out: Optional[str] = "equiangular",
        domain: Optional[str] = "half-line",
        inner_radius: Optional[float] = None,
        lmax: Optional[int] = None,
        mmax: Optional[int] = None,
        wmax: Optional[int] = None,
        npad: Optional[int] = 0,
        c: Optional[float] = 0.5,
        extension: Optional[str] = "circular",
        real_kernel: Optional[bool] = True,
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
        self.c = c
        self.extension = extension
        self.real_kernel = real_kernel

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

        # compute the radial truncation from the length of the extended log box
        if wmax is None:
            ntot_in, ntot_out = [2 * (nr - 1) if extension == "reflect" else nr + npad for nr in (nr_in, nr_out)]
            wmax = ntot_in // 2 + 1 if nr_in == nr_out else min(ntot_in, ntot_out) // 2

        # radial transforms. A real kernel transforms the real and imaginary parts of the SHT
        # coefficients separately, which are stacked on a trailing axis behind the radial one
        if real_kernel:
            mellin, imellin, rdim = RealMellinTransform, InverseRealMellinTransform, -4
        else:
            mellin, imellin, rdim = MellinTransform, InverseMellinTransform, -3
        self.mellin = mellin(nr_in, rmin, rmax, wmax=wmax, domain=domain, inner_radius=inner_radius, dim=rdim, npad=npad, c=c, extension=extension)
        self.imellin = imellin(nr_out, rmin, rmax, wmax=wmax, domain=domain, inner_radius=inner_radius, dim=rdim, npad=npad, c=c, extension=extension)
        self.wmax = wmax
        self.nw = self.mellin.nw

        # weight shape, diagonal in (omega, l)
        weight_shape = [num_groups, in_channels // num_groups, out_channels // num_groups, self.nw, self.lmax]

        # Compute scaling factor for correct initialization
        scale = math.sqrt(1.0 / (in_channels // num_groups)) * torch.ones(self.lmax, dtype=torch.complex64)
        # seemingly the first weight is not really complex, so we need to account for that
        scale[0] *= math.sqrt(2.0)
        self.weight = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.complex64))

        if bias:
            # a real kernel needs a separate bias for the real and imaginary parts
            bias_shape = [1, self.in_channels, self.nw, self.lmax, self.mmax] + ([2] if real_kernel else [])
            self.spectral_bias = nn.Parameter(torch.zeros(*bias_shape, dtype=torch.complex64))
            # volume mean rather than integral, so the bias stays trainable and scale covariant
            self.quadrature = QuadratureRadialS2(img_shape=in_shape, rmin=rmin, rmax=rmax, grid=grid_in, domain=domain, inner_radius=inner_radius, normalize=True)

    def extra_repr(self):
        return f"in_channels={self.in_channels}, out_channels={self.out_channels},\n lmax={self.lmax}, mmax={self.mmax}, nw={self.nw}, c={self.c}, extension={self.extension},\n real_kernel={self.real_kernel}, num_groups={self.num_groups}"

    def _apply(self, fn, *args, **kwargs):
        return super()._apply(_keep_complex(fn), *args, **kwargs)

    @torch.compile
    def _contract_diagonal(self, ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
        # the ellipsis covers the order m and, for a real kernel, the real/imaginary axis
        resc = torch.einsum("bgiwl...,giowl->bgowl...", ac, bc)
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

            # compute the volume mean in case the bias is used
            if hasattr(self, "spectral_bias"):
                mean = self.quadrature(x)

            # transform the angular axes, then the radial one
            x = self.sht(x)
            if self.real_kernel:
                x = torch.view_as_real(x)
            x = self.mellin(x).contiguous()

        # store the shapes, the trailing axes are (l, m) or (l, m, re/im)
        B, C = x.shape[:2]
        tail = x.shape[2:]

        # deal with bias, applied on the real view to keep the pointwise op off the complex dtype
        if hasattr(self, "spectral_bias"):
            bias = mean.reshape(B, C, *([1] * (x.dim() - 1))) * torch.view_as_real(self.spectral_bias)
            x = torch.view_as_complex(torch.view_as_real(x) + bias)

        # perform contraction
        x = x.reshape(B, self.num_groups, C // self.num_groups, *tail)
        xp = self._contract_diagonal(x, self.weight)
        # merge the group axes on the real view, as the einsum output can be non-contiguous
        x = torch.view_as_complex(torch.view_as_real(xp).reshape(B, self.out_channels, *tail, 2).contiguous())

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = self.imellin(x)
            if self.real_kernel:
                x = torch.view_as_complex(x.contiguous())
            x = self.isht(x)

        # convert datatype
        x = x.to(dtype=dtype)

        return x
