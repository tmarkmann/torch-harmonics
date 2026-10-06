# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
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

from .mellin import InverseMellinTransform
from .sht import InverseRealSHT


class GaussianRandomFieldS2(torch.nn.Module):
    r"""
    Gaussian random field on the sphere via Karhunen--Loève expansion.

    Samples realisations of a centred Gaussian random field on :math:`S^2`
    whose covariance operator has the Matérn-like power spectrum

    .. math::

        C_l = \sigma^2
              \left(\frac{l(l+1)}{R^2} + \tau^2\right)^{-\alpha}

    where :math:`l` is the spherical harmonic degree.  The field is generated
    by drawing i.i.d. standard-normal spectral coefficients, scaling them by
    :math:`\sqrt{C_l}`, and transforming to the spatial domain with an inverse
    SHT.

    Larger ``alpha`` produces smoother fields (steeper spectral roll-off);
    ``tau`` controls the transition scale between the flat low-:math:`l`
    plateau and the power-law decay.

    Parameters
    ----------
    nlat : int
        Number of latitudinal grid points (``nlon`` is set to ``2 * nlat``).
    alpha : float, optional
        Spectral exponent (smoothness).  Must be > 1 when ``sigma`` is not
        given.  Default ``2.0``.
    tau : float, optional
        Inverse correlation length scale.  Default ``3.0``.
    sigma : float, optional
        Overall amplitude.  If ``None`` (default), computed from ``alpha``
        and ``tau`` so that the field variance is :math:`\mathcal{O}(1)`.
    radius : float, optional
        Radius of the sphere.  Default ``1.0``.
    grid : str, optional
        Grid type for the inverse SHT (``"equiangular"``,
        ``"legendre-gauss"``, etc.).  Default ``"equiangular"``.
    dtype : torch.dtype, optional
        Floating-point dtype.  Default ``torch.float32``.

    Examples
    --------
    >>> import torch
    >>> from torch_harmonics.random_fields import GaussianRandomFieldS2
    >>> grf = GaussianRandomFieldS2(nlat=128, alpha=2.5, tau=5.0)
    >>> samples = grf(4)          # 4 independent realisations
    >>> samples.shape
    torch.Size([4, 128, 256])
    """

    def __init__(self, nlat, alpha=2.0, tau=3.0, sigma=None, radius=1.0, grid="equiangular", dtype=torch.float32):
        super().__init__()

        # Number of latitudinal modes.
        self.nlat = nlat

        # Default value of sigma if None is given.
        if sigma is None:
            if alpha <= 1.0:
                raise ValueError(f"Alpha must be greater than one, got {alpha}.")
            sigma = tau ** (0.5 * (2 * alpha - 2.0))

        # Inverse SHT
        self.isht = InverseRealSHT(self.nlat, 2 * self.nlat, grid=grid, norm="backward").to(dtype=dtype)

        lmax = self.isht.lmax
        mmax = self.isht.mmax

        # Square root of the eigenvalues of C.
        sqrt_eig = torch.as_tensor([j * (j + 1) for j in range(lmax)]).view(lmax, 1).repeat(1, mmax)
        sqrt_eig = torch.tril(sigma * (((sqrt_eig / radius**2) + tau**2) ** (-alpha / 2.0)))
        sqrt_eig[0, 0] = 0.0
        sqrt_eig = sqrt_eig.unsqueeze(0)
        self.register_buffer("sqrt_eig", sqrt_eig)

        # Save mean and var of the standard Gaussian.
        # Need these to re-initialize distribution on a new device.
        mean = torch.as_tensor([0.0]).to(dtype=dtype)
        var = torch.as_tensor([1.0]).to(dtype=dtype)
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)

        # Standard normal noise sampler.
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)

    def forward(self, N, xi=None):

        # Sample Gaussian noise.
        if xi is None:
            lmax = self.isht.lmax
            mmax = self.isht.mmax
            xi = self.gaussian_noise.sample(torch.Size((N, lmax, mmax, 2))).squeeze(-1)
            xi = torch.view_as_complex(xi)

        # Karhunen-Loeve expansion.
        u = self.isht(xi * self.sqrt_eig)

        return u

    # Override cuda and to methods so sampler gets initialized with mean
    # and variance on the correct device.
    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)

        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)

        return self


class GaussianRandomFieldRadialS2(torch.nn.Module):
    r"""
    Gaussian random field on :math:`(0, \infty) \times S^2` via Karhunen--Loève expansion.

    Samples realisations of a centred Gaussian random field whose covariance
    operator has the separable Matérn-like power spectrum

    .. math::

        C_{\omega l} = \sigma^2
              \left(\omega^2 + \tau_r^2\right)^{-\alpha_r}
              \left(l(l+1) + \tau_s^2\right)^{-\alpha_s}

    where :math:`l` is the spherical harmonic degree and :math:`\omega` is the
    Mellin frequency conjugate to :math:`x = \log r`. The radial factor is a 1D
    Matérn spectrum in :math:`x` and the angular factor the spectrum of
    :class:`GaussianRandomFieldS2`, so radial and angular smoothness are set
    independently.

    Parameters
    ----------
    nlat : int
        Number of latitudinal grid points.
    nr : int
        Number of radial grid points.
    rmin, rmax : float
        Bounds on :math:`r`, or on :math:`\rho = (r - R) / R` for the
        exterior domain.
    alpha_r : float, optional
        Radial spectral exponent.  Must be > 1/2 when ``sigma`` is not given.  Default ``2.0``.
    tau_r : float, optional
        Inverse correlation length in :math:`\log r`.  Default ``1.0``.
    alpha_s : float, optional
        Angular spectral exponent.  Must be > 1 when ``sigma`` is not given.
        Default ``2.0``.
    tau_s : float, optional
        Angular inverse correlation length scale.  Default ``3.0``.
    sigma : float, optional
        Overall amplitude.  If ``None`` (default), set to
        :math:`\tau_r^{\alpha_r - 1/2} \tau_s^{\alpha_s - 1}`.
    grid : str, optional
        Grid type for the inverse SHT (``"equiangular"``,
        ``"legendre-gauss"``, etc.).  Default ``"equiangular"``.
    domain : str, optional
        Either ``"half-line"`` or ``"exterior"``, by default ``"half-line"``.
    inner_radius : float, optional
        Inner radius, required for ``domain="exterior"``, by default None.
    lmax, wmax : int, optional
        Angular and radial truncation, by default inferred from the grids.
    npad : int, optional
        Number of radial nodes zero padded beyond ``rmax`` and discarded, by default 0.
    dtype : torch.dtype, optional
        Floating-point dtype.  Default ``torch.float32``.
    """

    def __init__(
        self,
        nr,
        nlat,
        rmin,
        rmax,
        alpha_r=2.0,
        tau_r=1.0,
        alpha_s=2.0,
        tau_s=3.0,
        sigma=None,
        grid="equiangular",
        domain="half-line",
        inner_radius=None,
        lmax=None,
        wmax=None,
        npad=0,
        dtype=torch.float32,
    ):
        super().__init__()

        # Number of latitudinal, longitudinal and radial points.
        self.nlat = nlat
        self.nlon = 2 * nlat
        self.nr = nr

        # Default value of sigma if None is given, one factor per dimension as in GaussianRandomFieldS2.
        if sigma is None:
            if alpha_r <= 0.5:
                raise ValueError(f"alpha_r must be greater than 1/2, got {alpha_r}.")
            if alpha_s <= 1.0:
                raise ValueError(f"alpha_s must be greater than one, got {alpha_s}.")
            sigma = tau_r ** (alpha_r - 0.5) * tau_s ** (alpha_s - 1.0)

        # Inverse transforms
        self.isht = InverseRealSHT(self.nlat, self.nlon, lmax=lmax, grid=grid, norm="backward").to(dtype=dtype)
        self.imellin = InverseMellinTransform(nr, rmin, rmax, wmax=wmax, domain=domain, inner_radius=inner_radius, dim=-3, npad=npad, c=0.0).to(dtype=dtype)

        lmax = self.isht.lmax
        mmax = self.isht.mmax

        # Mellin frequencies, signed and in torch.fft order.
        omega = self.imellin.omega.to(torch.float64)
        l = torch.arange(lmax, dtype=torch.float64)
        l = l * (l + 1)

        # Square root of the eigenvalues of C, separable in (omega, l)
        sqrt_eig_r = (omega**2 + tau_r**2) ** (-alpha_r / 2.0)
        sqrt_eig_s = (l + tau_s**2) ** (-alpha_s / 2.0)
        sqrt_eig = sigma * math.sqrt(self.imellin.length) * sqrt_eig_r.view(-1, 1, 1) * sqrt_eig_s.view(1, -1, 1)
        sqrt_eig = torch.tril(sqrt_eig.expand(-1, -1, mmax))
        sqrt_eig = sqrt_eig.unsqueeze(0).to(dtype=dtype)
        self.register_buffer("sqrt_eig", sqrt_eig)

        # Save mean and var of the standard Gaussian.
        mean = torch.as_tensor([0.0]).to(dtype=dtype)
        var = torch.as_tensor([1.0]).to(dtype=dtype)
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)

        # Standard normal noise sampler.
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)

    def forward(self, N, xi=None):

        # Sample Gaussian noise.
        if xi is None:
            nw = self.imellin.nw
            lmax = self.isht.lmax
            mmax = self.isht.mmax
            xi = self.gaussian_noise.sample(torch.Size((N, nw, lmax, mmax, 2))).squeeze(-1)
            xi = torch.view_as_complex(xi)

        # Karhunen-Loeve expansion
        u = self.isht(self.imellin(xi * self.sqrt_eig))

        return u

    # Override cuda and to methods so sampler gets initialized with mean
    # and variance on the correct device.
    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)

        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)

        return self
