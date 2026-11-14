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


import dataclasses

import torch

from torch_harmonics.integration import QuadratureS2
from torch_harmonics.radial import GeometricRadialGrid, SphericalProductGrid

from .poisson_equation import RadialPoissonSolver


class RadialPoissonDataset(torch.utils.data.Dataset):
    """Custom Dataset class for Poisson training data

    Parameters
    ----------
    grid : SphericalProductGrid
        Grid the samples are generated on, of shape (nr, nlat, nlon), on the half-line or the
        exterior domain; see RadialPoissonSolver
    source : str, optional
        Source type, by default "bump". Either "bump" or "ball"
    num_examples : int, optional
        Number of examples, by default 32
    device : torch.device, optional
        Device to use, by default torch.device("cpu")
    normalize : bool, optional
        Whether to normalize the input and target, by default True
    support : tuple, optional
        Interval ``(lo, hi)`` of radii that confines the "bump" blobs and the "ball" radius,
        by default None, which keeps the blobs off the ends of the grid and draws the ball
        radius from the middle of it
    oversampling : int, optional
        Radial refinement of the grid the sources are drawn and solved on, by default 1.
        Above 1, ``grid.radial`` must be a GeometricRadialGrid; the sources then live on the
        geometric grid with ``oversampling * (nr - 1) + 1`` nodes, which contains the nodes
        of ``grid`` as every ``oversampling``-th one

    Returns
    -------
    inp : torch.Tensor
        Source times r**2, shape (nr, nlat, nlon).
    tar : torch.Tensor
        Solution, shape (nr, nlat, nlon)
    """

    # new in v1.0.0rc1, so there is no signature before the grid descriptors to reject
    _legacy_signature_exempt = "new in v1.0.0rc1"

    def __init__(
        self,
        grid,
        source="bump",
        num_examples=32,
        device=torch.device("cpu"),
        normalize=True,
        support=None,
        oversampling=1,
    ):
        if source not in self._sources():
            raise ValueError(f"unknown source: {source}")
        if not isinstance(oversampling, int) or oversampling < 1:
            raise ValueError(f"oversampling must be a positive integer, got {oversampling!r}")

        self.num_examples = num_examples
        self.device = device
        self.normalize = normalize
        self.source = source
        self.support = None if support is None else tuple(support)
        self.oversampling = oversampling

        self.grid = grid
        self.nr, self.nlat, self.nlon = self.grid.shape

        # the solver keeps its buffers in float64 and solves in the precision of the source.
        # Oversampled, it draws and solves the sources on the fine grid and returns the
        # solution on the nodes of grid
        if oversampling == 1:
            self.solver = RadialPoissonSolver(grid).to(self.device)
        else:
            if not isinstance(grid, SphericalProductGrid) or not isinstance(grid.radial, GeometricRadialGrid):
                raise ValueError(f"oversampling needs a SphericalProductGrid with a GeometricRadialGrid, got {grid!r}")
            fine = dataclasses.replace(grid.radial, nr=oversampling * (self.nr - 1) + 1)
            self.solver = RadialPoissonSolver(SphericalProductGrid(grid.angular, fine), radial_out=grid.radial).to(self.device)

        # nodes of the target grid the ball radius is drawn from, as indices into it
        if self.support is None:
            self.ball_nodes = torch.arange(self.nr // 5, 4 * self.nr // 5)
        else:
            # the nodes are computed, so one meant to sit on a bound of the support may miss it by a rounding
            r, tol = self.solver.r_out, 1e-12
            self.ball_nodes = torch.nonzero((r >= self.support[0] * (1 - tol)) & (r <= self.support[1] * (1 + tol))).flatten().cpu()
            if source == "ball" and self.ball_nodes.numel() == 0:
                raise ValueError(f"support {self.support} contains no radial node to draw a ball radius from")

        # mean over the sphere, on every radial node alike, for the scale
        self.sphere_mean = QuadratureS2(self.grid, normalize=True).to(self.device)

    def __len__(self):
        return self.num_examples

    @classmethod
    def _sources(cls):
        """Source types _get_source draws; a subclass adding one extends both."""
        return ("bump", "ball")

    def _get_source(self, dtype):
        """Draw one source f on the solver's grid, in dtype."""

        if self.source == "bump":
            nblobs = int(torch.randint(1, 9, ()).item())
            return self.solver.random_bump_source(nblobs=nblobs, l_src=8, dtype=dtype, support=self.support)

        # random target node, taken from the solver's grid, which contains it
        ir = int(self.ball_nodes[torch.randint(self.ball_nodes.numel(), ())].item())
        r = self.solver.r[self.oversampling * ir].item()
        return self.solver.ball_source(radius=r, dtype=dtype)

    def _get_sample(self):
        """Get one unscaled (r**2 f, u) pair in float32, on the nodes of grid."""

        # the samples are float32, so they are solved in float32 too, which agrees with a
        # float64 solve to a few 1e-7 of the solution's maximum
        dtype = torch.float32

        f = self._get_source(dtype)
        u = self.solver.solve(f)
        r2f = self.solver.r_out.to(dtype).reshape(-1, 1, 1) ** 2 * f[:: self.oversampling]

        return r2f, u

    def scale(self, g):
        """Scale factor as the RMS, averaged over the sphere and over the radial nodes."""
        # squared relative to the largest value, which cannot overflow: r**2 f squares to
        # beyond float32 from radii of about 4e9 on, which a grid in physical units can reach
        m = g.abs().amax().clamp_min(torch.finfo(g.dtype).tiny)
        return m * self.sphere_mean((g / m) ** 2).mean().sqrt()

    def __getitem__(self, index):

        with torch.inference_mode():
            with torch.no_grad():
                inp, tar = self._get_sample()

                if self.normalize:
                    s = self.scale(inp)
                    inp, tar = inp / s, tar / s

        return inp.clone(), tar.clone()
