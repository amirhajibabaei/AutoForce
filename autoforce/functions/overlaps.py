# +
import autoforce.cfg as cfg
from autoforce.core import Function
from .harmonics import Harmonics
import torch
from math import sqrt, factorial as fac
from itertools import product
from typing import Optional


class Overlaps(Function):
    """
    Description:

    * The implementation for
        "Smooth Overlap of Atomic Positions (SOAP)"
      descriptor.

    * Smooth cutoffs:
      Often, this descriptor should be used along with a smooth
      cutoff function for preserving its continuity (at least up
      to 1st order derivatives) due to particles crossing the
      cutoff boundary. For maximum flexibility, instead of a
      cutoff function, the "forward" method accepts the optional
      weights wj which can be used for any general weighting
      mechanism. For example, given rij[nj, 3] and species[nj]:

      >>> soap = Overlaps(3, 3)
      >>> cut = CosineCut()
      >>> cutoff = 6.
      >>> dij = rij.norm(dim=1)
      >>> wj = cut(dij, cutoff)
      >>> y = soap(rij, species, wj=wj)

      A few cutoff functions are implemented in descriptors.cutoff.

    * The width of Gaussians:
      In calculating SOAP, a Gaussian density smearing is assumed
      around an atom (instead of sharp Dirac delta functions).
      The width of these Gaussians can be changed simply by
      rescaling of the displacement vectors:

          rij <- rij/alpha_j

      In general, "alpha_j" can be species dependent and optimized
      as hyperparameters.

    * When rescaling rij, one should rescale cutoff(s) as well.
      The caller to handle these parameters. This function accepts
      only rij, species and wj as inputs.

    """

    def __init__(self, lmax: int, nmax: int) -> None:
        """
        * lmax is the maximum l for the solid (spherical) harmonics {ylm}.

        * nmax is the maximum n for the {|rij|**(2*n)} sequence.

        * The desciptor is calculated by combining these sequences.

        """
        super().__init__()
        self.harmonics = Harmonics(lmax)
        self.nmax = nmax

        # Auxiliary Tensors
        self._nnl = _nnl(lmax, nmax)
        self._pow_2n = 2*torch.arange(nmax+1).to(torch.int)
        self._l = self.harmonics.l.reshape(-1).to(torch.int)
        self._c_m = torch.eye(lmax+1).neg().add(2).to(torch.int)

    @property
    def lmax(self):
        return self.harmonics.lmax

    def function(self, rij: torch.Tensor,
                 species: torch.Tensor,
                 wj: Optional[torch.Tensor] = None,
                 compress: Optional[bool] = True
                 ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        * rij: A float tensor with shape [:, 3] (displacement vectors).

        * species: An int tensor (1D) with the same length as rij (atomic numbers).

        * wj: An optional float tensor (1D) with the same length as rij (weights).


        * Returns:

            s_a, s_b, X

            Where, if ns is the number of unique species,
            the shape of X is

                compress = True -> [ns^2, (nmax+1)*(nmax+2)*(lmax+1)/2]

                compress = False -> [ns^2, nmax+1, nmax+1, lmax+1]

            and X[j, ...] is the joint descriptor for species
            of types (s_a[j], s_b[j]).

            Without compression components of X are symmetric
            wrt double permutations of axis:

                Y = X.reshape(ns, ns, nmax+1, nmax+1, lmax+1)
                0 = Y.permute(1, 0, 3, 2, 4) - Y

            and thus contain redundent numbers.

            After compression some components are multiplied
            by sqrt(2), so that the norm of the descriptor remains
            the same.

        """

        # 1. Sizes
        nj = rij.size(0)
        unique = torch.unique(species, sorted=True)
        ns = len(unique)

        # Dimensions:
        # _[s][n][l][m][j] -> [ns][nmax+1][lmax+1][lmax+1][nj]

        # 2. Mappings
        d_j = rij.norm(dim=1)
        q_j = (-0.5*d_j**2).exp()
        if wj is not None:
            q_j = q_j*wj
        y_lmj = self.harmonics.function(rij)

        # 3. Radial & Angular Coupling
        r_nj = (q_j*d_j[None]**self._pow_2n[:, None])
        f_nlmj = r_nj[:, None, None]*y_lmj[None]

        # 4. Density per species
        c_snlm = torch.zeros_like(f_nlmj[..., 0]).repeat(ns, 1, 1, 1)
        i_j = torch.arange(nj, dtype=torch.int)
        for k, z in enumerate(unique):
            c_snlm[k] = f_nlmj.index_select(-1, i_j[species == z]).sum(dim=-1)

        # 5. Sum over m (c_snlm & c^*_snlm product)
        _ssnnlm = c_snlm[None, :, None, ]*c_snlm[:, None, :, None]
        tmp = _ssnnlm.mul(self._c_m).flatten(-2, -1)
        _ssnnl = torch.zeros_like(_ssnnlm[..., 0]
                                  ).index_add(-1, self._l, tmp)*self._nnl

        # 6. Compress
        if compress:
            ui, uj = torch.triu_indices(self.nmax+1, self.nmax+1)
            _2sq = (2.0-torch.eye(self.nmax+1, dtype=rij.dtype)).sqrt()
            _ssnnl = _ssnnl[:, :, ui, uj, :]*_2sq[ui, uj, None]
            _ssnnl = _ssnnl.flatten(-2, -1)

        # 7. Bcast Species
        s_a, s_b = torch.tensor([x for x in product(unique, unique)]).T

        return s_a, s_b, _ssnnl.flatten(0, 1)


def _nnl(lmax, nmax):
    a = torch.tensor([[1/(sqrt(2*l+1)*2**(2*n+l)*fac(n)*fac(n+l))
                       for l in range(lmax+1)]
                      for n in range(nmax+1)
                      ])
    return (a[None]*a[:, None]).sqrt().to(cfg.float_t)
