# -*- coding: utf-8 -*-
"""Atom-centered symmetry functions method."""
import warnings
from typing import Union, Optional, Literal
import torch
from torch import Tensor
import numpy as np
from tbmalt import Basis, Geometry
from tbmalt.common.batch import pack
from tbmalt.data.units import length_units


class Acsf:
    """Construct Atom-centered symmetry functions (Acsf) method.

    This class is designed for batch calculations. Single geometry will be
    transfered to batch in the beginning. If `geometry` is in `__call__`,
    the code will update all initial information related to `geometry`.

    Arguments:
        geometry: Geometry instance with geometric information.
        basis: Basis object with orbital information.
        shell_dict: Dictionary with calculated orbital quantum number.
        g1_params: Cutoff of acsf functions.
        g2_params: Radical geometric parameters.
        g3_params: Radical geometric parameters.
        g4_params: Angular geometric parameters.
        g5_params: Angular geometric parameters.
        unit: Unit of input G parameters.
        element_resolve: If return element resolved G or sum of G value over
            all neighbouring atoms.
        form: This determines how to generate the output. If `atom`, return G
            values with the first dimension loop over all atoms; If `geometry`,
            return G values with the first dimension loop over all geometries;
            If `distance`, each atomic interaction will not be summed.
        distances_like: If determines if the output will keep the
            `Geometry.distances` shape, this means all `G_ij` will keep here.

    """

    def __init__(self,
                 geometry: Geometry,
                 basis: Basis,
                 shell_dict: dict,
                 g1_params: Union[float, Tensor],
                 g1_extra_params: Union[dict, Tensor] = None,
                 g2_params: Optional[Tensor] = None,
                 g3_params: Optional[Tensor] = None,
                 g4_params: Optional[Tensor] = None,
                 g5_params: Optional[Tensor] = None,
                 unit: Literal['bohr', 'angstrom'] = 'angstrom',
                 element_resolve: Optional[bool] = True,
                 form: Literal['geometry', 'atom', 'distance'] = 'atom',
                 **kwargs):
        self.geometry = geometry
        self.periodic = self.geometry.isperiodic
        self.unit = unit

        # ACSF parameters
        self.g1_params = g1_params * length_units[unit]  # in bohr
        self.g1_extra_params = g1_extra_params
        self.g2_params = g2_params
        self.g3_params = g3_params
        self.g4_params = g4_params
        self.g5_params = g5_params
        self.form = form
        self.element_resolve = element_resolve

        # Set atomic numbers batch like
        an = self.geometry.atomic_numbers
        self.atomic_numbers = an if an.dim() == 2 else an.unsqueeze(0)
        self.distances = self.geometry.distances if an.dim() == 2 else \
            self.geometry.distances.unsqueeze(0)
        self.unique_atomic_numbers = geometry.unique_atomic_numbers()

        # Check Basis
        self.basis = basis if basis.atomic_numbers.dim() == 2 else Basis(
            self.atomic_numbers, shell_dict)
        self.shell_dict = shell_dict

        # build orbital like atomic number matrix, expand size from
        # [n_batch, max_atom] to flatten [n_batch * max_atom, max_atom]
        ano = self.basis.atomic_number_matrix(form='atomic')[..., 1]
        self.ano = ano.view(-1, ano.shape[-1])

        # calculate G1, which is cutoff function
        self.fc, self.g1 = self.g1_func(self.g1_params)

        # transfer all geometric parameters to angstrom unit
        d_vect = self.geometry.distance_vectors / length_units['angstrom']
        self._d_vec = d_vect.unsqueeze(0) if d_vect.dim() == 3 else d_vect
        self._dist = self.distances / length_units['angstrom']

    def __call__(self, geometry: object = None):
        """Calculate G values with input parameters."""
        assert self.g1_params is not None, 'g1_params parameter is None'
        if geometry is not None:
            self._update_geo(geometry)

        if self.form == 'distance':
            _g = self.fc.clone()
        else:
            _g = self.g1.clone()

        if self.g2_params is not None:
            _g, self.g2 = self.g2_func(_g, self.g2_params)
        if self.g3_params is not None:
            _g, self.g3 = self.g3_func(_g, self.g3_params)
        if self.g4_params is not None:
            _g, self.g4 = self.g4_func(_g, self.g4_params)
        if self.g5_params is not None:
            _g, self.g5 = self.g5_func(_g, self.g5_params)

        # if atom_like is True, return g in sequence of each atom in batch,
        # else return g in sequence of each geometry
        if self.form == 'geometry':
            self.g = torch.zeros(*self.atomic_numbers.shape, _g.shape[-1])
            self.g[self.atomic_numbers.ne(0)] = _g
        else:
            self.g = _g

        return self.g

    def g1_func(self, g1_params):
        """Calculate G1 parameters."""
        _g1, self.mask = self._fc(self.distances, g1_params)
        if self.element_resolve:
            g1 = self._element_wise(_g1)
        else:
            g1 = _g1[self.atomic_numbers.ne(0)].sum(-1)

        # oprions of return type, if element_resolve, each atom specie
        # will be calculated seperatedly, else return the sum
        return _g1, g1

    def g2_func(self, g, g2_params):
        """Calculate G2 parameters."""
        _g2 = torch.zeros(self.distances.shape)
        _g2[self.mask] = torch.exp(
            -g2_params[..., 0] * ((g2_params[..., 1] - self._dist[self.mask])) ** 2)
        if self.form =='distance':
            g2 = _g2
        else:
            g2 = self._element_wise(_g2 * self.fc)

        g = g.unsqueeze(1) if g.dim() == 1 else g

        if self.form =='distance':
            return (torch.stack([g, g2], dim=-1), g2) if self.element_resolve \
                else (torch.cat([g, g2.sum(-1).unsqueeze(1)], dim=1), g2)
        else:
            return (torch.cat([g, g2], dim=1), g2) if self.element_resolve \
                else (torch.cat([g, g2.sum(-1).unsqueeze(1)], dim=1), g2)

    def g3_func(self, g, g3_params):
        """Calculate G3 parameters."""
        _g3 = torch.zeros(self.distances.shape)
        _g3[self.mask] = torch.cos(-g3_params[..., 0] * self._dist[self.mask])
        g3 = self._element_wise(_g3 * self.fc)
        g = g.unsqueeze(1) if g.dim() == 1 else g

        return (torch.cat([g, g3], dim=1), g3) if self.element_resolve else \
            (torch.cat([g, g3.sum(-1).unsqueeze(1)], dim=1), g3)

    def g4_func(self, g, g4_params, jk=True):
        """Calculate G4 parameters without element wise."""
        # parameter jk here is True, which will include j and k atomic
        # interactions when calculate G4 of i atom
        return self._angle(g, g4_params, jk=True)

    def g5_func(self, g, g5_params):
        """Calculate G5 parameters element wise."""
        # parameter jk here is False, which will not include j and k atomic
        # interactions when calculate G4 of i atom
        return self._angle(g, g5_params, jk=False)

    def _angle(self, g, g_params, jk=True):
        """Calculate G4 parameters."""
        eta, zeta, lamb = g_params.squeeze()
        d_vect_ijk = (self._d_vec.unsqueeze(-2) * self._d_vec.unsqueeze(-3)).sum(-1)

        # the dimension of d_ij * d_ik is [n_batch, n_atom_ij, n_atom_jk]
        print('_dist', self._dist.shape, self.distances.shape)
        _dist = self._dist if not self.periodic else self._dist.permute(0, -1, 1, 2)
        dist_ijk = _dist.unsqueeze(-1) * _dist.unsqueeze(-2)
        dist2_ijk = _dist.unsqueeze(-1) ** 2 + _dist.unsqueeze(-2) ** 2
        dist2_ijk = _dist.unsqueeze(-3) ** 2 + dist2_ijk if jk else dist2_ijk

        # Make sure distance_vec like ijj, ikk are zero, since these do not
        # contribute to angle
        mask_zero = torch.tril_indices(dist_ijk.shape[-1], dist_ijk.shape[-1])
        dist_ijk[:, :, mask_zero[0], mask_zero[0]] = 0

        # create the terms in G4 or G5
        cos = torch.zeros(dist_ijk.shape)
        exp = torch.zeros(dist_ijk.shape)
        mask = dist_ijk.ne(0)

        exp[mask] = torch.exp(-eta * dist2_ijk[mask])
        cos[mask] = d_vect_ijk[mask] / dist_ijk[mask]
        _fc = self.fc if not self.periodic else self.fc.permute(0, -1, 1, 2)
        fc = _fc.unsqueeze(-1) * _fc.unsqueeze(-2)
        fc = fc * _fc.unsqueeze(-3) if jk else fc

        ang = 0.5 * (2**(1 - zeta) * (1 + lamb * cos)**zeta * exp * fc)
        if not self.element_resolve:
            ang = ang.sum(-1)
            _g = ang.sum(-1)[self.atomic_numbers.ne(0)].unsqueeze(-1)
        else:
            if self.form == 'atom':
                _g = self._element_wise_ang(ang)[self.atomic_numbers != 0]
            else:
                _g = self._element_wise_ang(ang)
        print(g.shape, _g.shape)
        return torch.cat([g, _g.squeeze(0)], dim=1), _g

    def _element_wise(self, g):
        """Return g value with element wise for each atom in batch."""
        # return dimension [n_batch, max_atom, n_unique_atoms]
        if g.dim() == 3:  # For normal G1~G5
            g = g.view(-1, g.shape[-1])
            _g = torch.zeros(g.shape[0], len(self.unique_atomic_numbers))
        elif g.dim() == 4:  # If extra_params have multi dims
            d2, d1 = g.shape[-2], g.shape[-1]
            g = g.view(-1, d2, d1)
            _g = torch.zeros(g.shape[0], len(self.unique_atomic_numbers) * d1)

        # Use unique_atomic_numbers which will minimum using loops
        for i, ian in enumerate(self.unique_atomic_numbers):
            mask = self.ano == ian
            tmp = torch.zeros(g.shape)
            tmp[mask] = g[mask]

            if g.dim() == 2:
                _g[..., i] = tmp.sum(-1)
            elif g.dim() == 3:
                im = torch.arange(i, _g.shape[-1], len(self.unique_atomic_numbers))
                _g[..., im] = tmp.sum(-2)

        mask2 = self.atomic_numbers.flatten().ne(0)
        return _g[mask2]

    def _element_wise_ang(self, g):
        """Return g4, g5 values with element wise for each atom in batch."""
        uniq_atom_pair = []
        for i, ian in enumerate(self.unique_atomic_numbers):
            for j, jan in enumerate(self.unique_atomic_numbers[i:]):
                uniq_atom_pair.append(torch.tensor([ian, jan]))
        uniq_atom_pair = pack(uniq_atom_pair)
        anm = self.basis.atomic_number_matrix('atomic')

        g_res = []
        for iu in uniq_atom_pair:
            if not self.periodic:
                _ig = torch.zeros(*self.atomic_numbers.shape)
            else:
                _ig = torch.zeros(*self.atomic_numbers.shape,
                                  self._dist.shape[-1]).transpose(-1, -2)
            _im = torch.nonzero((anm == iu).all(dim=-1))

            # If atom pair is not homo, we have to consider inverse iu
            if iu[0] != iu[1]:
                _im = torch.cat([_im, torch.nonzero((anm == iu.flip(0)).all(dim=-1))])
                _im = _im[_im[..., 0].sort()[1]]

            # Select last two dims which equals to atom-pairs in _im
            g_im = g[_im[..., 0], :, _im[..., 1], _im[..., 2]]
            _imask, count = torch.unique_consecutive(_im[..., 0], return_counts=True)

            # If there is such atom pairs
            if count.shape[0] > 0:
                _ig[_imask] = pack(g_im.split(tuple(count))).sum(1)
            g_res.append(_ig)

        if not self.periodic:
            return pack(g_res).permute(1, 2, 0)
        else:
            return pack(g_res).permute(1, 3, 2, 0)

    def _fc(self, distances: Tensor, rcut: Tensor):
        """Cutoff function in acsf method."""
        fc = torch.zeros(distances.shape)
        mask = distances.lt(rcut) * distances.gt(0.0)
        fc[mask] = 0.5 * (torch.cos(
            np.pi * distances[mask] / rcut) + 1.0)

        return fc, mask
