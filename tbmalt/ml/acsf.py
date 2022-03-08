# -*- coding: utf-8 -*-
"""Atom-centered symmetry functions method."""
import warnings
from typing import Union, Optional, Literal
import torch
from torch import Tensor
import numpy as np
from tbmalt import Shell, Geometry
from tbmalt.common.batch import pack
from tbmalt.data.units import length_angstrom_units


class Acsf:
    """Construct Atom-centered symmetry functions (Acsf) method.

    This class is designed for batch calculations. Single geometry will be
    transfered to batch in the beginning. If `geometry` is in `__call__`,
    the code will update all initial information related to `geometry`.

    Arguments:
        geometry: Geometry instance with geometric information.
        basis: Shell object with orbital information.
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
            values with the first dimension loop over all atoms, the output
            shape will be [n_atom, n_acsf]; If `geometry`, return G values
            with the first dimension loop over all geometries, the output shape
            will be [n_geometry, n_atom, n_acsf]; If `distance`, each atomic
            interaction will not be summed, the output shape will be
            [n_geometry, n_atom, n_atom, n_acsf].

    Notes:
        The unit in geometry or basis is atomic unit, however, to have a better
        interface with other packages, the default unit here is angstrom.

    """

    def __init__(self,
                 geometry: Geometry,
                 basis: Shell,
                 shell_dict: dict,
                 g1_params: Union[float, Tensor],
                 g2_params: Optional[Tensor] = None,
                 g3_params: Optional[Tensor] = None,
                 g4_params: Optional[Tensor] = None,
                 g5_params: Optional[Tensor] = None,
                 unit: Literal['bohr', 'angstrom'] = 'angstrom',
                 element_resolve: Optional[bool] = True,
                 form: Literal['geometry', 'atom', 'distance'] = 'atom',
                 **kwargs):
        self.geometry = geometry
        self.shell_dict = shell_dict
        self.form = form
        self.element_resolve = element_resolve
        if self.form == 'distance' and not self.element_resolve:
            warnings.warn(
                'If form is distance, element_resolve is automatically True')
            self.element_resolve = False
        self.unique_atomic_numbers = geometry.unique_atomic_numbers
        d_vect = self.geometry.distance_vectors

        # Check batch
        if self.geometry.atomic_numbers.dim() == 1:
            self.batch = False
            self.atomic_numbers = self.geometry.atomic_numbers.unsqueeze(0)
            self.distances = self.geometry.distances.unsqueeze(0)
            self.basis = Shell(self.atomic_numbers, self.shell_dict)
            self.d_vec = d_vect.unsqueeze(0)
        elif self.geometry.atomic_numbers.dim() == 2:
            self.batch = True
            self.atomic_numbers = self.geometry.atomic_numbers
            self.distances = self.geometry.distances
            self.basis = basis
            self.d_vec = d_vect
        else:
            raise ValueError('atomic_numbers dimension error')

        # Check unit, to angstrom unit
        self.unit = unit
        self.distances = self.distances * length_angstrom_units['bohr']
        self.d_vec = self.d_vec * length_angstrom_units['bohr']

        # build orbital like atomic number matrix, this aims to help element
        # resolve issue
        self.ano = self.basis.atomic_number_matrix(form='atomic')[..., 1]

        # Check periodic
        self.isperiodic = self.geometry.isperiodic
        if self.isperiodic:
            self.periodic = kwargs.get('periodic', None)
            assert self.periodic is not None, 'Geometry is periodic, ' + \
                'periodic object should be offered'

            _dist = self.periodic.distances
            _d_vec = self.periodic.distance_vectors
            assert _dist.dim() in (3, 4), 'periodic distances diemension error'

            # ATTENTION, in periodic, all is in batch
            self.pe_distances = _dist
            self.pe_d_vec = _d_vec

            # Build periodic numbers and atomic matrix
            self.pe_atomic_numbers = self.atomic_numbers.repeat(
                _dist.shape[-1], 1, 1).transpose(0, 1)
            self.pe_ano = self.ano.repeat(
                _dist.shape[-1], 1, 1, 1).permute(1, 2, 3, 0)

            # To angstrom
            self.pe_distances = self.pe_distances * \
                length_angstrom_units['bohr']
            self.pe_d_vec = self.pe_d_vec * length_angstrom_units['bohr']

        # Check unit, to angstrom unit
        g1_params = g1_params * length_angstrom_units[unit]
        if g2_params is not None:
            self._g_error_message(g2_params, 2)
            g2_params[..., 1] = g2_params[..., 1] * length_angstrom_units[unit]
        if g3_params is not None:
            self._g_error_message(g3_params, 1)
            g3_params[..., 0] = g3_params[..., 0] / length_angstrom_units[unit]
        if g4_params is not None:
            self._g_error_message(g4_params, 3)
            g4_params[..., 0] = g4_params[..., 0] * length_angstrom_units[unit]
        if g5_params is not None:
            self._g_error_message(g5_params, 3)
            g5_params[..., 0] = g5_params[..., 0] * length_angstrom_units[unit]

        # ACSF parameters
        self.g1_params = g1_params
        self.g2_params = g2_params
        self.g3_params = g3_params
        self.g4_params = g4_params
        self.g5_params = g5_params

        # calculate G1, which is cutoff function
        self.g1 = self.g1_func(self.g1_params)
        self.g = self.g1.clone()

    def __call__(self, geometry: object = None):
        """Calculate G values with input parameters."""
        assert self.g1_params is not None, 'g1_params parameter is None'
        _an = self.pe_atomic_numbers if self.isperiodic else self.atomic_numbers
        if geometry is not None:
            self._update_geo(geometry)

        if self.g2_params is not None:
            self.g, self.g2 = self.g2_func(self.g, self.g2_params)
        if self.g3_params is not None:
            self.g, self.g3 = self.g3_func(self.g, self.g3_params)
        if self.g4_params is not None:
            self.g, self.g4 = self.g4_func(self.g, self.g4_params)
        if self.g5_params is not None:
            self.g, self.g5 = self.g5_func(self.g, self.g5_params)

    def _g_error_message(self, g, size):
        """Check the size of G parameters."""
        assert g.shape[-1] == size, f'There should be {size} parameters' + \
            f' in G, but get {g3_params.shape[-1]} parameters'

    def g1_func(self, g1_params):
        """Calculate G1 parameters."""
        self._fc(self.distances, g1_params)
        _g1 = self.fc.clone() if not self.isperiodic else self.pe_fc.clone()
        an = self.atomic_numbers if not self.isperiodic else self.pe_atomic_numbers

        if not self.form == 'distance':
            if self.element_resolve:
                _g1 = self._element_wise(_g1)
            else:
                _g1 = _g1.sum(-1)

            # Return form as 'atom' or 'geometry'
            _g1 = _g1[an.ne(0)] if self.form == 'atom' else _g1
        elif self.isperiodic and self.form == 'distance':
            # Shape: N_batch, n_cell, n_atom, n_atom
            _g1 = _g1.permute(0, -1, 1, 2)

        return _g1

    def g2_func(self, g, g2_params):
        """Calculate G2 parameters."""
        _fc = self.fc if not self.isperiodic else self.pe_fc
        an = self.pe_atomic_numbers if self.isperiodic else self.atomic_numbers
        _dist = self.pe_distances if self.isperiodic else self.distances
        _mask = _dist.ne(0)
        _g2 = torch.zeros(_dist.shape)
        _g2[_mask] = torch.exp(- g2_params[..., 0] *
                               ((g2_params[..., 1] - _dist[_mask])) ** 2)

        if self.form == 'distance':
            g2 = _g2 * _fc
            g2 = g2.permute(0, -1, 1, 2) if self.isperiodic else g2
            if g.dim() > g2.dim():
                return torch.cat([g, g2.unsqueeze(-1)], -1), g2
            elif g.dim() == g2.dim():
                return torch.cat([g.unsqueeze(-1), g2.unsqueeze(-1)], -1), g2
            else:
                raise ValueError('dimension error')
        else:
            if self.element_resolve:
                g2 = self._element_wise(_g2 * _fc)
            else:
                g2 = _g2.sum(-1)

            g2 = g2[an.ne(0)] if self.form == 'atom' else g2
            return torch.cat([g, g2], -1), g2

    def g3_func(self, g, g3_params):
        """Calculate G3 parameters."""
        _fc = self.fc if not self.isperiodic else self.pe_fc
        an = self.pe_atomic_numbers if self.isperiodic else self.atomic_numbers
        _dist = self.pe_distances if self.isperiodic else self.distances
        _mask = _dist.ne(0)
        _g3 = torch.zeros(_dist.shape)
        _g3[_mask] = torch.cos(-g3_params[..., 0] * _dist[_mask])

        if self.form == 'distance':
            g3 = _g3 * _fc
        else:
            if self.element_resolve:
                g3 = self._element_wise(_g3 * _fc)
            else:
                g3 = _g3.sum(-1)

            g3 = g3[an.ne(0)] if self.form == 'atom' else g3

        g = g.unsqueeze(1) if g.dim() == 1 else g

        return torch.cat([g, g3], -1), g3

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
        """Calculate angular parameters for G4, G5."""
        eta, zeta, lamb = g_params.squeeze()

        # the dimension of d_ij * d_ik is [n_batch, n_atom_ij, n_atom_jk]
        _dist = self.distances
        if not self.isperiodic:
            _an = self.atomic_numbers
            d_vect_ijk = (self.d_vec.unsqueeze(-2) *
                          self.d_vec.unsqueeze(-3)).sum(-1)
            dist_ijk = _dist.unsqueeze(-1) * _dist.unsqueeze(-2)
            dist2_ijk = _dist.unsqueeze(-1) ** 2 + _dist.unsqueeze(-2) ** 2
            dist2_ijk = _dist.unsqueeze(-3) ** 2 + \
                dist2_ijk if jk else dist2_ijk
            fc = self.fc.unsqueeze(-1) * self.fc.unsqueeze(-2)
            fc = fc * self.fc.unsqueeze(-3) if jk else fc
        else:
            _an = self.pe_atomic_numbers
            pe_dist = self.periodic.distances.permute(0, -1, 1, 2)
            d_vect_ijk = (self.d_vec.unsqueeze(-2).unsqueeze(1) *
                          self.pe_d_vec.unsqueeze(-3)).sum(-1)
            dist_ijk = _dist.unsqueeze(-1).unsqueeze(1) * pe_dist.unsqueeze(-2)
            dist2_ijk = _dist.unsqueeze(-1).unsqueeze(1) ** 2 + \
                pe_dist.unsqueeze(-2) ** 2
            _pe_fc = self.pe_fc.permute(0, -1, 1, 2).unsqueeze(-2)
            fc = self.fc.unsqueeze(-1).unsqueeze(1) * _pe_fc

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
        # _fc = self.fc if not self.isperiodic else self.fc.permute(0, -1, 1, 2)
        # fc = _fc.unsqueeze(-1) * _fc.unsqueeze(-2)
        # fc = fc * _fc.unsqueeze(-3) if jk else fc
        ang = 0.5 * (2 ** (1 - zeta) * (1 + lamb * cos) ** zeta * exp * fc)
        if self.form == 'distance':
            _g = ang.sum(-1)
            if g.dim() > _g.dim():
                return torch.cat([g, _g.unsqueeze(-1)], -1), _g
            elif g.dim() == _g.dim():
                return torch.cat([g.unsqueeze(-1), _g.unsqueeze(-1)], -1), _g
            else:
                raise ValueError('dimension error')
        else:
            if not self.element_resolve:
                ang = ang.sum(-1)
                _g = ang.sum(-1)[_an.ne(0)].unsqueeze(-1)
            else:
                _g = self._element_wise_ang(ang)
                _g = _g[_an.ne(0)] if self.form == 'atom' else _g

            return torch.cat([g, _g], dim=-1), _g

    def _element_wise(self, g):
        """Return g value with element wise for each atom in batch."""
        # return dimension [n_batch, max_atom, n_unique_atoms]
        an = self.atomic_numbers if not self.isperiodic \
            else self.pe_atomic_numbers
        ano = self.ano if not self.isperiodic else self.pe_ano
        _g = torch.zeros(*an.shape, len(self.unique_atomic_numbers))

        # Use unique_atomic_numbers which will minimum using loops
        for ii, ian in enumerate(self.unique_atomic_numbers):
            mask = ano == ian
            split = tuple(mask.sum(-1).flatten().tolist()) if not \
                self.isperiodic else tuple(mask.sum(-2).flatten().tolist())

            # Select the interaction of each atom with ian specie
            _g[..., ii] = pack(torch.split(g[mask], split)
                               ).sum(-1).reshape(an.shape)
        return _g

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
            if not self.isperiodic:
                _ig = torch.zeros(*self.atomic_numbers.shape)
            else:
                _ig = torch.zeros(self.pe_atomic_numbers.shape)
            _im = torch.nonzero((anm == iu).all(dim=-1))

            # If atom pair is not homo, we have to consider inverse iu
            if iu[0] != iu[1]:
                _im = torch.cat(
                    [_im, torch.nonzero((anm == iu.flip(0)).all(dim=-1))])
                _im = _im[_im[..., 0].sort()[1]]

            # Select last two dims which equals to atom-pairs in _im
            g_im = g[_im[..., 0], :, _im[..., 1], _im[..., 2]]
            _imask, count = torch.unique_consecutive(
                _im[..., 0], return_counts=True)

            # If there is such atom pairs
            if count.shape[0] > 0:
                _ig[_imask] = pack(g_im.split(tuple(count))).sum(1)
            g_res.append(_ig)

        if not self.isperiodic:
            return pack(g_res).permute(1, 2, 0)
        else:
            # Return shape: n_batch, n_cell, n_atom, n_atom
            return pack(g_res).permute(1, 2, 3, 0)

    def _fc(self, distances: Tensor, rcut: Tensor):
        """Return cutoff function"""
        self.fc = torch.zeros(distances.shape)
        mask = distances.lt(rcut) * distances.gt(0.0)
        self.fc[mask] = 0.5 * (torch.cos(np.pi * distances[mask] / rcut) + 1.0)

        if self.isperiodic:
            _pe_dist = self.pe_distances
            self.pe_fc = torch.zeros(_pe_dist.shape)
            mask_pe = _pe_dist.lt(rcut) * _pe_dist.gt(0.0)
            self.pe_fc[mask_pe] = 0.5 * (
                torch.cos(np.pi * _pe_dist[mask_pe] / rcut) + 1.0)
