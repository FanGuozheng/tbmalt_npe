#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of many-body tensor representation (MBTR)."""
import warnings
from typing import Literal
import torch
import numpy as np
from torch import Tensor
from tbmalt import Basis, Geometry
from tbmalt.structures.geometry import unique_atom_pairs
from tbmalt.common.batch import pack

# Define parameters
pi2 = np.sqrt(2 * np.pi)


class Mbtr:
    """Implementation of many-body tensor representation (MBTR).

    Arguments:
        geometry:
        basis:

    Reference:

    """

    def __init__(self,
                 geometry: Geometry,
                 basis: Basis,
                 g1: list,
                 g2: Tensor = None,
                 g3: Tensor = None,
                 form: Literal['atom', 'geometry', 'distance'] = 'geometry',
                 **kwargs):
        self.geometry = geometry
        self.atomic_numbers = self.geometry.atomic_numbers
        self.distances = self.geometry.distances
        self.batch = True if geometry.atomic_numbers.dim() == 2 else False
        self.unique_atomic_numbers = self.geometry.unique_atomic_numbers()
        self.unique_atom_pairs = unique_atom_pairs(
            self.geometry, repeat_pairs=False)
        self.basis = basis
        self.atomic_pairs = self.basis.atomic_number_matrix('atomic')
        self.form = form

        # weight of g1 set as 1
        self.g = self._get_g1(g1) * 1.0
        if g2 is not None:
            self.g = self._get_g2(self.g, g2)
        if g3 is not None:
            self.g = self._get_g3(self.g, g3)

    def _get_g1(self, g1: Tensor):
        r"""Check $\mathcal{D}_1$ function parameters."""
        # Check for the input, where g1 should include range start, end,
        # length and sigma ingaussian
        assert len(g1) == 4, f'len(g1) should be 4, but get {len(g1)}'
        _min, _max, _len, _sigma = g1

        # To get erf function difference, we need extra one length
        dx = (_max - _min) / _len
        _space = torch.linspace(_min - dx, _max + dx, _len + 1)

        # Use erf function difference to calculate delta Gaussian distribution
        if not self.batch:
            # The intensity of Gaussian function is determined by the number
            # of atoms of such element specie
            _map = [0.5 * (1.0 + torch.erf((_space - ii) * (
                self.atomic_numbers == ii).sum() / (2.0 * _sigma)))
                for ii in self.unique_atomic_numbers]

            # difference between erf function and sum over different elements
            _g1 = torch.cat([1 / (pi2 * _sigma) * (im[..., 1:] - im[..., :-1])
                             for im in _map])
        else:
            # TWO LOOPs HERE could be improved by parallize
            # For machine learning, the unique_atomic_numbers should be the
            # same for each system otherwise it's difficult to learn
            # _map = [[0.5 * (1.0 + torch.erf((_space - ii) / (2.0 * _sigma)
            #                                 * (inumber == ii).sum()))
            #         for ii in self.unique_atomic_numbers]
            #         for inumber in self.geometry.atomic_numbers]
            # _g1 = pack([torch.cat(
            #     [1 / (pi2 * _sigma) * (im[..., 1:] - im[..., :-1])
            #       for im in imap]) for imap in _map])

            if self.form in ('atom', 'geometry'):
                _g1 = torch.zeros(*self.atomic_numbers.shape, _len)
            else:
                _g1 = torch.zeros(*self.distances.shape, _len)
            for iua in self.unique_atomic_numbers:
                _mask = self.atomic_numbers == iua
                _map = 0.5 * (1.0 + torch.erf(
                    (_space - self.atomic_numbers[_mask].unsqueeze(-1)) / (2.0 * _sigma)))
                _g1[_mask] = _map[..., 1:] - _map[..., :-1]
            if self.form == 'atom':
                _g1 = _g1[self.atomic_numbers.ne(0)]

        return _g1  #/ torch.max(_g1)

    def _get_g2(self, g: Tensor, g2: Tensor = None):
        assert len(g2) == 4, f'len(g2) should be 4, but get {len(g2)}'
        _min, _max, _len, _sigma = g2
        if _min != 0:
            warnings.warn('min of g2 is not zero, reset it as 0')
            _min = 0
        assert _max > _min, 'max in g2 is smaller than 0'

        # Reset self.distances so that it ranges from 0 to _max
        _dist = self.distances / (torch.max(self.distances) / _max)

        # To get erf function difference, we need extra one length
        dx = (_max - _min) / _len
        _space = torch.linspace(_min - dx, _max + dx, _len + 1)

        _map = []
        if not self.batch:
            # Use erf function difference to calculate delta gussian
            for iuap in self.unique_atom_pairs:
                _mask = (self.atomic_pairs == iuap).sum(-1) == 2
                _idist = _dist[_mask]
                _map.append(0.5 * (1.0 + torch.erf(
                    (_space - _idist[_idist.ne(0)].unsqueeze(1)) / (2.0 * _sigma))))
            # _map = [0.5 * (1.0 + torch.erf(
            #     (_space - self.distances[  # ↓ mask over distances
            #         (self.atomic_pairs == iuap).sum(-1) == 2].unsqueeze(1))
            #     / (2.0 * _sigma)))
            # for iuap in _unique_atom_pairs]
            _g2 = torch.cat(
                [1 / (pi2 * _sigma) * (im[..., 1:] - im[..., :-1]).sum(0)
                 for im in _map])
            return torch.cat([g, _g2 / torch.max(_g2)])
        else:
            for ii, distance in enumerate(_dist):
                _imap = []
                for iuap in self.unique_atom_pairs:
                    _mask = (self.atomic_pairs == iuap).sum(-1) == 2
                    _idist = distance[_mask[ii]]
                    _imap.append(0.5 * (1.0 + torch.erf(
                        (_space - _idist[_idist.ne(0)].unsqueeze(1)) / (2.0 * _sigma))))
                _g2 = torch.cat(
                    [1 / (pi2 * _sigma) * (im[..., 1:] - im[..., :-1]).sum(0)
                     for im in _imap])

                _map.append(torch.cat([g[ii], _g2]))
            _g = pack(_map)
            return _g / torch.max(_g[..., g.shape[-1]:])

    def _get_g3(self, g, g3: Tensor = None):
        assert len(g3) == 4, f'len(g3) should be 4, but get {len(g3)}'
        _min, _max, _len, _sigma = g3
        assert _min >= -1 and _min < 1, 'min out of range (0, 1)'
        assert _max > _min, 'max is smaller than min'

        # To get erf function difference, we need extra one length
        dx = (_max - _min) / _len
        _space = torch.linspace(_min - dx, _max + dx, _len + 1)

        # For convenience, transfer single to batch
        if self.distances.dim() == 3:
            _dist = self.distances
            _d_vect = self.geometry.distance_vectors
            _atomic_numbers = self.atomic_numbers
            _atomic_pairs = self.atomic_pairs
        else:
            _dist = self.distances.unsqueeze(0)
            _d_vect = self.geometry.distance_vectors.unsqueeze(0)
            _atomic_numbers = self.atomic_numbers.unsqueeze(0)
            _atomic_pairs = self.atomic_pairs.unsqueeze(0)
            g = g.unsqueeze(0)

        d_vect_ijk = (_d_vect.unsqueeze(-2) * _d_vect.unsqueeze(-3)).sum(-1)

        # the dimension of d_ij * d_ik is [n_batch, n_atom_ij, n_atom_jk]
        dist_ijk = _dist.unsqueeze(-1) * _dist.unsqueeze(-2)
        # dist2_ijk = _dist.unsqueeze(-1) ** 2 + _dist.unsqueeze(-2) ** 2

        # create the terms in G4 or G5
        cos = torch.zeros(dist_ijk.shape)
        mask = dist_ijk.ne(0)
        cos[mask] = d_vect_ijk[mask] / dist_ijk[mask]

        # Set only lower diagonal is not zero to avoid repeat calculations
        ut = torch.unbind(torch.triu_indices(cos.shape[-1], cos.shape[-1], 0))
        cos.permute(2, 3, 0, 1)[ut] = 0.0

        # THIS could be improved by parallel or cpython
        _g3 = []
        for ii, (icos, number, atom_pair) in enumerate(
                zip(cos, _atomic_numbers, _atomic_pairs)):
            _map = []
            for iuan in self.unique_atomic_numbers:
                icos_this_u = icos[iuan == number]
                for iuap in self.unique_atom_pairs:

                    _mask = (atom_pair == iuap).sum(-1) == 2
                    _icos = icos_this_u.permute(1, 2, 0)[_mask].T
                    _map.append(0.5 * (1.0 + torch.erf(
                        (_space - _icos[_icos.ne(0)].unsqueeze(1)) / (2.0 * _sigma))))
                    _ig3 = torch.cat(
                        [1 / (pi2 * _sigma) * (im[..., 1:] - im[..., :-1]).sum(0)
                         for im in _map])

            _g3.append(torch.cat([g[ii], _ig3]))

        return pack(_g3).squeeze()
