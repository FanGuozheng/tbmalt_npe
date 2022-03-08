#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of many-body tensor representation (MBTR)."""
import warnings
from typing import Literal
import torch
import numpy as np
from torch import Tensor
from tbmalt import Shell, Geometry, Periodic
from tbmalt.structures.geometry import unique_atom_pairs
from tbmalt.common.batch import pack

# Define parameters
pi2 = np.sqrt(2 * np.pi)


class Mbtr:
    """Implementation of many-body tensor representation (MBTR).

    Arguments:
        geometry: Geometry from TBMaLT.
        shell: Shell object feom TBMaLT.

    Reference:

    """

    def __init__(self,
                 geometry: Geometry,
                 shell: Shell,
                 g1: list,
                 g2: Tensor = None,
                 g3: Tensor = None,
                 form: Literal['atom', 'geometry', 'distance'] = 'geometry',
                 **kwargs):
        self.geometry = geometry
        self.isperiodic = self.geometry.isperiodic
        self.batch = True if geometry.atomic_numbers.dim() == 2 else False

        if not self.batch:
            self.atomic_numbers = self.geometry.atomic_numbers.unsqueeze(0)
            self.distances = self.geometry.distances.unsqueeze(0)
            self.d_vect = self.geometry.distance_vectors.unsqueeze(0)
            self.atomic_pairs = shell.atomic_number_matrix(
                'atomic').unsqueeze(0)
        else:
            self.atomic_numbers = self.geometry.atomic_numbers
            self.distances = self.geometry.distances
            self.d_vect = self.geometry.distance_vectors
            self.atomic_pairs = shell.atomic_number_matrix('atomic')

        self.fc = torch.exp(2-self.distances * 0.5)
        self.n_batch = len(self.atomic_numbers)

        self.unique_atomic_numbers = self.geometry.unique_atomic_numbers
        self.unique_atom_pairs = unique_atom_pairs(
            self.geometry, repeat_pairs=False)
        self.shell = shell
        self.form = form

        if self.isperiodic:
            self.periodic = Periodic(geometry=self.geometry,
                                     latvec=self.geometry.cell,
                                     cutoff=kwargs.get('pe_cutoff', 10.0))
            self.fc_pe = torch.exp(2-self.periodic.distance_vectors * 0.5)

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
        _min, _max, _len, sigma = g1

        # To get erf function difference, we need extra one length
        dx = (_max - _min) / _len
        _space = torch.linspace(_min - dx, _max + dx, _len + 1)

        # Use erf function difference to calculate delta Gaussian distribution
        _map = []
        if self.form in ('atom', 'geometry'):
            for iua in self.unique_atomic_numbers:
                _mask = self.atomic_numbers == iua
                imap = 0.5 * (1.0 + torch.erf(
                    (_space - self.atomic_numbers[_mask].unsqueeze(-1)) / (2.0 * sigma)))
                _map.append(
                    pack(torch.split(imap, tuple(_mask.sum(-1)))).sum(1))

            g1 = torch.stack([(im[..., 1:] - im[..., :-1]) for im in _map], 1)
            if self.form == 'atom':
                g1 = g1[self.atomic_numbers.ne(0)]

        elif self.form == 'distances':
            imap = 0.5 * (1.0 + torch.erf(
                (_space - self.atomic_pairs.unsqueeze(-1)) / (2.0 * sigma)))
            g1 = (imap[..., 1:] - imap[..., :-1]).sum(-2)

        else:
            raise ValueError(f'Unkown form {self.form}')

        return g1  # / torch.max(_g1)

    def _get_g2(self, g: Tensor, g2: Tensor = None):
        """Get two-body tensor with distances."""
        assert len(g2) == 4, f'len(g2) should be 4, but get {len(g2)}'
        _min, _max, _len, sigma = g2
        if _min != 0:
            warnings.warn('min of g2 is not zero, reset it as 0')
            _min = 0
        assert _max > _min, 'max in g2 is smaller than 0'

        # Reset self.distances so that it ranges from 0 to _max
        _dist = self.distances / (torch.max(self.distances) / _max)

        # To get erf function difference, we need extra one length
        dx = (_max - _min) / _len
        _space = torch.linspace(_min - dx, _max + dx, _len + 1)

        if self.form in ('atom', 'geometry'):
            _map = []
            for iuap in self.unique_atom_pairs:
                _mask = ((self.atomic_pairs == iuap).sum(-1) == 2) * _dist.ne(0)
                imap = 1.0 / (pi2 * sigma) * 0.5 * (1.0 + torch.erf(
                    (_space - _dist[_mask].unsqueeze(1)) / (2.0 * sigma)))

                # Split the atomic pair distances in each geometries, return
                # shape from [n_pair, n_map] -> [n_batch, n_map]
                _map.append(pack(torch.split(imap, tuple(
                    _mask.sum(-1).sum(-1)))).sum(1))
                _g2 = torch.stack([(im[..., 1:] - im[..., :-1])
                                   for im in _map]).transpose(1, 0)

        elif self.form == 'distances':
            imap = 0.5 * (1.0 + torch.erf(
                (_space - self.distances.unsqueeze(-1)) / (2.0 * sigma)))
            _g2 = (imap[..., 1:] - imap[..., :-1])
            print('imap', imap.shape, _g2.shape)

        else:
            raise ValueError(f'Unkown form {self.form}')

        return torch.cat([g, _g2 / torch.max(_g2)], 1)

    def _get_g3(self, g, g3: Tensor = None, smear=10.0):
        """Get three-body tensor with angular parameters."""
        assert len(g3) == 4, f'len(g3) should be 4, but get {len(g3)}'
        _min, _max, _len, sigma = g3
        assert _min >= -1 and _min < 1, 'min out of range (0, 1)'
        assert _max > _min, 'max is smaller than min'

        # To get erf function difference, we need extra one length
        dx = (_max - _min) / _len
        _space = torch.linspace(_min - dx, _max + dx, _len + 1)
        pad_value = (_max + dx) * smear + 1

        # For convenience, transfer single to batch
        assert self.distances.dim() == 3
        _dist = self.distances
        _atomic_pairs = self.atomic_pairs
        d_vect_ijk = (self.d_vect.unsqueeze(-2) *
                      self.d_vect.unsqueeze(-3)).sum(-1)

        # the dimension of d_ij * d_ik is [n_batch, n_atom_ij, n_atom_jk]
        dist_ijk = _dist.unsqueeze(-1) * _dist.unsqueeze(-2)
        # dist2_ijk = _dist.unsqueeze(-1) ** 2 + _dist.unsqueeze(-2) ** 2

        # create the terms in G4 or G5
        # Set initial values as 2 is to exclude Atom1-Atom1 like angle
        cos = torch.ones(dist_ijk.shape) * pad_value
        mask = dist_ijk.ne(0)
        cos[mask] = d_vect_ijk[mask] / dist_ijk[mask]

        # Set only lower diagonal is not zero to avoid repeat calculations
        ut = torch.unbind(torch.triu_indices(cos.shape[-1], cos.shape[-1], 0))
        cos.permute(2, 3, 0, 1)[ut] = pad_value

        # THIS could be improved by parallel or cpython
        g3, _map, uniq_atom_pairs = [], [], []
        for i, ian in enumerate(self.unique_atomic_numbers):
            for j, jan in enumerate(self.unique_atomic_numbers[i:]):
                uniq_atom_pairs.append(torch.tensor([ian, jan]))
        uniq_atom_pairs = pack(uniq_atom_pairs)

        for u_atom_pair in uniq_atom_pairs:
            # if not self.isperiodic:
            #     ig = torch.ones(*self.atomic_numbers.shape) * 2
            # else:
            #     ig = torch.ones(self.pe_atomic_numbers.shape) * 2

            # Select ALL the interactions with u_atom_pair
            _im = torch.nonzero((self.atomic_pairs == u_atom_pair).all(dim=-1))

            # If atom pair is not homo, we have to consider inverse u_atom_pair
            if u_atom_pair[0] != u_atom_pair[1]:
                _im = torch.cat(
                    [_im, torch.nonzero((self.atomic_pairs == u_atom_pair.flip(0)).all(dim=-1))])
                _im = _im[_im[..., 0].sort()[1]]

            # Select last two dims which equals to atom-pairs in _im
            g_im = cos[_im[..., 0], :, _im[..., 1], _im[..., 2]]
            _imask, count = torch.unique_consecutive(
                _im[..., 0], return_counts=True)

            # If there is such atom pairs
            _g3 = []
            if count.shape[0] > 0:
                for jj in self.unique_atomic_numbers:
                    _g3.append(pack([ii[ii.le(1) * ia == jj] for ii, ia in zip(
                        g_im.split(tuple(count)), self.atomic_numbers)], value=pad_value))

            _g3 = pack(_g3, value=pad_value)
            # For each geometry, add the same atom pair in the last 2 dimension
            _map.append(1.0 / (pi2 * sigma) * 0.5 *
                        (1.0 + torch.erf((_space - _g3.unsqueeze(-1)) * smear) /
                         (2.0 * sigma)).sum(-2).transpose(1, 0))

        g3 = torch.cat([(im[..., 1:] - im[..., :-1]) for im in _map], -2)

        return torch.cat([g, g3], -2).squeeze()
