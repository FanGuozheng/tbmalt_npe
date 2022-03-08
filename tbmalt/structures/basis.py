#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 19:35:17 2022

@author: gz_fan
"""
from typing import Literal, Dict

import numpy as np
import torch
from torch import nn, Tensor
from scipy.special import factorial2 as fact2

from tbmalt import Geometry
from tbmalt.structures.geometry import unique_atom_pairs
from tbmalt.data.gto import _basis_type, _shell_dict, _get_coefficients
from tbmalt.common.maths.ints import int_over


class Basis(nn.Module):
    """Construct various basis."""

    def __init__(self,
                 geometry: Geometry,
                 shell_dict: Dict,
                 basis_type: Literal[_basis_type.keys()]):
        super(Basis, self).__init__()
        self.geometry = geometry
        self.atomic_numbers = self.geometry.atomic_numbers
        self.positions = self.geometry.positions
        self.shell_dict = shell_dict
        self._build_shell()
        self.coefficients = _get_coefficients(
            self.geometry.unique_atomic_numbers, basis_type)

        self._norm()

    def _norm(self):
        """Normalization of basis coefficients."""
        self.norm = {}
        print('self.shell', self.shell)
        for uan in self.geometry.unique_atomic_numbers:
            l, m, n = self.shell[int(uan)][..., 0], \
                self.shell[int(uan)][..., 1], self.shell[int(uan)][..., 2]
            alpha = self.coefficients[int(uan)][0][..., 0]

            # The fact2(lmn) should be without gradient here
            print('alpha', alpha.shape, l.shape)
            self.norm.update({int(uan): torch.sqrt(
                (2 * alpha / np.pi) ** 1.5 * (4 * alpha) ** (l + m + n)
                / fact2(2 * l - 1) / fact2(2 * m - 1) / fact2(2 * n - 1))})

    def forwad(self):
        pass

    def _get_S(self) -> Tensor:
        """Build overlap integrals from two `Basis` object."""
        for uap in unique_atom_pairs(self.geometry).tolist():
            ind_atom1 = uap[0] == self.atomic_numbers
            ind_atom2 = uap[1] == self.atomic_numbers
            atom1 = self.atomic_numbers[ind_atom1]
            atom2 = self.atomic_numbers[ind_atom2]

            coef1=self.coefficients[uap[0]][0][..., 1]
            coef2=self.coefficients[uap[1]][0][..., 1]
            alpha1=self.coefficients[uap[0]][0][..., 0]
            alpha2=self.coefficients[uap[1]][0][..., 0]

            # Repeat for each atom with standard basis input
            if (coef1).dim() == 1:
                coef1 = coef1.repeat(len(atom1), 1)
                coef1 = coef1.repeat(len(atom2), 1)
            if (alpha1).dim() == 1:
                alpha1 = alpha1.repeat(len(atom1), 1)
                alpha2 = alpha2.repeat(len(atom2), 1)

            S = int_over(self.norm[uap[0]], self.norm[uap[1]],
                         coef1, coef2, alpha1, alpha2,
                         self.shell[uap[0]], self.shell[uap[1]],
                         position1=self.positions[ind_atom1],
                         position2=self.positions[ind_atom2]
                         )
            print('S', S)

    def _get_H(self):
        pass

    def _get_K(self):
        pass

    def _build_shell(self):
        """"""
        self.shell = {}
        for uan in self.geometry.unique_atomic_numbers.tolist():
            if max(self.shell_dict[uan]) == 0:
                self.shell.update({uan: _shell_dict[0]})
            elif max(self.shell_dict[uan]) == 1:
                self.shell.update({uan: _shell_dict[0]})


class GTO(Basis):
    """"""

    def __init__(self,
                 geometry: Geometry,
                 shell_dict: Tensor,
                 basis_type: Literal[_basis_type.keys()] = 'STO-3G'):
        super().__init__(geometry, shell_dict, basis_type)


class STO:

    def __init__(self):
        pass


if __name__ == '__main__':
    """"""
    geo = Geometry(atomic_numbers=torch.tensor([1]),
                   positions=torch.tensor([[0., 0., 0.]]))
    shell_dict = {1: [0], 14: [0]}
    G1 = GTO(geo, shell_dict=shell_dict, basis_type='STO-3G')

    assert (torch.abs(G1.norm[1] - torch.tensor(
        [1.794441828121703, 0.500326475188964, 0.187735459537730])
        ) < 1E-6).all(), 'tolerance check of normalization in basis fails'

    geo = Geometry(atomic_numbers=torch.tensor([14]),
                   positions=torch.tensor([[0., 0., 0.]]))
    G1 = GTO(geo, shell_dict=shell_dict, basis_type='STO-3G')

    G1._get_S()

    from pyscf import gto
    mol = gto.Mole()
    mol.build(
        atom = '''Si 0 0 1''',
        basis = 'sto-3g')
    overlap = mol.intor('int1e_ovlp')
    print('overlap', overlap.shape)
