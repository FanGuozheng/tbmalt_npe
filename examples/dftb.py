#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Examples to perform DFTB1 and DFTB2 calculations."""
import torch
from ase.build import molecule
from tbmalt import Geometry, Dftb1, Dftb2
torch.set_default_dtype(torch.float64)


##############
# Parameters #
##############
task = 'dftb2'
path_to_skf = '../tests/unittests/data/slko/mio'
shell_dict = {1: [0], 7: [0, 1], 8: [0, 1]}


def dftb1(device=torch.device('cpu')):
    "Non-SCC DFTB calculations"
    numbers = torch.tensor([[8, 1, 1]])
    positions = torch.tensor([[[0., 0., 0.119262], [0., 0.763239, -0.477047],
                               [0., -0.763239, -0.477047]]])
    geometry = Geometry(numbers, positions, units='angstrom',  device=device)

    # Create DFTB2 object
    dftb2 = Dftb1(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')

    # Run SCC DFTB calculations
    dftb2()
    print('Calculated single water Mulliken charges:', dftb2.charge)

    # Create bacth geometry object
    numbers = torch.tensor([[8, 1, 1, 0], [7, 1, 1, 1]])
    positions = torch.tensor([
        [[0., 0., 0.119262], [0., 0.763239, -0.477047],
         [0., -0.763239, -0.477047], [0., 0., 0.]],
        [[0., 0., 0.116489], [0., 0.939731, -0.271808],
         [0.813831, -0.469865, -0.271808], [-0.813831, -0.469865, -0.271808]]])
    geometry = Geometry(numbers, positions, units='angstrom',  device=device)

    # Create DFTB2 object
    dftb2 = Dftb1(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')

    # Run SCC DFTB calculations
    dftb2()
    print('Calculated [H2O, NH3] Mulliken charges:', dftb2.charge)


def dftb2(device=torch.device('cpu')):
    "SCC DFTB calculations, where molecules are from ase."
    # Create single geometry object
    geometry = Geometry.from_ase_atoms([molecule('H2O')], device=device)

    # Create DFTB2 object
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')

    # Run SCC DFTB calculations
    dftb2()
    print('Calculated single water Mulliken charges:', dftb2.charge)

    # Create bacth geometry object
    geometry = Geometry.from_ase_atoms([
        molecule('H2O'), molecule('NH3')], device=device)

    # Create DFTB2 object
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf')

    # Run SCC DFTB calculations
    dftb2()
    print('Calculated [H2O, NH3] Mulliken charges:', dftb2.charge)


if __name__ == '__main__':
    locals()[task]()
