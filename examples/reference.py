#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""An example to show how to generate DFT reference for training."""
import torch
import os
import torch
from tbmalt.utils.reference.write_reference import CalReference
from tbmalt.io.hdf import LoadHdf


task = 'aims'
# For FHI-aims calculations, the executable binary and species_defaults should
# be defined, because the two files/directories are too huge, it is not offered
# in github, instead the generated reference is offered
path_to_aims = os.path.join(os.getcwd(), '../tests/unittests/aims/aims.x')
path_to_aims_specie = os.path.join(
    os.getcwd(), '../tests/unittests/aims/species_defaults/tight')
# Downloads: https://figshare.com/articles/dataset/ANI-1_data_set_20M_DFT_energies_for_non-equilibrium_small_molecules/5287732?backTo=/collections/_/3846712
path_to_input = '../../database/ani1/ani_gdb_s01.h5'
assert os.path.isfile(path_to_aims), 'There is no executable binary' + \
    'FHI-aims named `aims.x`'
assert os.path.isfile(path_to_input), f'file {path_to_input} is not found'
assert os.path.isdir(path_to_aims_specie), 'There is no' + \
    ' species_defaults/tight folder for FHI-aims calculations.'


def aims(device):
    """FHI-aims reference."""
    properties = ['charge', 'homo_lumo', 'energy']
    input_type = 'ANI-1'
    # How many to calculate, how many to load for test
    n_per_mol, reference_size = 6, 6
    periodic = False
    reference_type = 'aims'
    output_name = 'aims.hdf'

    w_aims_ani1 = CalReference(
        path_to_input, input_type, n_per_mol, reference_type,
        path_to_aims_specie=path_to_aims_specie, path_to_aims=path_to_aims)

    # calculate properties
    results = w_aims_ani1(properties, periodic=periodic)

    # write results (properties) to hdf
    CalReference.to_hdf(results, w_aims_ani1, properties, mode='w',
                        output_name=output_name)

    # test the hdf reference
    numbers, positions, data = LoadHdf.load_reference(
        output_name, reference_size, properties)

    # make sure the data type consistency
    LoadHdf.get_info(output_name)  # return dataset information


def dftb(device):
    """DFTB+ reference."""


if __name__ == '__main__':
    locals()[task](torch.device('cpu'))
