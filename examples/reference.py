#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""An example to show how to generate DFT reference for training."""
import torch
import os
import torch
from tbmalt.utils.reference.write_reference import CalReference
from tbmalt.io.hdf import LoadHdf


task = 'dftbplus'
# For FHI-aims calculations, the executable binary and species_defaults should
# be defined, because the two files/directories are too huge, it is not offered
# in github, instead the generated reference is offered
path_to_aims = os.path.join(os.getcwd(), '../tests/unittests/aims/aims.x')
path_to_aims_specie = os.path.join(
    os.getcwd(), '../tests/unittests/aims/species_defaults/tight')
# Downloads: https://figshare.com/articles/dataset/ANI-1_data_set_20M_DFT_energies_for_non-equilibrium_small_molecules/5287732?backTo=/collections/_/3846712
path_to_input = '../../database/ani1/ani_gdb_s01.h5'
assert os.path.isfile(path_to_aims), 'There is no executable binary' + \
    f' FHI-aims named `aims.x` in {path_to_aims}'
assert os.path.isfile(path_to_input), f'file {path_to_input} is not found'
assert os.path.isdir(path_to_aims_specie), f'{path_to_aims_specie}' + \
    ' is not found for FHI-aims calculations.'

# For dftb, the dataset is too huge, the link to download dataset is here:
# https://springernature.figshare.com/collections/The_ANI-1ccx_and_ANI-1x_data_sets_coupled-cluster_and_density_functional_theory_properties_for_molecules/4712477
if task == 'dftbplus':
    path_to_dftbplus = os.path.join(
        os.getcwd(), '../tests/unittests/dftbplus/dftb+')
    path_to_input = '../../database/ani1x-release.h5'
    path_to_skf = os.path.join(
        os.getcwd(), '../tests/unittests/data/slko/mio/')
    assert os.path.isfile(path_to_dftbplus), 'There is no executable ' + \
        f'binary DFTB+ named `dftb+` in {path_to_dftbplus}'
    assert os.path.isfile(
        path_to_input), f'file {path_to_input} is not found'
    assert os.path.isdir(path_to_skf), f'{path_to_skf} is not found.'


def aims(device):
    """FHI-aims reference."""
    properties = ['charge', 'homo_lumo', 'energy']
    dataset_type = 'ANI-1'
    # How many to calculate, how many to load for test
    n_per_mol, reference_size = 6, 6
    periodic = False
    calculator = 'aims'
    output_name = 'aims.hdf'

    w_aims_ani1 = CalReference(
        path_to_input, dataset_type, n_per_mol, calculator,
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


def dftbplus(device):
    """DFTB+ reference, here the dataset is ANI-x."""
    properties = ['charge', 'homo_lumo', 'energy']
    dataset_type = 'ANIx'
    # How many to calculate, how many to load for test
    n_per_mol, reference_size = 10, 100
    periodic = False
    calculator = 'dftbplus'
    output_name = 'dftb.hdf'

    w_aims_ani1 = CalReference(
        path_to_input, dataset_type, n_per_mol, calculator,
        path_to_skf=path_to_skf, path_to_dftbplus=path_to_dftbplus)

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


if __name__ == '__main__':
    locals()[task](torch.device('cpu'))
