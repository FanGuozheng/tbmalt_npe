#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example to run training."""
import os
import torch

from tbmalt.structures.geometry import Geometry
from tbmalt.ml.optim import OptSpline, OptVcr
from tbmalt.io.hdf import LoadHdf
torch.set_printoptions(15)
torch.set_default_dtype(torch.float64)


###################
# optimize params #
###################
task = 'mlvcr'
params = {'targets': ['charge'],
          'max_steps': 8,
          'loss_function': 'MSELoss',
          'optimizer': 'Adam',
          'charge_weight': 1.0,
          'min_steps': 6,
          'comp_r_min': 3.0,
          'comp_r_max': 4.0}
vcr = torch.tensor([3., 4.])
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}

# Check reference, see `referece.py` more about the generation of reference
dataset_aims = './aims.hdf'
assert os.path.isfile(dataset_aims), f'There is no reference: {dataset_aims}'

# Check Slater-Koster files, see `skf.py` more about the generation of SKF
path_to_vcr = './vcr.hdf'
assert os.path.isfile(path_to_vcr), f'There is no reference: {path_to_vcr}'


def mlspline(dataset_ref, size: int = 300, **kwargs):
    """Optimize spline parameters or compression radii."""
    params['lr'] = 0.001
    geo_opt, data_ref = _load_ref(dataset_ref, size, ['charge'])

    # optimize integrals with spline parameters
    path_to_skf = './skf.hdf'
    opt = OptSpline(geometry=geo_opt,
                    reference=data_ref,
                    path_to_skf=path_to_skf,
                    parameter=params,
                    shell_dict=shell_dict,
                    train_onsite=False)
    opt()


def mlvcr(dataset_ref, size: int = 300, dataset_dftb=None, **kwargs):
    """Optimize spline parameters or compression radii."""
    params['lr'] = 0.01

    geo_opt, data_ref = _load_ref(dataset_ref, size, ['charge'])
    if dataset_dftb is not None:
        geo_dftb, data_dftb = _load_ref(dataset_dftb, size, ['charge'])

    # optimize integrals with compression radii
    path_to_skf_homo = './skf.hdf'
    opt = OptVcr(geometry=geo_opt,
                 reference=data_ref,
                 path_to_skf_vcr=path_to_vcr,
                 path_to_skf_homo=path_to_skf_homo,
                 parameter=params,
                 vcr=vcr,
                 shell_dict=shell_dict,
                 h_compr_feed=True,
                 s_compr_feed=True,
                 interpolation='BicubInterp',
                 global_r=True,
                 train_onsite=False)
    opt()


def _load_ref(dataset, size, properties, units='angstrom', **kwargs):
    """Helper function to load dataset, return `Geometry` object, data."""
    numbers, positions, data = LoadHdf.load_reference(
        dataset, size, properties)
    cell = kwargs.get('cell', None)
    geo = Geometry(numbers[:2], positions[:2], units=units, cell=cell)

    return geo, {'charge': data['charge'][:2]}


if __name__ == '__main__':
    locals()[task](dataset_aims)
