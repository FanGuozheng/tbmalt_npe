#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""An example to show how to transfer Slater-Koster files to h5 files.

The `std` means standard Slater-Koster files which are the same in DFTB+.
The 'vcr' means the Slater-Koster files with various compression radii in
basis for each atom, this allow to train each atom locally.

"""
import os
import torch
from tbmalt.ml.skfeeds import SkfFeed, VcrFeed
torch.set_default_dtype(torch.float64)

## Param  ##
task = 'std'
############


def std(device):
    """Read, write Slater-Koster data from standard and h5 format."""
    path = '../tests/unittests/data/slko/mio'
    shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}
    ele_nums = torch.tensor([1, 6])
    std = SkfFeed.from_dir(
        path, shell_dict, skf_type='skf', element_numbers=ele_nums,
        integral_type='H', write_h5=True)

    path = './skf.hdf'
    std = SkfFeed.from_dir(
        path, shell_dict, skf_type='h5', element_numbers=ele_nums,
        integral_type='H')

    print('The keys of Hamiltonian table:', std.off_site_dict.keys())


def vcr(device):
    """Read, write Slater-Koster data with various compression radii
    from standard and h5 format."""
    path = './data/vcr/'
    path2 = '../tests/unittests/data/slko/mio'
    shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}
    vcr = torch.tensor([3., 4.])
    ele_nums = torch.tensor([1, 6])
    vcr = VcrFeed.from_dir(
        path, shell_dict, vcr=vcr, skf_type='skf', element_numbers=ele_nums,
        path_homo=path2, integral_type='H', interpolation='BicubInterp',
        write_h5=True)

    path = './vcr.hdf'
    vcr = VcrFeed.from_dir(
        path, shell_dict, vcr=vcr, skf_type='h5', element_numbers=ele_nums,
        integral_type='H', interpolation='BicubInterp')

    print('The keys of Hamiltonian table:', vcr.off_site_dict.keys())


if __name__ == '__main__':
    locals()[task](torch.device('cpu'))
