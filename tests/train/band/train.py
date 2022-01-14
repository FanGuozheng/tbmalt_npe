#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate dataset for PyTorch training.

This example includes the following part:
    1. With FHI-aims input, perform DFT calculations.
    2. Read data from FHI-aims calculations.
    3. Write data as the TBMaLT and PyTorch format.

"""
import os
import pickle
import torch
from tbmalt import Basis, Geometry, SkfFeed, SkfParamFeed, hs_matrix, Dftb1, Dftb2
from tbmalt.structures.periodic import Periodic
from tbmalt.io import Dataset, DataLoader
from tbmalt.ml.model import Hamiltonian
from tbmalt.ml import Acsf, Mbtr, Cdt
from tbmalt.common.batch import pack
from tbmalt.ml.feature import Dscribe
torch.set_default_dtype(torch.float64)
torch.set_printoptions(2)


############# Params #############
path_sk = '../slko'
descriptor_package = 'tbmalt'  # tbmalt, scikit-learn
feature_type = 'acsf'
shell_dict = {1: [0], 3: [0, 1], 6: [0, 1], 14: [0, 1, 2]}
orbital_resolve = True
neig_resolve = True
add_k = False
gen_ref, train = False, True


def train(device):
    # 4. Loading and Collect data
    dataset = Dataset.pkl(os.path.join(os.getcwd(), 'si.pkl'), 'pkl', ['band', 'occ'])
    loader = DataLoader(dataset, 2)
    geometry = dataset.geometry()
    band0, band, delta_band = _band(dataset.properties['band'], dataset.properties['occ'])
    basis = Basis(geometry.atomic_numbers, shell_dict)
    skparams = SkfParamFeed.from_dir(path_sk, geometry, skf_type='skf')
    # klines = torch.tensor([[[0., 0., 0., 0], [0.5, 0.5, 0.5, 20]]]).repeat(
    #     len(geometry.n_atoms), 1, 1)
    klines = torch.tensor([[[0., 0., 0., 0.5, 0.0, 0.0, 20]]]).repeat(
        len(geometry.n_atoms), 1, 1)

    # klines = torch.tensor([
    #     [[0., 0., 0., 0], [0.05, 0., 0., 2]],
    #     [[0.25, 0., 0., 0], [0.3, 0., 0., 2]],
    #     [[0., 0., 0., 0], [0.05, 0., 0., 2]],
    #     [[0.25, 0., 0., 0], [0.3, 0., 0., 2]]])

    periodic = Periodic(geometry, geometry.cell, cutoff=skparams.cutoff, klines=klines)

    # Set representation
    if descriptor_package == 'scikit-learn':
        X0 = Dscribe(geometry, feature_type=feature_type).features
    elif descriptor_package == 'tbmalt':
        if feature_type == 'acsf':
            acsf = Acsf(periodic, basis, shell_dict,
                        g1_params=6.0,
                        g2_params=torch.tensor([0.5, 1.0]),
                        # g4_params=torch.tensor([0.02, 1.0, -1.0]),
                        form='distance',
                        atom_like=False,
                        units='angstrom')
            acsf()
            x_train = acsf.g
        elif feature_type == 'mbtr':
            mbtr = Mbtr(
                periodic, basis, g1=[0, 8, 100, 0.1])
            x_train = mbtr.g
        elif feature_type == 'cdt':
            cdt = Cdt(
                periodic, basis, {1: 1}, g2=[0, 12, 100, 0.1])
            x_train = cdt.constant_distribution

    dftb = Dftb2(geometry, shell_dict, path_sk, skf_type='skf', klines=klines, maxiter=1)
    dftb()

    # X0 = norm_standardization(X0)
    # atomic_feature = x_train[mask_nums_block].flatten(0, -2)
    # _mask_orb = dftb.basis.orbs_per_atom[dftb.basis.orbs_per_atom.ne(0)]
    atomic_feature = x_train.flatten(0, -2)
    print('x_train', x_train.shape, 'atomic_feature', atomic_feature.shape)
    train = Hamiltonian(
        n_feature=atomic_feature.shape[-1], # + 1,
        activation='ReLU',
        nn_type='linear',
        neig_resolve=neig_resolve,
        dftb=dftb)
    train.pre_train(atomic_feature,
                    isperiodic=True)
    with open('train.pkl', "wb") as f:
        pickle.dump(train, f)

    train(atomic_feature,
          ref={'tot_band': band0, 'band': band, 'delta_band': delta_band},
          isperiodic=True)


def _band(band0, occ, n_band=[0, 20, 40, 60], n_band1=[1, 21, 41, 61]):
    # Make first band as zero
    band0[0] = band0[0] - torch.max(band0[0, :20, 13])
    # band0[1] = band0[1] - torch.max(band0[1, :20, 41])

    band = pack([band0[0, [0, 10, 18]][..., 10: 17]])
                 # band0[1, [0, 10]][..., 34: 45]])
    band1 = pack([band0[0, [1, 11, 19]][..., 10: 17]])
                  # band0[1, [1, 11]][..., 34: 45]])

    delta_band = band1 - band

    # band = pack([band0[0, [0, 10]][..., 10: 17]])
    # band1 = pack([band0[0, [1, 11]][..., 10: 17]])

    # delta_band = band1 - band

    return band0[:, :20], band, delta_band

if __name__ == '__main__':
    train(torch.device('cpu'))
