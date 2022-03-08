#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate dataset for PyTorch training.

This example includes the following part:
    1. With FHI-aims input, perform DFT calculations.
    2. Read data from FHI-aims calculations.
    3. Write data as the TBMaLT and PyTorch format.

"""
import os
import time
import pickle
import torch
import numpy as np
from torch import Tensor
from tbmalt import Shell, Geometry, SkfFeed, SkfParamFeed, hs_matrix, Dftb1, Dftb2
from tbmalt.structures.periodic import Periodic
from tbmalt.io import Dataset, DataLoader
from tbmalt.ml.model import Hamiltonian
from tbmalt.ml import Mbtr, Cdt, Acsf
from tbmalt.common.batch import pack
torch.set_default_dtype(torch.float64)
torch.set_printoptions(4)


############# Params #############
task = {'pre_train': True, 'train': True, 'test': True}
train_s = True
train_onsite = True
parallel = True
path_sk = '../slko'
feature_type = 'acsf'
shell_dict = {1: [0], 3: [0, 1], 6: [0, 1], 14: [0, 1, 2]}
orbital_resolve = True
neig_resolve = True
n_valence = 3
n_conduction = 0
n_band0 = [0, 10, 20, 30, 40, 50, 60, 70, 80]
n_band1 = [1, 11, 21, 31, 41, 51, 61, 71, 81]


def train(device):
    # 4. Loading and Collect data
    dataset = Dataset.pkl(os.path.join(
        os.getcwd(), 'si.pkl'), 'pkl', ['band', 'occ'])
    with open('geometry.pkl', "rb") as f:
        geo_raw = pickle.load(f)
    loader = DataLoader(dataset, 2)
    geo = dataset.geometry
    print('new n_atoms', geo.n_atoms)
    indices = [2, 7, 9]
    max_atoms = torch.max(geo.n_atoms[indices])

    geometry, basis, klines, ref, skparams, periodic = _select_geometry(
        geo, geo_raw, dataset, indices, max_atoms)
    x_train = build_representation(geometry, basis, periodic)

    dftb = Dftb2(geometry, shell_dict, path_sk,
                 skf_type='skf', klines=klines, maxiter=1)
    dftb()

    atomic_feature = x_train.flatten(0, -2)
    # atomic_feature = stdnorm(atomic_feature)
    onsite_feature = x_train.sum(-2).sum(-2)[geometry.atomic_numbers.ne(0)]

    if task['pre_train']:
        train = Hamiltonian(n_feature=atomic_feature.shape[-1],  # + 1,
                            activation='ReLU',
                            nn_type='linear',
                            neig_resolve=neig_resolve,
                            dftb=dftb,
                            train_s=train_s,
                            train_onsite=train_onsite,
                            onsite_feature=onsite_feature)
        train.pre_train(atomic_feature, isperiodic=True)
        with open('pre_train.pkl', "wb") as f:
            pickle.dump(train, f)

    if task['train']:
        with open('pre_train.pkl', "rb") as f:
            train = pickle.load(f)
        if parallel:
            time0 = time.time()
            net = torch.nn.DataParallel(train, device_ids=[0, 1, 2, 3])
            net(atomic_feature, ref,
                isperiodic=True, n_band0=n_band0, n_band1=n_band1,
                n_valence=n_valence, n_conduction=n_conduction,
                klines=klines, path_to_skf=path_sk)
            time1 = time.time()
            print('time parallel:', time1 - time0)
        else:
            time0 = time.time()
            train(atomic_feature, ref,
                  isperiodic=True, n_band0=n_band0, n_band1=n_band1,
                  n_valence=n_valence, n_conduction=n_conduction)
            time1 = time.time()
            print('time serial:', time1 - time0)

        with open('train.pkl', "wb") as f:
            pickle.dump(train, f)

    if task['test']:
        indices = [10]  # [0, 2, 7]
        # print(geo_raw.obj_dict)
        max_atoms = torch.max(geo.n_atoms[indices])
        geometry, basis, klines, ref, skparams, periodic = _select_geometry(
            geo, geo_raw, dataset, indices, max_atoms)
        x_test = build_representation(geometry, basis, periodic)
        atomic_feature = x_test.flatten(0, -2)
        onsite_feature = x_test.sum(-2).sum(-2)[geometry.atomic_numbers.ne(0)]

        dftb = Dftb2(geometry, shell_dict, path_sk,
                     skf_type='skf', klines=klines, maxiter=1)
        dftb()
        with open('train.pkl', "rb") as f:
            train = pickle.load(f)
        train.test(X=atomic_feature, dftb_test=dftb, ref=ref,
                   onsite_feature=onsite_feature, n_band0=n_band0, n_band1=n_band1,
                   n_valence=n_valence, n_conduction=n_conduction)


def _band(dataset, indices, max_orbs,
          n_band0, n_band1):
    band0 = dataset.properties['band'][indices]
    band0 = band0[..., :max_orbs]

    # Select valence band and conduction band
    ind0 = torch.repeat_interleave(torch.arange(
        len(band0)), n_valence + n_conduction)
    n_val = band0.sum(-2).lt(0).sum(-1)
    ind_1 = (torch.repeat_interleave(n_val, n_valence + n_conduction) +
             torch.arange(-n_valence, n_conduction).repeat(len(band0))).flatten()
    band0 = pack(band0[ind0, :, ind_1].split(
        tuple((torch.ones(len(band0)) * (n_valence + n_conduction)).int()), 0))

    occ = dataset.properties['occ'][indices]
    occ = occ[..., :max_orbs]

    # import matplotlib.pyplot as plt
    # for iband in band0:
    #     print(iband.shape)
    #     plt.plot(torch.arange(len(iband.T)), iband.T)
    #     plt.ylim(-5, 4)
    #     plt.show()

    band = band0[..., n_band0]
    delta_band = band0[..., n_band1] - band0[..., n_band0]

    return band0, band, delta_band


def _select_geometry(geo, geo_raw, dataset, indices, max_atoms):
    geometry = Geometry(geo.atomic_numbers[indices][..., :max_atoms],
                        geo.positions[indices][:, :max_atoms],
                        cell=geo.cell[indices],
                        units='bohr')
    print('geo.n_atoms', geometry.n_atoms)

    # band0, band, delta_band = _band(dataset.properties['band'], dataset.properties['occ'])
    basis = Shell(geometry.atomic_numbers, shell_dict)
    obj_dict = {ii: geo_raw.obj_dict[ii] for ii in indices}
    klines = pack([pack([torch.from_numpy(np.concatenate([ii, jj, np.array([10])]))
                   for ii, jj in zip(val['kpts'][:-1], val['kpts'][1:])])
                   for key, val in obj_dict.items()])[:, :len(n_band0)]

    band0, band, delta_band = _band(
        dataset, indices,
        max_orbs=basis.atomic_number_matrix('full').shape[-2],
        n_band0=n_band0, n_band1=n_band1)
    ref = {'tot_band': band0, 'band': band, 'delta_band': delta_band}

    skparams = SkfParamFeed.from_dir(path_sk, geometry, skf_type='skf')
    periodic = Periodic(geometry, geometry.cell,
                        cutoff=skparams.cutoff, klines=klines)

    return geometry, basis, klines, ref, skparams, periodic


def build_representation(geometry, basis, periodic):
    # Set representation
    if feature_type == 'acsf':
        acsf = Acsf(geometry=geometry,
                    basis=basis,
                    shell_dict=shell_dict,
                    g1_params=6.0,
                    g2_params=torch.tensor([0.5, 1.0]),
                    g5_params=torch.tensor([0.02, 1.0, -1.0]),
                    form='distance',
                    units='angstrom',
                    element_resolve=True,
                    periodic=periodic)
        acsf()
        x_train = acsf.g.permute(0, 2, 3, 1, -1)
    elif feature_type == 'mbtr':
        mbtr = Mbtr(
            geometry, basis,
            g1=[0, 15, 100, 0.1],
            g2=[0, 1, 100, 0.1],
            g3=[-1, 1, 100, 0.1]
        )

        # mbtr = Mbtr(
        #     periodic, basis, g1=[0, 8, 100, 0.1])
        x_train = mbtr.g
    elif feature_type == 'cdt':
        cdt = Cdt(
            periodic, basis, {1: 1}, g2=[0, 12, 100, 0.1])
        x_train = cdt.constant_distribution
    return x_train


def stdnorm(input: Tensor):
    assert input.dim() == 2, 'dimension error'
    size = torch.numel(input)
    mean = input.sum() / size
    standard_deviation = torch.sqrt(torch.sum((input - mean) ** 2) / size)
    return (input - mean) / standard_deviation


if __name__ == '__main__':
    train(torch.device('cpu'))
