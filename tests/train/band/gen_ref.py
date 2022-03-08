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
import matplotlib.pyplot as plt
import torch
from tbmalt.io import GeometryTo
from tbmalt.utils.reference.calculator import Calculator
from tbmalt.ml.preprocessing import norm_standardization
torch.set_default_dtype(torch.float64)
torch.set_printoptions(4)

# 1. geometry parameters, including path to geometries
geometry_path = '../data/si'
to_geometry_path = os.path.join(os.getcwd(), './geometry')

# 2. path and some input files for FHI-aims
# control_in = os.path.join(os.getcwd(), './control.in')
aims_bin = os.path.join(os.getcwd(), './aims.x')
calculation_properties = ['band']
path_sk = '../../../tests/unittests/data/slko/mio'
##################################
descriptor_package = 'tbmalt'  # tbmalt, scikit-learn
feature_type = 'acsf'
shell_dict = {1: [0], 3: [0, 1], 6: [0, 1], 14: [0, 1]}
orbital_resolve = True
neig_resolve = True
add_k = False
gen_ref, train = False, True


def gen_reference(device):
    """Generate reference for traning."""
    # 1. Generate geometry
    # geometry_in_files = [os.path.join(geometry_path, ii)
    #                      for ii in os.listdir(geometry_path)]
    geometry1 = ['cubic_2.cif', 'cubic_8.cif',
                 'hexagonal_4.cif',
                 'monoclinic_4.cif', 'monoclinic_8.cif', 'monoclinic_16.cif',
                 'orthorhombic_2.cif', 'orthorhombic_4.cif', 'orthorhombic_8.cif',
                 'tetragonal_4.cif', 'tetragonal_8.cif']
    geometry_in_files = [os.path.join(geometry_path, ii) for ii in geometry1]
    geot = GeometryTo(geometry_in_files,
                      path_to_input_template=os.path.join(os.getcwd(), 'control.in'),
                      to_geometry_type='aims',
                      to_geometry_path=to_geometry_path,
                      calculation_properties=calculation_properties)
    geot()
    with open('geometry.pkl', 'wb') as f:
        pickle.dump(geot, f)

    # 2. Run FHI-aims calculations
    control_in = []
    geometry_in = []
    for file in os.listdir(to_geometry_path):
        if file.startswith('control.in'):
            control_in.append(file)
        elif file.startswith('geometry.in'):
            geometry_in.append(file)

    # 3. Run FHI-aims and read calculated results
    calculator = Calculator.aims(control_in=control_in,
                                 geometry_in=geometry_in,
                                 aims_bin=aims_bin,
                                 env_path=to_geometry_path,
                                 properties=calculation_properties,
                                 obj_dict=geot.obj_dict)
    calculator.save(calculator)
    print([ii.shape for ii in calculator.results['band']])
    print(calculator.results['occ'])

    # 3. Save calculated data
    with open('ref.pkl', 'wb') as f:
        pickle.dump(calculator, f)


if __name__ == '__main__':
    gen_reference(torch.device('cpu'))
