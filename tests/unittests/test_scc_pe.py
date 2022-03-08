"""Test SCC DFTB."""
import torch
import pytest
import numpy as np
from ase.build import molecule
from tbmalt import Geometry, Dftb1, Dftb2
from tbmalt.io import read_band

torch.set_default_dtype(torch.float64)
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1],
              14: [0, 1], 22: [0, 1, 2]}


def test_h2_pe(device):
    """Test SCC DFTB for H2 with periodic boundary condition."""
    h2 = molecule('H2')
    h2.cell = [6.0, 6.0, 6.0]
    kpoints = torch.tensor([1, 1, 1])
    geometry = Geometry.from_ase_atoms([h2])
    path_to_skf = './data/slko/mio'

    dftb1 = Dftb1(geometry=geometry,
                  shell_dict=shell_dict,
                  path_to_skf=path_to_skf,
                  skf_type='skf',
                  kpoints=kpoints)
    dftb1()
    dftb2 = Dftb2(geometry=geometry,
                  shell_dict=shell_dict,
                  path_to_skf=path_to_skf,
                  skf_type='skf',
                  kpoints=kpoints)
    dftb2()

    assert torch.max(abs(dftb1.charge - torch.tensor([[1.0, 1.0]]))) < 1E-14, \
        'DFTB1 charge tolerance error'
    assert torch.max(abs(dftb2.charge - torch.tensor([[1.0, 1.0]]))) < 1E-14, \
        'DFTB2 charge tolerance error'

    kpoints2 = torch.tensor([2, 2, 2])
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='skf', kpoints=kpoints2)


@pytest.mark.skip(reason="Test SKF input too huge.")
def test_h2o_pe_vcr(device):
    """Test H2O with various compression radii."""
    path_to_skf = './tests/unittests/data/slko/vcr.h5'
    h2o = molecule('H2O')
    h2o.cell = [6.0, 6.0, 6.0]
    geometry = Geometry.from_ase_atoms([h2o])
    grids = torch.tensor([1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.])
    multi_varible = torch.tensor([[3.0, 3.0, 3.0]])
    dftb2 = Dftb2(geometry, shell_dict=shell_dict,
                  path_to_skf=path_to_skf, skf_type='h5', basis_type='vcr',
                  interpolation='BicubInterp',
                  grids=grids, multi_varible=multi_varible)
    dftb2()

    assert torch.max(abs(dftb2.charge - torch.tensor([[
        6.591468709378842, 0.704265645310579, 0.704265645310579]]))) < 1E-3


def test_ch4_pe(device):
    """Test SCC DFTB for ch4 with periodic boundary condition."""
    ch4 = molecule('CH4')
    ch4.cell = [6.0, 6.0, 6.0]
    kpoints = torch.tensor([1, 1, 1])
    geometry = Geometry.from_ase_atoms([ch4])
    path_to_skf = './tests/unittests/data/slko/mio'

    # dftb1 = Dftb1(geometry=geometry,
    #               shell_dict=shell_dict,
    #               path_to_skf=path_to_skf,
    #               skf_type='skf',
    #               kpoints=kpoints)
    # dftb1()
    dftb2 = Dftb2(geometry=geometry,
                  shell_dict=shell_dict,
                  path_to_skf=path_to_skf,
                  skf_type='skf',
                  kpoints=kpoints)
    dftb2()

    assert torch.max(abs(dftb2.charge - torch.tensor([[
        4.305475062065351, 0.923631234483662, 0.923631234483662,
        0.923631234483662, 0.923631234483662]]))) < 1E-9, 'tolerance error'


def test_c2h6_pe(device):
    """Test SCC DFTB for ch4 with periodic boundary condition."""
    c2h6 = molecule('C2H6')
    c2h6.cell = [6.0, 6.0, 6.0]
    kpoints = torch.tensor([1, 1, 1])
    geometry = Geometry.from_ase_atoms([c2h6])
    path_to_skf = './tests/unittests/data/slko/mio'

    # Non-SCC
    dftb1 = Dftb1(geometry=geometry,
                  shell_dict=shell_dict,
                  path_to_skf=path_to_skf,
                  skf_type='skf')
    dftb1()
    dftb1_pe = Dftb1(geometry=geometry,
                     shell_dict=shell_dict,
                     path_to_skf=path_to_skf,
                     skf_type='skf',
                     kpoints=kpoints)
    dftb1_pe()

    dftb2 = Dftb2(geometry=geometry,
                  shell_dict=shell_dict,
                  path_to_skf=path_to_skf,
                  skf_type='skf',
                  kpoints=kpoints)
    dftb2()


def test_pe_batch(device):
    """Test SCC DFTB for c2h6 with periodic boundary condition."""
    path_to_skf = './data/slko/mio'
    ch4 = molecule('CH4')
    h2o = molecule('H2O')
    c2h6 = molecule('C2H6')

    geometry = Geometry.from_ase_atoms([h2o, ch4, c2h6])
    dftb2 = Dftb2(geometry=geometry,
                  shell_dict=shell_dict,
                  path_to_skf=path_to_skf,
                  skf_type='skf')
    dftb2()
    print(dftb2.__class__.__dict__.keys())

    ch4.cell = [6.0, 6.0, 6.0]
    c2h6.cell = [6.0, 6.0, 6.0]
    h2o.cell = [6.0, 6.0, 6.0]
    kpoints = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    geometry = Geometry.from_ase_atoms([h2o, ch4, c2h6])
    dftb1 = Dftb1(geometry=geometry,
                  shell_dict=shell_dict,
                  path_to_skf=path_to_skf,
                  skf_type='skf',
                  kpoints=kpoints)
    # dftb1()

    dftb2 = Dftb2(geometry=geometry,
                  shell_dict=shell_dict,
                  path_to_skf=path_to_skf,
                  skf_type='skf',
                  kpoints=kpoints)

    dftb2()
    print('fermi', dftb2.fermi, dftb2.H0_energy)

    # assert torch.max(abs(dftb2.charge - torch.tensor([[
    #     4.305475062065351, 0.923631234483662, 0.923631234483662,
    #       0.923631234483662, 0.923631234483662]]))) < 1E-9


def test_si_pe(device):
    """Test SCC DFTB for c2h6 with periodic boundary condition."""
    band = read_band('./data/sk/si/band.out')
    bandd = read_band('./data/sk/si/band.out.d')

    geometry = Geometry(
        torch.tensor([[14, 14]]),
        torch.tensor([[[0., 0.,  0.], [1.356773, 1.356773, 1.356773]]]),
        cell=torch.tensor([[
            [2.713546, 2.713546, 0.0], [0.0, 2.713546, 2.713546],
            [2.713546, 0.0, 2.713546]]]),
        units='angstrom')
    klines = torch.tensor([[0.5, 0.5, -0.5, 0, 0, 0, 10],
                           [0, 0, 0, 0, 0.5, 0.25, 10],
                           [0, 0.5, 0.25, 0.25, 0.25, 0.25, 10]])
    path_to_skf = './data/slko'
    dftb2 = Dftb2(geometry=geometry,
                  shell_dict=shell_dict,
                  path_to_skf=path_to_skf,
                  skf_type='skf',
                  klines=klines)
    dftb2()

    shell_dict.update({14: [0, 1, 2]})
    dftb2d = Dftb2(geometry=geometry,
                   shell_dict=shell_dict,
                   path_to_skf=path_to_skf,
                   skf_type='skf',
                   klines=klines)
    dftb2d()
    check_band = torch.max(abs(dftb2.eigenvalue.squeeze() - band)) < 1E-3
    check_bandd = torch.max(abs(dftb2d.eigenvalue.squeeze() - bandd)) < 1E-7
    dftb2d = Dftb2(geometry, shell_dict=shell_dict, interpolation='Spline1d',
                   path_to_skf=path_to_skf, skf_type='skf', klines=klines)
    dftb2d()
    print(dftb2d.fermi, dftb2d.qzero, dftb2d.H0_energy.sum(-1),
          dftb2d.coulomb_energy)
    import matplotlib.pyplot as plt
    plt.plot(torch.arange(len(dftb2d.eigenvalue.squeeze())),
             dftb2d.eigenvalue.squeeze(), 'r')
    plt.plot(torch.arange(len(bandd)), bandd, 'g')
    plt.ylim(-5, 5)
    plt.show()
    print('dftb2.eigenvalue.squeeze() - band', bandd.shape,
          torch.max(abs(dftb2d.eigenvalue.squeeze() - bandd)),
          torch.max(abs(dftb2.eigenvalue.squeeze() - band)))
    assert check_band
    assert check_bandd


def test_band_batch(device):
    """Test SCC DFTB for c2h6 with periodic boundary condition."""


def _get_matrix(filename, device):
    """Read DFTB+ hamsqr1.dat and oversqr.dat."""
    return torch.from_numpy(np.loadtxt(filename)).to(device)


if __name__ == '__main__':
    test_si_pe(torch.device('cpu'))
