"""Test periodic Hamiltonian and overlap.

The second derivative in SK integral result in slightly accuracy decrease."""
import torch
from tbmalt import Geometry, SkfParamFeed, hs_matrix
from ase import Atoms
from ase.build import molecule
from tbmalt.ml.skfeeds import VcrFeed, TvcrFeed
from tbmalt.data.units import length_units
from tbmalt.structures.periodic import Periodic
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)

shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}


def _get_cell_trans(latVec, cutoff, negExt=1, posExt=1, unit='bohr'):
    """Reproduce code originally from DFTB+ for test TBMaLT.

    This code is for single geometry and not vectorized, retain loop as DFTB+,
    to act as a reference for cell translation code in TBMaLT."""
    if unit == 'angstrom':
        latVec = latVec / 0.529177249
    recVec = torch.inverse(latVec)

    # get ranges of periodic boundary condition from negative to positive
    ranges = torch.zeros((2, 3), dtype=torch.int8)
    for ii in range(3):
        iTmp = torch.floor(cutoff * torch.sqrt(sum(recVec[:, ii] ** 2)))
        ranges[0, ii] = -negExt - iTmp
        ranges[1, ii] = posExt + iTmp

    # Length of the first, second and third column in ranges
    leng1, leng2, leng3 = ranges[1, :] - ranges[0, :] + 1
    ncell = leng1 * leng2 * leng3  # -> Number of lattice cells

    # Cell translation vectors in relative coordinates
    cellvec = torch.zeros(ncell, 3)
    col3 = torch.linspace(ranges[0, 2], ranges[1, 2], leng3)
    col2 = torch.linspace(ranges[0, 1], ranges[1, 1], leng2)
    col1 = torch.linspace(ranges[0, 0], ranges[1, 0], leng1)
    cellvec[:, 2] = col3.repeat(int(ncell.numpy() / leng3.numpy()))
    col2 = col2.repeat(leng3, 1)
    col2 = torch.cat([(col2[:, ii]) for ii in range(leng2)])
    cellvec[:, 1] = col2.repeat(int(ncell.numpy() / (leng2 * leng3).numpy()))
    col1 = col1.repeat(leng3 * leng2, 1)
    cellvec[:, 0] = torch.cat([(col1[:, ii]) for ii in range(leng1)])

    # Cell translation vectors in absolute units
    rcellvec = torch.stack([torch.matmul(
        torch.transpose(latVec, 0, 1), cellvec[ii]) for ii in range(ncell)])

    return cellvec.T, rcellvec


def test_pe_normal_ch4():
    """Test CH4 Hamiltonian and ovelap in periodic geometry."""
    latvec = torch.tensor([[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]])
    positions = torch.tensor([[[.5, .5, .5], [.6, .6, .6], [.4, .6, .6],
                              [.6, .4, .6], [.6, .6, .4]]])
    path = './data/slko/mio'
    cutoff = torch.tensor([9.98])
    numbers = torch.tensor([[6, 1, 1, 1, 1]])
    cellvec_ref, rcellvec_ref = _get_cell_trans(latvec, cutoff + 1, unit='bohr')
    geo = Geometry(numbers, positions, latvec)

    skparams = SkfParamFeed.from_dir(path, geo, skf_type='skf')
    periodic = Periodic(geo, geo.cell, skparams.cutoff)

    # skt = SKT(geo, sktable, periodic)
    assert torch.max(abs(periodic.cellvec[0] - cellvec_ref)) < 1E-14
    assert torch.max(abs(periodic.rcellvec[0] - rcellvec_ref)) < 1E-14


def test_phase():
    # Get packed selected cell_vector within the cutoff [n_batch, max_cell, 3]
    return pack([torch.exp((0. + 1.0j) * torch.bmm(
        ik[mask[0]].unsqueeze(1), cell_vec[mask[0]]))
        for ik in kpoint.permute(1, 0, -1)]).squeeze(2)


def test_pe_normal_co2():
    """Test CO2 Hamiltonian and ovelap in periodic geometry."""
    latvec = torch.tensor([[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]])
    positions = torch.tensor([[.5, .5, .5], [.55, .55, .55], [.45, .45, .45]])
    numbers = torch.tensor([6, 8, 8])
    cutoff = torch.tensor([9.98])
    cellvec_ref, rcellvec_ref = _get_cell_trans(latvec / length_units['angstrom'], cutoff + 1)
    geo = Geometry(numbers, positions, latvec)
    periodic = Periodic(geo, geo.cell, sktable.cutoff, unit='bohr')
    assert torch.max(abs(periodic.cellvec[0] - cellvec_ref)) < 1E-14
    assert torch.max(abs(periodic.rcellvec[0] - rcellvec_ref)) < 1E-14
