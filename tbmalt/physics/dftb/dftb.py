"""DFTB calculator."""
from typing import Literal, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor

import tbmalt.common.maths as maths
from tbmalt import Shell, Geometry, SkfFeed, SkfParamFeed, hs_matrix
from tbmalt.physics.dftb.repulsive import Repulsive
from tbmalt.common.maths.mixer import Simple, Anderson
from tbmalt.physics.properties import mulliken, dos, pdos, band_pass_state_filter
from tbmalt.common.batch import pack
from tbmalt.data.units import _Hartree__eV, energy_units
from tbmalt.ml.skfeeds import VcrFeed, TvcrFeed
from tbmalt.physics.electrons import Gamma
from tbmalt.structures.periodic import Periodic
from tbmalt.physics.coulomb import Coulomb
from tbmalt.physics.filling import fermi_search, fermi_smearing


class Dftb(nn.Module):
    """Dftb template module which inherited from nn.Module.

    Arguments:
        geometry: Geometry object in TBMaLT.
        shell_dict: A dictionary which offer angular momenta shell of each atom.
        repulsive: Boolean parameter determines if perform repulsive
            calculations.
        skf_type: The type Slater-Kpster table.
        periodic: Periodic object in TBMaLT.
        mixer: Mixer type for Mulliken charges.

    """

    def __init__(
            self,
            geometry: Geometry,
            shell_dict: Dict[int, List[int]],
            path_to_skf: str,
            repulsive: bool = False,
            skf_type: Literal['h5', 'skf'] = 'h5',
            basis_type: str = 'normal',
            periodic: Periodic = None,
            mixer: str = 'Anderson',
            temperature: Union[Tensor, float] = 300.0,
            **kwargs):
        super(Dftb, self).__init__()
        self.skf_type = skf_type
        self.shell_dict = shell_dict
        self.path_to_skf = path_to_skf
        self.geometry = geometry
        self.repulsive = repulsive
        self.mixer_type = mixer
        self.temperature = temperature
        self.interpolation = kwargs.get('interpolation', 'PolyInterpU')
        self.batch = True if self.geometry.distances.dim() == 3 else False
        hsfeed = {'normal': SkfFeed, 'vcr': VcrFeed,
                  'tvcr': TvcrFeed}[basis_type]

        _grids = kwargs.get('grids', None)
        _interp = kwargs.get('interpolation', 'PolyInterpU')

        self.basis = Shell(self.geometry.atomic_numbers, self.shell_dict)
        self.h_feed = hsfeed.from_dir(path=path_to_skf,
                                      shell_dict=shell_dict,
                                      vcr=_grids,
                                      skf_type=skf_type,
                                      geometry=geometry,
                                      interpolation=_interp,
                                      integral_type='H')
        self.s_feed = hsfeed.from_dir(path=path_to_skf,
                                      shell_dict=shell_dict,
                                      vcr=_grids,
                                      skf_type=skf_type,
                                      geometry=geometry,
                                      interpolation=_interp,
                                      integral_type='S')
        self.skparams = SkfParamFeed.from_dir(path=path_to_skf,
                                              geometry=geometry,
                                              skf_type=skf_type,
                                              repulsive=self.repulsive)

    def init_dftb(self, **kwargs):
        """Initialization of DFTB calculations."""
        self._n_batch = self.geometry._n_batch if self.geometry._n_batch \
            is not None else 1
        self.dtype = self.geometry.positions.dtype if \
            not self.isperiodic else torch.complex128
        self.qzero = self.skparams.qzero
        self.nelectron = self.qzero.sum(-1)

        if self.isperiodic:
            self.periodic = Periodic(geometry=self.geometry,
                                     latvec=self.geometry.cell,
                                     cutoff=self.skparams.cutoff,
                                     **kwargs)
            self.coulomb = Coulomb(geometry=self.geometry,
                                   periodic=self.periodic,
                                   method='search')
            self.distances = self.periodic.periodic_distances
            self.u = self._expand_u(self.skparams.U)
            self.max_nk = torch.max(self.periodic.n_kpoints)
        else:
            self.distances = self.geometry.distances
            self.u = self.skparams.U  # self.skt.U
            self.periodic, self.coulomb = None, None

        # if self.method in ('dftb2', 'Dftb3', 'xlbomd'):
        self.method = kwargs.get('gamma_method', 'read')
        self.short_gamma = Gamma(
            self.u, self.distances, self.geometry.atomic_numbers,
            self.periodic, method=self.method).gamma

        self.atom_orbitals = self.basis.orbs_per_atom
        self.inv_dist = self._inv_distance(self.geometry.distances)

    def _inv_distance(self, distance):
        """Return inverse distance."""
        inv_distance = torch.zeros(*distance.shape)
        inv_distance[distance.ne(0.0)] = 1.0 / distance[distance.ne(0.0)]
        return inv_distance

    def _update_shift(self, this: dict, charge: Tensor = None):
        """Update shift."""
        _charge = self.charge if charge is None else charge
        return torch.einsum('ij, ijk-> ik',
                            (_charge - self.qzero)[this['mask']],
                            self.shift[this['mask']])

    def forward(self, hamiltonian, overlap, this):
        """A template for DFTB forward calculations."""
        # calculate the eigen-values & vectors via a Cholesky decomposition
        if self.isperiodic:
            self.ie, eigvec, nocc, density, q_new = [], [], [], [], []
            self._mask_k = []
            # Loop over all K-points
            for ik in range(self.max_nk):

                # calculate the eigen-values & vectors
                iep, ieig = maths.eighb(hamiltonian[..., ik], overlap[..., ik])

                Ef = fermi_search(
                    eigenvalues=iep,
                    n_electrons=self.qzero.sum(-1)[this['mask']],
                    e_mask=(self.basis.on_atoms != -1)[this['mask']],
                    kT=self._kt(),
                )
                occ = fermi_smearing(iep, Ef, self._kt()) * 2.0
                iden = torch.sqrt(occ).unsqueeze(1).expand_as(ieig) * ieig
                self.ie.append(iep), eigvec.append(ieig)
                self._update_scc_ik(
                    iep, ieig, self.over[..., ik],
                    this, ik, torch.max(self.periodic.n_kpoints))

                irho = (torch.conj(iden) @ iden.transpose(1, 2))  # -> density
                density.append(irho)

                # calculate mulliken charges for each system in batch
                iq = mulliken(self.over[this['mask'], :this['size'], :this['size'], ik],
                              irho, self.atom_orbitals[this['mask']])
                _q = iq.real
                q_new.append(_q)

            # nocc = pack(nocc).T
            self.rho = pack(density).permute(1, 2, 3, 0)
            if this['iter'] == 0:
                # self.nocc = torch.zeros(*nocc.shape)
                self._density = torch.zeros(
                    *self.rho.shape, dtype=self.rho.dtype)
            q_new = (pack(q_new).permute(2, 1, 0) *
                     self.periodic.k_weights[this['mask']]).sum(-1).T
            # charge_mix, _mask = self.mixer(q_new)
            epsilon = pack(self.ie).permute(1, 0, -1)
            eigvec = pack(eigvec).permute(1, 0, -2, -1)

        else:
            epsilon, eigvec = maths.eighb(hamiltonian, overlap)
            Ef = fermi_search(
                eigenvalues=epsilon,
                n_electrons=self.qzero.sum(-1)[this['mask']],
                e_mask=(self.basis.on_atoms != -
                        1)[this['mask']][..., :this['size']],
                kT=self._kt(),
            )
            occ = fermi_smearing(epsilon, Ef, self._kt()) * 2.0

            # eigenvector with Fermi-Dirac distribution
            c_scaled = torch.sqrt(occ).unsqueeze(1).expand_as(
                eigvec) * eigvec
            self.rho = c_scaled @ c_scaled.transpose(1, 2)  # -> density

            q_new = mulliken(overlap, self.rho,
                             self.atom_orbitals[this['mask']])

        if this['iter'] == 0 or not self.batch:
            self._density = self.rho.clone()
            self._occ = occ.clone()
            self._epsilon = epsilon.clone()
            self._eigvec = eigvec.clone()
        elif this['iter'] > 0 and self.batch:
            self._occ[this['mask'], :occ.shape[-1]] = occ
            if self.periodic:
                self._epsilon[this['mask'], :epsilon.shape[-2],
                              : epsilon.shape[-1]] = epsilon
                self._eigvec[this['mask'], : eigvec.shape[-3], : eigvec.shape[-2],
                             : eigvec.shape[-1]] = eigvec.clone()
            else:
                self._epsilon[this['mask'], : epsilon.shape[-1]] = epsilon
                self._eigvec[this['mask'], : eigvec.shape[-1],
                             : eigvec.shape[-1]] = eigvec.clone()

        return q_new

    def __add__(self):
        pass

    def __repr__(self):
        pass

    def __getitem__(self):
        pass

    def __hs__(self, hamiltonian, overlap, **kwargs):
        """Hamiltonian or overlap feed."""
        multi_varible = kwargs.get('multi_varible', None)

        if self.isperiodic:
            hs_obj = self.periodic
        else:
            hs_obj = self.geometry
        if hamiltonian is None:
            self.ham = hs_matrix(
                hs_obj, self.basis, self.h_feed, multi_varible=multi_varible)
        else:
            self.ham = hamiltonian
        if overlap is None:
            self.over = hs_matrix(
                hs_obj, self.basis, self.s_feed, multi_varible=multi_varible)
        else:
            self.over = overlap

    def _next_geometry(self, geometry, **kwargs):
        """Update geometry for DFTB calculations."""
        if (self.geometry.atomic_numbers != geometry.atomic_numbers).any():
            raise ValueError('Atomic numbers in new geometry have changed.')

        self.geometry = geometry

        if self.isperiodic:
            self.periodic = Periodic(self.geometry, self.geometry.cell,
                                     cutoff=self.skparams.cutoff, **kwargs)
            self.coulomb = Coulomb(
                self.geometry, self.periodic, method='search')
            self.max_nk = torch.max(self.periodic.n_kpoints)

        # calculate short gamma
        if self.method in ('dftb2', 'dftb3'):
            self.short_gamma = Gamma(
                self.u, self.distances, self.geometry.atomic_numbers,
                self.periodic, method=self.method).gamma

        self.atom_orbitals = self.basis.orbs_per_atom
        self.inv_dist = self._inv_distance(self.geometry.distances)

    def _get_shift(self) -> Tensor:
        """Return shift term for periodic and non-periodic."""
        if not self.isperiodic:
            return self.inv_dist - self.short_gamma
        else:
            return self.coulomb.invrmat - self.short_gamma

    def _second_order_ham(self, this: dict, charge: Tensor = None):
        """Build second order Gamma and Fock."""
        # Update shift with latest charges
        shift = self._update_shift(this)
        self.shift_orb = pack([
            ishif.repeat_interleave(iorb)
            for iorb, ishif in zip(self.atom_orbitals[this['mask']], shift)])
        shift_mat = torch.stack([
            torch.unsqueeze(ishift, 1) + ishift for ishift in self.shift_orb])

        if self.isperiodic:  # -> Repeat over k-Path dimension
            shift_mat = shift_mat.repeat(torch.max(
                self.periodic.n_kpoints), 1, 1, 1).permute(1, 2, 3, 0)
            size = shift_mat.shape[-2]   # the new shape
        else:
            size = shift_mat.shape[-1]   # the new shape

        # Return masked H & S
        S = self.over[this['mask'], :size, :size]
        H = self.ham[this['mask'], :size, :size] + 0.5 * S * shift_mat

        return size, shift_mat, H, S

    def _update_scc_ik(self, epsilon, eigvec, _over, this, ik=None, n_kpoints=None):
        """Update data for each kpoints."""
        if this['iter'] == 0:
            if ik is None:
                # self._epsilon = torch.zeros(*epsilon.shape)
                self.eigenvector = torch.zeros(*eigvec.shape, dtype=self.dtype)
            elif ik == 0:
                self.epsilon = torch.zeros(*epsilon.shape)
                self.eigenvector = torch.zeros(*eigvec.shape, dtype=self.dtype)

        if ik is None:
            # self._epsilon[self.mask, :epsilon.shape[1]] = epsilon
            self.eigenvector[
                this['mask'], : eigvec.shape[1], : eigvec.shape[2]] = eigvec
        else:
            self.epsilon[this['mask'], : epsilon.shape[1]] = epsilon
            self.eigenvector[
                this['mask'], : eigvec.shape[1], : eigvec.shape[2]] = eigvec

    def _onsite_population(self) -> Tensor:
        """Get onsite population for CPA DFTB."""
        nb = self.geometry._n_batch
        ns = self.geometry.n_atoms
        acum = torch.cat([torch.zeros(self.atom_orbitals.shape[0]).unsqueeze(0),
                          torch.cumsum(self.atom_orbitals, dim=1).T]).T.long()
        denmat = [idensity.diag() for idensity in self.density]

        # get onsite population
        return pack([torch.stack(
            [torch.sum(denmat[ib][acum[ib][iat]: acum[ib][iat + 1]])
             for iat in range(ns[ib])]) for ib in range(nb)])

    def _expand_u(self, u) -> Tensor:
        """Expand Hubbert U for periodic system."""
        shape_cell = self.distances.shape[1]
        return u.repeat(shape_cell, 1, 1).transpose(0, 1)

    def unit(self, unit: str = 'au') -> str:
        """Set general unit for DFTB output."""
        return unit

    def _kt(self) -> Union[Tensor, float]:
        return self.temperature * energy_units['k'] * _Hartree__eV

    @property
    def isperiodic(self) -> bool:
        """Return if the whole system is periodic conditions."""
        return self.geometry.isperiodic

    @property
    def init_charge(self) -> Tensor:
        """Return initial charge."""
        return self.qzero

    @property
    def homo_lumo(self) -> Tensor:
        """Return dipole moments."""
        # get HOMO-LUMO, not orbital resolved
        return torch.stack([
            ieig[int(iocc) - 1:int(iocc) + 1]
            for ieig, iocc in zip(self._epsilon, self.nocc)])

    @property
    def cpa(self) -> Tensor:
        """Get onsite population for CPA DFTB.

        J. Chem. Phys. 144, 151101 (2016)
        """
        onsite = self._onsite_population()
        nat = self.geometry.n_atoms
        numbers = self.geometry.atomic_numbers

        return pack([1.0 + (onsite[ib] - self.qzero[ib])[:nat[ib]] / numbers[
            ib][:nat[ib]] for ib in range(self.geometry._n_batch)])

    @property
    def eigenvalue(self, unit: str = 'eV') -> Tensor:
        """Return eigenvalue."""
        sca = _Hartree__eV if unit == 'eV' else 1.0
        return self._epsilon * sca

    @property
    def charge(self) -> Tensor:
        return self._charge

    @property
    def occupation(self) -> Tensor:
        return self._occ

    @property
    def total_energy(self) -> Tensor:
        return self.electronic_energy + self.repulsive_energy

    @property
    def band_energy(self) -> Tensor:
        """Return H0 energy."""
        return (self._epsilon * self._occ).sum(-1)

    @property
    def H0_energy(self) -> Tensor:
        """Return H0 energy."""
        if not self.isperiodic:
            return (self.ham * self._density).sum((-1, -2))
        else:
            return (self.ham * self._density).sum((-1, -2, -3)).real

    @property
    def coulomb_energy(self) -> Tensor:
        """Calculate Coulomb energy (atom resolved charge)."""
        _q = self.charge - self.qzero
        deltaq = _q.unsqueeze(1) * _q.unsqueeze(2)
        return 0.5 * (self.shift * deltaq).sum((-1, -2))

    @property
    def electronic_energy(self) -> Tensor:
        """Return electronic energy."""
        return self.H0_energy + self.coulomb_energy

    @property
    def repulsive_energy(self) -> Tensor:
        """Return repulsive energy."""
        return self.cal_repulsive().repulsive_energy

    @property
    def dipole(self) -> Tensor:
        """Return dipole moments."""
        return torch.sum((self.qzero - self._charge).unsqueeze(-1) *
                         self.geometry.positions, 1)

    def cal_repulsive(self) -> Repulsive:
        return Repulsive(self.geometry, self.skparams, self.basis)

    @property
    def density(self):
        return self._density

    @property
    def shift_mat(self):
        return self._shift_mat

    @property
    def energy_weighted_density(self):
        '''Return weighted density.'''
        mask = self._occ.ne(0).unsqueeze(1).expand_as(self._eigvec)
        _eig = torch.zeros(self._eigvec.shape)
        _eig[mask] = self._eigvec[mask]

        dm1 = _eig @ _eig.transpose(1, 2)

        _eps = torch.zeros(self._occ.shape)
        _mask = self._occ.ne(0)
        _eps = self._occ * self._epsilon
        shift = torch.min(_eps) - 0.1
        _eps[_mask] = _eps[_mask] - shift
        c_scaled = torch.sqrt(_eps).unsqueeze(
            1).expand_as(self._eigvec) * self._eigvec  # * sign

        dm2 = c_scaled @ c_scaled.transpose(1, 2)

        return dm2 + dm1 * shift

    @property
    def U(self):
        return self.skparams.U

    @property
    def E_fermi(self):
        kweight = self.periodic.k_weights if self.isperiodic else None
        return fermi_search(eigenvalues=self.eigenvalue,
                            n_electrons=self.qzero.sum(-1).numpy(),
                            e_mask=self.basis,
                            kT=self._kt(),
                            k_weights=kweight)


class Dftb1(Dftb):
    """Density-functional tight-binding method without high order correction."""

    def __init__(self,
                 geometry: object,
                 shell_dict: Dict[int, List[int]],
                 path_to_skf: str,
                 repulsive: bool = True,
                 skf_type: Literal['h5', 'skf'] = 'h5',
                 basis_type: str = 'normal',
                 periodic: Periodic = None,
                 mixer: str = 'Anderson',
                 temperature: Union[Tensor, float] = 300.0,
                 **kwargs):
        self.method = 'dftb1'
        self.maxiter = kwargs.get('maxiter', 1)
        super().__init__(
            geometry=geometry,
            shell_dict=shell_dict,
            path_to_skf=path_to_skf,
            repulsive=repulsive,
            skf_type=skf_type,
            basis_type=basis_type,
            periodic=periodic,
            mixer=mixer,
            temperature=temperature,
            **kwargs
        )
        super().init_dftb(**kwargs)

    def forward(self,
                charge: Tensor = None,  # -> Initial charge
                geometry: Geometry = None,  # Update Geometry
                hamiltonian: Tensor = None,
                overlap: Tensor = None,
                **kwargs):
        if geometry is not None:
            super()._next_geometry(geometry, **kwargs)

        # Mask is noly for consistency with batch SCC-method
        self.mask = torch.tensor([True]).repeat(self._n_batch)
        super().__hs__(hamiltonian, overlap, **kwargs)
        this = {'iter': 0, 'size': hamiltonian.shape[-1]}

        # One diagonalization with given charges
        if charge is not None:
            self.shift = self._get_shift()
            _, shift_mat, H, S = super()._second_order_ham(charge)
            charge_mul = super().forward(H, S, this)

        # Standard non-SCC-DFTB
        else:
            charge_mul = super().forward(self.ham, self.over, this)

        self._charge = charge_mul


class Dftb2(Dftb):
    """Self-consistent-charge density-functional tight-binding method."""

    def __init__(self,
                 geometry: object,
                 shell_dict: Dict[int, List[int]],
                 path_to_skf: str,
                 repulsive: bool = True,
                 skf_type: Literal['h5', 'skf'] = 'h5',
                 basis_type: str = 'normal',
                 periodic: Periodic = None,
                 mixer: str = 'Anderson',
                 temperature: Union[Tensor, float] = 300.0,
                 **kwargs
                 ):
        self.method = 'dftb2'
        self.maxiter = kwargs.get('maxiter', 60)
        super().__init__(geometry=geometry,
                         shell_dict=shell_dict,
                         path_to_skf=path_to_skf,
                         repulsive=repulsive,
                         skf_type=skf_type,
                         basis_type=basis_type,
                         periodic=periodic,
                         temperature=temperature,
                         **kwargs)
        super().init_dftb(**kwargs)

    def forward(self,
                charge: Tensor = None,  # -> Initial charge
                geometry: Geometry = None,  # Update Geometry
                hamiltonian: Tensor = None,
                overlap: Tensor = None,
                **kwargs):
        """Perform SCC-DFTB calculation."""
        self.converge_number = []
        self.mixer = globals()[self.mixer_type](q_init=self.qzero,
                                                return_convergence=True)
        self.shift = self._get_shift()

        if geometry is not None:
            super()._next_geometry(geometry)

        this = {'mask': torch.tensor([True]).repeat(self._n_batch)}
        super().__hs__(hamiltonian, overlap, **kwargs)
        self._charge = self.qzero.clone() if charge is None else charge

        # Loop for DFTB2
        for iiter in range(self.maxiter):

            this = self._single_loop(iiter, this)
            if not this['mask'].any():
                break

    def _single_loop(self, iiter, this: dict):
        """Perform each single SCC-DFTB loop."""
        # get shift and repeat shift according to number of orbitals
        self.atm_size = torch.max(self.geometry.n_atoms[this['mask']])
        size, _shift_mat, hamiltonian, overlap = super()._second_order_ham(this)
        this.update({'iter': iiter, 'size': size})

        if iiter == 0:
            self._shift_mat = _shift_mat.clone()
        else:
            self._shift_mat[this['mask'], :size, :size] = _shift_mat

        # Diagonalization, calculate charges, Mix charges
        q_new = super().forward(hamiltonian, overlap, this)
        charge_mix, _mask = self.mixer(q_new)
        self._charge[this['mask']] = charge_mix
        self._density[this['mask'], :self.rho.shape[1],
                      :self.rho.shape[2]] = self.rho
        this.update({'mask': ~_mask})
        self.converge_number.append(_mask.sum().tolist())

        return this

    @property
    def dos_energy(self, unit='eV', ext=1, grid=1000):
        """Energy distribution of (P)DOS.

        Arguments:
            unit: The unit of distribution of (P)DOS energy.

        """
        self.unit = unit
        e_min = torch.min(self._epsilon.detach()) - ext
        e_max = torch.max(self._epsilon.detach()) + ext

        if unit in ('eV', 'EV', 'ev'):
            return torch.linspace(e_min, e_max, grid)
        elif unit in ('hartree', 'Hartree'):
            return torch.linspace(e_min, e_max, grid) * _Hartree__eV
        else:
            raise ValueError('unit of energy in DOS should be eV or Hartree.')

    @property
    def pdos(self):
        """Return PDOS."""
        energy = torch.linspace(-1, 1, 200)
        return pdos(self.eigenvector, self.over, self._epsilon, energy)

    @property
    def dos(self):
        """Return energy distribution and DOS with fermi energy correction."""
        sigma = self.params.dftb_params['sigma']
        energy = self.dos_energy
        energy = energy.repeat(self.system.size_batch, 1)  # -> to batch

        # make sure the 1st dimension is batch
        if self.unit in ('eV', 'EV', 'ev'):
            return dos((self._epsilon - self.fermi.unsqueeze(1)),
                       energy, sigma)  # , mask=self.band_filter)
        elif self.unit in ('hartree', 'Hartree'):
            return dos(self._epsilon - self.fermi.unsqueeze(1) * _Hartree__eV,
                       energy, sigma)  # , mask=self.band_filter)

    @property
    def band_filter(self, n_homo=torch.tensor([3]), n_lumo=torch.tensor([3]),
                    band_filter=True) -> Tensor:
        """Return filter of states."""
        if band_filter:
            n_homo = n_homo.repeat(self._epsilon.shape[0])
            n_lumo = n_lumo.repeat(self._epsilon.shape[0])
            return band_pass_state_filter(self._epsilon, n_homo, n_lumo, self.fermi)
        else:
            return None


class Dftb3(Dftb):
    """Density functional tight binding method with third order."""

    def __init__(self):
        pass
