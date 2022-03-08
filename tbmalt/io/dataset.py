#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""An interface to datset reading, loading and writing."""
from typing import List, Union, Literal
import os
import pickle
import scipy
import numpy as np
import torch
import h5py
import ase
import ase.io as io
from torch import Tensor
from tbmalt import Geometry
from torch.utils.data import Dataset as _Dataset
from tbmalt.structures.geometry import to_atomic_numbers
from tbmalt.common.batch import pack
from tbmalt.common.logger import get_logger


class Dataset(_Dataset):
    """An interface to dataset inherited from `torch.utils.data.Dataset`.

    Arguments:
        properties: A list of atomic or geometric or electronic properties.

    """

    def __init__(self, properties):
        self.properties = properties

    @property
    def geometry(self) -> Geometry:
        """Create `Geometry` object in TBMaLT."""
        atomic_numbers = self.properties['atomic_numbers']
        assert atomic_numbers.dim() == 2, 'do not support single `Geometry`.'
        positions = self.properties['positions']
        cell = self.properties['cell'] if 'cell' in self.properties.keys() \
            else torch.zeros(len(atomic_numbers), 3, 3)

        return Geometry(atomic_numbers, positions, cell=cell, units='angstrom')

    def __getitem__(self, index: Tensor) -> dict:
        """Return properties with selected indices from original samples."""
        return {key: val[index] for key, val in self.properties.items()}

    def __len__(self):
        return len(self.properties['numbers'])

    def __add__(self):
        pass

    def __repr__(self):
        """Representation of "Dataset" object."""
        if not self.geometry.isperiodic:
            _pe = 'molecule'
        elif not self.geometry.periodic_list.all():
            _pe = 'mixture'
        else:
            _pe = 'solid'

        return f'{self.__class__.__name__}({len(self.geometry.positions)} {_pe})'

    def metadata(self):
        pass

    def qm9(self):
        pass

    def qm7(self):
        dataset = scipy.io.loadmat(self.dataset)
        n_dataset_ = self.n_dataset
        coor_ = dataset['R']
        qatom_ = dataset['Z']
        positions = []
        for idata in range(n_dataset_):
            icoor = coor_[idata]
            natom_ = 0
            symbols_ = []

            number = torch.from_numpy(qatom_[idata][:natom_])
            coor = torch.from_numpy(icoor[:natom_, :])
            positions.append(coor)

    @classmethod
    def pkl(cls,
            path_to_data, type: str,
            properties: List[str],
            **kwargs) -> 'Dataset':
        """Read data from pkl object with geometry and properties.

        This input pkl files could be either from TBMaLT or the code with
        attributes including: atomic_numbers, positions, cell and the input
        properties.

        Arguments:
            path_to_data: Path to binary data which contains geometry
                information, atomic numbers, positions and atomic or
                geometric properties.
            properties: A list of atomic or geometric properties.
            to_geometry, If transfer atomic numbers and positions to
            `Geometry` object.
        """
        try:
            with open(path_to_data, 'rb') as f:
                data = pickle.load(f)
                atomic_numbers = pack(
                    [torch.from_numpy(ii) for ii in data.atomic_numbers])
                positions = pack(
                    [torch.from_numpy(ii) for ii in data.positions])
                cell = pack(
                    [torch.from_numpy(np.asarray(ii)) for ii in data.cell])
                properties = {
                    iproperty: pack([torch.from_numpy(ii)
                                     for ii in data.results[iproperty]])
                    for iproperty in properties}

            properties.update({
                'atomic_numbers': atomic_numbers,
                'positions': positions,
                'cell': cell
            })
            return cls(properties)
        except Exception:
            get_logger.error(f'Fails to open {path_to_data}')

    def ani1(self, **kwargs):
        """Load the data from ANI-1 dataset."""
        dtype = kwargs.get('dtype', np.float64)

        # define the output
        numbers, positions = [], []

        # symbols for each molecule, global atom specie
        symbolsl = []

        # temporal coordinates for all
        _coorall = []

        # temporal molecule species for all
        _specie, _number = [], []

        # temporal number of molecules in all molecule species
        n_molecule = []

        # load each ani_gdb_s0*.h5 data in datalist
        adl = AniDataloader(self.dataset)
        self.in_size = round(self.size / adl.size())  # each group size

        # such as for ani_gdb_s01.h5, there are 3 species: CH4, NH3, H2O
        for iadl, data in enumerate(adl):

            # get each molecule specie size
            size_ani = len(data['coordinates'])
            isize = min(self.in_size, size_ani)

            # size of each molecule specie
            n_molecule.append(isize)

            # selected coordinates of each molecule specie
            _coorall.append(torch.from_numpy(
                data['coordinates'][:isize].astype(dtype)))

            # add atom species in each molecule specie
            _specie.append(data['species'])
            _number.append(to_atomic_numbers(data['species']).squeeze())

        for ispe, isize in enumerate(n_molecule):
            numbers.extend([_number[ispe]] * isize)

            # add coordinates
            positions.extend([icoor for icoor in _coorall[ispe][:isize]])

        return numbers, positions

    @classmethod
    def from_caculator(cls, calculator, properties):
        for ipro in properties:
            assert ipro in calculator.results.keys(), \
                f'{ipro} is not in {calculator.results.keys()}'
        for ipro in properties:
            return cls(calculator.results[ipro], ipro)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        pass


class AniDataloader:
    """Interface to ANI-1 data."""

    def __init__(self, input):
        if not os.path.exists(input):
            exit('Error: file not found - ' + input)
        self.input = h5py.File(input, 'r')

    def iterator(self, g, prefix=''):
        """Group recursive iterator."""
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            keys = [i for i in item.keys()]
            if isinstance(item[keys[0]], h5py.Dataset):  # test for dataset
                data = {'path': path}
                for k in keys:
                    if not isinstance(item[k], h5py.Group):
                        dataset = np.array(item[k][()])
                        if type(dataset) is np.ndarray:
                            if dataset.size != 0:
                                if type(dataset[0]) is np.bytes_:
                                    dataset = [a.decode('ascii')
                                               for a in dataset]

                        data.update({k: dataset})

                yield data
            else:
                yield from self.iterator(item, path)

    def __iter__(self):
        """Default class iterator (iterate through all data)."""
        for data in self.iterator(self.input):
            yield data

    def size(self):
        count = 0
        for g in self.input.values():
            count = count + len(g.items())
        return count


class GeometryTo(object):
    """Transfer and write various input geometries.

    Arguments:
        in_geometry_files: Single or batch input geometry files.
        to_geometry_type: Output geometry type.
        to_geometry_path: Output geometry path.

    """

    def __init__(self,
                 in_geometry_files: List[str],
                 path_to_input_template: str,
                 to_geometry_type: Literal['h5', 'ase', 'geometry'] = 'cif',
                 to_geometry_path: str = './',
                 calculation_properties=['energy'],
                 **kwargs):
        self.in_geometry_files = in_geometry_files
        self.path_to_input_template = path_to_input_template
        self.to_geometry_type = to_geometry_type
        self.to_geometry_path = to_geometry_path

        if 'band' in calculation_properties:
            self.bandpath = True
            self.npoints = kwargs.get('npoints', 10)
            self.n_band_grid = kwargs.get('n_band_grid', 10)
        else:
            self.bandpath = False

    def __call__(self, idx: Tensor = None):
        """Transfer geometry."""
        if os.path.isdir(self.to_geometry_path):
            if len(os.listdir(self.to_geometry_path)) != 0:
                get_logger(self.__class__.__name__).info(
                    f'{self.to_geometry_path} is not empty, all files will be removed in this path.')

                # Remove and Build output dir
                os.system('rm -r ' + self.to_geometry_path)
                os.system('mkdir -p ' + self.to_geometry_path)
        else:
            get_logger(self.__class__.__name__).info(
                f'{self.to_geometry_path} do not exist, build now ...')
            os.system('mkdir -p ' + self.to_geometry_path)

        # Select geometries with input indices
        if idx is not None:
            self.in_geometry_files = [self.in_geometry_files[ii] for ii in idx]

        # Create geometric files and input files
        if isinstance(self.in_geometry_files, list):
            try:
                # To list ASE object
                _in = [io.read(ii) for ii in self.in_geometry_files]
            except:
                get_logger(self.__class__.__name__).error(
                    f'could not load {self.in_geometry_files}')

            self._obj_dict = {}
            if self.to_geometry_type == 'geometry':
                return Geometry.from_ase_atoms(_in)

            elif self.to_geometry_type == 'aims':
                for ii, iin in enumerate(_in):
                    # Generate geometry file and copy to target dir
                    io.write(os.path.join(
                        self.to_geometry_path, 'geometry.in.' + str(ii)), iin, format='aims')

                    # Copy tempalte input files and modify
                    self._in_file = os.path.join(
                        self.to_geometry_path, 'control.in.' + str(ii))

                    # Deal with band structures
                    if self.bandpath:
                        _obj = self._aims_band(iin)
                        self._obj_dict.update({ii: _obj.todict()})

            elif self.to_geometry_type == 'dftbplus':
                for ii, iin in enumerate(_in):
                    io.write(os.path.join(
                        self.to_geometry_path, 'geo.gen.' + str(ii)), iin, format='dftb')

                    # Copy tempalte input files and modify
                    self._in_file = os.path.join(
                        self.to_geometry_path, 'dftb_in.hsd.' + str(ii))

                    # Deal with band structures
                    if self.bandpath:
                        _obj = self._dftb_band(iin)
                        self._obj_dict.update({ii: _obj.todict()})

    def _aims_band(self, ase_aims_obj):
        """Modify control.in file and return ase object with band strucutres."""
        _obj = ase_aims_obj.cell.bandpath(npoints=self.npoints)
        kpts = _obj.kpts
        kpts_val = ''.join([
            'output band ' + str(ib)[1:-1] + ' ' + str(ie)[1:-1] + ' ' + str(
                self.n_band_grid) + '\n' for ib, ie in zip(kpts[:-1], kpts[1:])])
        with open(self.path_to_input_template, 'r') as f:
            data = f.read()
            try:
                data = data.replace('kpts', kpts_val)
            except:
                logger.error('could not replace "kpts" with real band path,' +
                             ' check if there is keyword "kpts" in template')

        # Write modified data to new control.in
        with open(self._in_file, 'w') as f:
            f.write(data)

        return _obj

    def _dftb_band(self, ase_aims_obj):
        """Modify control.in file and return ase object with band strucutres."""
        _obj = ase_aims_obj.cell.bandpath(npoints=self.npoints)
        kpts = _obj.kpts
        kpts_val = ''.join([
            '1 ' + str(ib)[1:-1] + ' \n' + str(self.n_band_grid - 1) + ' '
            + str(ie)[1:-1] + '\n' for ib, ie in zip(kpts[:-1], kpts[1:])])
        with open(self.path_to_input_template, 'r') as f:
            data = f.read()
            try:
                data = data.replace('kpts', kpts_val)
            except:
                logger.error('could not replace "kpts" with real band path,' +
                             ' check if there is keyword "kpts" in template')

        # Write modified data to new control.in
        with open(self._in_file, 'w') as f:
            f.write(data)

        return _obj

    @property
    def obj_dict(self):
        """ASE object dictionary."""
        return self._obj_dict
