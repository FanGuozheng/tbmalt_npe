# -*- coding: utf-8 -*-
"""Elemental reference data.

Reference data pertaining to chemical elements & their properties are located
here. As the `chemical_symbols` & `atomic_numbers` attributes are frequently
used they have been made accessible from the `tbmalt.data` namespace.

Attributes:
    chemical_symbols (List[str]): List of chemical symbols whose indices are
        the atomic numbers of the associated elements; i.e.
        `chemical_symbols[6]` will yield `"C"`.
    atomic_numbers (Dict[str, int]): Dictionary keyed by chemical symbols &
        valued by atomic numbers. This is used to get the atomic number
        associated with a given chemical symbol.

"""
from typing import List, Dict

# Chemical symbols of the elements. Neutronium is included to ensure the index
# matches the atomic number and to assist with batching behaviour.
chemical_symbols: List[str] = [
    # Period zero
    'X' ,
    # Period one
    'H' , 'He',
    # Period two
    'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
    # Period three
    'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar',
    # Period four
    'K',  'Ca', 'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # Period five
    'Rb', 'Sr', 'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I' , 'Xe',
    # Period six
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt',
    'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    # Period seven
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
    'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

# Dictionary for looking up an element's atomic number.
atomic_numbers: Dict[str, int] = {sym: z for z, sym in
                                  enumerate(chemical_symbols)}

# chemical masses from DFTB+
chemical_masses: List[float] = [
    # Period zero
    0.0,
    # Period one
    1.00794075, 4.00260193,
    # Period two
    6.94003660, 9.01218307, 10.81102805, 12.01073590, 14.00670321, 15.99940492,
    18.99840316, 20.18004638,
    # Period three
    22.98976928, 24.30505162, 26.98153853, 28.08549871, 30.97376200,
    32.06478741, 35.45293758, 39.94779856,
    # Period four
    39.09830091, 40.07802251, 44.95590828, 47.86674496, 50.94146504,
    51.99613176, 54.93804391, 55.84514443, 58.93319429, 58.69334711,
    63.54603995, 65.37778253, 69.72306607, 72.62755016, 74.92159457,
    78.95938856, 79.90352778, 83.79800000,
    # Period five
    85.46766360, 87.61664447, 88.90584030, 91.22364160, 92.90637300,
    95.95978854, 97.90721240, 101.06494014, 102.90549800, 106.41532751,
    107.86814963, 112.41155782, 114.81808663, 118.71011259, 121.75978367,
    127.60312648, 126.90447190, 131.29276145,
    # Period six
    132.90545196, 137.32689163, 138.90546887, 140.11573074, 140.90765760,
    144.24159603, 144.91275590, 150.36635571, 151.96437813, 157.25213065,
    158.92535470, 162.49947282, 164.93032880, 167.25908265, 168.93421790,
    173.05415017, 174.96681496, 178.48497872, 180.94787564, 183.84177755,
    186.20670455, 190.22485963, 192.21605165, 195.08445686, 196.96656879,
    200.59916703, 204.38341284, 207.21690806, 208.98039910, 208.98243080,
    209.98714790, 222.01757820,
    # Period seven
    223.01973600, 226.02541030, 227.02775230, 232.03805580, 231.03588420,
    238.02891046, 237.04817360, 244.06420530, 243.06138130, 247.07035410,
    247.07030730, 251.07958860, 252.08298000, 257.09510610, 258.09843150,
    259.10103000, 262.10961000, 267.12179000, 269.12791000, 271.13393000,
    270.13336000, 276.14846000, 276.15159000, 280.16131000, 282.16912000,
    284.17416000, 284.17873000, 289.19042000, 288.19274000, 293.20449000,
    292.20746000, 294.21392000]

atomic_masses: Dict[str, int] = {
    sym: z for z, sym in enumerate(chemical_masses)}
