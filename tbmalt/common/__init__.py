# -*- coding: utf-8 -*-
"""A collection of useful code abstractions.

All modules that are not specifically associated with any one component of the
code, such as generally mathematical operations, are located here.
"""
from typing import Tuple, Union, List
import torch
from torch import Tensor

# Types
float_like = Union[Tensor, float]
bool_like = Union[Tensor, bool]
