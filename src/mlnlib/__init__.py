"""
mlnlib - Memory-efficient Python library for multilayer networks

This package provides tools for working with large multilayer networks
efficiently on computers with limited resources.

Main Classes:
-------------
- MultiLayerNetwork: Main class for multilayer network analysis
- RawCSVtoMLN: Utility class for converting raw CSV files to MLN format

Authors: Eszter Bokányi, Rachel de Jong, Yuliia Kazmina
Contact: e.bokanyi@liacs.leidenuniv.nl
License: MIT
"""

from .mln import MultiLayerNetwork
from .preparation import RawCSVtoMLN

__version__ = "0.3.0"
__all__ = ["MultiLayerNetwork", "RawCSVtoMLN"]
