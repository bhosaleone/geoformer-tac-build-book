"""
5D and N-D Toroidal Crystal Topology
===================================
Constructs multi-dimensional lattice graphs with periodic boundary conditions (toroidal topology)
for mapping continuous thermodynamic fields.

Authors: Shrikant Bhosale (Atmabhan Pandit)
"""

import numpy as np
from typing import Tuple, List, Dict
from functools import lru_cache


class CrystalTopology5D:
    """
    Standard 5-dimensional toroidal lattice of 120 nodes (2 x 2 x 2 x 3 x 5 = 120).
    Every node connects to 2 neighbors along each of the 5 dimensions,
    yielding a regular degree-10 torus graph.
    """
    DIMS = (2, 2, 2, 3, 5)
    STRIDES = (60, 30, 15, 5, 1)

    def __init__(self, num_nodes: int = 120):
        self.num_nodes = num_nodes
        self.adjacency_list: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        self._build_5d_lattice()

    @staticmethod
    def coords_to_1d(c0: int, c1: int, c2: int, c3: int, c4: int) -> int:
        strides = (60, 30, 15, 5, 1)
        return c0 * strides[0] + c1 * strides[1] + c2 * strides[2] + c3 * strides[3] + c4

    @staticmethod
    def d1_to_coords(idx: int) -> Tuple[int, int, int, int, int]:
        dims = (2, 2, 2, 3, 5)
        c4 = idx % dims[4]; idx //= dims[4]
        c3 = idx % dims[3]; idx //= dims[3]
        c2 = idx % dims[2]; idx //= dims[2]
        c1 = idx % dims[1]; idx //= dims[1]
        c0 = idx
        return (c0, c1, c2, c3, c4)

    def _build_5d_lattice(self):
        dims = self.DIMS
        for i in range(self.num_nodes):
            coords = list(self.d1_to_coords(i))
            neighbors = set()
            for d in range(5):
                for delta in [-1, 1]:
                    neighbor_coords = list(coords)
                    neighbor_coords[d] = (neighbor_coords[d] + delta) % dims[d]
                    neighbor_idx = self.coords_to_1d(*neighbor_coords)
                    if neighbor_idx != i:
                        neighbors.add(neighbor_idx)
            self.adjacency_list[i] = sorted(list(neighbors))


class FlexibleCrystal5D:
    """
    Generalized 5-dimensional geometric lattice supporting arbitrary dimensions.
    """
    def __init__(self, dims: Tuple[int, int, int, int, int] = (10, 10, 10, 5, 2)):
        self.dims = tuple(dims)
        self.num_nodes = int(np.prod(dims))

        # Compute strides
        strides = []
        curr = 1
        for d in reversed(self.dims):
            strides.insert(0, curr)
            curr *= d
        self.strides = tuple(strides)

        self.adjacency_list: Dict[int, List[int]] = {i: [] for i in range(self.num_nodes)}
        self._build_lattice()

    def coords_to_1d(self, coords: List[int]) -> int:
        return sum(c * s for c, s in zip(coords, self.strides))

    def d1_to_coords(self, idx: int) -> List[int]:
        res = []
        rem = idx
        for s in self.strides:
            c, rem = divmod(rem, s)
            res.append(c)
        return res

    def _build_lattice(self):
        for i in range(self.num_nodes):
            coords = self.d1_to_coords(i)
            neighbors = set()
            for d in range(5):
                for delta in [-1, 1]:
                    nc = list(coords)
                    nc[d] = (nc[d] + delta) % self.dims[d]
                    n_idx = self.coords_to_1d(nc)
                    if n_idx != i:
                        neighbors.add(n_idx)
            self.adjacency_list[i] = sorted(list(neighbors))
