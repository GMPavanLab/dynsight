from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import MDAnalysis
    import numpy as np
    from numpy.typing import NDArray

import dynsight


class Insight:
    def __init__(self, data: NDArray[np.float64], param: float) -> None:
        self.dataset = data
        self.param = param

    def spatial_average(self, trajectory: Trj) -> None:
        pass

    def get_onion(self, delta_t: int) -> None:
        pass

    def save(self) -> None:
        pass


class Trj:
    def __init__(self, universe: MDAnalysis.Universe) -> None:
        self.trj = universe
        self.dt = universe.trajectory.dt

    def get_lens(self, r_cut: float) -> Insight:
        neigcounts = dynsight.lens.list_neighbours_along_trajectory(
            input_universe=self.trj,
            cutoff=r_cut,
        )
        lens, *_ = dynsight.lens.neighbour_change_in_time(neigcounts)
        return Insight(lens, r_cut)

    def save(self) -> None:
        pass
