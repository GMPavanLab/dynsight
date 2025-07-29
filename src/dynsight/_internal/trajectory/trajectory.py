from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from MDAnalysis import AtomGroup
    from numpy.typing import NDArray

import MDAnalysis
from MDAnalysis.coordinates.memory import MemoryReader

import dynsight
from dynsight.logs import logger
from dynsight.trajectory import Insight

UNIVAR_DIM = 2


@dataclass(frozen=True)
class Trj:
    """Contains a trajectory.

    Attributes:
        universe: a MDAnalysis.Universe containing the trajectory.

    .. warning::

        This class is under development. The name and type of the "universe"
        attribute may change in the future.
    """

    universe: MDAnalysis.Universe = field()
    trajslice: slice | None = None
    n_atoms: int = field(init=False)
    n_frames: int = field(init=False)

    def __post_init__(self) -> None:
        n_atoms = len(self.universe.atoms)
        if self.trajslice is None:
            n_frames = len(self.universe.trajectory)
        else:
            n_frames = sum(1 for _ in self.universe.trajectory[self.trajslice])
        object.__setattr__(self, "n_atoms", n_atoms)
        object.__setattr__(self, "n_frames", n_frames)

    @classmethod
    def init_from_universe(cls, universe: MDAnalysis.Universe) -> Trj:
        """Initialize Trj object from MDAnalysis.Universe.

        See https://docs.mdanalysis.org/2.9.0/documentation_pages/core/universe.html#MDAnalysis.core.universe.Universe.
        """
        logger.log("Created Trj from MDAnalysis.Universe.")
        return Trj(universe)

    @classmethod
    def init_from_xyz(cls, traj_file: Path, dt: float) -> Trj:
        """Initialize Trj object from .xyz file.

        See https://docs.mdanalysis.org/2.9.0/documentation_pages/core/universe.html#MDAnalysis.core.universe.Universe.

        Parameters:
            dt:
                the trajectory's time-step.
        """
        logger.log(f"Created Trj from {traj_file} with dt = {dt}.")
        universe = MDAnalysis.Universe(traj_file, dt=dt)
        return Trj(universe)

    @classmethod
    def init_from_xtc(cls, traj_file: Path, topo_file: Path) -> Trj:
        """Initialize Trj object from .gro and .xtc files.

        See https://docs.mdanalysis.org/2.9.0/documentation_pages/core/universe.html#MDAnalysis.core.universe.Universe.
        """
        logger.log(f"Created Trj from {traj_file}, {topo_file}.")
        universe = MDAnalysis.Universe(topo_file, traj_file)
        return Trj(universe)

    def get_coordinates(self, selection: str) -> NDArray[np.float64]:
        """Returns the coordinates as an array.

        The array has shape (n_frames, n_atoms, n_coordinates).
        """
        atoms = self.universe.select_atoms(selection)
        trajslice = slice(None) if self.trajslice is None else self.trajslice

        attr_dict = {"selection": selection}
        logger.log(f"Extracted coordinates array with args {attr_dict}.")
        return np.array(
            [
                atoms.positions.copy()
                for ts in self.universe.trajectory[trajslice]
            ]
        )

    def with_slice(self, trajslice: slice | None) -> Trj:
        """Returns a Trj with a different frames' slice."""
        attr_dict = {"trajslice": trajslice}
        logger.log(f"Created a sliced Trj with args {attr_dict}.")
        return Trj(self.universe, trajslice=trajslice)

    def get_slice(self, start: int, stop: int, step: int) -> Trj:
        """Returns a Trj with a subset of frames.

        .. warning::

            This function could fill up the memory in case of large
            trajectories and it's deprecated. Use Trj.with_slice() instead.
        """
        n_atoms = self.universe.atoms.n_atoms

        # Get array of positions from all but the last frame
        frame_indices = list(range(start, stop, step))
        coords = np.empty((len(frame_indices), n_atoms, 3), dtype=np.float32)
        for i, ts in enumerate(self.universe.trajectory[start:stop:step]):
            coords[i] = ts.positions

        mem_reader = MemoryReader(coords, order="fac")
        u_new = MDAnalysis.Universe(topology=self.universe._topology)  # noqa: SLF001
        u_new.trajectory = mem_reader

        attr_dict = {"start": start, "stop": stop, "step": step}
        logger.log(f"Created a sliced Trj with args {attr_dict}.")
        return Trj(u_new)

    def get_coord_number(
        self,
        r_cut: float,
        delay: int = 1,
        selection: str = "all",
        neigcounts: list[list[AtomGroup]] | None = None,
    ) -> tuple[list[list[AtomGroup]], Insight]:
        """Compute coordination number on the trajectory.

        Returns:
            tuple:
                * neighcounts: a list[list[AtomGroup]], it can be used to
                    speed up subsequent descriptors' computations.
                * An Insight containing the number of neighbors. It has the
                    following meta: r_cut, selection.
        """
        if neigcounts is None:
            neigcounts = dynsight.lens.list_neighbours_along_trajectory(
                input_universe=self.universe,
                cutoff=r_cut,
                selection=selection,
                trajslice=self.trajslice,
            )
        _, nn, *_ = dynsight.lens.neighbour_change_in_time(
            neigh_list_per_frame=neigcounts,
            delay=delay,
        )

        attr_dict = {"r_cut": r_cut, "delay": delay, "selection": selection}
        logger.log(f"Computed coord_number using args {attr_dict}.")
        return neigcounts, Insight(
            dataset=nn.astype(np.float64),
            meta=attr_dict,
        )

    def get_lens(
        self,
        r_cut: float,
        delay: int = 1,
        selection: str = "all",
        neigcounts: list[list[AtomGroup]] | None = None,
    ) -> tuple[list[list[AtomGroup]], Insight]:
        """Compute LENS on the trajectory.

        Returns:
            tuple:
                * neighcounts: a list[list[AtomGroup]], it can be used to
                    speed up subsequent descriptors' computations.
                * An Insight containing LENS. It has the following meta:
                    r_cut, selection.
        """
        if neigcounts is None:
            neigcounts = dynsight.lens.list_neighbours_along_trajectory(
                input_universe=self.universe,
                cutoff=r_cut,
                selection=selection,
                trajslice=self.trajslice,
            )
        lens, *_ = dynsight.lens.neighbour_change_in_time(
            neigh_list_per_frame=neigcounts,
            delay=delay,
        )

        attr_dict = {"r_cut": r_cut, "delay": delay, "selection": selection}
        logger.log(f"Computed LENS using args {attr_dict}.")

        return neigcounts, Insight(
            dataset=lens[:, 1:],
            meta=attr_dict,
        )

    def get_soap(
        self,
        r_cut: float,
        n_max: int,
        l_max: int,
        selection: str = "all",
        centers: str = "all",
        respect_pbc: bool = True,
        n_core: int = 1,
    ) -> Insight:
        """Compute SOAP on the trajectory.

        The returned Insight contains the following meta: r_cut, n_max, l_max,
        respect_pbc, centers, selection.
        """
        soap = dynsight.soap.saponify_trajectory(
            self.universe,
            soaprcut=r_cut,
            soapnmax=n_max,
            soaplmax=l_max,
            selection=selection,
            soap_respectpbc=respect_pbc,
            centers=centers,
            n_core=n_core,
            trajslice=self.trajslice,
        )
        attr_dict = {
            "r_cut": r_cut,
            "n_max": n_max,
            "l_max": l_max,
            "respect_pbc": respect_pbc,
            "selection": selection,
            "centers": centers,
        }
        logger.log(f"Computed SOAP with args {attr_dict}.")
        return Insight(dataset=soap, meta=attr_dict)

    def get_orientational_op(
        self,
        r_cut: float,
        order: int = 6,
        selection: str = "all",
        neigcounts: list[list[AtomGroup]] | None = None,
    ) -> tuple[list[list[AtomGroup]], Insight]:
        """Compute the magnitude of the orientational order parameter.

        Returns:
            tuple:
                * neighcounts: a list[list[AtomGroup]], it can be used to
                    speed up subsequent descriptors' computations.
                * An Insight containing the orientational order parameter.
                    It has the following meta: r_cut, order, selection.
        """
        if neigcounts is None:
            neigcounts = dynsight.lens.list_neighbours_along_trajectory(
                input_universe=self.universe,
                cutoff=r_cut,
                selection=selection,
                trajslice=self.trajslice,
            )
        psi = dynsight.descriptors.orientational_order_param(
            self.universe,
            neigh_list_per_frame=neigcounts,
            order=order,
        )

        attr_dict = {"r_cut": r_cut, "order": order, "selection": selection}

        logger.log(
            f"Computed orientational order parameter using args {attr_dict}."
        )

        return neigcounts, Insight(
            dataset=psi,
            meta=attr_dict,
        )

    def get_velocity_alignment(
        self,
        r_cut: float,
        selection: str = "all",
        neigcounts: list[list[AtomGroup]] | None = None,
    ) -> tuple[list[list[AtomGroup]], Insight]:
        """Compute the average velocity alignment.

        Returns:
            tuple:
                * neighcounts: a list[list[AtomGroup]], it can be used to
                    speed up subsequent descriptors' computations.
                * An Insight containing the average velocities alignment.
                    It has the following meta: r_cut, selection.
        """
        if neigcounts is None:
            neigcounts = dynsight.lens.list_neighbours_along_trajectory(
                input_universe=self.universe,
                cutoff=r_cut,
                selection=selection,
                trajslice=self.trajslice,
            )
        phi = dynsight.descriptors.velocity_alignment(
            self.universe,
            neigh_list_per_frame=neigcounts,
        )

        attr_dict = {"r_cut": r_cut, "selection": selection}

        logger.log(
            f"Computed average velocity alignment using args {attr_dict}."
        )

        return neigcounts, Insight(
            dataset=phi,
            meta=attr_dict,
        )

    def get_rdf(
        self,
        distances_range: list[float],
        s1: str = "all",
        s2: str = "all",
        exclusion_block: list[int] | None = None,
        nbins: int = 200,
        norm: Literal["rdf", "density", "none"] = "rdf",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute the radial distribution function g(r).

        See https://docs.mdanalysis.org/1.1.1/documentation_pages/analysis/rdf.html.

        Returns:
            tuple:
                * A list of values of the interparticle distance r
                * The corresponding list of values of g(r)
        """
        trajslice = slice(None) if self.trajslice is None else self.trajslice
        bins, rdf = dynsight.analysis.compute_rdf(
            universe=self.universe,
            distances_range=distances_range,
            s1=s1,
            s2=s2,
            exclusion_block=exclusion_block,
            nbins=nbins,
            norm=norm,
            start=trajslice.start,
            stop=trajslice.stop,
            step=trajslice.step,
        )
        attr_dict = {
            "distances_range": distances_range,
            "s1": s1,
            "s2": s2,
            "exclusion_block": exclusion_block,
            "nbins": nbins,
            "norm": norm,
        }
        logger.log(f"Computed g(r) with args {attr_dict}.")
        return bins, rdf
