from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from MDAnalysis import AtomGroup
    from numpy.typing import NDArray

import logging

import MDAnalysis
from MDAnalysis.coordinates.memory import MemoryReader

import dynsight
from dynsight.logs import logger
from dynsight.trajectory import Insight

UNIVAR_DIM = 2
logging.getLogger("MDAnalysis").setLevel(logging.ERROR)


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
        centers: str = "all",
        selection: str = "all",
        respect_pbc: bool = True,
        neigcounts: list[list[AtomGroup]] | None = None,
        n_jobs: int = 1,
    ) -> tuple[list[list[AtomGroup]], Insight]:
        """Compute coordination number on the trajectory.

        Returns:
            tuple:
                * neighcounts: a list[list[AtomGroup]], it can be used to
                    speed up subsequent descriptors' computations.
                * An Insight containing the number of neighbors. It has the
                    following meta: name, r_cut, centers, selection.
        """
        if neigcounts is None:
            neigcounts = dynsight.lens.list_neighbours_along_trajectory(
                universe=self.universe,
                r_cut=r_cut,
                centers=centers,
                selection=selection,
                trajslice=self.trajslice,
                respect_pbc=respect_pbc,
                n_jobs=n_jobs,
            )

        n_frames = len(neigcounts)
        n_atoms = len(neigcounts[0])
        counts = np.zeros((n_atoms, n_frames), dtype=int)

        for f, frame in enumerate(neigcounts):
            for a, atom_group in enumerate(frame):
                counts[a, f] = len(atom_group)

        attr_dict = {
            "name": "coord_number",
            "r_cut": r_cut,
            "centers": centers,
            "selection": selection,
        }
        logger.log(f"Computed coord_number using args {attr_dict}.")
        return neigcounts, Insight(
            dataset=counts.astype(np.float64),
            meta=attr_dict,
        )

    def get_lens(
        self,
        r_cut: float,
        delay: int = 1,
        centers: str = "all",
        selection: str = "all",
        respect_pbc: bool = True,
        n_jobs: int = 1,
    ) -> Insight:
        """Compute LENS on the trajectory.

        Returns:
            Insight
                An Insight containing LENS. It has the following meta:
                name, r_cut, delay, centers, selection.
        """
        lens = dynsight.lens.compute_lens(
            universe=self.universe,
            r_cut=r_cut,
            delay=delay,
            centers=centers,
            selection=selection,
            trajslice=self.trajslice,
            respect_pbc=respect_pbc,
            n_jobs=n_jobs,
        )

        attr_dict = {
            "name": "lens",
            "r_cut": r_cut,
            "delay": delay,
            "centers": centers,
            "selection": selection,
        }
        logger.log(f"Computed LENS using args {attr_dict}.")

        return Insight(
            dataset=lens,
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
        n_jobs: int = 1,
    ) -> Insight:
        """Compute SOAP on the trajectory.

        The returned Insight contains the following meta: name, r_cut, n_max,
        l_max, respect_pbc, selection, centers.
        """
        soap = dynsight.soap.saponify_trajectory(
            self.universe,
            soaprcut=r_cut,
            soapnmax=n_max,
            soaplmax=l_max,
            selection=selection,
            soap_respectpbc=respect_pbc,
            centers=centers,
            n_core=n_jobs,
            trajslice=self.trajslice,
        )
        attr_dict = {
            "name": "soap",
            "r_cut": r_cut,
            "n_max": n_max,
            "l_max": l_max,
            "respect_pbc": respect_pbc,
            "selection": selection,
            "centers": centers,
        }
        logger.log(f"Computed SOAP with args {attr_dict}.")
        return Insight(dataset=soap, meta=attr_dict)

    def get_timesoap(
        self,
        r_cut: float | None = None,
        n_max: int | None = None,
        l_max: int | None = None,
        soap_insight: Insight | None = None,
        selection: str = "all",
        centers: str = "all",
        respect_pbc: bool = True,
        n_jobs: int = 1,
        delay: int = 1,
    ) -> tuple[Insight, Insight]:
        """Compute SOAP and then timeSOAP on the trajectory.

        The returned Insights (soap and timesoap) contain the following meta:
        name, r_cut, n_max, l_max, respect_pbc, selection, centers.
        Regarding the timeSOAP Insight, the delay used is also included.
        """
        if soap_insight is not None:
            if getattr(soap_insight, "meta", {}).get("name") != "soap":
                msg = (
                    f"soap_insight.meta['name'] must be 'soap', found: "
                    f"{soap_insight.meta.get('name', None)}"
                )
                raise ValueError(msg)
            msg = (
                "Loaded existing soap_insight: parameters r_cut, n_max, l_max,"
                " selection, centers, and respect_pbc will be ignored."
            )
            logger.log(msg)
            soap = soap_insight
        else:
            if r_cut is None or n_max is None or l_max is None:
                msg = (
                    "r_cut, n_max e l_max cannot be None"
                    " if the soap_insight is not provided."
                )
                raise ValueError(msg)

            soap = self.get_soap(
                r_cut=r_cut,
                n_max=n_max,
                l_max=l_max,
                selection=selection,
                centers=centers,
                respect_pbc=respect_pbc,
                n_jobs=n_jobs,
            )
            logger.log(f"Computed SOAP with args {soap.meta}.")

        timesoap = soap.get_angular_velocity(delay=delay)

        logger.log(f"Computed timeSOAP with args {timesoap.meta}.")
        return soap, timesoap

    def get_orientational_op(
        self,
        r_cut: float,
        order: int = 6,
        centers: str = "all",
        selection: str = "all",
        respect_pbc: bool = True,
        neigcounts: list[list[AtomGroup]] | None = None,
        n_jobs: int = 1,
    ) -> tuple[list[list[AtomGroup]], Insight]:
        """Compute the magnitude of the orientational order parameter.

        Returns:
            tuple:
                * neighcounts: a list[list[AtomGroup]], it can be used to
                    speed up subsequent descriptors' computations.
                * An Insight containing the orientational order parameter.
                    It has the following meta: name, r_cut, order, centers,
                    selection.
        """
        if neigcounts is None:
            neigcounts = dynsight.lens.list_neighbours_along_trajectory(
                universe=self.universe,
                r_cut=r_cut,
                centers=centers,
                selection=selection,
                trajslice=self.trajslice,
                respect_pbc=respect_pbc,
                n_jobs=n_jobs,
            )
        psi = dynsight.descriptors.orientational_order_param(
            self.universe,
            neigh_list_per_frame=neigcounts,
            order=order,
        )

        attr_dict = {
            "name": "orientational_op",
            "r_cut": r_cut,
            "order": order,
            "centers": centers,
            "selection": selection,
        }

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
        centers: str = "all",
        selection: str = "all",
        respect_pbc: bool = True,
        neigcounts: list[list[AtomGroup]] | None = None,
        n_jobs: int = 1,
    ) -> tuple[list[list[AtomGroup]], Insight]:
        """Compute the average velocity alignment.

        Returns:
            tuple:
                * neighcounts: a list[list[AtomGroup]], it can be used to
                    speed up subsequent descriptors' computations.
                * An Insight containing the average velocities alignment.
                    It has the following meta: name, r_cut, centers, selection.
        """
        if neigcounts is None:
            neigcounts = dynsight.lens.list_neighbours_along_trajectory(
                universe=self.universe,
                r_cut=r_cut,
                centers=centers,
                selection=selection,
                trajslice=self.trajslice,
                respect_pbc=respect_pbc,
                n_jobs=n_jobs,
            )

        phi = dynsight.descriptors.velocity_alignment(
            self.universe,
            neigh_list_per_frame=neigcounts,
        )

        attr_dict = {
            "name": "velocity_alignement",
            "r_cut": r_cut,
            "centers": centers,
            "selection": selection,
        }

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
            "name": "rdf",
            "distances_range": distances_range,
            "s1": s1,
            "s2": s2,
            "exclusion_block": exclusion_block,
            "nbins": nbins,
            "norm": norm,
        }
        logger.log(f"Computed g(r) with args {attr_dict}.")
        return bins, rdf

    def dump_colored_trj(
        self,
        labels: NDArray[np.int64],
        file_path: Path,
    ) -> None:
        """Save an .xyz file with the labels for each atom.

        The output file has columns: atom_type, x, y, z, label.
        """
        trajslice = slice(None) if self.trajslice is None else self.trajslice

        if labels.shape != (self.n_atoms, self.n_frames):
            msg = (
                f"Shape mismatch: ClusterInsight should have "
                f"{self.n_atoms} atoms, {self.n_frames} frames, but has "
                f"{labels.shape[0]} atoms, {labels.shape[1]} frames."
            )
            logger.log(msg)
            raise ValueError(msg)

        lab_new = labels + 2
        with file_path.open("w") as f:
            for i, ts in enumerate(self.universe.trajectory[trajslice]):
                f.write(f"{self.n_atoms}\n")
                if ts.dimensions is not None:
                    box_str = " ".join(f"{x:.5f}" for x in ts.dimensions)
                else:
                    box_str = "0.0 0.0 0.0 0.0 0.0 0.0"
                f.write(
                    f"Lattice={box_str} "
                    f"Properties=species:S:1:pos:R:3:type:I:1\n"
                )
                for atom_idx in range(self.n_atoms):
                    label = str(lab_new[atom_idx, i])
                    x, y, z = ts.positions[atom_idx]
                    f.write(
                        f"{self.universe.atoms[atom_idx].name} {x:.5f}"
                        f" {y:.5f} {z:.5f} {label}\n"
                    )
        logger.log(f"Colored trj saved to {file_path}.")
