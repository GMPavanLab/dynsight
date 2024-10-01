from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import MDAnalysis
    import numpy as np
    import numpy.typing as npt

from MDAnalysis.analysis.rdf import InterRDF


def compute_rdf(
    universe: MDAnalysis.Universe,
    distances_range: list[float],
    s1: str = "all",
    s2: str = "all",
    exclusion_block: list[int] | None = None,
    nbins: int = 200,
    start: int | None = None,
    stop: int | None = None,
    step: int = 1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Radial Distribution Function between two types of particles.

    The RDF between two types of particles `a` and `b` is defined as:

    .. math::
        g_{ab}(r) = (N_a N_b)^{-1} \sum_{i=1}^{N_a}\sum_{j=1}^{N_b}
        \langle \sigma(|\mathbf{r_i}\mathbf{r_j}| - r)\rangle

    The radial distribution function is calculated by histogramming distances
    between two groups of atom `s1` and `s2`. Periodic boundary conditions are
    taken into account via the minimum-image convention. More information
    concerning this function can be found
    `here <https://docs.mdanalysis.org/1.1.1/documentation_pages/analysis/rdf.html>`_.

    Parameters:
        universe:
            Simulation MDAnalysis universe.
        s1:
            First atom group.
        s2:
            Second atom group.
        distances_range:
            Initial and final distance within which compute the RDF.
        exclusion_block:
            A tuple specifying the size of blocks
            (e.g., molecules) to exclude distances between atoms within the
            same block. If `s1` and `s2` are equal, it prevents
            self-interactions by default with (1, 1).
        nbins:
            The number of bins used to divide the distance range for histogram
            the RDF.
        start:
            Initial molecular dynamics step.
        stop:
            Final molecular dynamics step.
        step:
            Frequency at which the dynamic is sampled.

    Returns:
        Two arrays where the pair separation distance and the RDF values
        are stored respectively.
    """
    selection_1 = universe.select_atoms(s1)
    selection_2 = universe.select_atoms(s2)

    if exclusion_block is None and selection_1 == selection_2:
        exclusion_block = [1, 1]

    rdf = InterRDF(
        g1=selection_1,
        g2=selection_2,
        nbins=nbins,
        range=distances_range,
        exclusion_block=exclusion_block,
    )

    rdf.run(verbose=True, start=start, stop=stop, step=step)
    return rdf.bins, rdf.rdf
