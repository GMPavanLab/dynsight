from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import MDAnalysis
    import numpy as np

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
    verbose: bool = True,
) -> tuple[np.ndarray[int, Any], np.ndarray[float, Any]]:
    r"""Radial Distribution Function between two types of particles.

    prova prova.

    $$
    g_{ab}(r) = \frac{\langle \rho_b(r) \rangle}{\rho_b}
    $$
    bye.
    """
    if exclusion_block is None:
        exclusion_block = [1, 1]

    selection_1 = universe.select_atoms(s1)
    selection_2 = universe.select_atoms(s2)

    rdf = InterRDF(
        g1=selection_1,
        g2=selection_2,
        nbins=nbins,
        range=distances_range,
        exclusion_block=exclusion_block,
    )

    rdf.run(verbose=verbose, start=start, stop=stop, step=step)
    return rdf.bins, rdf.rdf
