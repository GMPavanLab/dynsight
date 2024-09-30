from __future__ import annotations
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF

def compute_rdf(
        universe: mda.Universe,
        distances_range: list[float],
        s1: str = "all",
        s2: str = "all",
        exclusion_block: list[int] = [1,1],
        nbins: int = 200,
        start: int = None,
        stop: int = None,
        step: int = 1,
        verbose: bool = True
        ):
    selection_1 = universe.select_atoms(s1)
    selection_2 = universe.select_atoms(s2)
    rdf = InterRDF(
        g1=selection_1,
        g2=selection_2,
        nbins=nbins,
        range=distances_range,
        exclusion_block=exclusion_block
    )
    rdf.run(
        verbose=verbose,
        start=start,
        stop=stop,
        step=step
    )
    return rdf.bins, rdf.rdf
