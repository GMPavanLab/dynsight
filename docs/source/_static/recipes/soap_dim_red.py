"""Copiable code from Recipe #2."""

from __future__ import annotations

from pathlib import Path

from sklearn.decomposition import PCA

import dynsight
from dynsight.trajectory import Insight, Trj
from dynsight.utilities import load_or_compute_soap


def compute_soap_pca(
    trj: Trj,
    r_cut: float,
    n_max: int,
    l_max: int,
    n_components: int,
    soap_path: Path | None = None,
    pca_path: Path | None = None,
    selection: str = "all",
    centers: str = "all",
    respect_pbc: bool = True,
    n_core: int = 1,
) -> Insight:
    """Computes the PCA of SOAP on a Trajectory."""
    if pca_path is not None and pca_path.exists():
        return Insight.load_from_json(pca_path)

    soap = load_or_compute_soap(
        trj=trj,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        selection=selection,
        centers=centers,
        respect_pbc=respect_pbc,
        n_core=n_core,
        soap_path=soap_path,
    )

    n_atom, n_frames, n_dims = soap.dataset.shape
    reshaped_soap = soap.dataset.reshape(n_atom * n_frames, n_dims)
    pca = PCA(n_components=n_components)
    transformed_soap = pca.fit_transform(reshaped_soap)
    pca_ds = transformed_soap.reshape(n_atom, n_frames, -1)

    soap_pca = Insight(pca_ds, meta=soap.meta.copy())

    if pca_path is not None:
        soap_pca.dump_to_json(pca_path)

    return soap_pca


def compute_soap_tica(
    trj: Trj,
    r_cut: float,
    n_max: int,
    l_max: int,
    lag_time: int,
    tica_dim: int,
    soap_path: Path | None = None,
    tica_path: Path | None = None,
    selection: str = "all",
    centers: str = "all",
    respect_pbc: bool = True,
    n_core: int = 1,
) -> Insight:
    """Computes the tICA of SOAP on a Trajectory."""
    if tica_path is not None and tica_path.exists():
        return Insight.load_from_json(tica_path)

    soap = load_or_compute_soap(
        trj=trj,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        selection=selection,
        centers=centers,
        respect_pbc=respect_pbc,
        n_core=n_core,
        soap_path=soap_path,
    )

    rel_times, _, tica_ds = dynsight.descriptors.many_body_tica(
        soap.dataset,
        lag_time=lag_time,
        tica_dim=tica_dim,
    )

    meta = soap.meta.copy()
    meta.update(
        {
            "lag_time": lag_time,
            "rel_times": rel_times,
        }
    )
    soap_tica = Insight(tica_ds, meta=meta)

    if tica_path is not None:
        soap_tica.dump_to_json(tica_path)

    return soap_tica


def compute_timesoap(
    trj: Trj,
    r_cut: float,
    n_max: int,
    l_max: int,
    delay: int = 1,
    soap_path: Path | None = None,
    tsoap_path: Path | None = None,
    selection: str = "all",
    centers: str = "all",
    respect_pbc: bool = True,
    n_core: int = 1,
) -> Insight:
    """Computes timeSOAP on a Trajectory."""
    if tsoap_path is not None and tsoap_path.exists():
        return Insight.load_from_json(tsoap_path)

    soap = load_or_compute_soap(
        trj=trj,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        selection=selection,
        centers=centers,
        respect_pbc=respect_pbc,
        n_core=n_core,
        soap_path=soap_path,
    )

    tsoap = soap.get_angular_velocity(delay=delay)

    if tsoap_path is not None:
        tsoap.dump_to_json(tsoap_path)

    return tsoap


def main() -> None:
    """Copiable code from Recipe #2."""
    files_path = Path("../tests/systems/")
    trj = Trj.init_from_xtc(
        traj_file=files_path / "balls_7_nvt.xtc",
        topo_file=files_path / "balls_7_nvt.gro",
    )

    _ = compute_soap_pca(
        trj=trj,
        r_cut=2.0,
        n_max=4,
        l_max=4,
        n_components=1,
    )

    _ = compute_soap_tica(
        trj=trj,
        r_cut=10.0,
        n_max=4,
        l_max=4,
        lag_time=10,
        tica_dim=1,
    )

    _ = compute_timesoap(
        trj=trj,
        r_cut=10.0,
        n_max=4,
        l_max=4,
    )


if __name__ == "__main__":
    main()
