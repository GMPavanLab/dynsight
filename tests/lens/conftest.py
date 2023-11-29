from __future__ import annotations

import pathlib  # noqa: TCH003

import dynsight
import MDAnalysis
import numpy as np
import pytest


def fewframeuniverse(
    trajectory: list[list[list[int]]],
    dimensions: list[int],
) -> MDAnalysis.Universe:
    nat = np.shape(trajectory)[1]
    toret = MDAnalysis.Universe.empty(
        n_atoms=nat,
        n_residues=nat,
        n_segments=1,
        atom_resindex=np.arange(nat),
        residue_segindex=[1] * nat,
        trajectory=True,
    )
    toret.load_new(
        np.asarray(trajectory),
        format=MDAnalysis.coordinates.memory.MemoryReader,
        dimensions=dimensions,
    )
    return toret


@pytest.fixture(
    scope="module",
    params=[1, 2],
)
def input1_2(request: pytest.FixtureRequest) -> int:
    return request.param


def giveuniverse(
    angles: tuple[float, float, float] = (90.0, 90.0, 90.0),
    repeatframes: int = 1,
) -> MDAnalysis.Universe:
    traj = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ],
            [
                [0.1, 0.1, 0.1],
                [1.1, 1.1, 1.1],
                [2.1, 2.1, 2.1],
                [3.1, 3.1, 3.1],
            ],
            [
                [0.2, 0.2, 0.2],
                [1.2, 1.2, 1.2],
                [2.2, 2.2, 2.2],
                [3.2, 3.2, 3.2],
            ],
            [
                [0.3, 0.3, 0.3],
                [1.3, 1.3, 1.3],
                [2.3, 2.3, 2.3],
                [3.3, 3.3, 3.3],
            ],
            [
                [0.4, 0.4, 0.4],
                [1.4, 1.4, 1.4],
                [2.4, 2.4, 2.4],
                [3.4, 3.4, 3.4],
            ],
        ]
        * repeatframes
    )
    u = MDAnalysis.Universe.empty(
        4, trajectory=True, atom_resindex=[0, 0, 0, 0], residue_segindex=[0]
    )

    u.add_TopologyAttr("type", ["H"] * 4)
    u.atoms.positions = traj[0]
    u.trajectory = MDAnalysis.coordinates.memory.MemoryReader(
        traj,
        order="fac",
        # this tests the non orthogonality of the box
        dimensions=np.array(
            [[6.0, 6.0, 6.0, angles[0], angles[1], angles[2]]] * traj.shape[0]
        ),
    )
    # adding this for recognisign this univers during tests:
    u.myUsefulName = "FixedBox"
    return u


def giveuniverse_changingbox(
    angles: tuple[float, float, float] = (90.0, 90.0, 90.0)
) -> MDAnalysis.Universe:
    traj = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ],
            [
                [0.1, 0.1, 0.1],
                [1.1, 1.1, 1.1],
                [2.1, 2.1, 2.1],
                [3.1, 3.1, 3.1],
            ],
            [
                [0.2, 0.2, 0.2],
                [1.2, 1.2, 1.2],
                [2.2, 2.2, 2.2],
                [3.2, 3.2, 3.2],
            ],
            [
                [0.3, 0.3, 0.3],
                [1.3, 1.3, 1.3],
                [2.3, 2.3, 2.3],
                [3.3, 3.3, 3.3],
            ],
            [
                [0.4, 0.4, 0.4],
                [1.4, 1.4, 1.4],
                [2.4, 2.4, 2.4],
                [3.4, 3.4, 3.4],
            ],
        ]
    )
    u = MDAnalysis.Universe.empty(
        4, trajectory=True, atom_resindex=[0, 0, 0, 0], residue_segindex=[0]
    )
    dimensions = np.array(
        [[6.0, 6.0, 6.0, angles[0], angles[1], angles[2]]] * traj.shape[0]
    )
    multiplier = np.array(
        [
            [1.5 - 0.5 * np.cos(k / traj.shape[0] * 2 * np.pi)]
            for k in range(traj.shape[0])
        ]
    )
    dimensions[:, :3] *= multiplier

    u.add_TopologyAttr("type", ["H"] * 4)
    u.atoms.positions = traj[0]
    u.trajectory = MDAnalysis.coordinates.memory.MemoryReader(
        traj,
        order="fac",
        # this tests the non orthogonality of the box
        dimensions=dimensions,
    )
    # adding this for recognisign this univers during tests:
    u.myUsefulName = "ChangingBox"
    return u


def giveuniverse_longchangingbox(
    angles: tuple[float, float, float] = (90.0, 90.0, 90.0)
) -> MDAnalysis.Universe:
    trajlen = 300
    traj = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]]
        * trajlen
    )

    rng = np.random.default_rng(12345)
    traj[1:] += rng.random(size=traj[1:].shape) * 2 - 1
    u = MDAnalysis.Universe.empty(
        4, trajectory=True, atom_resindex=[0, 0, 0, 0], residue_segindex=[0]
    )
    dimensions = np.array(
        [[6.0, 6.0, 6.0, angles[0], angles[1], angles[2]]] * traj.shape[0]
    )
    multiplier = np.array(
        [
            [1.5 - 0.5 * np.cos(k / 20 * 2 * np.pi)]
            for k in range(traj.shape[0])
        ]
    )
    dimensions[:, :3] *= multiplier

    u.add_TopologyAttr("type", ["H"] * 4)
    u.atoms.positions = traj[0]
    u.trajectory = MDAnalysis.coordinates.memory.MemoryReader(
        traj,
        order="fac",
        # this tests the non orthogonality of the box
        dimensions=dimensions,
    )
    # adding this for recognisign this univers during tests:
    u.myUsefulName = "ChangingBox"
    return u


@pytest.fixture(
    scope="session",
    params=[
        giveuniverse,
        giveuniverse_changingbox,
        giveuniverse_longchangingbox,
    ],
)
def input_universe(request: pytest.FixtureRequest) -> MDAnalysis.Universe:
    return request.param


@pytest.fixture(scope="session")
def hdf5_file(
    tmp_path_factory: pytest.TempdirFactory,
    input_universe: MDAnalysis.Universe,
) -> tuple[pathlib.Path, MDAnalysis.Universe]:
    fouratomsfiveframes = input_universe((90.0, 90.0, 90.0))

    testfname = (
        tmp_path_factory.mktemp("data")
        / f"test{fouratomsfiveframes.myUsefulName}.hdf5"
    )

    dynsight.hdf5er.mda_to_hdf5(
        fouratomsfiveframes, testfname, "4Atoms5Frames", override=True
    )

    return testfname, fouratomsfiveframes


def lensiszerofixtures() -> MDAnalysis.Universe:
    # no change in NN
    return (
        fewframeuniverse(
            trajectory=[
                [[0, 0, 0], [0, 0, 1], [5, 5, 5], [5, 5, 6]],
                [[0, 0, 0], [0, 0, 1], [5, 5, 5], [5, 5, 6]],
            ],
            dimensions=[10, 10, 10, 90, 90, 90],
        ),
        [0] * 4,
    )


def lensiszeronnfixtures() -> MDAnalysis.Universe:
    # Zero NN
    return (
        fewframeuniverse(
            trajectory=[
                [[0, 0, 0], [5, 5, 5]],
                [[0, 0, 0], [5, 5, 5]],
            ],
            dimensions=[10, 10, 10, 90, 90, 90],
        ),
        [0] * 2,
    )


def lensisonefixtures() -> MDAnalysis.Universe:
    # all NN changes
    return (
        fewframeuniverse(
            trajectory=[
                [[0, 0, 0], [0, 0, 1], [5, 5, 5], [0, 1, 0]],
                [[0, 0, 0], [5, 5, 6], [5, 5, 5], [5, 6, 5]],
            ],
            dimensions=[10, 10, 10, 90, 90, 90],
        ),
        [1] * 4,
    )


getuni = {
    "LENSISZERO": lensiszerofixtures(),
    "LENSISZERONN": lensiszeronnfixtures(),
    "LENSISONE": lensisonefixtures(),
}


@pytest.fixture(
    scope="module",
    params=["LENSISZERO", "LENSISZERONN", "LENSISONE"],
)
def lensfixtures(request: pytest.FixtureRequest) -> MDAnalysis.Universe:
    return getuni[request.param]
