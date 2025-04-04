"""How to use the code for the information gain in clustering analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

import dynsight


def energy_landscape_1(x: float, y: float) -> float:
    """A potential energy landscape with 2 minima."""
    sigma = 0.12  # Width of the Gaussian wells
    gauss1 = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gauss2 = np.exp(-(x**2 + (y - 1) ** 2) / (2 * sigma**2))
    return -np.log(gauss1 + gauss2 + 1e-6)


def energy_landscape_2(x: float, y: float) -> float:
    """A potential energy landscape with 4 minima."""
    sigma = 0.12  # Width of the Gaussian wells
    gauss1 = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gauss2 = np.exp(-((x - 1) ** 2 + y**2) / (2 * sigma**2))
    gauss3 = np.exp(-(x**2 + (y - 1) ** 2) / (2 * sigma**2))
    gauss4 = np.exp(-((x - 1) ** 2 + (y - 1) ** 2) / (2 * sigma**2))
    return -np.log(gauss1 + gauss2 + gauss3 + gauss4 + 1e-6)


def numerical_gradient(
    f: Callable[[float, float], float], x: float, y: float, h: float = 1e-5
) -> tuple[float, float]:
    """Compute numerical gradient using finite differences."""
    grad_x = (f(x + h, y) - f(x - h, y)) / (2 * h)
    grad_y = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return -grad_x, -grad_y


def create_trajectory(
    energy_landscape: Callable[[float, float], float], file_name: Path
) -> NDArray[np.float64]:
    """Simulate Langevin Dynamics on a given energy landscape."""
    rng = np.random.default_rng(0)
    n_atoms = 100
    time_steps = 10000
    dt = 0.01  # Time step
    diffusion_coeff = 0.6  # Diffusion coefficient (random noise strength)

    if energy_landscape == energy_landscape_1:
        particles = rng.standard_normal((n_atoms, 2)) * 0.2
        particles[n_atoms // 2 :, 1] += 1.0
    else:
        particles = rng.standard_normal((n_atoms, 2)) * 0.2
        n_group = n_atoms // 4
        particles[n_group : 2 * n_group, 1] += 1  # (0, 1)
        particles[2 * n_group : 3 * n_group, 0] += 1  # (1, 0)
        particles[3 * n_group :, 0] += 1  # (1, 1)
        particles[3 * n_group :, 1] += 1

    trajectory = np.zeros((time_steps, n_atoms, 2))
    for t in range(time_steps):
        for i in range(n_atoms):
            x, y = particles[i]
            fx, fy = numerical_gradient(energy_landscape, x, y)
            noise_x = np.sqrt(2 * diffusion_coeff * dt) * rng.standard_normal()
            noise_y = np.sqrt(2 * diffusion_coeff * dt) * rng.standard_normal()

            # Update position with deterministic force and stochastic term
            particles[i, 0] += fx * dt + noise_x
            particles[i, 1] += fy * dt + noise_y

            trajectory[t, i] = particles[i]

    plt.figure()
    plt.plot(trajectory[:, :, 0], trajectory[:, :, 1])
    plt.show()

    dataset = np.transpose(trajectory, (1, 0, 2))
    np.save(file_name, dataset)
    return dataset


def main() -> None:
    """How to use the code for the information gain in clustering analysis."""
    cwd = Path.cwd()
    folder_name = "info_gain"
    folder_path = cwd / folder_name
    if not folder_path.exists():
        folder_path.mkdir()

    # Let's build bidimensional variables - Langevin Dynamics in 2D
    file_1 = folder_path / "trj_2.npy"  #  With 2 minima
    file_2 = folder_path / "trj_4.npy"  #  With 4 minima

    if not file_1.exists():
        dataset_1 = create_trajectory(energy_landscape_1, file_1)
    dataset_1 = np.load(file_1)

    if not file_2.exists():
        dataset_2 = create_trajectory(energy_landscape_2, file_2)
    dataset_2 = np.load(file_2)

    results = []
    env0 = []
    delta_t_list = np.unique(np.geomspace(2, 1000, 45, dtype=int))
    example_delta_t = 4  #  Choosing a ∆t which works well to plot results

    for i, dataset in enumerate([dataset_1, dataset_2]):
        n_atoms, n_frames, n_dims = dataset.shape

        # We can do clustering on one single variable:
        y_positions = dataset[:, :, 1]
        info_gain_y = np.zeros(delta_t_list.size)
        tmp_env0 = np.zeros(delta_t_list.size)

        for j, delta_t in enumerate(delta_t_list):
            reshaped_data = dynsight.onion.helpers.reshape_from_nt(
                y_positions, delta_t
            )
            state_list, labels = dynsight.onion.onion_uni(reshaped_data)
            tmp_env0[j] = np.sum(labels == -1) / labels.size

            if j == example_delta_t:
                dynsight.onion.plot.plot_output_uni(
                    f"info_gain/tmp_{i}_1D.png",
                    reshaped_data,
                    n_atoms,
                    state_list,
                )

            # and compute the information gain:
            info_gain_y[j], *_ = dynsight.analysis.compute_entropy_gain(
                reshaped_data, labels, n_bins=40
            )
        results.append(info_gain_y)
        env0.append(tmp_env0)

        # Or we can do clustering on both variables:
        info_gain_xy = np.zeros(delta_t_list.size)
        tmp_env0 = np.zeros(delta_t_list.size)
        tmp1_dataset = np.transpose(dataset, (2, 0, 1))
        for j, delta_t in enumerate(delta_t_list):
            reshaped_data = dynsight.onion.helpers.reshape_from_dnt(
                tmp1_dataset, delta_t
            )
            state_list, labels = dynsight.onion.onion_multi(reshaped_data)
            tmp_env0[j] = np.sum(labels == -1) / labels.size

            if j == example_delta_t:
                dynsight.onion.plot.plot_output_multi(
                    f"info_gain/tmp_{i}_2D.png",
                    tmp1_dataset,
                    state_list,
                    labels,
                    delta_t,
                )

            # and compute the information gain:
            # We need an array (n_samples, n_dims), and labels (n_samples,)
            n_sequences = int(n_frames / delta_t)
            long_labels = np.repeat(labels, delta_t)
            tmp = dataset[:, : n_sequences * delta_t, :]
            ds_reshaped = tmp.reshape((-1, n_dims))

            info_gain_xy[j], *_ = dynsight.analysis.compute_entropy_gain_multi(
                ds_reshaped, long_labels, n_bins=[40, 40]
            )
        # Need to multiply by two because it's 2 dimensional, and the output
        # of the info_gain functions is normalized by the log volume of the
        # phase space, which is 2D is doubled
        info_gain_xy *= 2
        results.append(info_gain_xy)
        env0.append(tmp_env0)

    colorlist = ["C0", "C2", "C1", "C3"]
    markerlist = ["s", "o", "d", "o"]
    labellist = [
        "2 peaks - 1D clustering",
        "2 peaks - 2D clustering",
        "4 peaks - 1D clustering",
        "4 peaks - 2D clustering",
    ]

    fig, ax = plt.subplots()
    for i, system in enumerate(results):
        ax.plot(
            delta_t_list,
            system,
            label=labellist[i],
            c=colorlist[i],
            marker=markerlist[i],
        )

    ax.set_xlabel(r"Time resolution $\Delta t$ [frame]")
    ax.set_ylabel(r"Information gain $\Delta H$ [bit]")
    ax.set_xscale("log")
    ax.legend()
    fig.savefig("info_gain/Information_gains.png", dpi=600)


if __name__ == "__main__":
    main()
