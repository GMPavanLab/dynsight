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
    gauss2 = np.exp(-((x - 1) ** 2 + y**2) / (2 * sigma**2))
    return -np.log(gauss1 + gauss2 + 1e-6)


def energy_landscape_2(x: float, y: float) -> float:
    """A potential energy landscape with 3 minima."""
    sigma = 0.12  # Width of the Gaussian wells
    gauss1 = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gauss2 = np.exp(-((x - 1) ** 2 + y**2) / (2 * sigma**2))
    gauss3 = np.exp(-(x**2 + (y - 1) ** 2) / (2 * sigma**2))
    return -np.log(gauss1 + gauss2 + gauss3 + 1e-6)


def numerical_gradient(
    f: Callable[[float, float], float], x: float, y: float, h: float = 1e-5
) -> tuple[float, float]:
    """Compute numerical gradient using finite differences."""
    grad_x = (f(x + h, y) - f(x - h, y)) / (2 * h)
    grad_y = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return -grad_x, -grad_y


def create_trajectory(
    energy_landscape: Callable[[float, float], float], name: str
) -> NDArray[np.float64]:
    """Simulate a Langevin Dynamics on a given energy landscape."""
    rng = np.random.default_rng(0)
    n_atoms = 100
    time_steps = 10000
    dt = 0.01  # Time step
    diffusion_coeff = 0.6  # Diffusion coefficient (random noise strength)

    particles = rng.standard_normal((n_atoms, 2)) * 0.2
    particles[n_atoms // 2 :, 0] += 1

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
    np.save(f"info_gain/trj_{name}.npy", dataset)
    return dataset


def main() -> None:
    """How to use the code for the information gain in clustering analysis."""
    # Let's build bidimensional variables - Langevin Dynamics in 2D
    file_1 = Path("info_gain/trj_1.npy")  #  With 2 minima
    file_2 = Path("info_gain/trj_2.npy")  #  With 3 minima
    if file_1.exists():
        dataset_1 = np.load(file_1)
    else:
        dataset_1 = create_trajectory(energy_landscape_1, "1")
    if file_2.exists():
        dataset_2 = np.load(file_2)
    else:
        dataset_2 = create_trajectory(energy_landscape_2, "2")

    results = []
    delta_t_list = np.unique(np.geomspace(2, 300, 40, dtype=int))
    example_delta_t = 4  #  Choosing a ∆t which works well to show results

    for k, dataset in enumerate([dataset_1, dataset_2]):
        n_atoms, n_frames, n_dims = dataset.shape

        # We can do clustering on one single variable:
        x_positions = dataset[:, :, 0]
        info_gain_x = np.zeros(delta_t_list.size)

        for i, delta_t in enumerate(delta_t_list):
            reshaped_data = dynsight.onion.helpers.reshape_from_nt(
                x_positions, delta_t
            )
            state_list, labels = dynsight.onion.onion_uni(reshaped_data)

            if i == example_delta_t:
                dynsight.onion.plot.plot_output_uni(
                    f"info_gain/tmp_{k}_1D.png",
                    reshaped_data,
                    n_atoms,
                    state_list,
                )

            # and compute the relative information gain:
            info_gain_x[i], *_ = dynsight.analysis.compute_entropy_gain(
                reshaped_data, labels, n_bins=40
            )
        results.append(info_gain_x)

        # We can do clustering on both variables:
        info_gain_xy = np.zeros(delta_t_list.size)
        tmp1_dataset = np.transpose(dataset, (2, 0, 1))
        for i, delta_t in enumerate(delta_t_list):
            reshaped_data = dynsight.onion.helpers.reshape_from_dnt(
                tmp1_dataset, delta_t
            )
            state_list, labels = dynsight.onion.onion_multi(reshaped_data)

            if i == example_delta_t:
                dynsight.onion.plot.plot_output_multi(
                    f"info_gain/tmp_{k}_2D.png",
                    tmp1_dataset,
                    state_list,
                    labels,
                    delta_t,
                )

            # and compute the relative information gain:
            # We need an array (n_samples, n_dims)
            n_sequences = int(n_frames / delta_t)
            long_labels = np.repeat(labels, delta_t)
            tmp = dataset[:, : n_sequences * delta_t, :]
            ds_reshaped = tmp.reshape((-1, n_dims))

            info_gain_xy[i], *_ = dynsight.analysis.compute_multivariate_gain(
                ds_reshaped, long_labels, n_bins=[40, 40]
            )
        info_gain_xy *= 2  # Need to multiply by two because it's 2 dimensional
        results.append(info_gain_xy)

    fig, ax = plt.subplots()
    ax.plot(
        delta_t_list, results[0], label="2 minima - 1D clustering", marker="o"
    )
    ax.plot(
        delta_t_list,
        results[1],
        label="2 minima - 2D clustering",
        marker="o",
        c="C2",
    )
    ax.plot(
        delta_t_list,
        results[2],
        label="3 minima - 1D clustering",
        marker="o",
        ls="--",
        c="C1",
    )
    ax.plot(
        delta_t_list,
        results[3],
        label="3 minima - 2D clustering",
        marker="o",
        c="C3",
    )
    ax.set_xlabel(r"Time resolution $\Delta t$ [frame]")
    ax.set_ylabel(r"Information gain $\Delta H$ [bit]")
    ax.set_xscale("log")
    ax.legend()
    plt.show()
    fig.savefig("info_gain/Information_gains.png", dpi=600)


if __name__ == "__main__":
    main()
