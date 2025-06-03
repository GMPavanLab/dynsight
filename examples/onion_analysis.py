"""Analysis of particles moving between Gaussian energy minima.

In this example, a dataset is created composed by the (x, y) coordinates
of particles moving in a 2D energy landscape with 4 energy minima.

Onion clustering is applied initially to the univariate dataset of x
positions, finding only two clusters.

Then, applying it on the full (x, y) bivariate dataset, all the 4 minima are
found.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynsight.trajectory import Insight


def energy_landscape(x: float, y: float) -> float:
    """A 2-dimensional potential energy landscape with 4 minima."""
    sigma = 0.12  # Width of the Gaussian wells
    gauss1 = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gauss2 = np.exp(-((x - 1) ** 2 + y**2) / (2 * sigma**2))
    gauss3 = np.exp(-(x**2 + (y - 1) ** 2) / (2 * sigma**2))
    gauss4 = np.exp(-((x - 1) ** 2 + (y - 1) ** 2) / (2 * sigma**2))
    return -np.log(gauss1 + gauss2 + gauss3 + gauss4 + 1e-6)


def numerical_gradient(
    x: float, y: float, h: float = 1e-5
) -> tuple[float, float]:
    """Compute numerical gradient using finite differences."""
    grad_x = (energy_landscape(x + h, y) - energy_landscape(x - h, y)) / (
        2 * h
    )
    grad_y = (energy_landscape(x, y + h) - energy_landscape(x, y - h)) / (
        2 * h
    )
    return -grad_x, -grad_y


def create_trajectory(
    n_atoms: int,
    time_steps: int,
    file_path: Path,
) -> Insight:
    """Simulate Langevin Dynamics on a given energy landscape."""
    rng = np.random.default_rng(0)
    dt = 0.01  # Time step
    diffusion_coeff = 0.8  # Diffusion coefficient (random noise strength)

    # Initialize particles' positions
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
            fx, fy = numerical_gradient(x, y)
            noise_x = np.sqrt(2 * diffusion_coeff * dt) * rng.standard_normal()
            noise_y = np.sqrt(2 * diffusion_coeff * dt) * rng.standard_normal()

            # Update position with deterministic force and stochastic term
            particles[i, 0] += fx * dt + noise_x
            particles[i, 1] += fy * dt + noise_y

            trajectory[t, i] = particles[i]

    dataset = np.transpose(trajectory, (1, 0, 2))
    insight = Insight(dataset)
    insight.dump_to_json(file_path)

    return insight


def main() -> None:
    """Analysis of particles moving between Gaussian energy minima."""
    data_path = Path("onion_analysis")

    # Load or create the input dataset
    # Here we are using the (x,y) coordinates as the descriptor for the system
    n_atoms = 100
    n_frames = 10000
    file_path = data_path / "data.json"
    if file_path.exists():
        coord_2d = Insight.load_from_json(file_path)
    else:
        coord_2d = create_trajectory(n_atoms, n_frames, file_path)
    coord_1d = Insight(coord_2d.dataset[:, :, 0])

    # Test onion clustering on a wide range of time resolutions
    delta_t_list, n_clust, unclass_frac = coord_1d.get_onion_analysis(
        fig1_path=data_path / "time-res_1d.png",
        fig2_path=data_path / "pop_fracs_1d.png",
    )

    # Perform onion clustering at delta_t = 100
    onion_results = coord_1d.get_onion_smooth(delta_t=100)
    onion_results.plot_output(data_path / "output_1d.png", coord_1d)
    onion_results.plot_one_trj(data_path / "example_1d.png", coord_1d, 10)
    onion_results.plot_state_populations(data_path / "state_pops_1d.png")

    # Test onion clustering on a wide range of time resolutions
    delta_t_list, n_clust, unclass_frac = coord_2d.get_onion_analysis(
        fig1_path=data_path / "time-res_2d.png",
        fig2_path=data_path / "pop_fracs_2d.png",
    )

    # Perform onion clustering at delta_t = 10
    onion_results = coord_2d.get_onion_smooth(delta_t=10)
    onion_results.plot_output(data_path / "output_2d.png", coord_2d)
    onion_results.plot_one_trj(data_path / "example_2d.png", coord_2d, 10)
    onion_results.plot_state_populations(data_path / "state_pops_2d.png")

    plt.show()


if __name__ == "__main__":
    main()
