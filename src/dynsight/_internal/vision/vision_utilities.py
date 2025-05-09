from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

if TYPE_CHECKING:
    import pathlib

    from numpy.typing import NDArray


def find_outliers(
    distribution: NDArray[np.float64],
    save_path: pathlib.Path,
    fig_name: str,
    thr: float = 1e-5,
) -> NDArray[np.float64]:
    """Detects outliers in a distribution by fitting a normal distribution."""
    if distribution.size == 0:
        msg = "Distribution is empty or contains only NaNs/Infs."
        raise ValueError(msg)
    if np.std(distribution) == 0:
        return np.array([])

    def _gaussian(
        x: NDArray[np.float64], mu: float, sigma: float, amplitude: float
    ) -> NDArray[np.float64]:
        return amplitude * norm.pdf(x, mu, sigma)

    # Compute histogram and bin centers
    hist, bin_edges = np.histogram(distribution, bins="auto", density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mask = np.isfinite(hist) & (hist > 0)
    hist = hist[mask]
    bin_centers = bin_centers[mask]
    min_n_bins = 3
    if len(hist) < min_n_bins:
        return np.array([])
    # Fit the Gaussian curve to the histogram data
    popt, _ = curve_fit(
        _gaussian,
        bin_centers,
        hist,
        p0=[np.mean(distribution), np.std(distribution), np.max(hist)],
    )
    mu, sigma, amplitude = popt

    # Generate fitted Gaussian curve for plotting
    x = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    fitted_curve = _gaussian(x, mu, sigma, amplitude)

    # Calculate PDF threshold-based cutoffs
    base_pdf = amplitude / (sigma * np.sqrt(2 * np.pi))
    x_threshold_min = mu - np.sqrt(-2 * sigma**2 * np.log(thr / base_pdf))
    x_threshold_max = mu + np.sqrt(-2 * sigma**2 * np.log(thr / base_pdf))

    # Identify outliers using numpy boolean indexing
    outliers: NDArray[np.float64] = distribution[
        (distribution < x_threshold_min) | (distribution > x_threshold_max)
    ]

    # Plot histogram, fitted curve, and threshold lines
    plt.hist(
        distribution,
        bins="auto",
        density=True,
        alpha=0.6,
        color="g",
        label="Histogram",
    )
    plt.plot(
        x,
        fitted_curve,
        "k-",
        linewidth=2,
        label=rf"Gaussian fit  $\mu={mu:.2f},\ \sigma={sigma:.2f}$",
    )
    plt.axvline(
        x_threshold_min,
        color="r",
        linestyle="--",
        label=f"Threshold Min = {x_threshold_min:.2f}",
    )
    plt.axvline(
        x_threshold_max,
        color="b",
        linestyle="--",
        label=f"Threshold Max = {x_threshold_max:.2f}",
    )
    plt.legend(loc="best")
    plt.title("Histogram with Gaussian Fit and Thresholds")
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(save_path / fig_name)
    plt.close()

    return outliers
