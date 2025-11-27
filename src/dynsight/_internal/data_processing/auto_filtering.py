from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import os

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

import io
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore[import-untyped]
from numpy.fft import fft, fftfreq
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt

from dynsight.trajectory import Insight

# Type alias for 64-bit float numpy arrays
ArrayF64: TypeAlias = NDArray[np.float64]


# --------------------------- Constants ---------------------------

# Frequency conversion constants
FREQ_TERA = 1e12  # Terahertz in Hz
FREQ_GIGA = 1e9  # Gigahertz in Hz
FREQ_MEGA = 1e6  # Megahertz in Hz
FREQ_KILO = 1e3  # Kilohertz in Hz

# Default parameters for filtering
DEFAULT_FRAMES_TO_REMOVE = 20  # Frames to trim from each end
DEFAULT_FILTER_ORDER = 4  # Butterworth filter order

# Image processing constants
IMG_NDIM_GRAYSCALE = 2  # Number of dimensions for grayscale
IMG_CHANNELS_RGBA = 4  # Number of channels for RGBA images

# Numerical constants
SMALL_EPSILON = 1e-9  # Small number to avoid division by zero
NDIM_EXPECTED = 2  # Expected number of dimensions for input
MIN_FRAMES_TO_DROP = 2  # Minimum frames needed to drop first frame

# Initialize logger for this module
logger = logging.getLogger(__name__)


# --------------------------- Result container ---------------------------


@dataclass(frozen=True)
class AutoFiltInsight:
    """Container for auto-filtering results.

    Stores all outputs from the filtering workflow including paths,
    cutoff frequencies, and metadata.

    Attributes:
        output_dir: Base directory where all outputs are saved
        video_path: Path to forward video showing filter evolution
        cutoffs: List of cutoff frequencies used (Hz)
        filtered_files: Dict mapping cutoff freq to saved .npy path
        meta: Dictionary of metadata (parameters used)
        filtered_collection: Tuple of filtered signal arrays
    """

    # Non-default fields must come first
    output_dir: Path
    collection_path: Path

    # Optional fields
    video_path: Path | None = None
    fft_video_path: Path | None = None

    # Default fields (hide large arrays from repr)
    cutoffs: list[float] = field(default_factory=list, repr=False)
    filtered_files: dict[float, Path] = field(default_factory=dict, repr=False)
    meta: dict[str, Any] = field(default_factory=dict, repr=False)
    filtered_collection: tuple[ArrayF64, ...] = field(
        default_factory=tuple, repr=False
    )


# --------------------------- Helper Functions ---------------------------


def _resolve_dataset_path(user_path: str | os.PathLike[str]) -> Path:
    """Resolve user input to a concrete dataset file path."""
    p = Path(user_path).expanduser().resolve()

    if p.is_file():
        return p

    if not p.exists():
        msg = f"Path does not exist: {p}"
        raise FileNotFoundError(msg)

    if p.is_dir():
        for ext in (".json", ".npy", ".npz"):
            hits = sorted(p.glob(f"*{ext}"))

            if len(hits) == 1:
                return hits[0]

            if len(hits) > 1:
                names = ", ".join(h.name for h in hits)
                msg = f"Multiple {ext} files in {p}: {names}"
                raise ValueError(msg)

        msg = f"No .json/.npy/.npz found in {p}"
        raise FileNotFoundError(msg)

    msg = f"Unsupported path: {p}"
    raise FileNotFoundError(msg)


def _load_array_any(
    path: Path,
    *,
    mmap_mode: Literal["r+", "r", "w+", "c"] | None = None,
    enforce_2d: bool = True,
) -> NDArray[np.float64]:
    """Load dataset from .json, .npy, or .npz file."""
    sfx = path.suffix.lower()

    if sfx == ".json":
        arr1 = np.load(path, mmap_mode=mmap_mode)
        ins = Insight(arr1)
    elif sfx == ".npy":
        arr = np.load(path, mmap_mode=mmap_mode)
        ins = Insight(dataset=np.asarray(arr), meta={"source": path.name})
    elif sfx == ".npz":
        z = np.load(path, mmap_mode=mmap_mode)

        if not z.files:
            msg = "Empty .npz file."
            raise ValueError(msg)

        key = z.files[0]
        ins = Insight(
            dataset=np.asarray(z[key]), meta={"source": path.name, "key": key}
        )
    else:
        msg = f"Unsupported file type: {sfx}"
        raise ValueError(msg)

    if enforce_2d and ins.dataset.ndim != NDIM_EXPECTED:
        msg = f"Expected 2D array (series x frames), got {ins.dataset.shape}"
        raise ValueError(msg)

    return np.asarray(ins.dataset)


def _make_dir_safe(directory: Path) -> None:
    """Create directory and all parent directories if they don't exist."""
    directory.mkdir(parents=True, exist_ok=True)


def _freq_label_for_folder(freq_hz: float) -> str:
    """Convert frequency in Hz to human-readable string with units."""
    if freq_hz >= FREQ_TERA:
        return f"{freq_hz / FREQ_TERA:.3f}THz"
    if freq_hz >= FREQ_GIGA:
        return f"{freq_hz / FREQ_GIGA:.3f}GHz"
    if freq_hz >= FREQ_MEGA:
        return f"{freq_hz / FREQ_MEGA:.3f}MHz"
    if freq_hz >= FREQ_KILO:
        return f"{freq_hz / FREQ_KILO:.3f}kHz"
    return f"{freq_hz:.3f}Hz"


# --------------------------- Main Class ---------------------------


class AutoFilteringPipeline:
    """Automatic multi-level Butterworth lowpass filtering pipeline.

    This class provides a modular interface for applying multi-level
    filtering to time series data. Each output type can be generated
    independently.

    Example:
        >>> import numpy as np
        >>> from dynsight._internal.data_processing.auto_filtering import (
        ...     AutoFilteringPipeline,
        ... )
        >>> # Create dummy data: 10 series, 100 frames
        >>> data = np.random.randn(10, 100)
        >>> pipeline = AutoFilteringPipeline(
        ...     signals=data,
        ...     dt_ps=100.0,
        ...     levels=5,
        ...     out_dir="./outputs"
        ... )
        >>> pipeline.compute_fft_and_cutoffs()  # doctest: +SKIP
        >>> pipeline.apply_filtering()  # doctest: +SKIP
        >>> result = pipeline.save_filtered_collection()  # doctest: +SKIP
        >>> pipeline.save_fft_plots()  # doctest: +SKIP
        >>> pipeline.save_cutoff_folders()  # doctest: +SKIP
        >>> pipeline.create_signal_video()  # doctest: +SKIP
        >>> pipeline.create_fft_video()  # doctest: +SKIP
    """

    def __init__(
        self,
        signals: NDArray[np.float64] | None = None,
        path: str | Path = ".",
        dt_ps: float = 100.0,
        levels: int = 50,
        out_dir: str | Path | None = None,
        reuse_existing: bool = True,
        frames_to_remove: int = DEFAULT_FRAMES_TO_REMOVE,
        low_frac: float = 0.20,
        low_ratio: float = 2.0,
        seed: int = 42,
        drop_first_frame: bool = True,
    ) -> None:
        """Initialize the filtering pipeline.

        Args:
            signals: Pre-loaded signals array (series x frames), or None
            path: Path to dataset file/folder (used if signals is None)
            dt_ps: Time step in picoseconds
            levels: Number of filter cutoff frequencies to test
            out_dir: Output directory (default: ./autofilter_outputs)
            reuse_existing: Reuse existing filtered files if available
            frames_to_remove: Frames to trim from edges
            low_frac: Cumulative FFT fraction defining "low frequency"
            low_ratio: Ratio of low-freq to high-freq cutoffs
            seed: Random seed for reproducibility
            drop_first_frame: Drop first frame from input
        """
        # Validate and store parameters
        self.dt_ps = dt_ps
        self.levels = levels
        self.frames_to_remove = frames_to_remove
        self.low_frac = low_frac
        self.low_ratio = low_ratio
        self.seed = seed
        self.reuse_existing = reuse_existing

        # Validate parameters and compute derived values
        self.dt, self.fs = self._validate_params()

        # Load or validate signals
        self.signals = self._resolve_signals(signals, path, drop_first_frame)
        self.n_series, self.n_frames = self.signals.shape

        # Set up output directory
        self.output_dir = self._select_output_dir(out_dir)

        # Initialize state variables (computed later)
        self.freq: NDArray[np.float64] | None = None
        self.mag: NDArray[np.float64] | None = None
        self.cutoffs: list[float] = []
        self.original_trim: NDArray[np.float64] | None = None
        self.filtered_collection: list[ArrayF64] = []
        self.filtered_paths: dict[float, Path] = {}
        self.global_ymin: float | None = None
        self.global_ymax: float | None = None

        logger.info("Pipeline initialized: %s", self.output_dir)

    def _validate_params(self) -> tuple[float, float]:
        """Validate input parameters and compute derived values."""
        if self.dt_ps <= 0:
            msg = "dt_ps must be > 0"
            raise ValueError(msg)

        if self.levels < 1:
            msg = "levels must be >= 1"
            raise ValueError(msg)

        dt = self.dt_ps * 1e-12
        fs = 1.0 / dt

        logger.info(
            "dt = %.3g ps (%.3e s) | fs = %.3e Hz | Nyquist = %.3e Hz",
            self.dt_ps,
            dt,
            fs,
            0.5 * fs,
        )

        return dt, fs

    def _resolve_signals(
        self,
        signals: NDArray[np.float64] | None,
        path: str | Path,
        drop_first_frame: bool,
    ) -> NDArray[np.float64]:
        """Load or validate input signals."""
        if signals is None:
            ds_path = _resolve_dataset_path(path)
            signals = _load_array_any(ds_path)

        if signals.ndim != NDIM_EXPECTED:
            msg = f"Expected 2D array (series x frames), got {signals.shape}"
            raise ValueError(msg)

        if drop_first_frame:
            if signals.shape[1] < MIN_FRAMES_TO_DROP:
                msg = f"Need at least {MIN_FRAMES_TO_DROP} frames."
                raise ValueError(msg)
            signals = signals[:, 1:]

        logger.info("Using signals -> shape %s", signals.shape)

        return signals

    def _select_output_dir(self, out_dir: str | Path | None) -> Path:
        """Create and return output directory path."""
        base = (
            Path(out_dir)
            if out_dir is not None
            else Path.cwd() / "autofilter_outputs"
        )

        _make_dir_safe(base)

        return base

    def compute_fft_and_cutoffs(self) -> None:
        """Compute FFT and determine cutoff frequencies.

        This must be called before apply_filtering().
        """
        logger.info("[STEP] Computing summed FFT (original data) ...")
        self.freq, self.mag = self._compute_fft_summed(self.signals, self.dt)

        logger.info(
            "Frequency bins: %d | Min/Max freq: %.3e/%.3e Hz",
            len(self.freq),
            self.freq.min(),
            self.freq.max(),
        )

        logger.info(
            "Selecting %d cutoff(s) with low-freq bias (<=%.2f cum |FFT|)",
            self.levels,
            self.low_frac,
        )
        self.cutoffs = self._find_cutoffs_biased(
            self.freq,
            self.mag,
            self.levels,
        )

        logger.info(
            "Cutoffs (Hz, ascending): %s",
            [f"{c:.2e}" for c in self.cutoffs],
        )

    def apply_filtering(self) -> None:
        """Apply Butterworth filtering at all cutoff frequencies.

        This must be called after compute_fft_and_cutoffs().
        """
        if not self.cutoffs:
            msg = "Must call compute_fft_and_cutoffs() first"
            raise RuntimeError(msg)

        # Trim original signals
        self.original_trim = self._remove_filter_artifacts(
            self.signals,
            frames_to_remove=self.frames_to_remove,
        )

        # Initialize global y-limits from original data
        self.global_ymin = float(np.nanmin(self.original_trim))
        self.global_ymax = float(np.nanmax(self.original_trim))

        logger.info(
            "Processing %d cutoff level(s); fs=%.3e Hz, Nyquist=%.3e Hz",
            len(self.cutoffs),
            self.fs,
            0.5 * self.fs,
        )

        # Process each cutoff level
        for i, cutoff in enumerate(sorted(self.cutoffs), start=1):
            filtered_trim = self._apply_single_cutoff(i, cutoff)

            # Update global y-limits
            ymin_filt = float(np.nanmin(filtered_trim))
            self.global_ymin = min(self.global_ymin, ymin_filt)
            ymax_filt = float(np.nanmax(filtered_trim))
            self.global_ymax = max(self.global_ymax, ymax_filt)

            self.filtered_collection.append(filtered_trim)

        logger.info(
            "Filtering complete: %d levels processed",
            len(self.cutoffs),
        )

    def _apply_single_cutoff(self, level_idx: int, cutoff: float) -> ArrayF64:
        """Apply filtering for a single cutoff (core logic only)."""
        logger.info(
            "[LEVEL %d/%d] cutoff=%.3e Hz -> COMPUTE",
            level_idx,
            len(self.cutoffs),
            cutoff,
        )

        # Apply filter to each series
        t0 = time.time()
        filtered = np.array(
            [
                self._butter_lowpass_filter(row, cutoff, self.fs)
                for row in self.signals
            ]
        )
        logger.info(
            "Applied Butterworth via filtfilt in %.2fs",
            time.time() - t0,
        )

        # Remove edge artifacts
        return self._remove_filter_artifacts(
            filtered,
            frames_to_remove=self.frames_to_remove,
        )

    def save_filtered_collection(
        self, filename: str = "filtered_collection.npz"
    ) -> Path:
        """Save all filtered arrays to single NPZ file (MANDATORY OUTPUT).

        Returns:
            Path to saved NPZ file
        """
        if not self.filtered_collection:
            msg = "Must call apply_filtering() first"
            raise RuntimeError(msg)

        npz_path = self.output_dir / filename
        np.savez_compressed(npz_path, *self.filtered_collection)

        logger.info(
            "[OUT] Saved filtered collection (%d levels): %s",
            len(self.filtered_collection),
            npz_path,
        )

        return npz_path

    def save_cutoff_folders(
        self,
        save_fft_plots: bool = True,
        save_kde_plots: bool = True,
        save_comparison_plots: bool = True,
        n_comparison_atoms: int = 3,
    ) -> dict[float, Path]:
        """Save individual folders for each cutoff with diagnostics.

        Args:
            save_fft_plots: Save FFT plot for each cutoff
            save_kde_plots: Save signals + KDE plot for each cutoff
            save_comparison_plots: Save original vs filtered comparison
            n_comparison_atoms: Number of random series to compare

        Returns:
            Dictionary mapping cutoff frequency to folder path
        """
        if not self.filtered_collection:
            msg = "Must call apply_filtering() first"
            raise RuntimeError(msg)

        if self.original_trim is None:
            msg = "original_trim is None"
            raise RuntimeError(msg)

        folders = {}

        for i, (cutoff, filtered_trim) in enumerate(
            zip(self.cutoffs, self.filtered_collection), start=1
        ):
            label = _freq_label_for_folder(cutoff)
            folder = self.output_dir / f"cutoff_{label}"
            _make_dir_safe(folder)

            # Save filtered signals
            out_path = folder / "filtered_signals.npy"
            np.save(out_path, filtered_trim)

            # Save FFT plot
            if save_fft_plots:
                f_filt, mag_filt = self._compute_fft_summed(
                    filtered_trim, self.dt
                )
                self._plot_fft(
                    f_filt,
                    mag_filt,
                    f"Summed FFT (filtered, cutoff={cutoff:.2e} Hz)",
                    folder / "fft_plot.png",
                )

            # Save KDE plot
            if save_kde_plots:
                self._plot_signals_with_kde(
                    filtered_trim,
                    "Filtered Data + KDE",
                    folder / "filt_kde.png",
                )

            # Save comparison plots
            if save_comparison_plots:
                n_pick = min(n_comparison_atoms, self.n_series)
                random.seed(self.seed)
                rand_atoms = random.sample(range(self.n_series), n_pick)

                length = filtered_trim.shape[1]
                original_aligned = self.original_trim[:, -length:]

                for idx in rand_atoms:
                    self._plot_single_atom_comparison(
                        original_aligned,
                        filtered_trim,
                        idx,
                        folder / f"atom_{idx}_comparison.png",
                    )

            folders[cutoff] = folder
            self.filtered_paths[cutoff] = out_path

            logger.info(
                "[OUT] Saved cutoff folder [%d/%d]: %s",
                i,
                len(self.cutoffs),
                folder,
            )

        return folders

    def save_fft_plots(self, mark_cutoffs: bool = True) -> Path:
        """Save FFT plot of original signals (OPTIONAL).

        Args:
            mark_cutoffs: Whether to mark cutoff frequencies on plot

        Returns:
            Path to saved plot
        """
        if self.freq is None or self.mag is None:
            msg = "Must call compute_fft_and_cutoffs() first"
            raise RuntimeError(msg)

        plot_path = self.output_dir / "original_fft.png"

        self._plot_fft(
            self.freq,
            self.mag,
            "Summed FFT of Original Signals",
            plot_path,
            mark_freqs=self.cutoffs if mark_cutoffs else None,
        )

        logger.info("[OUT] Saved original FFT plot: %s", plot_path)

        return plot_path

    def create_signal_video(
        self,
        max_overlay_traces: int = 5,
        frame_duration: float = 0.25,
        filename: str = "filtered_evolution",
    ) -> Path:
        """Create video showing signal evolution during filtering.

        Args:
            max_overlay_traces: Max number of individual traces to overlay
            frame_duration: Duration of each frame in seconds
            filename: Base filename (without extension)

        Returns:
            Path to saved video file
        """
        if not self.filtered_collection:
            msg = "Must call apply_filtering() first"
            raise RuntimeError(msg)

        if self.original_trim is None:
            msg = "original_trim is None"
            raise RuntimeError(msg)

        if self.global_ymin is None or self.global_ymax is None:
            msg = "global y-limits are None"
            raise RuntimeError(msg)

        # Add padding to y-limits
        y_range = self.global_ymax - self.global_ymin
        pad = 1e-7 * y_range if y_range > 0 else 1.0
        yl = (self.global_ymin - pad, self.global_ymax + pad)

        video_path = self._render_video(
            self.original_trim,
            self.filtered_collection,
            self.cutoffs,
            out_path_base=self.output_dir / filename,
            y_limits=yl,
            max_overlay_traces=max_overlay_traces,
            seed=self.seed,
            frame_duration=frame_duration,
        )

        logger.info("[OUT] Saved signal evolution video: %s", video_path)

        return video_path

    def create_fft_video(
        self,
        frame_duration: float = 0.25,
        filename: str = "fft_evolution",
    ) -> Path:
        """Create video showing FFT evolution during filtering.

        Args:
            frame_duration: Duration of each frame in seconds
            filename: Base filename (without extension)

        Returns:
            Path to saved video file
        """
        if not self.filtered_collection:
            msg = "Must call apply_filtering() first"
            raise RuntimeError(msg)

        if self.freq is None or self.mag is None:
            msg = "Must call compute_fft_and_cutoffs() first"
            raise RuntimeError(msg)

        fft_video_path = self._render_fft_evolution_video(
            self.freq,
            self.mag,
            self.filtered_collection,
            self.cutoffs,
            self.dt,
            out_path_base=self.output_dir / filename,
            frame_duration=frame_duration,
        )

        logger.info("[OUT] Saved FFT evolution video: %s", fft_video_path)

        return fft_video_path

    def get_result(self) -> AutoFiltInsight:
        """Get result container with all metadata and paths.

        Returns:
            AutoFiltInsight object with all result information
        """
        # Find paths if they exist
        collection_path = self.output_dir / "filtered_collection.npz"
        video_path = self.output_dir / "filtered_evolution.avi"
        fft_video_path = self.output_dir / "fft_evolution.avi"

        meta = {
            "dt_ps": self.dt_ps,
            "frames_to_remove": self.frames_to_remove,
            "levels": self.levels,
            "low_frac": self.low_frac,
            "low_ratio": self.low_ratio,
            "seed": self.seed,
        }

        # Ensure collection_path exists before creating result
        if not collection_path.exists():
            msg = "filtered_collection.npz does not exist"
            raise RuntimeError(msg)

        return AutoFiltInsight(
            cutoffs=self.cutoffs,
            output_dir=self.output_dir,
            collection_path=collection_path,
            filtered_files=self.filtered_paths,
            video_path=video_path if video_path.exists() else None,
            fft_video_path=(
                fft_video_path if fft_video_path.exists() else None
            ),
            meta=meta,
            filtered_collection=tuple(self.filtered_collection),
        )

    # --------------------------- Private Helper Methods ---------------

    def _compute_fft_summed(
        self, signals: NDArray[np.float64], dt: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute FFT along time axis and sum magnitudes across series."""
        _n_series, n_frames = signals.shape

        f_all = fftfreq(n_frames, d=dt)
        pos_mask = f_all > 0

        fft_vals = fft(signals, axis=1)[:, pos_mask]

        mag_sum: NDArray[np.float64] = np.asarray(
            np.abs(fft_vals), dtype=np.float64
        ).sum(axis=0)

        freq: NDArray[np.float64] = np.asarray(
            f_all[pos_mask], dtype=np.float64
        )

        return freq, mag_sum

    def _find_cutoffs_biased(
        self,
        freq: NDArray[np.float64],
        mag: NDArray[np.float64],
        num_levels: int,
    ) -> list[float]:
        """Find cutoff frequencies biased toward low frequencies."""
        cum = np.cumsum(mag)
        total = cum[-1] if cum.size else 0.0

        if total == 0.0:
            warn_msg = (
                "[WARN] Summed magnitude is all zeros; "
                "using max frequency as single cutoff."
            )
            logger.warning(warn_msg)
            return [float(freq[-1])] if freq.size else []

        cum = cum / total

        ratio_sum = self.low_ratio + 1.0
        n_low = max(1, round(num_levels * (self.low_ratio / ratio_sum)))
        n_high = max(1, num_levels - n_low)

        min_frac = 0.05
        max_frac = 0.95
        low_hi = max(min(self.low_frac, max_frac - 1e-6), min_frac + 1e-6)

        th_low = np.linspace(min_frac, low_hi, n_low, endpoint=True)
        th_high = np.linspace(low_hi + 1e-6, max_frac, n_high, endpoint=True)

        thresholds = np.concatenate([th_low, th_high])

        cutoffs: list[float] = []
        for th in thresholds:
            idx = np.searchsorted(cum, th, side="left")

            if idx >= len(freq):
                idx = len(freq) - 1

            c = float(freq[int(idx)])

            if len(cutoffs) == 0 or not np.isclose(c, cutoffs[-1]):
                cutoffs.append(c)

        cutoffs = sorted(set(cutoffs))

        idx_split = np.searchsorted(cum, self.low_frac, side="left")
        f_split = (
            float(freq[min(int(idx_split), len(freq) - 1)])
            if len(freq)
            else float("nan")
        )
        info_msg = (
            f"[INFO] Biased cutoffs: low_frac={self.low_frac:.2f} "
            f"(~f={f_split:.3e} Hz) | total unique={len(cutoffs)}"
        )
        logger.info(info_msg)

        return cutoffs

    def _butter_lowpass_filter(
        self,
        signal: NDArray[np.float64],
        cutoff: float,
        fs: float,
        order: int = DEFAULT_FILTER_ORDER,
    ) -> NDArray[np.float64]:
        """Apply Butterworth lowpass filter to signal."""
        nyq = 0.5 * fs

        if cutoff >= nyq:
            warn_msg = (
                f"[WARN] cutoff {cutoff:.3e} >= Nyquist {nyq:.3e}; "
                "passing signal through."
            )
            logger.warning(warn_msg)
            return signal

        b, a = butter(order, cutoff / nyq, btype="low")

        return filtfilt(b, a, signal)

    def _remove_filter_artifacts(
        self,
        signals: NDArray[np.float64],
        frames_to_remove: int = DEFAULT_FRAMES_TO_REMOVE,
    ) -> NDArray[np.float64]:
        """Remove edge frames affected by filtering artifacts."""
        _n_series, n_frames = signals.shape

        if n_frames <= 2 * frames_to_remove:
            warn_msg = (
                f"[WARN] Not enough frames ({n_frames}) to remove "
                f"{frames_to_remove} per side. Skipping trim."
            )
            logger.warning(warn_msg)
            return signals

        trimmed = signals[:, frames_to_remove:-frames_to_remove]

        logger.info(
            f"[STEP] Removed {frames_to_remove} frames per side "
            f"-> new shape {trimmed.shape}"
        )

        return trimmed

    def _plot_fft(
        self,
        freq: NDArray[np.float64],
        mag: NDArray[np.float64],
        title: str,
        path: Path,
        mark_freqs: list[float] | None = None,
    ) -> None:
        """Create and save FFT magnitude plot."""
        plt.figure()

        plt.plot(freq / FREQ_GIGA, mag, lw=1.5, label="Summed |FFT|")

        if mark_freqs:
            y_interp = np.interp(mark_freqs, freq, mag)
            plt.scatter(
                np.array(mark_freqs) / FREQ_GIGA,
                y_interp,
                s=30,
                label="Cutoffs",
            )

        plt.title(title)
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Summed Magnitude |FFT|")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(path, dpi=200)
        plt.close()

    def _plot_signals_with_kde(
        self, signals: NDArray[np.float64], title: str, path: Path
    ) -> None:
        """Create dual-panel plot: signals + KDE distribution."""
        mean_signal = np.mean(signals, axis=0)
        all_values = signals.ravel()

        _fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

        ax1.plot(signals.T, lw=0.3, alpha=0.35, c="gray")
        ax1.plot(mean_signal, color="red", lw=1.0, label="Mean")
        ax1.set_title(title)
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Signal")
        ax1.grid(alpha=0.3)
        ax1.legend()

        sns.kdeplot(y=all_values, ax=ax2, fill=True, alpha=0.3)
        ax2.set_title("KDE Distribution")

        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    def _plot_single_atom_comparison(
        self,
        orig: NDArray[np.float64],
        filt: NDArray[np.float64],
        atom_idx: int,
        path: Path,
    ) -> None:
        """Plot original vs filtered signal for a single series."""
        plt.figure()

        plt.plot(orig[atom_idx], label="Original", lw=1.0)
        plt.plot(filt[atom_idx], label="Filtered", lw=1.0)

        plt.title(f"Atom/Series {atom_idx}: Original vs Filtered")
        plt.xlabel("Frame")
        plt.ylabel("Signal")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(path, dpi=200)
        plt.close()

    def _render_video(
        self,
        raw_trim: NDArray[np.float64],
        filtered_list: list[NDArray[np.float64]],
        cutoffs: list[float],
        out_path_base: Path,
        y_limits: tuple[float, float] | None = None,
        max_overlay_traces: int = 10,
        seed: int = 42,
        frame_duration: float = 0.25,
    ) -> Path:
        """Create videos showing filter evolution."""
        rng = random.Random(seed)  # noqa: S311
        frames: list[NDArray[np.uint8]] = []

        yl = None if y_limits is None else y_limits

        # RAW frame (original, unfiltered)
        picks = (
            rng.sample(
                range(raw_trim.shape[0]),
                k=min(max_overlay_traces, raw_trim.shape[0]),
            )
            if raw_trim.shape[0]
            else []
        )
        traces = [raw_trim[i] for i in picks]

        frames.append(
            self._render_frame_array(
                raw_trim.mean(axis=0),
                raw_trim.std(axis=0),
                y_limits=yl,
                title_override="Original (unfiltered, trimmed)",
                overlay=traces,
                dist_values=raw_trim.ravel(),
                show_legend=False,
            )
        )

        # FILTERED frames
        for filt, cutoff in zip(filtered_list, cutoffs):
            n_series = filt.shape[0]
            picks = (
                rng.sample(
                    range(n_series), k=min(max_overlay_traces, n_series)
                )
                if n_series
                else []
            )
            traces = [filt[i] for i in picks]

            title = f"Filtered @ {_freq_label_for_folder(cutoff)}"

            frames.append(
                self._render_frame_array(
                    filt.mean(axis=0),
                    filt.std(axis=0),
                    y_limits=yl,
                    title_override=title,
                    overlay=traces,
                    dist_values=filt.ravel(),
                    show_legend=False,
                )
            )

        fps = 1.0 / max(frame_duration, SMALL_EPSILON)

        avi_path = out_path_base.with_suffix(".avi")

        try:
            writer = imageio.get_writer(
                avi_path,
                format="FFMPEG",  # type: ignore[arg-type]
                mode="I",
                fps=fps,
                codec="mpeg4",
                bitrate="10M",
                macro_block_size=None,
                output_params=["-pix_fmt", "yuv420p"],
            )
            for f in frames[::-1]:
                writer.append_data(f)
            writer.close()
        except TypeError as e:
            msg = f"Video creation failed for {avi_path}: {e}"
            raise RuntimeError(msg) from e

        logger.info(f"[OUT ] Saved AVI ({len(frames)} frames): {avi_path}")

        return avi_path

    def _render_frame_array(
        self,
        mean: NDArray[np.float64],
        std: NDArray[np.float64],
        y_limits: tuple[float, float] | None = None,
        title_override: str | None = None,
        overlay: list[NDArray[np.float64]] | None = None,
        dist_values: NDArray[np.float64] | None = None,
        show_legend: bool = False,
        kde_bw: float = 1.0,
    ) -> NDArray[np.uint8]:
        """Render a single video frame as a numpy image array."""
        x = np.arange(mean.size, dtype=np.float64)

        fig, (ax1, ax2) = plt.subplots(
            ncols=2,
            figsize=(10, 4),
            sharey=True,
            facecolor="white",
        )

        self._draw_left_panel(
            ax1,
            x,
            mean,
            std,
            overlay,
            title_override,
            y_limits,
            show_legend,
        )

        self._draw_kde_panel(ax2, dist_values, kde_bw)

        fig.tight_layout()

        return self._finalize_frame(fig)

    def _draw_left_panel(
        self,
        ax: Axes,
        x: NDArray[np.float64],
        mean: NDArray[np.float64],
        std: NDArray[np.float64],
        overlay: list[NDArray[np.float64]] | None,
        title: str | None,
        y_limits: tuple[float, float] | None,
        show_legend: bool,
    ) -> None:
        """Draw left panel of video frame showing signal traces."""
        ax.plot(x, mean, lw=1.2, label="Mean")

        ax.fill_between(
            x, mean - std, mean + std, alpha=0.25, label="+/- 1 sigma"
        )

        if overlay:
            for tr in overlay:
                ax.plot(x, tr, lw=0.6, alpha=0.35)

        ax.set_title(title or "Filtered")
        ax.set_xlabel("Frame (trimmed)")
        ax.set_ylabel("Signal")
        ax.grid(alpha=0.3)

        if show_legend:
            ax.legend()

        if y_limits is not None:
            ax.set_ylim(*y_limits)

    def _draw_kde_panel(
        self,
        ax: Axes,
        dist_values: NDArray[np.float64] | None,
        kde_bw: float,
    ) -> None:
        """Draw right panel of video frame showing KDE distribution."""
        ax.set_title("KDE")
        ax.set_xlabel("Density")
        ax.grid(alpha=0.3)

        if dist_values is None:
            return

        vals = np.asarray(dist_values)
        vals = vals[np.isfinite(vals)]

        if vals.size > 1 and np.nanstd(vals) > 0:
            sns.kdeplot(
                y=vals, ax=ax, fill=True, alpha=0.3, bw_adjust=kde_bw
            )
        elif vals.size > 0:
            ax.axhline(float(vals[0]), ls="--", alpha=0.6)

        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    def _finalize_frame(self, fig: Figure) -> NDArray[np.uint8]:
        """Convert matplotlib figure to RGB numpy array."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=220, facecolor="white")
        plt.close(fig)

        buf.seek(0)
        img_raw = imageio.imread(buf)

        if img_raw.ndim == IMG_NDIM_GRAYSCALE:
            img: NDArray[np.uint8] = np.stack(
                [img_raw, img_raw, img_raw], axis=2
            )
        else:
            img = img_raw

        if img.shape[2] == IMG_CHANNELS_RGBA:
            img = img[:, :, :3]

        h, w = img.shape[:2]
        if h % 2 or w % 2:
            img = img[: h - (h % 2), : w - (w % 2), :]

        return img

    def _render_fft_evolution_video(
        self,
        freq_original: NDArray[np.float64],
        mag_original: NDArray[np.float64],
        filtered_list: list[NDArray[np.float64]],
        cutoffs: list[float],
        dt: float,
        out_path_base: Path,
        frame_duration: float = 0.25,
    ) -> Path:
        """Create video showing FFT evolution during filtering."""
        frames: list[NDArray[np.uint8]] = []

        logger.info("[FFT VIDEO] Rendering original frame (no filtering)...")

        frames.append(
            self._render_fft_frame(
                freq_original,
                mag_original,
                mag_original,
                cutoff=None,
                title_left="Original FFT (unfiltered)",
                title_right="Original FFT (unfiltered)",
            )
        )

        for i, (filt, cutoff) in enumerate(
            zip(filtered_list, cutoffs), start=1
        ):
            logger.info(
                "[FFT VIDEO] Rendering frame %d/%d (cutoff=%.3e Hz)...",
                i,
                len(cutoffs),
                cutoff,
            )

            freq_filt, mag_filt = self._compute_fft_summed(filt, dt)

            mag_filt_interp = np.interp(
                freq_original,
                freq_filt,
                mag_filt,
                left=0.0,
                right=0.0,
            )

            cutoff_label = _freq_label_for_folder(cutoff)
            frames.append(
                self._render_fft_frame(
                    freq_original,
                    mag_original,
                    mag_filt_interp,
                    cutoff=cutoff,
                    title_left=f"Original FFT (cutoff @ {cutoff_label})",
                    title_right=f"Filtered FFT (cutoff @ {cutoff_label})",
                )
            )

        fps = 1.0 / max(frame_duration, SMALL_EPSILON)

        avi_path = out_path_base.with_suffix(".avi")

        try:
            writer = imageio.get_writer(
                avi_path,
                format="FFMPEG",  # type: ignore[arg-type]
                mode="I",
                fps=fps,
                codec="mpeg4",
                bitrate="10M",
                macro_block_size=None,
                output_params=["-pix_fmt", "yuv420p"],
            )
            for f in frames[::-1]:
                writer.append_data(f)
            writer.close()
        except TypeError as e:
            msg = f"FFT video creation failed for {avi_path}: {e}"
            raise RuntimeError(msg) from e

        logger.info(
            "[OUT ] Saved FFT evolution video (%d frames): %s",
            len(frames),
            avi_path,
        )

        return avi_path

    def _render_fft_frame(
        self,
        freq: NDArray[np.float64],
        mag_original: NDArray[np.float64],
        mag_filtered: NDArray[np.float64],
        cutoff: float | None,
        title_left: str,
        title_right: str,
    ) -> NDArray[np.uint8]:
        """Render a single FFT comparison frame."""
        fig, (ax1, ax2) = plt.subplots(
            ncols=2,
            figsize=(12, 4),
            facecolor="white",
        )

        freq_ghz = freq / FREQ_GIGA

        ax1.plot(
            freq_ghz,
            mag_original,
            lw=1.5,
            label="Original FFT",
            color="blue",
        )

        if cutoff is not None:
            cutoff_ghz = cutoff / FREQ_GIGA
            y_interp = np.interp(cutoff, freq, mag_original)

            ax1.axvline(cutoff_ghz, color="red", ls="--", lw=1.5, alpha=0.7)

            cutoff_label = _freq_label_for_folder(cutoff)
            ax1.scatter(
                [cutoff_ghz],
                [y_interp],
                s=80,
                color="red",
                zorder=5,
                label=f"Cutoff: {cutoff_label}",
            )

        ax1.set_title(title_left)
        ax1.set_xlabel("Frequency (GHz)")
        ax1.set_ylabel("Summed Magnitude |FFT|")
        ax1.grid(alpha=0.3)
        ax1.legend()

        ax2.plot(
            freq_ghz,
            mag_filtered,
            lw=1.5,
            label="Filtered FFT",
            color="green",
        )
        ax2.set_title(title_right)
        ax2.set_xlabel("Frequency (GHz)")
        ax2.set_ylabel("Summed Magnitude |FFT|")
        ax2.grid(alpha=0.3)
        ax2.legend()

        fig.tight_layout()

        return self._finalize_frame(fig)


# --------------------------- Convenience Function ---------------------------


def auto_filtering(
    signals: NDArray[np.float64] | None = None,
    path: str | Path = ".",
    dt_ps: float = 100.0,
    levels: int = 50,
    out_dir: str | Path | None = None,
    reuse_existing: bool = True,
    frames_to_remove: int = DEFAULT_FRAMES_TO_REMOVE,
    low_frac: float = 0.20,
    low_ratio: float = 2.0,
    seed: int = 42,
    max_overlay_traces: int = 5,
    frame_duration: float = 0.25,
    drop_first_frame: bool = True,
    save_cutoff_folders: bool = True,
    save_fft_plots: bool = True,
    save_signal_video: bool = True,
    save_fft_video: bool = True,
) -> AutoFiltInsight:
    """Convenience function for complete automatic filtering workflow.

    This function provides the same interface as the original code but
    uses the new class-based implementation internally. Each output type
    can be controlled via parameters.

    Args:
        signals: Pre-loaded signals array (series x frames), or None
        path: Path to dataset file/folder (used if signals is None)
        dt_ps: Time step in picoseconds
        levels: Number of filter cutoff frequencies to test
        out_dir: Output directory (default: ./autofilter_outputs)
        reuse_existing: Reuse existing filtered files if available
        frames_to_remove: Frames to trim from edges
        low_frac: Cumulative FFT fraction defining "low frequency"
        low_ratio: Ratio of low-freq to high-freq cutoffs
        seed: Random seed for reproducibility
        max_overlay_traces: Max traces to overlay in video frames
        frame_duration: Duration of each video frame in seconds
        drop_first_frame: Drop first frame from input
        save_cutoff_folders: Save individual cutoff folders
        save_fft_plots: Save FFT plots
        save_signal_video: Create signal evolution video
        save_fft_video: Create FFT evolution video

    Returns:
        AutoFiltInsight object with all results and metadata
    """
    # Create pipeline
    pipeline = AutoFilteringPipeline(
        signals=signals,
        path=path,
        dt_ps=dt_ps,
        levels=levels,
        out_dir=out_dir,
        reuse_existing=reuse_existing,
        frames_to_remove=frames_to_remove,
        low_frac=low_frac,
        low_ratio=low_ratio,
        seed=seed,
        drop_first_frame=drop_first_frame,
    )

    # Run core analysis
    pipeline.compute_fft_and_cutoffs()
    pipeline.apply_filtering()

    # Save filtered collection (MANDATORY)
    pipeline.save_filtered_collection()

    # Save optional outputs based on parameters
    if save_fft_plots:
        pipeline.save_fft_plots()

    if save_cutoff_folders:
        pipeline.save_cutoff_folders()

    if save_signal_video:
        pipeline.create_signal_video(
            max_overlay_traces=max_overlay_traces,
            frame_duration=frame_duration,
        )

    if save_fft_video:
        pipeline.create_fft_video(frame_duration=frame_duration)

    # Return result
    return pipeline.get_result()
