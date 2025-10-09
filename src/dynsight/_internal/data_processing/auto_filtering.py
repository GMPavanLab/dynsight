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
FREQ_GIGA = 1e9   # Gigahertz in Hz
FREQ_MEGA = 1e6   # Megahertz in Hz
FREQ_KILO = 1e3   # Kilohertz in Hz

# Default parameters for filtering
DEFAULT_FRAMES_TO_REMOVE = 20  # Frames to trim from each end
DEFAULT_FILTER_ORDER = 4        # Butterworth filter order

# Image processing constants
IMG_NDIM_GRAYSCALE = 2   # Number of dimensions for grayscale
IMG_CHANNELS_RGBA = 4    # Number of channels for RGBA images

# Numerical constants
SMALL_EPSILON = 1e-9     # Small number to avoid division by zero
NDIM_EXPECTED = 2        # Expected number of dimensions for input
MIN_FRAMES_TO_DROP = 2   # Minimum frames needed to drop first frame

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
    video_path: Path | None

    # Default fields (hide large arrays from repr)
    cutoffs: list[float] = field(
        default_factory=list, repr=False)
    filtered_files: dict[float, Path] = field(
        default_factory=dict, repr=False)
    meta: dict[str, Any] = field(
        default_factory=dict, repr=False)
    filtered_collection: tuple[ArrayF64, ...] = field(
        default_factory=tuple, repr=False
    )


# --------------------------- Helpers (I/O, plots) ---------------------------


def _resolve_dataset_path(user_path: str | os.PathLike[str]) -> Path:
    """Resolve user input to a concrete dataset file path.

    Accepts either a file or folder. For folders, looks for a single
    file with preference: .json > .npy > .npz.

    Args:
        user_path: User-provided path (file or directory)

    Returns:
        Resolved Path to a dataset file

    Raises:
        FileNotFoundError: If path doesn't exist or no valid files found
        ValueError: If multiple files of same type found (ambiguous)
    """
    # Expand ~ and resolve to absolute path
    p = Path(user_path).expanduser().resolve()

    # If it's already a file, return it
    if p.is_file():
        return p

    # Check if path exists at all
    if not p.exists():
        msg = f"Path does not exist: {p}"
        raise FileNotFoundError(msg)

    # If it's a directory, search for dataset files
    if p.is_dir():
        # Try each extension in preference order
        for ext in (".json", ".npy", ".npz"):
            # Find all files with this extension
            hits = sorted(p.glob(f"*{ext}"))

            # Exactly one file found - use it
            if len(hits) == 1:
                return hits[0]

            # Multiple files found - ambiguous
            if len(hits) > 1:
                names = ", ".join(h.name for h in hits)
                msg = f"Multiple {ext} files in {p}: {names}"
                raise ValueError(msg)

        # No valid files found
        msg = f"No .json/.npy/.npz found in {p}"
        raise FileNotFoundError(msg)

    # Shouldn't reach here (not file, not dir, but exists?)
    msg = f"Unsupported path: {p}"
    raise FileNotFoundError(msg)


def _load_array_any(
    path: Path,
    *,
    mmap_mode: Literal["r+", "r", "w+", "c"] | None = None,
    enforce_2d: bool = True,
) -> NDArray[np.float64]:
    """Load dataset from .json, .npy, or .npz file.

    Wraps loaded data into an Insight object for validation,
    then returns the underlying array.

    Args:
        path: Path to dataset file
        mmap_mode: Memory-mapping mode for numpy.load
        enforce_2d: If True, raise error if not 2D array

    Returns:
        Loaded numpy array

    Raises:
        ValueError: If file type unsupported, empty .npz, or wrong
            dimensions
    """
    # Get file extension (lowercase)
    sfx = path.suffix.lower()

    # Load based on file type
    if sfx == ".json":
        # Load from JSON format
        arr1 = np.load(path, mmap_mode=mmap_mode)
        ins = Insight(arr1)
    elif sfx == ".npy":
        # Load from numpy binary format
        arr = np.load(path, mmap_mode=mmap_mode)
        ins = Insight(dataset=np.asarray(arr), meta={"source": path.name})
    elif sfx == ".npz":
        # Load from compressed numpy format
        z = np.load(path, mmap_mode=mmap_mode)

        # Check if npz is empty
        if not z.files:
            msg = "Empty .npz file."
            raise ValueError(msg)

        # Use first key in npz
        key = z.files[0]
        ins = Insight(
            dataset=np.asarray(z[key]), meta={"source": path.name, "key": key}
        )
    else:
        # Unsupported file type
        msg = f"Unsupported file type: {sfx}"
        raise ValueError(msg)

    # Validate dimensions if requested
    if enforce_2d and ins.dataset.ndim != NDIM_EXPECTED:
        msg = f"Expected 2D array (series x frames), got {ins.dataset.shape}"
        raise ValueError(msg)

    # Return as numpy array
    return np.asarray(ins.dataset)


def _make_dir_safe(directory: Path) -> None:
    """Create directory and all parent directories if they don't exist.

    Args:
        directory: Path to directory to create
    """
    # Create directory with parents, don't error if exists
    directory.mkdir(parents=True, exist_ok=True)


def _freq_label_for_folder(freq_hz: float) -> str:
    """Convert frequency in Hz to human-readable string with units.

    Chooses appropriate unit (THz, GHz, MHz, kHz, Hz) based on
    magnitude.

    Args:
        freq_hz: Frequency in Hertz

    Returns:
        Formatted string like "1.234GHz"
    """
    # Choose unit based on frequency magnitude
    if freq_hz >= FREQ_TERA:
        return f"{freq_hz / FREQ_TERA:.3f}THz"
    if freq_hz >= FREQ_GIGA:
        return f"{freq_hz / FREQ_GIGA:.3f}GHz"
    if freq_hz >= FREQ_MEGA:
        return f"{freq_hz / FREQ_MEGA:.3f}MHz"
    if freq_hz >= FREQ_KILO:
        return f"{freq_hz / FREQ_KILO:.3f}kHz"
    return f"{freq_hz:.3f}Hz"


def _plot_fft(
    freq: NDArray[np.float64],
    mag: NDArray[np.float64],
    title: str,
    path: Path,
    mark_freqs: list[float] | None = None,
) -> None:
    """Create and save FFT magnitude plot.

    Plots frequency vs magnitude with optional markers for cutoff
    frequencies.

    Args:
        freq: Frequency array in Hz
        mag: Magnitude array (summed across all series)
        title: Plot title
        path: Where to save the plot
        mark_freqs: Optional list of frequencies to mark with scatter
            points
    """
    # Create new figure
    plt.figure()

    # Plot FFT magnitude vs frequency (in GHz)
    plt.plot(freq / FREQ_GIGA, mag, lw=1.5, label="Summed |FFT|")

    # Add markers for cutoff frequencies if provided
    if mark_freqs:
        # Interpolate magnitude values at cutoff frequencies
        y_interp = np.interp(mark_freqs, freq, mag)
        # Plot cutoff markers
        plt.scatter(
            np.array(mark_freqs) / FREQ_GIGA,
            y_interp,
            s=30,
            label="Cutoffs",
        )

    # Format plot
    plt.title(title)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Summed Magnitude |FFT|")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save and close
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_signals_with_kde(
    signals: NDArray[np.float64], title: str, path: Path
) -> None:
    """Create dual-panel plot: signals + KDE distribution.

    Left panel shows all signal traces plus mean.
    Right panel shows KDE of all signal values.

    Args:
        signals: 2D array (series x frames)
        title: Plot title
        path: Where to save the plot
    """
    # Calculate mean signal across all series
    mean_signal = np.mean(signals, axis=0)

    # Flatten all values for KDE
    all_values = signals.ravel()

    # Create figure with 2 columns
    _fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    # Left panel: plot all traces in gray
    ax1.plot(signals.T, lw=0.3, alpha=0.35, c="gray")
    # Overlay mean in red
    ax1.plot(mean_signal, color="red", lw=1.0, label="Mean")
    ax1.set_title(title)
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Signal")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Right panel: KDE of all values
    sns.kdeplot(y=all_values, ax=ax2, fill=True, alpha=0.3)
    ax2.set_title("KDE Distribution")

    # Save and close
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_single_atom_comparison(
    orig: NDArray[np.float64],
    filt: NDArray[np.float64],
    atom_idx: int,
    path: Path,
) -> None:
    """Plot original vs filtered signal for a single series.

    Args:
        orig: Original signals (series x frames)
        filt: Filtered signals (series x frames)
        atom_idx: Index of series to plot
        path: Where to save the plot
    """
    # Create figure
    plt.figure()

    # Plot original signal
    plt.plot(orig[atom_idx], label="Original", lw=1.0)
    # Plot filtered signal
    plt.plot(filt[atom_idx], label="Filtered", lw=1.0)

    # Format plot
    plt.title(f"Atom/Series {atom_idx}: Original vs Filtered")
    plt.xlabel("Frame")
    plt.ylabel("Signal")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save and close
    plt.savefig(path, dpi=200)
    plt.close()


# --------------------------- Filt helpers ---------------------------


def _compute_fft_summed(
    signals: NDArray[np.float64], dt: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute FFT along time axis and sum magnitudes across series.

    Only keeps positive frequencies. Useful for finding dominant
    frequency components across all signals.

    Args:
        signals: 2D array (series x frames)
        dt: Time step in seconds

    Returns:
        freq: Positive frequency array
        mag_sum: Summed magnitude across all series
    """
    # Get shape
    _n_series, n_frames = signals.shape

    # Compute frequency bins
    f_all = fftfreq(n_frames, d=dt)

    # Keep only positive frequencies
    pos_mask = f_all > 0

    # Compute FFT along time axis (axis=1)
    fft_vals = fft(signals, axis=1)[:, pos_mask]

    # Sum magnitudes across all series

    mag_sum: NDArray[np.float64] = np.asarray(np.abs(fft_vals),
                                              dtype=np.float64).sum(axis=0)

    # Get positive frequencies
    freq: NDArray[np.float64] = np.asarray(f_all[pos_mask], dtype=np.float64)


    return freq, mag_sum


def _find_cutoffs_biased(
    freq: NDArray[np.float64],
    mag: NDArray[np.float64],
    num_levels: int,
    low_frac: float = 0.20,
    low_ratio: float = 2.0,
    min_frac: float = 0.05,
    max_frac: float = 0.95,
) -> list[float]:
    """Find cutoff frequencies biased toward low frequencies.

    Uses cumulative FFT magnitude to select frequencies. Puts more
    cutoffs in the low-frequency region (below low_frac) by a ratio
    of low_ratio:1.

    Args:
        freq: Frequency array
        mag: Magnitude array
        num_levels: Total number of cutoffs to find
        low_frac: Cumulative fraction defining "low frequency" region
        low_ratio: Ratio of low-freq to high-freq cutoffs
        min_frac: Minimum cumulative fraction to consider
        max_frac: Maximum cumulative fraction to consider

    Returns:
        List of cutoff frequencies (sorted, unique)
    """
    # Compute cumulative sum of magnitude
    cum = np.cumsum(mag)

    # Get total magnitude
    total = cum[-1] if cum.size else 0.0

    # Handle all-zero case
    if total == 0.0:
        warn_msg = (
            "[WARN] Summed magnitude is all zeros; "
            "using max frequency as single cutoff."
        )
        logger.warning(warn_msg)
        return [float(freq[-1])] if freq.size else []

    # Normalize to cumulative fraction
    cum = cum / total

    # Calculate how many cutoffs in each region
    n_low = max(1, round(num_levels * (low_ratio / (low_ratio + 1.0))))
    n_high = max(1, num_levels - n_low)

    # Define boundary for low-frequency region
    low_hi = max(min(low_frac, max_frac - 1e-6), min_frac + 1e-6)

    # Create thresholds for low-frequency region
    th_low = np.linspace(min_frac, low_hi, n_low, endpoint=True)

    # Create thresholds for high-frequency region
    th_high = np.linspace(low_hi + 1e-6, max_frac, n_high, endpoint=True)

    # Combine all thresholds
    thresholds = np.concatenate([th_low, th_high])

    # Find frequency for each threshold
    cutoffs: list[float] = []
    for th in thresholds:
        # Find index where cumulative reaches threshold
        idx = np.searchsorted(cum, th, side="left")

        # Clamp to valid range
        if idx >= len(freq):
            idx = len(freq) - 1

        # Get frequency at this index
        c = float(freq[int(idx)])

        # Only add if not duplicate
        if len(cutoffs) == 0 or not np.isclose(c, cutoffs[-1]):
            cutoffs.append(c)

    # Sort and remove duplicates
    cutoffs = sorted(set(cutoffs))

    # Log information about split point
    idx_split = np.searchsorted(cum, low_frac, side="left")
    f_split = (
        float(freq[min(int(idx_split), len(freq) - 1)])
        if len(freq)
        else float("nan")
    )
    info_msg = (
        f"[INFO] Biased cutoffs: low_frac={low_frac:.2f} "
        f"(~f={f_split:.3e} Hz) | total unique={len(cutoffs)}"
    )
    logger.info(info_msg)

    return cutoffs


def _butter_lowpass_filter(
    signal: NDArray[np.float64],
    cutoff: float,
    fs: float,
    order: int = DEFAULT_FILTER_ORDER,
) -> NDArray[np.float64]:
    """Apply Butterworth lowpass filter to signal.

    Uses zero-phase filtering (filtfilt) to avoid phase distortion.

    Args:
        signal: 1D signal array
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order (higher = sharper cutoff)

    Returns:
        Filtered signal array
    """
    # Calculate Nyquist frequency
    nyq = 0.5 * fs

    # Check if cutoff is valid
    if cutoff >= nyq:
        warn_msg = (
            f"[WARN] cutoff {cutoff:.3e} >= Nyquist {nyq:.3e}; "
            "passing signal through."
        )
        logger.warning(warn_msg)
        return signal

    # Design Butterworth filter
    b, a = butter(order, cutoff / nyq, btype="low")

    # Apply zero-phase filter
    return filtfilt(b, a, signal)


def _remove_filter_artifacts(
    signals: NDArray[np.float64],
    frames_to_remove: int = DEFAULT_FRAMES_TO_REMOVE,
) -> NDArray[np.float64]:
    """Remove edge frames affected by filtering artifacts.

    Trims the same number of frames from both start and end.

    Args:
        signals: 2D array (series x frames)
        frames_to_remove: Number of frames to remove from each end

    Returns:
        Trimmed signal array
    """
    # Get shape
    _n_series, n_frames = signals.shape

    # Check if we have enough frames to trim
    if n_frames <= 2 * frames_to_remove:
        warn_msg = (
            f"[WARN] Not enough frames ({n_frames}) to remove "
            f"{frames_to_remove} per side. Skipping trim."
        )
        logger.warning(warn_msg)
        return signals

    # Trim frames from both ends
    trimmed = signals[:, frames_to_remove:-frames_to_remove]

    # Log the operation
    logger.info(
        f"[STEP] Removed {frames_to_remove} frames per side "
        f"-> new shape {trimmed.shape}"
    )

    return trimmed


# --------------------------- Video helpers ---------------------------


def _draw_left_panel(
    ax: Axes,
    x: NDArray[np.float64],
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    overlay: list[NDArray[np.float64]] | None,
    title: str | None,
    y_limits: tuple[float, float] | None,
    show_legend: bool,
) -> None:
    """Draw left panel of video frame showing signal traces.

    Args:
        ax: Matplotlib axes to draw on
        x: X-axis values (frame indices)
        mean: Mean signal across all series
        std: Standard deviation across all series
        overlay: Optional list of individual traces to overlay
        title: Panel title
        y_limits: Optional (ymin, ymax) to fix y-axis
        show_legend: Whether to show legend
    """
    # Plot mean signal
    ax.plot(x, mean, lw=1.2, label="Mean")

    # Fill area for +/- 1 standard deviation
    ax.fill_between(x, mean - std, mean + std, alpha=0.25, label="+/- 1 sigma")

    # Overlay individual traces if provided
    if overlay:
        for tr in overlay:
            ax.plot(x, tr, lw=0.6, alpha=0.35)

    # Set labels and title
    ax.set_title(title or "Filtered")
    ax.set_xlabel("Frame (trimmed)")
    ax.set_ylabel("Signal")
    ax.grid(alpha=0.3)

    # Add legend if requested
    if show_legend:
        ax.legend()

    # Set y-limits if provided
    if y_limits is not None:
        ax.set_ylim(*y_limits)


def _draw_kde_panel(
    ax: Axes,
    dist_values: NDArray[np.float64] | None,
    kde_bw: float,
) -> None:
    """Draw right panel of video frame showing KDE distribution.

    Args:
        ax: Matplotlib axes to draw on
        dist_values: Array of all signal values for KDE
        kde_bw: Bandwidth adjustment for KDE
    """
    # Set title and labels
    ax.set_title("KDE")
    ax.set_xlabel("Density")
    ax.grid(alpha=0.3)

    # Return early if no data
    if dist_values is None:
        return

    # Convert to array and remove non-finite values
    vals = np.asarray(dist_values)
    vals = vals[np.isfinite(vals)]

    # Plot KDE if we have valid data with variance
    if vals.size > 1 and np.nanstd(vals) > 0:
        sns.kdeplot(y=vals, ax=ax, fill=True, alpha=0.3, bw_adjust=kde_bw)
    # Just draw horizontal line if constant value
    elif vals.size > 0:
        ax.axhline(float(vals[0]), ls="--", alpha=0.6)

    # Move y-axis to right side
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")


def _render_frame_array(
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    y_limits: tuple[float, float] | None = None,
    title_override: str | None = None,
    overlay: list[NDArray[np.float64]] | None = None,
    dist_values: NDArray[np.float64] | None = None,
    show_legend: bool = False,
    kde_bw: float = 1.0,
) -> NDArray[np.uint8]:
    """Render a single video frame as a numpy image array.

    Creates a two-panel figure (signals + KDE) and converts to RGB
    array.

    Args:
        mean: Mean signal
        std: Standard deviation of signal
        y_limits: Optional y-axis limits
        title_override: Title for left panel
        overlay: Optional traces to overlay
        dist_values: Values for KDE plot
        show_legend: Whether to show legend
        kde_bw: KDE bandwidth adjustment

    Returns:
        RGB image array (height x width x 3)
    """
    # Create x-axis values
    x = np.arange(mean.size, dtype=np.float64)

    # Create figure with 2 columns sharing y-axis
    fig, (ax1, ax2) = plt.subplots(
        ncols=2,
        figsize=(10, 4),
        sharey=True,
        facecolor="white",
    )

    # Draw left panel (signals)
    _draw_left_panel(
        ax1, x, mean, std, overlay, title_override, y_limits, show_legend
    )

    # Draw right panel (KDE)
    _draw_kde_panel(ax2, dist_values, kde_bw)

    # Adjust layout
    fig.tight_layout()

    # Convert to image array
    return _finalize_frame(fig)


def _finalize_frame(fig: Figure) -> NDArray[np.uint8]:
    """Convert matplotlib figure to RGB numpy array.

    Ensures dimensions are even (required for some video codecs).

    Args:
        fig: Matplotlib figure

    Returns:
        RGB image array (height x width x 3) with even dimensions
    """
    # Save figure to bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, facecolor="white")
    plt.close(fig)

    # Read back as image
    buf.seek(0)
    img_raw = imageio.imread(buf)

    # Convert grayscale to RGB if needed
    if img_raw.ndim == IMG_NDIM_GRAYSCALE:
        img: NDArray[np.uint8] = np.stack(
            [img_raw, img_raw, img_raw], axis=2
        )
    else:
        img = img_raw

    # Remove alpha channel if present
    if img.shape[2] == IMG_CHANNELS_RGBA:
        img = img[:, :, :3]

    # Ensure even dimensions (crop if needed)
    h, w = img.shape[:2]
    if h % 2 or w % 2:
        img = img[: h - (h % 2), : w - (w % 2), :]

    return img


def _render_video(
    raw_trim: NDArray[np.float64],
    filtered_list: list[NDArray[np.float64]],
    cutoffs: list[float],
    out_path_base: Path,
    y_limits: tuple[float, float] | None = None,
    max_overlay_traces: int = 10,
    seed: int = 42,
    frame_duration: float = 0.25,
) -> Path:
    """Create videos showing filter evolution.

    First frame is original (unfiltered), subsequent frames show
    progressively filtered signals.

    Args:
        raw_trim: Original trimmed signals
        filtered_list: List of filtered signal arrays (one per cutoff)
        cutoffs: List of cutoff frequencies
        out_path_base: Base path for output videos (without extension)
        y_limits: Optional fixed y-axis limits for all frames
        max_overlay_traces: Max number of individual traces to overlay
        seed: Random seed for trace selection
        frame_duration: Duration of each frame in seconds

    Returns:
        Tuple of (video_path)
    """
    # Initialize random number generator
    rng = random.Random(seed)  # noqa: S311
    frames: list[NDArray[np.uint8]] = []

    # Simplify y_limits variable
    yl = None if y_limits is None else y_limits

    # ---- RAW frame (original, unfiltered) ----

    # Randomly select traces to overlay
    picks = (
        rng.sample(
            range(raw_trim.shape[0]),
            k=min(max_overlay_traces, raw_trim.shape[0]),
        )
        if raw_trim.shape[0]
        else []
    )
    traces = [raw_trim[i] for i in picks]

    # Render original data frame
    frames.append(
        _render_frame_array(
            raw_trim.mean(axis=0),
            raw_trim.std(axis=0),
            y_limits=yl,
            title_override="Original (unfiltered, trimmed)",
            overlay=traces,
            dist_values=raw_trim.ravel(),
            show_legend=False,
        )
    )

    # ---- FILTERED frames ----

    # Create one frame for each cutoff level
    for filt, cutoff in zip(filtered_list, cutoffs):
        # Randomly select traces to overlay
        picks = (
            rng.sample(
                range(filt.shape[0]), k=min(max_overlay_traces, filt.shape[0])
            )
            if filt.shape[0]
            else []
        )
        traces = [filt[i] for i in picks]

        # Create title with cutoff frequency
        title = f"Filtered @ {_freq_label_for_folder(cutoff)}"

        # Render filtered frame
        frames.append(
            _render_frame_array(
                filt.mean(axis=0),
                filt.std(axis=0),
                y_limits=yl,
                title_override=title,
                overlay=traces,
                dist_values=filt.ravel(),
                show_legend=False,
            )
        )

    # Calculate FPS from frame duration
    fps = 1.0 / max(frame_duration, SMALL_EPSILON)

    # Define output paths
    avi_path = out_path_base.with_suffix(".avi")

    # Try to write videos with detailed parameters
    try:

        # Write video
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

    # Log success
    logger.info(f"[OUT ] Saved AVI ({len(frames)} frames): {avi_path}")

    return avi_path


# --------------------------- Core workflow ---------------------------


def _validate_params(dt_ps: float, levels: int) -> tuple[float, float]:
    """Validate input parameters and compute derived values.

    Args:
        dt_ps: Time step in picoseconds
        levels: Number of filter levels

    Returns:
        Tuple of (dt_seconds, sampling_frequency_hz)

    Raises:
        ValueError: If parameters are invalid
    """
    # Check dt is positive
    if dt_ps <= 0:
        msg = "dt_ps must be > 0"
        raise ValueError(msg)

    # Check levels is at least 1
    if levels < 1:
        msg = "levels must be >= 1"
        raise ValueError(msg)

    # Convert dt to seconds
    dt = dt_ps * 1e-12

    # Calculate sampling frequency
    fs = 1.0 / dt

    # Log parameters
    logger.info(
        "dt = %.3g ps (%.3e s) | fs = %.3e Hz | Nyquist = %.3e Hz",
        dt_ps,
        dt,
        fs,
        0.5 * fs,
    )

    return dt, fs


def _resolve_signals(
    signals: NDArray[np.float64] | None,
    path: str | Path,
    drop_first_frame: bool,
) -> NDArray[np.float64]:
    """Load or validate input signals.

    If signals array is provided, use it. Otherwise load from path.
    Optionally drops first frame to remove initialization artifacts.

    Args:
        signals: Optional pre-loaded signals array
        path: Path to load signals from (if signals is None)
        drop_first_frame: Whether to drop the first frame

    Returns:
        Validated 2D signals array

    Raises:
        ValueError: If array is not 2D or too few frames to drop
    """
    # Load signals if not provided
    if signals is None:
        ds_path = _resolve_dataset_path(path)
        signals = _load_array_any(ds_path)

    # Validate dimensions
    if signals.ndim != NDIM_EXPECTED:
        msg = f"Expected 2D array (series x frames), got {signals.shape}"
        raise ValueError(msg)

    # Drop first frame if requested
    if drop_first_frame:
        # Check we have enough frames
        if signals.shape[1] < MIN_FRAMES_TO_DROP:
            msg = f"Need at least {MIN_FRAMES_TO_DROP} frames."
            raise ValueError(msg)
        # Remove first frame
        signals = signals[:, 1:]

    # Log final shape
    logger.info("Using signals -> shape %s", signals.shape)

    return signals


def _select_output_dir(out_dir: str | Path | None) -> Path:
    """Create and return output directory path.

    Uses provided path or creates default in current directory.

    Args:
        out_dir: Optional output directory path

    Returns:
        Path to output directory (created if doesn't exist)
    """
    # Use provided path or create default
    base = (
        Path(out_dir)
        if out_dir is not None
        else Path.cwd() / "autofilter_outputs"
    )

    # Create directory if needed
    _make_dir_safe(base)

    return base


def _fft_and_cutoffs(
    signals: NDArray[np.float64],
    dt: float,
    levels: int,
    low_frac: float,
    low_ratio: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[float]]:
    """Compute FFT and determine cutoff frequencies.

    Args:
        signals: 2D signals array
        dt: Time step in seconds
        levels: Number of cutoff levels to find
        low_frac: Fraction defining low-frequency region
        low_ratio: Ratio of low to high frequency cutoffs

    Returns:
        Tuple of (frequency_array, magnitude_array, cutoff_list)
    """
    # Compute FFT
    logger.info("[STEP] Computing summed FFT (original data) ...")
    freq, mag = _compute_fft_summed(signals, dt)

    # Log frequency range
    logger.info(
        "Frequency bins: %d | Min/Max freq: %.3e/%.3e Hz",
        len(freq),
        freq.min(),
        freq.max(),
    )

    # Find cutoff frequencies
    logger.info(
        "Selecting %d cutoff(s) with low-freq bias (<=%.2f cum |FFT|)",
        levels,
        low_frac,
    )
    cutoffs = _find_cutoffs_biased(
        freq,
        mag,
        levels,
        low_frac=low_frac,
        low_ratio=low_ratio,
        min_frac=0.05,
        max_frac=0.95,
    )

    # Log selected cutoffs
    logger.info("Cutoffs (Hz, ascending): %s", [f"{c:.2e}" for c in cutoffs])

    return freq, mag, cutoffs


def _process_level(
    i: int,
    total: int,
    cutoff: float,
    signals: NDArray[np.float64],
    fs_hz: float,
    dt: float,
    folder: Path,
    original_trim: NDArray[np.float64],
    n_series: int,
    frames_to_remove: int,
    reuse_existing: bool,
) -> NDArray[np.float64]:
    """Process a single filter level (cutoff frequency).

    Applies Butterworth filter, removes edge artifacts, and saves
    filtered signals along with diagnostic plots.

    Args:
        i: Current level index (1-based)
        total: Total number of levels
        cutoff: Cutoff frequency for this level
        signals: Original signals array
        fs_hz: Sampling frequency in Hz
        dt: Time step in seconds
        folder: Output folder for this level
        original_trim: Trimmed original signals (for comparison)
        n_series: Number of series (for random sampling)
        frames_to_remove: Number of frames to trim from edges
        reuse_existing: Whether to reuse existing filtered file

    Returns:
        Filtered and trimmed signals array
    """
    # Create output folder
    _make_dir_safe(folder)
    out_path = folder / "filtered_signals.npy"

    # Reuse existing file if requested and available
    if out_path.exists() and reuse_existing:
        logger.info("[LEVEL %d/%d] cutoff=%.3e Hz -> REUSE", i, total, cutoff)
        return np.load(out_path)

    # Log start of computation
    logger.info(
        "[LEVEL %d/%d] cutoff=%.3e Hz -> COMPUTE & SAVE to: %s",
        i,
        total,
        cutoff,
        folder,
    )

    # Apply filter to each series
    t0 = time.time()
    filtered = np.array(
        [_butter_lowpass_filter(row, cutoff, fs_hz) for row in signals]
    )
    logger.info("Applied Butterworth via filtfilt in %.2fs", time.time() - t0)

    # Remove edge artifacts
    filtered_trim = _remove_filter_artifacts(
        filtered,
        frames_to_remove=frames_to_remove,
    )

    # Save filtered signals
    np.save(out_path, filtered_trim)
    logger.info(
        "Saved filtered signals: %s -> %s",
        filtered_trim.shape,
        out_path,
    )

    # Create FFT plot of filtered data
    f_filt, mag_filt = _compute_fft_summed(filtered_trim, dt)
    _plot_fft(
        f_filt,
        mag_filt,
        f"Summed FFT (filtered, cutoff={cutoff:.2e} Hz)",
        folder / "fft_plot.png",
    )

    # Create signals + KDE plot
    kde_path = folder / "filt_kde.png"
    _plot_signals_with_kde(
        filtered_trim,
        "Filt_Data + KDE",
        kde_path,
    )

    # Create comparison plots for random series
    n_pick = min(3, n_series)  # Pick up to 3 series
    random.seed(42)
    rand_atoms = random.sample(range(n_series), n_pick)

    # Align original to same length as filtered
    length = filtered_trim.shape[1]
    original_aligned = original_trim[:, -length:]

    # Plot original vs filtered for selected series
    for idx in rand_atoms:
        _plot_single_atom_comparison(
            original_aligned,
            filtered_trim,
            idx,
            folder / f"atom_{idx}_comparison.png",
        )

    # Log completion
    logger.info(
        "Saved %d Original vs Filtered overlays: %s",
        n_pick,
        rand_atoms,
    )

    return filtered_trim


def auto_filtering(
    signals: NDArray[np.float64] | None = None,
    *,
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
) -> AutoFiltInsight:
    """Automatic multi-level Butterworth lowpass filtering.

    Main workflow:
    1. Load/validate input signals
    2. Compute FFT to find frequency content
    3. Select multiple cutoff frequencies (biased to low freq)
    4. Apply Butterworth filter at each cutoff
    5. Create diagnostic plots and videos

    Args:
        signals: Pre-loaded signals array (series x frames), or None
            to load from path
        path: Path to dataset file/folder (used if signals is None)
        dt_ps: Time step in picoseconds
        levels: Number of filter cutoff frequencies to test
        out_dir: Output directory (default: ./autofilter_outputs)
        reuse_existing: Reuse existing filtered files if available
        frames_to_remove: Frames to trim from edges (removes filter
            artifacts)
        low_frac: Cumulative FFT fraction defining "low frequency"
            (<0.20 = low)
        low_ratio: Ratio of low-freq to high-freq cutoffs (2.0 = twice
            as many low)
        seed: Random seed for reproducibility
        max_overlay_traces: Max traces to overlay in video frames
        frame_duration: Duration of each video frame in seconds
        drop_first_frame: Drop first frame from input (removes
            initialization)

    Returns:
        AutoFiltInsight object containing:
            - cutoffs: List of cutoff frequencies used
            - output_dir: Path to output directory
            - filtered_files: Dict mapping cutoff to saved .npy file
            - video_path: Path to forward evolution video
            - meta: Dictionary of parameters used
            - filtered_collection: Tuple of all filtered arrays
    """
    # Validate parameters and compute derived values
    dt, fs = _validate_params(dt_ps, levels)

    # Load or validate signals
    signals = _resolve_signals(signals, path, drop_first_frame)

    # Set up output directory
    base = _select_output_dir(out_dir)

    # Compute FFT and determine cutoff frequencies
    freq, mag, cutoffs = _fft_and_cutoffs(
        signals,
        dt,
        levels,
        low_frac,
        low_ratio,
    )

    # Plot original FFT with cutoff markers
    _plot_fft(
        freq,
        mag,
        "Summed FFT of Original Signals",
        base / "original_fft.png",
        mark_freqs=cutoffs,
    )
    logger.info("[PLOT] Saved original FFT with cutoff markers.")

    # Set up for filtering loop
    fs_hz = fs  # Sampling frequency
    n_series, _ = signals.shape
    logger.info(
        "Processing %d cutoff level(s); fs=%.3e Hz, Nyquist=%.3e Hz",
        len(cutoffs),
        fs_hz,
        0.5 * fs_hz,
    )

    # Trim original signals (for comparison and video)
    original_trim = _remove_filter_artifacts(
        signals,
        frames_to_remove=frames_to_remove,
    )

    # Initialize result containers
    filtered_collection: list[NDArray[np.float64]] = []
    used_cutoffs: list[float] = []
    filtered_paths: dict[float, Path] = {}

    # Initialize global y-limits from original data
    global_ymin = float(np.nanmin(original_trim))
    global_ymax = float(np.nanmax(original_trim))

    # Process each cutoff level
    for i, cutoff in enumerate(sorted(cutoffs), start=1):
        # Create folder for this level
        folder = base / f"cutoff_{_freq_label_for_folder(cutoff)}"

        # Filter signals at this cutoff
        filtered_trim = _process_level(
            i,
            len(cutoffs),
            cutoff,
            signals,
            fs_hz,
            dt,
            folder,
            original_trim,
            n_series,
            frames_to_remove,
            reuse_existing,
        )

        # Update global y-limits to encompass this level
        global_ymin = min(global_ymin, float(np.nanmin(filtered_trim)))
        global_ymax = max(global_ymax, float(np.nanmax(filtered_trim)))

        # Store results
        filtered_collection.append(filtered_trim)
        used_cutoffs.append(cutoff)
        filtered_paths[cutoff] = folder / "filtered_signals.npy"

    # Add padding to y-limits
    pad = (
        1e-7 * (global_ymax - global_ymin)
        if global_ymax > global_ymin
        else 1.0
    )
    yl = (global_ymin - pad, global_ymax + pad)

    # Create evolution videos with consistent y-limits
    video_path = _render_video(
        original_trim,
        filtered_collection,
        used_cutoffs,
        out_path_base=base / "filtered_evolution",
        y_limits=yl,
        max_overlay_traces=max_overlay_traces,
        seed=seed,
        frame_duration=frame_duration,
    )

    # Save all filtered arrays to single compressed NPZ file
    npz_path = base / "filtered_collection.npz"
    np.savez_compressed(npz_path, *filtered_collection)

    # Collect metadata
    meta = {
        "dt_ps": dt_ps,
        "frames_to_remove": frames_to_remove,
        "levels": levels,
        "low_frac": low_frac,
        "low_ratio": low_ratio,
        "seed": seed,
        "frame_duration_s": frame_duration,
    }

    # Return results container
    return AutoFiltInsight(
        cutoffs=used_cutoffs,
        output_dir=base,
        filtered_files=filtered_paths,
        video_path=video_path,
        meta=meta,
        filtered_collection=tuple(filtered_collection)
    )
