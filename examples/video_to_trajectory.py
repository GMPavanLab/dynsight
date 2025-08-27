"""Virtualization of a video in order to extract trajectory information.

In this example, we show how to use the dynsight ``vision`` module
combined with the track module to obtain a trajectory file from a video.

A very simple video is used to demonstrate the workflow, for this reason the
default detection model is used. For more complex videos, it is recommended to
exploit the ``label_tool`` to create a synthesized dataset as a starting point
to train the initial model and then follow the subsequent steps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from dynsight.track import track_xyz
from dynsight.vision import VisionInstance


def plot_results(
    instance: VisionInstance,
    output_path: Path,
    name: str,
) -> None:
    if instance.prediction_results is None:
        msg = "No prediction results found"
        raise ValueError(msg)

    n_detections = [len(result) for result in instance.prediction_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(n_detections, marker="o")
    ax1.set_title("N° Detections in Time")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("N° Detections")

    ax2.hist(n_detections, bins="auto", edgecolor="black")
    ax2.set_title("Detection Distribution")
    ax2.set_xlabel("N° Detections")
    ax2.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(output_path / f"{name}.png", dpi=600)
    plt.close()


def main() -> None:
    video_path = Path("video_to_trajectory/example_video.mp4")
    n_iterations = 5
    instance = VisionInstance(
        source=video_path,
        output_path=Path("output"),
        device="0",  # select GPU id, "cpu" or "mps" for MacOS users.
        workers=8,  # number of cores used.
    )
    for it in range(n_iterations):
        instance.predict(prediction_title=f"prediction_{it}")
        plot_results(
            instance=instance,
            output_path=instance.output_path,
            name=f"results_plot_{it}",
        )
        instance.create_dataset_from_predictions(
            dataset_name=f"dataset_{it}",
        )
        instance.set_training_dataset(
            training_data_yaml=instance.output_path
            / f"dataset_{it}"
            / "dataset.yaml",
        )
        instance.train(title=f"train_{it}")
    traj_path = instance.export_prediction_to_xyz(
        file_name=Path("trajectory.xyz")
    )
    track_xyz(
        input_xyz=traj_path,
        output_xyz=Path("output/tracked_traj.xyz"),
        search_range=10,
    )


if __name__ == "__main__":
    main()
