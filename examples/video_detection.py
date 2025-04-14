import argparse
from pathlib import Path

import dynsight


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynsight video detection")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare synthesized data",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train and predict",
    )
    args = parser.parse_args()
    input_video = Path("video.mp4")
    frames = dynsight.vision.Video(input_video)
    output_folder = Path("output_folder")
    output_folder.mkdir(exist_ok="True")

    # Check if data synthesised exists.
    if (
        args.prepare is False
        and not (output_folder / "synthesized_data").exist()
    ):
        raise RuntimeError

    detection = dynsight.vision.Detect(
        input_frames=frames,
        project_folder=output_folder,
    )
    dataset = detection.synthesize(
        n_epochs=2,
        dataset_dimension=1000,
        reference_img="0.png",
        training_set_fraction=0.7,
        sample_from="gui",
    )
    if args.train:
        trained_model = detection.fit(input=dataset)
    else:
        trained_model = detection.load_model()

    detection.predict(model=trained_model)


if __name__ == "__main__":
    main()
