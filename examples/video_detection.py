import argparse
from pathlib import Path

import dynsight


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_video = args.input
    video = dynsight.vision.Video(input_video)
    output_folder = Path("output_folder")
    output_folder.mkdir(exist_ok=True)

    # Check if synthetic data exists if not preparing
    if not args.prepare and not (output_folder / "synthetic_dataset").exists():
        msg = "Synthetic dataset not found and --prepare not set."
        raise RuntimeError(msg)

    detection = dynsight.vision.Detect(
        input_video=video,
        project_folder=output_folder,
    )
    # dataset = detection.synthesize()

    # if args.train:
    #    detection.fit(initial_dataset=dataset)


if __name__ == "__main__":
    main()
