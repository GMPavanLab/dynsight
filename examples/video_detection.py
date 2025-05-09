import argparse
from pathlib import Path

import dynsight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynsight video detection")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare synthesized data.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Predict the video.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output folder.",
    )
    parser.add_argument(
        "--maxcycle",
        type=int,
        required=False,
        default=5,
        help="Number of fitting cycle procedures.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=100,
        help="Number of epochs for each training session.",
    )
    parser.add_argument(
        "--patiece",
        type=int,
        required=False,
        default=10,
        help="Patience in terms of epochs to earlystop the procedure.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        required=False,
        default=2,
        help="BatchSize for training.",
    )
    parser.add_argument(
        "--workers", type=int, required=True, help="Number of CPUs used."
    )
    parser.add_argument(
        "--gpu",
        type=tuple,
        required=False,
        default=None,
        help="ID of the GPU(s) used.",
    )
    parser.add_argument(
        "--step",
        type=int,
        required=False,
        default=1,
        help="Step for frame to be used.",
    )
    parser.add_argument(
        "--detect_model",
        type=Path,
        required=False,
        default=None,
        help="Model to be used for predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_video = args.input
    output_project = args.output
    video = dynsight.vision.Video(input_video)
    detection = dynsight.vision.Detect(
        input_video=video,
        project_folder=output_project,
    )
    if args.prepare:
        detection.synthesize()
    if args.train:
        detection.fit(
            initial_dataset=detection.get_project_path() / "synthetic_dataset",
            max_sessions=args.maxcycle,
            training_epochs=args.epochs,
            training_patience=args.patience,
            batch_size=args.batchsize,
            workers=args.workers,
            device=args.gpu,
            frame_reading_step=args.step,
        )
    if args.predict:
        detection.predict_frames(model_path=args.detect_model)
        detection.compute_xyz(
            detection_folder_path="prediction",
            output_path=Path.cwd(),
        )


if __name__ == "__main__":
    main()
