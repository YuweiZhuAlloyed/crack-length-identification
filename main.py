import os
import argparse
from pathlib import Path
from src.trad_crack_mask_extractor import GMMCrackMaskExtractor, ThresholdCrackMaskExtractor


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract masks for all images in a directory.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the input directory containing images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory to save masks.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_name in os.listdir(input_dir):
        input_path = input_dir / image_name
        output_path = output_dir / f"mask_{image_name}"
        extractor = GMMCrackMaskExtractor(
            image_path=input_path.as_posix(), save_path=output_path.as_posix())
        extractor.extract()
        print(f"Processed {image_name}, saved mask to {output_path}")
