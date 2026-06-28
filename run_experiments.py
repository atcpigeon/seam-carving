"""Batch experiments for seam carving project.

This script automatically runs the main experiment setting:
- removal ratios: 10%, 20%, 30% of image width
- methods: standard resize, center crop, seam carving
- outputs: method results, red-bordered comparison images, runtime CSV summary
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from seam_carving import (
    center_crop,
    load_image,
    make_comparison_image,
    resize_image,
    save_image,
    standard_resize,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_RATIOS = [0.10, 0.20, 0.30]


def iter_images(input_dir: Path):
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def parse_ratios(ratio_text: str) -> list[float]:
    ratios = []
    for part in ratio_text.split(","):
        value = float(part.strip())
        if value > 1:
            value /= 100
        if value <= 0 or value >= 1:
            raise ValueError("Each ratio must be between 0 and 1, or between 0 and 100 as a percentage.")
        ratios.append(value)
    return ratios


def method_runtime(method_name: str, image, target_width: int, target_height: int, verbose: bool = False):
    start = time.time()
    if method_name == "standard_resize":
        result = standard_resize(image, target_width, target_height)
    elif method_name == "center_crop":
        result = center_crop(image, target_width, target_height)
    elif method_name == "seam_carving":
        result = resize_image(image, target_width, target_height, verbose=verbose)
    else:
        raise ValueError(f"Unknown method: {method_name}")
    return result, time.time() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 10%, 20%, 30% seam-carving experiments")
    parser.add_argument("--input-dir", default="input", help="folder containing input images")
    parser.add_argument("--output-dir", default="output/experiments", help="folder for experiment outputs")
    parser.add_argument(
        "--ratios",
        default=",".join(str(r) for r in DEFAULT_RATIOS),
        help="comma-separated removal ratios, e.g. 0.1,0.2,0.3 or 10,20,30",
    )
    parser.add_argument(
        "--height-ratio",
        type=float,
        default=0.0,
        help="optional height removal ratio; default 0 means only width is reduced",
    )
    parser.add_argument("--verbose", action="store_true", help="print seam removal progress")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = parse_ratios(args.ratios)
    height_ratio = args.height_ratio / 100 if args.height_ratio > 1 else args.height_ratio
    if height_ratio < 0 or height_ratio >= 1:
        raise ValueError("Height ratio must be between 0 and 1, or between 0 and 100 as a percentage.")

    rows = []
    image_paths = list(iter_images(input_dir))
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    for image_path in image_paths:
        image = load_image(image_path)
        input_height, input_width = image.shape[:2]
        stem = image_path.stem
        suffix = image_path.suffix

        for ratio in ratios:
            removed_vertical = max(1, round(input_width * ratio))
            removed_horizontal = round(input_height * height_ratio)
            target_width = max(1, input_width - removed_vertical)
            target_height = max(1, input_height - removed_horizontal)
            ratio_label = f"{round(ratio * 100)}pct"
            case_dir = output_dir / stem / ratio_label
            case_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"Processing {image_path.name} at {ratio_label}: "
                f"{input_width}x{input_height} -> {target_width}x{target_height}"
            )

            resize_result, resize_runtime = method_runtime("standard_resize", image, target_width, target_height)
            crop_result, crop_runtime = method_runtime("center_crop", image, target_width, target_height)
            seam_result, seam_runtime = method_runtime(
                "seam_carving", image, target_width, target_height, verbose=args.verbose
            )

            original_path = case_dir / f"{stem}_original{suffix}"
            resize_path = case_dir / f"{stem}_{ratio_label}_standard_resize{suffix}"
            crop_path = case_dir / f"{stem}_{ratio_label}_center_crop{suffix}"
            seam_path = case_dir / f"{stem}_{ratio_label}_seam_carving{suffix}"
            comparison_path = case_dir / f"{stem}_{ratio_label}_comparison{suffix}"

            save_image(image, original_path)
            save_image(resize_result, resize_path)
            save_image(crop_result, crop_path)
            save_image(seam_result, seam_path)
            save_image(
                make_comparison_image(
                    [
                        ("Original", image),
                        ("Standard resize", resize_result),
                        ("Center crop", crop_result),
                        ("Seam carving", seam_result),
                    ]
                ),
                comparison_path,
            )

            method_rows = [
                ("standard_resize", resize_runtime, resize_path),
                ("center_crop", crop_runtime, crop_path),
                ("seam_carving", seam_runtime, seam_path),
            ]
            for method, runtime, output_path in method_rows:
                rows.append(
                    {
                        "image": image_path.name,
                        "ratio": f"{ratio:.2f}",
                        "ratio_percent": round(ratio * 100),
                        "method": method,
                        "input_width": input_width,
                        "input_height": input_height,
                        "target_width": target_width,
                        "target_height": target_height,
                        "removed_vertical_seams": input_width - target_width,
                        "removed_horizontal_seams": input_height - target_height,
                        "runtime_seconds": f"{runtime:.4f}",
                        "output_path": str(output_path),
                    }
                )

            print(
                f"Saved {ratio_label} outputs. Runtime: "
                f"resize={resize_runtime:.4f}s, crop={crop_runtime:.4f}s, seam={seam_runtime:.4f}s"
            )

    csv_path = output_dir / "runtime_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image",
            "ratio",
            "ratio_percent",
            "method",
            "input_width",
            "input_height",
            "target_width",
            "target_height",
            "removed_vertical_seams",
            "removed_horizontal_seams",
            "runtime_seconds",
            "output_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved runtime summary to {csv_path}")


if __name__ == "__main__":
    main()
