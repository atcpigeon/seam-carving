# Content-Aware Image Resizing via Seam Carving

This repository contains a Python implementation of seam carving for content-aware image resizing, together with experiment scripts and a short project report.

## Features

- Load and save RGB images
- Compute pixel energy from image gradients
- Find minimum-energy vertical seams with dynamic programming
- Remove seams to resize images without uniform distortion
- Resize width and height
- Optional seam visualization for the first seam
- Standard resize and center-crop baselines
- Side-by-side comparison images
- Runtime reporting for experiment analysis
- Batch experiment script for all images in `input/`
- Simple object removal with a user-specified rectangular mask

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Basic usage

Resize an image to a new width.
If the input file is inside the `input/` folder, you can pass just the filename.
By default, the output keeps the original filename and is saved into the `output/` folder:

```bash
python seam_carving.py sample.png --width 600
```

This command creates:

```text
output/sample.png
output/sample_standard_resize.png
output/sample_center_crop.png
output/sample_comparison.png
```

Save a preview with the first seam highlighted:

```bash
python seam_carving.py sample.png --width 600 --show-seam
```

## Object removal

You can remove an object by giving a rectangle mask with `x y width height`.
The rectangle is converted into a low-energy region so seams are encouraged to pass through it.

```bash
python seam_carving.py sample.png --remove-object 120 80 140 100
```

This removes the masked region and saves the result.

If you want to remove the object and then restore the original width with ordinary resizing:

```bash
python seam_carving.py sample.png --remove-object 120 80 140 100 --restore-width
```

## Baseline outputs

For each run, the script also saves:

- standard resize baseline
- center crop baseline
- comparison image with red borders around the four panels

## Batch experiments

Run all images in `input/` with the recommended final experiment setup:

- 10%, 20%, 30% width reduction
- standard resize
- center crop
- seam carving
- runtime summary

```bash
python run_experiments.py
```

Outputs go to:

```text
output/experiments/
```

Each image gets subfolders like:

```text
output/experiments/sample/10pct/
output/experiments/sample/20pct/
output/experiments/sample/30pct/
```

Each ratio folder contains:

- original image
- standard resize result
- center crop result
- seam carving result
- side-by-side comparison image

The script also writes:

```text
output/experiments/runtime_summary.csv
```

You can customize the ratios:

```bash
python run_experiments.py --ratios 10,20,30
```

or:

```bash
python run_experiments.py --ratios 0.1,0.2,0.3
```

## Notes

- Horizontal resizing is implemented by transposing the image and reusing the vertical seam logic.
- The implementation uses NumPy and Pillow. OpenCV is optional for Sobel-gradient computation; otherwise a NumPy finite-difference implementation is used.
- Object removal is a simplified extension based on lowering the energy inside a user-specified rectangular mask.
