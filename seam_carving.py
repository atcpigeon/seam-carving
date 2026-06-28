"""Content-aware image resizing via seam carving.

This module implements the functionality described in the project proposal:
- image loading and saving
- energy computation from image gradients
- dynamic-programming seam search
- seam removal and resizing
- optional seam visualization helpers

The implementation is intentionally self-contained so it can be used as the
core of a CS240 project report/demo.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except Exception:  # pragma: no cover - optional fallback
    cv2 = None


Array = np.ndarray


@dataclass
class SeamResult:
    seam: List[Tuple[int, int]]
    energy: float


def load_image(path: str | Path) -> Array:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_image(image: Array, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(output_path)


def resolve_input_path(path: str | Path) -> Path:
    input_path = Path(path)
    if input_path.exists():
        return input_path

    input_folder_path = Path("input") / input_path
    if input_folder_path.exists():
        return input_folder_path

    raise FileNotFoundError(
        f"Cannot find input image '{path}'. Tried '{input_path}' and '{input_folder_path}'."
    )


def compute_energy(image: Array) -> Array:
    gray = image.astype(np.float32)
    gray = 0.299 * gray[:, :, 0] + 0.587 * gray[:, :, 1] + 0.114 * gray[:, :, 2]

    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return np.zeros_like(gray)

    if cv2 is not None:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.abs(gx) + np.abs(gy)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gx[:, 0] = gray[:, 1] - gray[:, 0]
    gx[:, -1] = gray[:, -1] - gray[:, -2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gy[0, :] = gray[1, :] - gray[0, :]
    gy[-1, :] = gray[-1, :] - gray[-2, :]
    return np.abs(gx) + np.abs(gy)


def apply_energy_mask(energy: Array, mask: Array, bias: float = -1e6) -> Array:
    adjusted = energy.astype(np.float64).copy()
    adjusted[mask] += bias
    return adjusted


def make_rectangle_mask(shape: Tuple[int, int], x: int, y: int, width: int, height: int) -> Array:
    h, w = shape
    if width <= 0 or height <= 0:
        raise ValueError("Object removal rectangle must have positive width and height.")
    if x < 0 or y < 0 or x >= w or y >= h:
        raise ValueError("Object removal rectangle origin must be inside the image.")
    x2 = min(x + width, w)
    y2 = min(y + height, h)
    mask = np.zeros((h, w), dtype=bool)
    mask[y:y2, x:x2] = True
    return mask


def remove_object_by_mask(image: Array, mask: Array, verbose: bool = False) -> Array:
    result = image.copy()
    working_mask = mask.copy()
    removed = 0

    while working_mask.any():
        energy = compute_energy(result)
        adjusted_energy = apply_energy_mask(energy, working_mask)
        seam_result = find_vertical_seam(adjusted_energy)
        seam = seam_result.seam
        result = remove_vertical_seam(result, seam)
        working_mask = remove_vertical_seam(working_mask[:, :, None].astype(np.uint8), seam)[:, :, 0].astype(bool)
        removed += 1
        if verbose and (removed == 1 or removed % 10 == 0 or not working_mask.any()):
            print(f"Removed {removed} seams while erasing object; remaining masked pixels = {int(working_mask.sum())}")

    return result


def find_vertical_seam(energy: Array) -> SeamResult:
    h, w = energy.shape
    m = energy.astype(np.float64).copy()
    backtrack = np.zeros((h, w), dtype=np.int32)

    for i in range(1, h):
        prev = m[i - 1]
        left = np.empty_like(prev)
        mid = prev
        right = np.empty_like(prev)

        left[0] = np.inf
        left[1:] = prev[:-1]
        right[-1] = np.inf
        right[:-1] = prev[1:]

        candidates = np.stack((left, mid, right), axis=0)
        choice = np.argmin(candidates, axis=0)
        offsets = np.array([-1, 0, 1], dtype=np.int32)
        backtrack[i] = np.arange(w, dtype=np.int32) + offsets[choice]
        m[i] += np.min(candidates, axis=0)

    j = int(np.argmin(m[-1]))
    seam: List[Tuple[int, int]] = []
    total = float(m[-1, j])
    for i in range(h - 1, -1, -1):
        seam.append((i, j))
        if i > 0:
            j = int(backtrack[i, j])
    seam.reverse()
    return SeamResult(seam=seam, energy=total)


def remove_vertical_seam(image: Array, seam: List[Tuple[int, int]]) -> Array:
    h, w, c = image.shape
    mask = np.ones((h, w), dtype=bool)
    for i, j in seam:
        mask[i, j] = False
    return image[mask].reshape(h, w - 1, c)


def transpose_image(image: Array) -> Array:
    return np.transpose(image, (1, 0, 2))


def carve_width(image: Array, target_width: int, verbose: bool = False) -> Array:
    result = image.copy()
    original_width = result.shape[1]
    seams_to_remove = original_width - target_width

    for removed in range(seams_to_remove):
        energy = compute_energy(result)
        seam = find_vertical_seam(energy).seam
        result = remove_vertical_seam(result, seam)

        if verbose and (removed == 0 or (removed + 1) % 10 == 0 or removed + 1 == seams_to_remove):
            print(f"Removed {removed + 1}/{seams_to_remove} vertical seams; current width = {result.shape[1]}")

    return result


def carve_height(image: Array, target_height: int, verbose: bool = False) -> Array:
    transposed = transpose_image(image)
    carved = carve_width(transposed, target_height, verbose=verbose)
    return transpose_image(carved)


def resize_image(
    image: Array,
    target_width: int | None = None,
    target_height: int | None = None,
    verbose: bool = False,
) -> Array:
    result = image
    if target_width is not None and target_width < result.shape[1]:
        result = carve_width(result, target_width, verbose=verbose)
    if target_height is not None and target_height < result.shape[0]:
        result = carve_height(result, target_height, verbose=verbose)
    return result


def mark_seam(image: Array, seam: List[Tuple[int, int]], color: Tuple[int, int, int] = (255, 0, 0)) -> Array:
    marked = image.copy()
    for i, j in seam:
        marked[i, j] = color
    return marked


def make_output_path(input_path: Path, output_arg: str | None = None) -> Path:
    if output_arg is not None:
        output_path = Path(output_arg)
        if output_path.parent == Path("."):
            return Path("output") / output_path
        return output_path
    return Path("output") / input_path.name


def standard_resize(image: Array, target_width: int, target_height: int | None = None) -> Array:
    if target_height is None:
        target_height = image.shape[0]
    pil_img = Image.fromarray(image.astype(np.uint8))
    resized = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return np.array(resized, dtype=np.uint8)


def center_crop(image: Array, target_width: int | None = None, target_height: int | None = None) -> Array:
    h, w, _ = image.shape
    if target_width is None:
        target_width = w
    if target_height is None:
        target_height = h
    if target_width > w or target_height > h:
        raise ValueError("Center crop target size cannot be larger than input size.")

    start_x = (w - target_width) // 2
    start_y = (h - target_height) // 2
    return image[start_y : start_y + target_height, start_x : start_x + target_width, :]


def make_comparison_image(images: List[Tuple[str, Array]]) -> Array:
    pil_images = [(title, Image.fromarray(img.astype(np.uint8))) for title, img in images]
    display_height = max(img.height for _, img in pil_images)
    separator_width = 24
    border_width = 6
    margin = 12
    title_height = 36

    display_items = []
    for title, img in pil_images:
        display_width = round(img.width * display_height / img.height)
        display_img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        display_items.append((title, display_img, display_width))

    canvas_width = sum(width for _, _, width in display_items) + separator_width * (len(display_items) - 1) + margin * 2
    canvas_height = display_height + title_height + margin * 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    red = (255, 0, 0)
    black = (0, 0, 0)

    x = margin
    y = margin + title_height
    for title, display_img, width in display_items:
        draw.text((x, margin), title, fill=black)
        canvas.paste(display_img, (x, y))
        draw.rectangle(
            [x, y, x + width - 1, y + display_height - 1],
            outline=red,
            width=border_width,
        )
        x += width + separator_width

    return np.array(canvas, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Content-aware image resizing via seam carving")
    parser.add_argument("input", help="input image path or filename inside the input folder")
    parser.add_argument("output", nargs="?", default=None, help="optional output path; defaults to output/<input filename>")
    parser.add_argument("--width", type=int, default=None, help="target width")
    parser.add_argument("--height", type=int, default=None, help="target height")
    parser.add_argument("--show-seam", action="store_true", help="save a preview image with the first seam highlighted")
    parser.add_argument("--seam-preview", default=None, help="path for seam preview image")
    parser.add_argument("--remove-object", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"), help="remove an object inside a rectangle by lowering its seam energy")
    parser.add_argument("--restore-width", action="store_true", help="after object removal, resize the result back to the original width using ordinary resizing")
    parser.add_argument("--verbose", action="store_true", help="print progress while removing seams")
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    image = load_image(input_path)

    if args.width is not None and args.width <= 0:
        raise ValueError("Target width must be positive.")
    if args.height is not None and args.height <= 0:
        raise ValueError("Target height must be positive.")
    input_width = image.shape[1]
    input_height = image.shape[0]

    print(f"Loaded {input_path} with size {input_width}x{input_height}")

    if args.width is not None and args.width > input_width:
        raise ValueError(
            "This implementation removes seams only, so target width cannot be larger than input width. "
            f"Input width: {input_width}; target width: {args.width}."
        )
    if args.height is not None and args.height > input_height:
        raise ValueError(
            "This implementation removes seams only, so target height cannot be larger than input height. "
            f"Input height: {input_height}; target height: {args.height}."
        )
    if args.remove_object is not None and args.height is not None:
        raise ValueError("Object removal currently supports vertical seam removal only, so --height cannot be used with --remove-object.")
    if args.width is not None:
        print(f"Target width: {args.width}; seams to remove: {image.shape[1] - args.width}")
    if args.height is not None:
        print(f"Target height: {args.height}; seams to remove: {image.shape[0] - args.height}")

    output_path = make_output_path(input_path, args.output)
    comparison_path = output_path.with_name(f"{output_path.stem}_comparison{output_path.suffix}")

    if args.show_seam:
        energy = compute_energy(image)
        seam = find_vertical_seam(energy).seam
        preview_path = Path(args.seam_preview) if args.seam_preview else output_path.with_name(f"{output_path.stem}_seam{output_path.suffix}")
        if preview_path.parent == Path("."):
            preview_path = Path("output") / preview_path
        save_image(mark_seam(image, seam), preview_path)
        print(f"Saved seam preview to {preview_path}")

    target_width = args.width or image.shape[1]
    target_height = args.height or image.shape[0]

    start = time.time()
    if args.remove_object is not None:
        x, y, rect_width, rect_height = args.remove_object
        object_mask = make_rectangle_mask((input_height, input_width), x, y, rect_width, rect_height)
        print(f"Removing object rectangle: x={x}, y={y}, width={rect_width}, height={rect_height}")
        object_removed = remove_object_by_mask(image, object_mask, verbose=args.verbose)
        if args.restore_width:
            resized = standard_resize(object_removed, input_width, input_height)
        elif args.width is not None and args.width < object_removed.shape[1]:
            resized = resize_image(object_removed, args.width, None, verbose=args.verbose)
        else:
            resized = object_removed
    else:
        resized = resize_image(image, args.width, args.height, verbose=args.verbose)
    elapsed = time.time() - start

    target_width = resized.shape[1]
    target_height = resized.shape[0]
    baseline_resize = standard_resize(image, target_width, target_height)
    baseline_crop = center_crop(image, target_width, target_height)

    save_image(resized, output_path)
    save_image(baseline_resize, output_path.with_name(f"{output_path.stem}_standard_resize{output_path.suffix}"))
    save_image(baseline_crop, output_path.with_name(f"{output_path.stem}_center_crop{output_path.suffix}"))
    comparison = make_comparison_image(
        [
            ("Original", image),
            ("Standard resize", baseline_resize),
            ("Center crop", baseline_crop),
            ("Seam/Object result", resized),
        ]
    )
    save_image(comparison, comparison_path)
    print(f"Saved resized image to {output_path} with size {resized.shape[1]}x{resized.shape[0]}")
    print(f"Saved standard resize baseline to {output_path.with_name(f'{output_path.stem}_standard_resize{output_path.suffix}')}")
    print(f"Saved center crop baseline to {output_path.with_name(f'{output_path.stem}_center_crop{output_path.suffix}')}")
    print(f"Saved comparison image to {comparison_path}")
    print(f"Runtime: {elapsed:.4f} seconds")


if __name__ == "__main__":
    main()
