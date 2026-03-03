# Screenshot normaliser

Detects the application window in each screenshot, crops, and outputs all images at a consistent size with uniform padding - useful for a set of screenshots taken at slightly different sizes (e.g.: free-hand screenshot tool).

## Features

- Window detection via luminance projection (Otsu threshold), background colour distance, or edge/Hough-line fallback
- Consensus sizing: all output images share the same dimensions (median of detected window sizes)
- Uniform wallpaper-coloured border sampled from the original background

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)

## Quick start

```sh
uv run normaliser.py
```

`uv run` reads the inline script metadata and installs `opencv-python`, `Pillow`, and `numpy` into a temporary environment.

## Project venv

```sh
uv sync
uv run python normaliser.py
```

## Usage

1. Click **Add images** to load one or more screenshots.
2. Click **Detect** — the app highlights the detected window in each image.
3. Adjust **Padding** (pixels added around the detected window) and **Sensitivity** if needed.
4. Click **Normalise** — output files are saved alongside the originals as `<name>_normalised<ext>`.

## How it works

Detection uses a row/column projection approach:

1. Convert to greyscale and apply Otsu's threshold to separate the window from the wallpaper.
2. Compute the fraction of "window" pixels in each row and column.
3. Find the **longest contiguous run** above a threshold in each axis — this robustly handles stray edge pixels that would otherwise collapse a bounding-box approach.
4. If luminance fails, retry using background-colour distance from image corners, then Hough-line edge detection as a last resort.

All detected windows are resized to the median width and height (LANCZOS4), and the border is flood-filled with a colour sampled from the original wallpaper corners.

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Thresholding, edge detection |
| `Pillow` | Image I/O, resize, compositing |
| `numpy` | Array operations |

## License

[MIT](LICENSE) © 2026 Michael Champanis
