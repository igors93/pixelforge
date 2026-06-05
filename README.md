# PixelForge

PixelForge is a modular command-line application for procedural image generation.
It creates deterministic mathematical artwork from dimensions, a generator name,
and an optional seed.

The application separates generation, encoding, file writing, validation, and CLI
concerns. New visual algorithms can be added without changing the main generation
flow.

## Requirements

- Python 3.11 or newer
- NumPy 1.26 or newer
- Pillow 10 or newer

## Installation

Create a virtual environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Available generators

- `harmonic-waves`: domain-warped sine waves with smooth cosine palettes
- `plasma-flow`: fluid plasma built from interacting trigonometric fields
- `radial-bloom`: flower-like polar patterns with petals and radial ripples
- `mandelbrot-dream`: a smoothly colored Mandelbrot fractal
- `random-noise`: independent random RGB pixels kept as a baseline generator

## Generate an image

`harmonic-waves` is the default generator:

```bash
pixelforge generate \
  --width 512 \
  --height 512 \
  --seed 42 \
  --output output/harmonic-waves.png
```

Select another generator:

```bash
pixelforge generate \
  --width 800 \
  --height 800 \
  --generator radial-bloom \
  --seed 7 \
  --output output/radial-bloom.png
```

Generate a Mandelbrot variation:

```bash
pixelforge generate \
  --width 1000 \
  --height 700 \
  --generator mandelbrot-dream \
  --seed 120 \
  --output output/mandelbrot-dream.png
```

The same generator, size, and seed always produce the same pixel data.

Overwrite an existing file explicitly:

```bash
pixelforge generate --output output/image.png --overwrite
```

List registered generators:

```bash
pixelforge list-generators
```

You can also run the package without installing the console command:

```bash
PYTHONPATH=src python -m pixel_forge generate --output output/image.png
```

## Current limits

- Minimum width and height: 1 pixel
- Maximum width and height: 1000 pixels
- Output format: PNG
- Color mode: RGB

The limits are centralized in `src/pixel_forge/core/config/settings.py`.

## Adding another generator

1. Create a package under `src/pixel_forge/generators/`.
2. Inherit from `SeededArrayGenerator` when the algorithm returns a NumPy RGB array.
3. Implement `name` and `render`.
4. Register the generator in `build_default_registry`.
5. Add deterministic tests for seeded output.

The service, encoder, writer, and CLI do not need algorithm-specific changes.

## Development

Install development dependencies and run the test suite:

```bash
python -m pip install -e ".[dev]"
pytest
```

Run the quality checks:

```bash
ruff check .
mypy src
```

## Architecture

See `docs/architecture/overview.md` for the dependency flow and extension points.
