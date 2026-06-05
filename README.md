# PixelForge

PixelForge is a modular command-line application for generating procedural images. The first release contains one generator: uniform RGB random noise.

The project deliberately separates generation, encoding, file writing, validation, and CLI concerns. This makes it possible to add new algorithms or output formats without rewriting the application flow.

## Requirements

- Python 3.11 or newer
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

## Generate an image

```bash
pixelforge generate \
  --width 512 \
  --height 512 \
  --generator random-noise \
  --output output/noise.png
```

Use a seed to reproduce the same image:

```bash
pixelforge generate \
  --width 512 \
  --height 512 \
  --seed 42 \
  --output output/noise-seed-42.png
```

Overwrite an existing file explicitly:

```bash
pixelforge generate --output output/noise.png --overwrite
```

List registered generators:

```bash
pixelforge list-generators
```

You can also run the package without installing the console command:

```bash
PYTHONPATH=src python -m pixel_forge generate --output output/noise.png
```

## Current limits

- Minimum width and height: 1 pixel
- Maximum width and height: 1000 pixels
- Output format: PNG
- Color mode: RGB

The limits are centralized in `src/pixel_forge/core/config/settings.py`.

## Development

Install development dependencies and run the test suite:

```bash
python -m pip install -e ".[dev]"
pytest
```

Optional quality checks:

```bash
ruff check .
mypy src
```

## Architecture

See `docs/architecture/overview.md` for the dependency flow and extension points.
