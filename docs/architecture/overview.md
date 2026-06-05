# Architecture Overview

PixelForge uses a small layered architecture with explicit dependency boundaries.

```text
CLI
  -> Generation service
       -> Request validator
       -> Generator registry
            -> Image generator
       -> Image encoder
       -> Binary writer
```

## Main responsibilities

- `cli`: parses terminal arguments and presents results or errors;
- `core`: contains settings, exceptions, domain models, and protocols;
- `generators`: contains independent procedural generation algorithms;
- `image/encoders`: converts generated RGB data into a file format;
- `image/writers`: persists encoded bytes to storage;
- `services`: coordinates the complete generation use case;
- `shared`: contains focused validation and path utilities.

## Dependency rule

Core models and protocols do not depend on the CLI or concrete infrastructure.
Concrete generators, encoders, and writers implement contracts consumed by the
application service.

## Seeded NumPy generators

The artistic generators inherit from `SeededArrayGenerator`. The base class
centralizes:

- automatic seed creation when no seed is supplied;
- deterministic NumPy random generator creation;
- RGB shape and dtype validation;
- conversion to the immutable `GeneratedImage` model.

Reusable mathematical helpers live in `pixel_forge.generators.common`:

- `fields.py`: Cartesian and polar coordinate fields with aspect correction;
- `color.py`: vectorized HSV conversion and cosine color palettes;
- `types.py`: shared typed NumPy array aliases;
- `base.py`: the common seeded-array generator lifecycle.

Each concrete generator remains focused on its mathematical formula.

## Adding a generator

1. Create a package under `pixel_forge/generators`.
2. Implement the `ImageGenerator` protocol or inherit from `SeededArrayGenerator`.
3. Register one instance in `build_default_registry`.
4. Add deterministic tests for seeded output.

The service and CLI do not need algorithm-specific changes.
