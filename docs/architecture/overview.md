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

- `cli`: Parses terminal arguments and presents results or errors.
- `core`: Contains settings, exceptions, domain models, and protocols.
- `generators`: Contains independent procedural generation algorithms.
- `image/encoders`: Converts generated pixel data into a file format.
- `image/writers`: Persists encoded bytes to storage.
- `services`: Coordinates the complete generation use case.
- `shared`: Contains focused validation and path utilities.

## Dependency rule

Core models and protocols do not depend on CLI or concrete infrastructure. Concrete generators, encoders, and writers implement the protocols consumed by the service.

## Adding a generator

1. Create a package under `pixel_forge/generators`.
2. Implement the `ImageGenerator` protocol.
3. Register one instance in `build_default_registry`.
4. Add deterministic tests for seeded output.

The service and CLI do not need algorithm-specific changes.
