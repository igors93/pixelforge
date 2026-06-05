# 0002 - Use vectorized NumPy fields for mathematical generators

## Status

Accepted.

## Context

PixelForge originally generated independent RGB noise with Python's standard
random module. Mathematical patterns require evaluating trigonometric, radial,
and fractal formulas for every pixel. Python-level pixel loops become difficult
to read and unnecessarily slow at the configured 1000 x 1000 limit.

## Decision

Artistic generators use NumPy arrays and vectorized operations. A shared
`SeededArrayGenerator` owns seed management, validates the returned RGB array,
and converts it to the existing domain model.

Coordinate and color helpers are shared, while each generator keeps its own
visual formula.

## Consequences

- NumPy becomes a runtime dependency.
- Mathematical generators remain concise and fast enough for the current size limit.
- Seeded outputs are reproducible.
- New generators can reuse coordinate fields and color mapping without changing
  the application service or CLI.
