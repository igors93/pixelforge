# ADR 0001: Store generated images as raw RGB bytes

## Status

Accepted

## Context

The generation domain should not depend directly on Pillow. A generator only needs to create pixel data, while encoding belongs to an infrastructure concern.

## Decision

`GeneratedImage` stores immutable raw RGB bytes, dimensions, generator name, and seed metadata. The PNG encoder is responsible for converting those bytes into a PNG file.

## Consequences

- Generators remain easy to test without filesystem access.
- New encoders can reuse the same generated pixel data.
- The current RGB model uses exactly three bytes per pixel.
