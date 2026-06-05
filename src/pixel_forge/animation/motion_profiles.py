"""Motion profile constants for each animated generator.

Each profile name is a plain string stored in AnimationRecipe so it can be
serialised to JSON without an import dependency. Centralising the valid
names here prevents typos and allows the service to validate CLI input.
"""

from __future__ import annotations

# ─── Harmonic Waves ───────────────────────────────────────────────────────────

HARMONIC_PROFILES: tuple[str, ...] = (
    "phase-drift",       # wave phases advance; default
    "rotating-field",    # coordinate system rotates
    "breathing-waves",   # frequency scale oscillates
    "dual-harmonic",     # two frequency bands drift at different speeds
    "color-orbit",       # palette phase cycles
)

HARMONIC_DEFAULT_PROFILE = "phase-drift"

# ─── Radial Bloom ─────────────────────────────────────────────────────────────

RADIAL_PROFILES: tuple[str, ...] = (
    "bloom-pulse",       # petal envelope breathes; default
    "radial-rotation",   # whole field rotates
    "orbital-bloom",     # crown rings orbit outward and inward
    "spiral-breath",     # ripple phase advances
    "eclipse-pulse",     # halo ring pulses
)

RADIAL_DEFAULT_PROFILE = "bloom-pulse"

# ─── Plasma Flow ──────────────────────────────────────────────────────────────

PLASMA_PROFILES: tuple[str, ...] = (
    "flow-cycle",        # domain warp phase advances; default
    "vortex-orbit",      # vortex positions circle
    "filament-breath",   # filament field pulses
    "plasma-tide",       # flow direction bias sweeps
)

PLASMA_DEFAULT_PROFILE = "flow-cycle"

# ─── Mandelbrot Dream ─────────────────────────────────────────────────────────

MANDELBROT_PROFILES: tuple[str, ...] = (
    "color-cycle",       # color_cycle offset advances; default
    "micro-orbit",       # camera orbits a small circle around the region centre
    "fractal-breath",    # zoom pulsates slightly around the recipe zoom
    "orbit-trap-cycle",  # orbit-trap phase shifts
)

MANDELBROT_DEFAULT_PROFILE = "color-cycle"

# ─── All profiles by generator ────────────────────────────────────────────────

ALL_PROFILES_BY_GENERATOR: dict[str, tuple[str, ...]] = {
    "harmonic-waves":   HARMONIC_PROFILES,
    "radial-bloom":     RADIAL_PROFILES,
    "plasma-flow":      PLASMA_PROFILES,
    "mandelbrot-dream": MANDELBROT_PROFILES,
}

DEFAULT_PROFILE_BY_GENERATOR: dict[str, str] = {
    "harmonic-waves":   HARMONIC_DEFAULT_PROFILE,
    "radial-bloom":     RADIAL_DEFAULT_PROFILE,
    "plasma-flow":      PLASMA_DEFAULT_PROFILE,
    "mandelbrot-dream": MANDELBROT_DEFAULT_PROFILE,
}
