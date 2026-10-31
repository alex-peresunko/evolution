"""Procedurally drawn, pre-rotated, cached sprites.

Every sprite is drawn once at SS× resolution and smoothscaled down (anti-aliasing), then cached
by quantized (on-screen size, health, walk frame, colour gene, angle); a frame is mostly blits.
Zooming scales positions and sprite sizes, so zoomed sprites are drawn natively."""
from .world_renderer import WorldRenderer

__all__ = ["WorldRenderer"]
