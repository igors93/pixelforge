"""High-quality antialiased shape drawing utilities based on Pillow."""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFilter

from pixel_forge.generators.geometric_orbit.parameters import Color, ShapeSpec

RGBA = tuple[int, int, int, int]


def _rgba(color: Color, alpha: int = 255) -> RGBA:
    return color[0], color[1], color[2], alpha


def _draw_geometry(
    draw: ImageDraw.ImageDraw,
    *,
    kind: str,
    box: tuple[int, int, int, int],
    fill: RGBA,
    outline: RGBA | None,
    outline_width: int,
) -> None:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top

    if kind == "circle":
        draw.ellipse(box, fill=fill, outline=outline, width=outline_width)
        return

    if kind == "triangle":
        points = (
            ((left + right) // 2, top),
            (right, bottom),
            (left, bottom),
        )
        draw.polygon(points, fill=fill)
        if outline is not None and outline_width > 0:
            draw.line((*points, points[0]), fill=outline, width=outline_width, joint="curve")
        return

    if kind == "capsule":
        radius = max(1, min(width, height) // 2)
        draw.rounded_rectangle(
            box,
            radius=radius,
            fill=fill,
            outline=outline,
            width=outline_width,
        )
        return

    if kind == "line":
        radius = max(1, height // 2)
        draw.rounded_rectangle(
            box,
            radius=radius,
            fill=fill,
            outline=outline,
            width=outline_width,
        )
        return

    draw.rectangle(box, fill=fill, outline=outline, width=outline_width)


def draw_shape(
    canvas: Image.Image,
    *,
    shape: ShapeSpec,
    scale: int,
    shadow_color: Color,
) -> None:
    """Draw one shape on an RGBA canvas using a rotated local layer."""

    width_px = max(4, round(shape.width * scale))
    height_px = max(4, round(shape.height * scale))
    outline_px = max(1, round(shape.outline_width * scale))
    blur_radius = max(2, round(0.007 * scale))
    padding = max(10, blur_radius * 4 + outline_px * 2)
    side = max(width_px, height_px) + padding * 2

    shape_layer = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    shadow_layer = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    center = side // 2
    box = (
        center - width_px // 2,
        center - height_px // 2,
        center + width_px // 2,
        center + height_px // 2,
    )

    shadow_draw = ImageDraw.Draw(shadow_layer)
    shadow_offset = max(2, round(0.006 * scale))
    shadow_box = tuple(value + shadow_offset for value in box)
    _draw_geometry(
        shadow_draw,
        kind=shape.kind,
        box=shadow_box,
        fill=_rgba(shadow_color, shape.shadow_opacity),
        outline=None,
        outline_width=0,
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(blur_radius))

    shape_draw = ImageDraw.Draw(shape_layer)
    _draw_geometry(
        shape_draw,
        kind=shape.kind,
        box=box,
        fill=_rgba(shape.fill),
        outline=_rgba(shape.outline),
        outline_width=outline_px,
    )

    combined = Image.alpha_composite(shadow_layer, shape_layer)
    rotated = combined.rotate(
        shape.rotation_degrees,
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )

    center_x = round(shape.center_x * scale)
    center_y = round(shape.center_y * scale)
    paste_x = center_x - rotated.width // 2
    paste_y = center_y - rotated.height // 2
    canvas.alpha_composite(rotated, (paste_x, paste_y))


def draw_center_disk(
    canvas: Image.Image,
    *,
    center_x: int,
    center_y: int,
    radius: int,
    fill: Color,
    outline: Color,
    highlight: Color,
    shadow: Color,
    outline_width: int,
    shadow_opacity: int,
    highlight_strength: int,
) -> None:
    """Draw the dominant center disk with shadow, rim, and soft highlight."""

    shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    shadow_offset = max(3, radius // 24)
    shadow_box = (
        center_x - radius + shadow_offset,
        center_y - radius + shadow_offset,
        center_x + radius + shadow_offset,
        center_y + radius + shadow_offset,
    )
    shadow_draw.ellipse(shadow_box, fill=_rgba(shadow, shadow_opacity))
    shadow_layer = shadow_layer.filter(
        ImageFilter.GaussianBlur(max(3, radius // 16))
    )
    canvas.alpha_composite(shadow_layer)

    draw = ImageDraw.Draw(canvas)
    disk_box = (
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
    )
    draw.ellipse(
        disk_box,
        fill=_rgba(fill),
        outline=_rgba(outline),
        width=outline_width,
    )

    if highlight_strength > 0:
        highlight_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        highlight_draw = ImageDraw.Draw(highlight_layer)
        highlight_radius = int(radius * 0.80)
        highlight_box = (
            center_x - int(radius * 0.58),
            center_y - int(radius * 0.62),
            center_x - int(radius * 0.58) + highlight_radius * 2,
            center_y - int(radius * 0.62) + highlight_radius * 2,
        )
        highlight_draw.ellipse(
            highlight_box,
            fill=_rgba(highlight, highlight_strength),
        )
        highlight_layer = highlight_layer.filter(
            ImageFilter.GaussianBlur(max(5, radius // 7))
        )
        mask = Image.new("L", canvas.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse(disk_box, fill=255)
        highlight_layer.putalpha(Image.composite(highlight_layer.getchannel("A"), Image.new("L", canvas.size, 0), mask))
        canvas.alpha_composite(highlight_layer)

        # Redraw the rim after the highlight so it stays crisp.
        draw = ImageDraw.Draw(canvas)
        draw.ellipse(disk_box, outline=_rgba(outline), width=outline_width)


def draw_panel(
    canvas: Image.Image,
    *,
    center_x: int,
    center_y: int,
    width: int,
    height: int,
    rotation_degrees: float,
    color: Color,
    corner_radius: int,
    shadow_color: Color,
    shadow_opacity: int,
) -> None:
    """Draw one clean rotated background panel with a subtle shadow."""

    padding = max(20, round(max(width, height) * 0.10))
    side = max(width, height) + padding * 2
    layer = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    shadow_layer = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    center = side // 2
    box = (
        center - width // 2,
        center - height // 2,
        center + width // 2,
        center + height // 2,
    )

    shadow_draw = ImageDraw.Draw(shadow_layer)
    offset = max(3, round(side * 0.006))
    shadow_box = tuple(value + offset for value in box)
    shadow_draw.rounded_rectangle(
        shadow_box,
        radius=corner_radius,
        fill=_rgba(shadow_color, shadow_opacity),
    )
    shadow_layer = shadow_layer.filter(
        ImageFilter.GaussianBlur(max(4, round(side * 0.01)))
    )

    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle(
        box,
        radius=corner_radius,
        fill=_rgba(color),
    )

    combined = Image.alpha_composite(shadow_layer, layer)
    rotated = combined.rotate(
        rotation_degrees,
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )
    canvas.alpha_composite(
        rotated,
        (center_x - rotated.width // 2, center_y - rotated.height // 2),
    )
