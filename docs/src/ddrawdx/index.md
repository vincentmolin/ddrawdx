# Package exports

Import the public API directly from `ddrawdx`. The [drawing API](ddraw.md)
describes the canvas, drawing primitives, transformations, and display helper.

`Mesh` is a list of two coordinate arrays and `Image` is a JAX array. The color
constants `BLACK`, `WHITE`, `DARKGRAY`, and `YELLOW` contain three RGB values;
use a scalar or a one-element vector when drawing on grayscale canvases.
`format_channels` maps `RGB`, `GRAY`, and `GREY` to their channel counts.
