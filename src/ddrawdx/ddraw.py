"""Differentiable drawing primitives; apply JAX transformations at the call site."""

from typing import List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

format_channels = {"GRAY": 1, "GREY": 1, "RGB": 3}

Mesh = List[jnp.ndarray]
Image = jnp.ndarray

DARKGRAY = jnp.array([0.2, 0.2, 0.2])
YELLOW = jnp.array([255, 222, 52], jnp.float32) / 255.0
BLACK = jnp.array([0.0, 0.0, 0.0])
WHITE = jnp.array([1.0, 1.0, 1.0])


class Canvas(NamedTuple):
    """Pixel values in a (height, width, channels) image and two coordinate meshes."""

    image: Image
    mesh: Mesh


def show(c: Canvas) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Display a canvas with matplotlib, returning the figure and axes."""
    fig, ax = plt.subplots()
    if c.image.shape[-1] > 1:
        ax.imshow(c.image)
    else:
        ax.imshow(c.image[..., 0], cmap="gray", vmin=0, vmax=1)
    ax.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    return fig, ax


def normalize(x: jnp.ndarray) -> jnp.ndarray:
    """Normalize a vector, with a finite result and derivative at zero."""
    x = jnp.asarray(x)
    x = x.astype(jnp.result_type(x, jnp.float32))
    eps = jnp.finfo(x.dtype).eps
    return x / jnp.sqrt(jnp.sum(x**2) + eps**2)


def rotmat(angle: float) -> jnp.ndarray:
    """Return the 2D rotation matrix for an angle in radians."""
    s = jnp.sin(angle)
    c = jnp.cos(angle)
    return jnp.array([[c, -s], [s, c]])


def _color(color, channels):
    color = jnp.asarray(color)
    if color.ndim == 0:
        return jnp.broadcast_to(color, (channels,))
    if color.shape != (channels,):
        raise ValueError(f"Color must be a scalar or have shape ({channels},).")
    return color


def canvas(
    width: int,
    height: Optional[int] = None,
    format: str = "RGB",
    background: Optional[jnp.ndarray] = None,
) -> Canvas:
    """Construct a white canvas over [0, 1] x [0, 1], with origin at lower left.

    The image has shape (height, width, channels). Backgrounds and drawing colors
    may be scalars or vectors with one value per channel.
    """
    height = width if height is None else height
    if width <= 0 or height <= 0:
        raise ValueError("Canvas dimensions must be positive.")
    if format not in format_channels:
        raise ValueError("Format must be RGB, GRAY, or GREY.")
    channels = format_channels[format]
    image = jnp.ones((height, width, channels))
    if background is not None:
        image = image * _color(background, channels)
    mesh = jnp.meshgrid(jnp.linspace(0, 1, width), jnp.linspace(1, 0, height))
    return Canvas(image=image, mesh=mesh)


def origin(c: Canvas) -> Tuple[Canvas, Mesh]:
    """Reset the mesh to [-1, 1] x [-1, 1]; return the canvas and previous mesh."""
    h, w, _ = c.image.shape
    mesh = jnp.meshgrid(jnp.linspace(-1, 1, w), jnp.linspace(1, -1, h))
    return Canvas(c.image, mesh), c.mesh


def scale(c: Canvas, xscale: float, yscale: float) -> Tuple[Canvas, Mesh]:
    """Scale the mesh, returning the canvas and previous mesh for restore."""
    mesh = c.mesh
    return Canvas(image=c.image, mesh=[mesh[0] / xscale, mesh[1] / yscale]), mesh


def translate(c: Canvas, dx: float, dy: float) -> Tuple[Canvas, Mesh]:
    """Translate the mesh by (dx, dy); return the canvas and previous mesh."""
    mesh = c.mesh
    return Canvas(image=c.image, mesh=[mesh[0] - dx, mesh[1] - dy]), mesh


def rotate(c: Canvas, angle: float) -> Tuple[Canvas, Mesh]:
    """Rotate the mesh by angle radians; return the canvas and previous mesh."""
    m = jnp.stack(c.mesh, axis=-1) @ rotmat(angle).T
    return Canvas(c.image, [m[..., 0], m[..., 1]]), c.mesh


def restore(c: Canvas, mesh: Mesh) -> Canvas:
    """Restore coordinates to an earlier mesh without changing pixel values."""
    return Canvas(c.image, mesh)


def _bump_1d(x, x0, x1, sharpness: float = 100.0):
    return jax.nn.sigmoid(sharpness * (x - x0)) * jax.nn.sigmoid(-sharpness * (x - x1))


def _fill(image: Image, alpha: jnp.ndarray, color):
    color = _color(color, image.shape[-1])
    alpha = alpha[..., None]
    return alpha * color + (1 - alpha) * image


def _rot90(v: jnp.ndarray):
    return jnp.array([-v[1], v[0]])


def _distance(squared):
    # Smooth the norm at zero, where sqrt otherwise gives NaN gradients. The
    # subtraction keeps the distance zero on an edge or at a circle's center.
    eps = jnp.finfo(squared.dtype).eps
    return jnp.sqrt(squared + eps**2) - eps


def _ray_crossings(x, y, starts, ends):
    """Edges crossed by a ray in the positive x direction."""
    y0, y1 = starts[:, 1, None, None], ends[:, 1, None, None]
    dy = y1 - y0
    intersection_x = starts[:, 0, None, None] + (
        (y - y0) * (ends[:, 0] - starts[:, 0])[:, None, None]
        / jnp.where(dy != 0, dy, 1)
    )
    return ((y0 > y) != (y1 > y)) & (x < intersection_x)


def fill_rect(
    c: Canvas,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: jnp.ndarray,
    sharpness: float = 100.0,
) -> Canvas:
    """Fill an axis-aligned rectangle with lower and upper corners (x0,y0), (x1,y1)."""
    alpha = _bump_1d(c.mesh[0], x0, x1, sharpness) * _bump_1d(
        c.mesh[1], y0, y1, sharpness
    )
    return Canvas(_fill(c.image, alpha, color), c.mesh)


def fill_poly(
    c: Canvas, ps: jnp.ndarray, color=0.0, sharpness: float = 300.0
) -> Canvas:
    """Fill a convex or concave polygon with vertices in either order.

    Use an (n, 2) array with at least three vertices. The even-odd rule determines
    the interior; a sigmoid of signed distance to the nearest edge feathers it.
    """
    ps = jnp.asarray(ps)
    ps = ps.astype(jnp.result_type(ps, jnp.float32))
    if ps.ndim != 2 or ps.shape[1] != 2 or ps.shape[0] < 3:
        raise ValueError("Polygon vertices must have shape (n, 2), with n >= 3.")

    ends = jnp.roll(ps, -1, axis=0)
    edges = ends - ps
    offsets = jnp.stack(c.mesh, axis=-1)[None, ...] - ps[:, None, None, :]
    lengths_sq = jnp.sum(edges**2, axis=-1)
    # A repeated vertex is a zero-length edge; treat it as its endpoint.
    denominators = jnp.where(lengths_sq > 0, lengths_sq, 1)
    projection = jnp.clip(
        jnp.sum(offsets * edges[:, None, None, :], axis=-1)
        / denominators[:, None, None],
        0,
        1,
    )
    nearest = offsets - projection[..., None] * edges[:, None, None, :]
    squared = jnp.sum(nearest**2, axis=-1)
    closest = jnp.argmin(squared, axis=0)
    distance = _distance(jnp.min(squared, axis=0))

    x, y = c.mesh
    crossings = _ray_crossings(x, y, ps, ends)
    inside = jnp.sum(crossings, axis=0) % 2 == 1
    signed_distance = jnp.where(inside, distance, -distance)

    # Along an edge interior, use its signed perpendicular distance directly:
    # taking the norm first loses the derivative when a pixel lies on the edge.
    edge = edges[closest]
    offset = jnp.stack(c.mesh, axis=-1) - ps[closest]
    line_distance = (
        edge[..., 0] * offset[..., 1] - edge[..., 1] * offset[..., 0]
    ) / jnp.sqrt(denominators[closest])
    direction = jnp.where(inside, 1, -1) * jnp.sign(line_distance)

    # On the edge itself, determine the inward normal from the other ray
    # crossings. A vertical ray handles horizontal edges. This uses local
    # parity, so it also works for either winding and self-intersections.
    other_edges = jnp.arange(ps.shape[0])[:, None, None] != closest
    right_inside = jnp.sum(crossings & other_edges, axis=0) % 2 == 1
    vertical = _ray_crossings(y, x, ps[:, ::-1], ends[:, ::-1])
    above_inside = jnp.sum(vertical & other_edges, axis=0) % 2 == 1
    boundary_direction = jnp.where(
        edge[..., 1] != 0,
        jnp.where(right_inside, -1, 1) * jnp.sign(edge[..., 1]),
        jnp.where(above_inside, 1, -1) * jnp.sign(edge[..., 0]),
    )
    direction = jnp.where(line_distance == 0, boundary_direction, direction)
    t = jnp.take_along_axis(projection, closest[None, ...], axis=0)[0]
    signed_distance = jnp.where(
        (t > 0) & (t < 1), direction * line_distance, signed_distance
    )
    alpha = jax.nn.sigmoid(sharpness * signed_distance)
    return Canvas(_fill(c.image, alpha, color), c.mesh)


def draw_line(
    c: Canvas,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    lineweight: float = 0.01,
    color=0.0,
    sharpness: float = 400.0,
) -> Canvas:
    """Draw a line with square caps and half-width lineweight."""
    p0 = jnp.array([x0, y0])
    p1 = jnp.array([x1, y1])
    v = normalize(p1 - p0) * lineweight
    n = _rot90(v)
    ps = jnp.array([p0 - v + n, p1 + v + n, p1 + v - n, p0 - v - n])
    return fill_poly(c, ps, color=color, sharpness=sharpness)


def _circle_alpha(mesh, cx, cy, r, sharpness):
    sqdist = (mesh[0] - cx) ** 2 + (mesh[1] - cy) ** 2
    return jax.nn.sigmoid(sharpness * (r - _distance(sqdist)))


def fill_circle(
    c: Canvas,
    cx: float,
    cy: float,
    r: float,
    color: jnp.ndarray,
    sharpness: float = 400.0,
) -> Canvas:
    """Fill a circle of radius r centered at (cx, cy)."""
    alpha = _circle_alpha(c.mesh, cx, cy, r, sharpness)
    return Canvas(_fill(c.image, alpha, color), c.mesh)


def draw_circle(
    c: Canvas,
    cx: float,
    cy: float,
    r: float,
    lineweight: float = 0.01,
    color=0.0,
    sharpness: float = 400.0,
) -> Canvas:
    """Draw a circle of radius r with half-width lineweight."""
    inner = _circle_alpha(c.mesh, cx, cy, r - lineweight, sharpness)
    outer = _circle_alpha(c.mesh, cx, cy, r + lineweight, sharpness)
    alpha = (1 - inner) * outer
    return Canvas(_fill(c.image, alpha, color), c.mesh)


def _clockwise_sweep(a0, a1):
    delta = jnp.asarray(a0) - jnp.asarray(a1)
    sweep = jnp.mod(delta, 2 * jnp.pi)
    turns = delta / (2 * jnp.pi)
    # Subtracting arbitrary start/end angles can round a whole turn slightly
    # above or below 2*pi. Recognize it within floating-point precision.
    eps = jnp.finfo(sweep.dtype).eps
    whole_turn = (jnp.round(turns) != 0) & (
        jnp.abs(turns - jnp.round(turns))
        <= 4 * eps * jnp.maximum(1, jnp.abs(turns))
    )
    return jnp.where(whole_turn, 2 * jnp.pi, sweep)


def draw_arc(
    c: Canvas,
    cx: float,
    cy: float,
    r: float,
    a0: float,
    a1: float,
    lineweight: float = 0.01,
    color=0.0,
    sharpness: float = 400.0,
) -> Canvas:
    """Draw a clockwise arc from a0 to a1 in radians, wrapping modulo 2*pi.

    Equal angles draw nothing; an explicit whole turn draws a full circle.
    """
    inner = _circle_alpha(c.mesh, cx, cy, r - lineweight, sharpness)
    outer = _circle_alpha(c.mesh, cx, cy, r + lineweight, sharpness)
    sweep = _clockwise_sweep(a0, a1)
    x, y = c.mesh[0] - cx, c.mesh[1] - cy
    start = -jnp.sin(a0) * x + jnp.cos(a0) * y
    end = jnp.sin(a1) * x - jnp.cos(a1) * y
    # Short sweeps intersect the endpoint half-planes; long sweeps unite them.
    distance = jnp.where(
        sweep <= jnp.pi, jnp.maximum(start, end), jnp.minimum(start, end)
    )
    angular_alpha = jax.nn.sigmoid(-sharpness * distance)
    angular_alpha = jnp.where(sweep == 0, 0, angular_alpha)
    angular_alpha = jnp.where(sweep == 2 * jnp.pi, 1, angular_alpha)
    alpha = (1 - inner) * outer * angular_alpha
    return Canvas(_fill(c.image, alpha, color), c.mesh)


def fill_arc(
    c: Canvas,
    cx: float,
    cy: float,
    r: float,
    a0: float,
    a1: float,
    color=0.0,
    sharpness: float = 400.0,
) -> Canvas:
    """Fill the convex hull of a clockwise arc, bounded by its chord and circle.

    Angles wrap modulo 2*pi. Equal angles fill nothing; an explicit whole turn
    fills the full disk.
    """
    circ = _circle_alpha(c.mesh, cx, cy, r, sharpness)
    sweep = _clockwise_sweep(a0, a1)
    midpoint = a0 - sweep / 2
    distance = (
        (c.mesh[0] - cx) * jnp.cos(midpoint)
        + (c.mesh[1] - cy) * jnp.sin(midpoint)
        - r * jnp.cos(sweep / 2)
    )
    chord_alpha = jax.nn.sigmoid(sharpness * distance)
    chord_alpha = jnp.where(sweep == 0, 0, chord_alpha)
    chord_alpha = jnp.where(sweep == 2 * jnp.pi, 1, chord_alpha)
    return Canvas(_fill(c.image, circ * chord_alpha, color), c.mesh)
