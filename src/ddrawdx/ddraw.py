import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes

from typing import Optional, NamedTuple, Tuple, List

format_channels = {"GRAY": 1, "RGB": 3}

Mesh = List[jnp.ndarray]
Image = jnp.ndarray


class Canvas(NamedTuple):
    """
    A Canvas is a tuple with pixel values in the width*height*channels array 'image'
    and coordinate meshes in 'mesh'. Construct with 'ddrawdx.canvas'
    """

    image: jnp.ndarray
    mesh: List[jnp.ndarray]


def show(c: Canvas) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Simple 'matplotlib.pyplot.imshow' wrapper.
    """
    fig, ax = plt.subplots()
    if c.image.shape[-1] > 1:
        ax.imshow(c.image)
    else:
        ax.imshow(c.image, cmap="gray", vmin=0, vmax=1)
    ax.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    return fig, ax


@jax.jit
def normalize(x: jnp.ndarray) -> jnp.ndarray:
    return x / jnp.linalg.norm(x)


@jax.jit
def rotmat(angle: float) -> jnp.ndarray:
    """
    2D rotation matrix
    """
    s = jnp.sin(angle)
    c = jnp.cos(angle)
    return jnp.array([[c, -s], [s, c]])


def canvas(
    width: int,
    height: Optional[int] = None,
    format: str = "RGB",
    background: Optional[jnp.ndarray] = None,
) -> Canvas:
    """
    Constructs a canvas of dimensions ```width x height```, with coordinates [0,...,1] x [0,...,1]
    and origin in the lower left corner.
    """
    height = height or width
    channels = format_channels[format]
    image = jnp.ones((width, height, channels))
    if background is not None:
        assert channels == len(background)
        image = image * jnp.expand_dims(background, axis=[0, 1])
    mesh = jnp.meshgrid(jnp.linspace(0, 1, width), jnp.linspace(1, 0, height))
    return Canvas(image=image, mesh=mesh)


@jax.jit
def origin(c: Canvas) -> Tuple[Canvas, Mesh]:
    """
    Translates the origin of 'c' to the center, and rescales the mesh to [-1,...,1]x[-1,...,1].
    Returns the new Canvas and the old mesh for ```restore```
    """
    w, h, _ = c.image.shape
    mesh = jnp.meshgrid(jnp.linspace(-1, 1, w), jnp.linspace(1, -1, h))
    return Canvas(c.image, mesh), c.mesh


@jax.jit
def scale(c: Canvas, xscale: float, yscale: float) -> Tuple[Canvas, Mesh]:
    """Scale mesh, returning the new Canvas and old mesh for ```restore```"""
    mesh = c.mesh
    return Canvas(image=c.image, mesh=[mesh[0] / xscale, mesh[1] / yscale]), mesh


@jax.jit
def translate(c: Canvas, dx: float, dy: float) -> Tuple[Canvas, Mesh]:
    """Translate mesh 'dx','dy' units"""
    mesh = c.mesh
    return Canvas(image=c.image, mesh=[mesh[0] - dx, mesh[1] - dy]), mesh


@jax.jit
def rotate(c: Canvas, angle: float) -> Tuple[Canvas, Mesh]:
    """Rotate mesh 'angle' radians"""
    m = jnp.stack(c.mesh, axis=-1)
    m = jnp.reshape(m, (-1, 2))
    rm = rotmat(angle)
    m = jnp.dot(rm, jnp.transpose(m))
    m = jnp.reshape(jnp.transpose(m), (c.mesh[0].shape[0], -1, 2))
    return Canvas(c.image, [m[:, :, 0], m[:, :, 1]]), c.mesh


@jax.jit
def restore(c: Canvas, mesh: Mesh) -> Canvas:
    """Restores coordinates to earlier mesh"""
    return Canvas(c.image, mesh)


@jax.jit
def _bump_1d(x, x0, x1, sharpness: float = 100.0):
    return jax.nn.sigmoid(sharpness * (x - x0)) * jax.nn.sigmoid(-sharpness * (x - x1))


@jax.jit
def _orth_bump(x, y, x0, y0, x1, y1, sharpness: float = 100.0):
    return _bump_1d(x, x0, x1, sharpness) * _bump_1d(y, y0, y1, sharpness)


@jax.jit
def _m_orth_bump(mesh, x0, y0, x1, y1, sharpness: float = 100.0):
    return _orth_bump(mesh[0], mesh[1], x0, y0, x1, y1, sharpness)


@jax.jit
def _fill(image: jnp.ndarray, alpha: jnp.ndarray, color: jnp.ndarray):
    add = jnp.tensordot(alpha, color, axes=0)
    keep = (1 - jnp.expand_dims(alpha, 2)) * image
    return add + keep


@jax.jit
def _rot90(v: jnp.ndarray):
    return jnp.array([-v[1], v[0]])


@jax.jit
def fill_rect(
    c: Canvas,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: jnp.ndarray,
    sharpness: float = 100.0,
) -> Canvas:
    """Fill axis-parallell rectangle with corners in (x0,y0), (x1,y1)."""
    alpha = _m_orth_bump(c.mesh, x0, y0, x1, y1, sharpness)
    return Canvas(_fill(c.image, alpha, color), c.mesh)


@jax.jit
def fill_poly(
    c: Canvas, ps: jnp.array, color=jnp.array([0.0, 0.0, 0.0]), sharpness: float = 300.0
) -> Canvas:
    """Fill polygon with clockwise oriented corners in 'ps'"""
    msh = jnp.stack(c.mesh, axis=-1)
    alpha = jnp.ones(msh.shape[:-1])
    nps = ps.shape[0]
    for i in range(nps):
        v = normalize(ps[(i + 1) % nps] - ps[i])
        n = _rot90(v)
        d = jnp.dot(n, ps[i])
        act = jnp.dot(msh, n) - d
        alpha = alpha * jax.nn.sigmoid(-sharpness * act)
    return Canvas(_fill(c.image, alpha, color), c.mesh)


@jax.jit
def draw_line(
    c: Canvas,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    lineweight: float = 0.01,
    color: jnp.ndarray = jnp.array([0.0, 0.0, 0.0]),
    sharpness: float = 400.0,
) -> Canvas:
    """Draw a line between (x0,y0) and (x1,y1)"""
    p0 = jnp.array([x0, y0])
    p1 = jnp.array([x1, y1])
    v = normalize(p1 - p0) * lineweight
    n = _rot90(v)
    ps = jnp.array([p0 - v + n, p1 + v + n, p1 + v - n, p0 - v - n])
    return fill_poly(c, ps, sharpness, color)


@jax.jit
def fill_circle(
    c: Canvas,
    cx: float,
    cy: float,
    r: float,
    color: jnp.ndarray,
    sharpness: float = 400.0,
):
    sqdist = (c.mesh[0] - cx) ** 2 + (c.mesh[1] - cy) ** 2
    dist = jnp.sqrt(sqdist)
    alpha = jax.nn.sigmoid(sharpness * (r - dist))
    return Canvas(_fill(c.image, alpha, color), c.mesh)
