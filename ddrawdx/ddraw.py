import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt

from typing import Optional, NamedTuple

format_channels = {"GRAY": 1, "RGB": 3}


class Canvas(NamedTuple):
    image: jnp.ndarray
    mesh: jnp.ndarray


def show(c: Canvas):
    if c.image.ndim > 1:
        return plt.imshow(c.image)
    else:
        return plt.imshow(c.image, cmap="gray", vmin=0, vmax=1)


@jax.jit
def normalize(x: jnp.ndarray):
    return x / jnp.linalg.norm(x)


@jax.jit
def rotmat(angle: float):
    s = jnp.sin(angle)
    c = jnp.cos(angle)
    return jnp.array([[c, -s], [s, c]])


def canvas(width: int, height: Optional[int] = None, format: str = "GRAY"):
    height = height or width
    channels = format_channels[format]
    image = jnp.ones((width, height, channels))
    mesh = jnp.meshgrid(jnp.linspace(0, 1, width), jnp.linspace(1, 0, height))
    return Canvas(image=image, mesh=mesh)

@jax.jit
def origin(c: Canvas):
    w, h, _ = c.image.shape
    mesh = jnp.meshgrid(jnp.linspace(-1,1,w), jnp.linspace(1,-1,h))
    return Canvas(c.image, mesh), c.mesh

@jax.jit
def scale(c: Canvas, xs, ys):
    mesh = c.mesh
    return Canvas(image=c.image, mesh=[mesh[0] * xs, mesh[1] * ys]), mesh


@jax.jit
def translate(c: Canvas, dx, dy):
    mesh = c.mesh
    return Canvas(image=c.image, mesh=[mesh[0] - dx, mesh[1] - dy]), mesh


@jax.jit
def rotate(c: Canvas, angle: float):
    m = jnp.stack(c.mesh, axis=-1)
    m = jnp.reshape(m, (-1, 2))
    rm = rotmat(angle)
    m = jnp.dot(rm, jnp.transpose(m))
    m = jnp.reshape(jnp.transpose(m), (c.mesh[0].shape[0], -1, 2))
    return Canvas(c.image, [m[:, :, 0], m[:, :, 1]]), c.mesh


@jax.jit
def restore(c: Canvas, mesh):
    return Canvas(c.image, mesh)


@jax.jit
def _bump_1d(x, x0, x1, sharpness=100.0):
    return jax.nn.sigmoid(sharpness * (x - x0)) * jax.nn.sigmoid(-sharpness * (x - x1))


@jax.jit
def _orth_bump(x, y, x0, y0, x1, y1, sharpness=100.0):
    return _bump_1d(x, x0, x1, sharpness) * _bump_1d(y, y0, y1, sharpness)


@jax.jit
def _m_orth_bump(mesh, x0, y0, x1, y1, sharpness=100.0):
    return _orth_bump(mesh[0], mesh[1], x0, y0, x1, y1, sharpness)


@jax.jit
def _fill(image: jnp.ndarray, alpha: jnp.ndarray, color: jnp.ndarray):
    add = jnp.tensordot(alpha, color, axes=0)
    keep = (1 - jnp.expand_dims(alpha, 2)) * image
    return add + keep


@jax.jit
def rot90(v: jnp.ndarray):
    return jnp.array([-v[1], v[0]])


@jax.jit
def fill_rect(c: Canvas, x0, y0, x1, y1, color: jnp.ndarray, sharpness=100.0):
    alpha = _m_orth_bump(c.mesh, x0, y0, x1, y1, sharpness)
    return Canvas(_fill(c.image, alpha, color), c.mesh)


@jax.jit
def fill_poly(c: Canvas, ps, sharpness=300.0, color=jnp.array([0.0, 0.0, 0.0])):
    msh = jnp.stack(c.mesh, axis=-1)
    alpha = jnp.ones(msh.shape[:-1])
    nps = ps.shape[0]
    for i in range(nps):
        v = normalize(ps[(i + 1) % nps] - ps[i])
        n = rot90(v)
        d = jnp.dot(n, ps[i])
        act = jnp.dot(msh, n) - d
        alpha = alpha * jax.nn.sigmoid(-sharpness * act)
    return Canvas(_fill(c.image, alpha, color), c.mesh)


@jax.jit
def draw_line(
    c: Canvas,
    x0,
    y0,
    x1,
    y1,
    lineweight=0.01,
    color: jnp.ndarray = jnp.array([0.0, 0.0, 0.0]),
    sharpness=400.0,
):
    p0 = jnp.array([x0, y0])
    p1 = jnp.array([x1, y1])
    v = normalize(p1 - p0) * lineweight
    n = rot90(v)
    ps = jnp.array([p0 - v + n, p1 + v + n, p1 + v - n, p0 - v - n])
    return fill_poly(c, ps, sharpness, color)


if __name__ == "__main__":
    col = jnp.array([0.8, 0.2, 0.1])
    c = canvas(200, 200, "RGB")
    c, r = origin(c)
    def flower(c):
        for s in jnp.arange(0.0, 0.82, 0.05):
            c, _ = rotate(c, s)
            c = fill_rect(c, -0.9 + s, -0.9 + s, 0.9 - s, 0.9 - s, col * (1 -  s), 400 * (0.88 - s))
        return c
    flower = jax.jit(flower)
    c = flower(c)
    show(c)
    c = fill_rect(c, 0.2, 0.2, 0.8, 0.6, jnp.array([0.0, 1.0, 0.0]))
    show(c)
    ps = jnp.array(
        [jnp.array([x, y]) for (x, y) in [(0.1, 0.1), (0.1, 0.5), (0.5, 0.1)]]
    )
    c = fill_poly(c, ps, color=jnp.array([1.0, 0.0, 0.0]))
    show(c)
    c, r = translate(c, 0.5,0.5)
    c = draw_line(c, 0.0, 0.0, 0.25, 0.25)
    show(c)
    c, _ = rotate(c, jnp.pi/2)
    c = draw_line(c, 0.0, 0.0, 0.25, 0.25)
    show(c)
