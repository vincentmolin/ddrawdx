import ddrawdx as drx
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

reddish = jnp.array([0.8, 0.2, 0.1])
c = drx.canvas(200, 200, "RGB")
c, r = drx.origin(c)

def flower(c):
    for s in jnp.arange(0.0, 0.8, 0.05):
        c, _ = drx.rotate(c, 1.2 * s)
        c = drx.fill_rect(c, -0.9 + s, -0.9 + s, 0.9 - s, 0.9 - s, reddish * (1 -  s), 400 * (0.88 - s))
    return c
flower = jax.jit(flower)
c = flower(c)
fig, ax = drx.show(c)
ax.set_title("Box flower")
fig.savefig("boxflower.png")

c = drx.canvas(200, 200, "RGB")
c = drx.fill_rect(c, 0.2, 0.2, 0.8, 0.6, jnp.array([0.0, 1.0, 0.0]))
ps = jnp.array(
    [[0.1, 0.1], [0.1, 0.5], [0.5, 0.1]]
)
c = drx.fill_poly(c, ps, color=jnp.array([1.0, 0.0, 0.0]))
c, r = drx.translate(c, 0.5, 0.5)
c = drx.draw_line(c, 0.0, 0.0, 0.25, 0.25)
c, _ = drx.rotate(c, jnp.pi/2)
c = drx.draw_line(c, 0.0, 0.0, 0.25, 0.25)
c = drx.restore(c, r)
fig, ax = drx.show(c)
ax.set_title("Clock")
fig.savefig("clock.png")

c = drx.canvas(200, 200, "GRAY")
c = drx.fill_rect(c, 0.3, 0.3, 0.7, 0.4, jnp.array([0.2]))
fig, ax = drx.show(c)
ax.set_title("Grayscale")
fig.savefig("gray.png")