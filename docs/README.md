# ddrawdx

A small library of differentiable drawing primitives in JAX.

Construct a canvas, draw shapes on its coordinate mesh, and optionally compile
your drawing sequence with `jax.jit`. Library functions do not apply JIT
themselves. Mesh transformations return the changed canvas and the previous mesh
so that coordinates can be restored after drawing.

- [Package exports](src/ddrawdx/index.md)
- [Drawing API](src/ddrawdx/ddraw.md)

```python
import ddrawdx as drx
import jax

def draw(c, radius):
    return drx.fill_circle(c, 0.5, 0.5, radius, drx.YELLOW)

c = jax.jit(draw)(drx.canvas(320, 180), 0.2)
fig, ax = drx.show(c)
```
