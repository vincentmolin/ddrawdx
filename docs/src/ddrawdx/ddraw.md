# Ddraw

[Ddrawdx Index](../../README.md#ddrawdx-index) /
`src` /
[Ddrawdx](./index.md#ddrawdx) /
Ddraw

> Auto-generated documentation for [src.ddrawdx.ddraw](../../../src/ddrawdx/ddraw.py) module.

- [Ddraw](#ddraw)
  - [Canvas](#canvas)
  - [canvas](#canvas)
  - [draw_line](#draw_line)
  - [fill_circle](#fill_circle)
  - [fill_poly](#fill_poly)
  - [fill_rect](#fill_rect)
  - [normalize](#normalize)
  - [origin](#origin)
  - [restore](#restore)
  - [rotate](#rotate)
  - [rotmat](#rotmat)
  - [scale](#scale)
  - [show](#show)
  - [translate](#translate)

## Canvas

[Show source in ddraw.py:16](../../../src/ddrawdx/ddraw.py#L16)

A Canvas is a tuple with pixel values in the width*height*channels array 'image'
and coordinate meshes in 'mesh'. Construct with 'ddrawdx.canvas'

#### Signature

```python
class Canvas(NamedTuple):
    ...
```



## canvas

[Show source in ddraw.py:56](../../../src/ddrawdx/ddraw.py#L56)

Constructs a canvas of dimensions ```width x height```, with coordinates [0,...,1] x [0,...,1]
and origin in the lower left corner.

#### Signature

```python
def canvas(
    width: int,
    height: Optional[int] = None,
    format: str = "RGB",
    background: Optional[jnp.ndarray] = None,
) -> Canvas:
    ...
```

#### See also

- [Canvas](#canvas)



## draw_line

[Show source in ddraw.py:177](../../../src/ddrawdx/ddraw.py#L177)

Draw a line between (x0,y0) and (x1,y1)

#### Signature

```python
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
    ...
```

#### See also

- [Canvas](#canvas)



## fill_circle

[Show source in ddraw.py:197](../../../src/ddrawdx/ddraw.py#L197)

#### Signature

```python
@jax.jit
def fill_circle(
    c: Canvas,
    cx: float,
    cy: float,
    r: float,
    color: jnp.ndarray,
    sharpness: float = 400.0,
):
    ...
```

#### See also

- [Canvas](#canvas)



## fill_poly

[Show source in ddraw.py:160](../../../src/ddrawdx/ddraw.py#L160)

Fill polygon with clockwise oriented corners in 'ps'

#### Signature

```python
@jax.jit
def fill_poly(
    c: Canvas, ps: jnp.array, color=jnp.array([0.0, 0.0, 0.0]), sharpness: float = 300.0
) -> Canvas:
    ...
```

#### See also

- [Canvas](#canvas)



## fill_rect

[Show source in ddraw.py:145](../../../src/ddrawdx/ddraw.py#L145)

Fill axis-parallell rectangle with corners in (x0,y0), (x1,y1).

#### Signature

```python
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
    ...
```

#### See also

- [Canvas](#canvas)



## normalize

[Show source in ddraw.py:41](../../../src/ddrawdx/ddraw.py#L41)

#### Signature

```python
@jax.jit
def normalize(x: jnp.ndarray) -> jnp.ndarray:
    ...
```



## origin

[Show source in ddraw.py:76](../../../src/ddrawdx/ddraw.py#L76)

Translates the origin of 'c' to the center, and rescales the mesh to [-1,...,1]x[-1,...,1].
Returns the new Canvas and the old mesh for [restore](#restore)

#### Signature

```python
@jax.jit
def origin(c: Canvas) -> Tuple[Canvas, Mesh]:
    ...
```

#### See also

- [Canvas](#canvas)
- [Mesh](#mesh)



## restore

[Show source in ddraw.py:112](../../../src/ddrawdx/ddraw.py#L112)

Restores coordinates to earlier mesh

#### Signature

```python
@jax.jit
def restore(c: Canvas, mesh: Mesh) -> Canvas:
    ...
```

#### See also

- [Canvas](#canvas)
- [Mesh](#mesh)



## rotate

[Show source in ddraw.py:101](../../../src/ddrawdx/ddraw.py#L101)

Rotate mesh 'angle' radians

#### Signature

```python
@jax.jit
def rotate(c: Canvas, angle: float) -> Tuple[Canvas, Mesh]:
    ...
```

#### See also

- [Canvas](#canvas)
- [Mesh](#mesh)



## rotmat

[Show source in ddraw.py:46](../../../src/ddrawdx/ddraw.py#L46)

2D rotation matrix

#### Signature

```python
@jax.jit
def rotmat(angle: float) -> jnp.ndarray:
    ...
```



## scale

[Show source in ddraw.py:87](../../../src/ddrawdx/ddraw.py#L87)

Scale mesh, returning the new Canvas and old mesh for [restore](#restore)

#### Signature

```python
@jax.jit
def scale(c: Canvas, xscale: float, yscale: float) -> Tuple[Canvas, Mesh]:
    ...
```

#### See also

- [Canvas](#canvas)
- [Mesh](#mesh)



## show

[Show source in ddraw.py:26](../../../src/ddrawdx/ddraw.py#L26)

Simple 'matplotlib.pyplot.imshow' wrapper.

#### Signature

```python
def show(c: Canvas) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    ...
```

#### See also

- [Canvas](#canvas)



## translate

[Show source in ddraw.py:94](../../../src/ddrawdx/ddraw.py#L94)

Translate mesh 'dx','dy' units

#### Signature

```python
@jax.jit
def translate(c: Canvas, dx: float, dy: float) -> Tuple[Canvas, Mesh]:
    ...
```

#### See also

- [Canvas](#canvas)
- [Mesh](#mesh)