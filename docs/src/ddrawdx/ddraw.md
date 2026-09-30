# Drawing API

Drawing and mesh transformations return new canvases and can be composed with
`jax.jit` and `jax.grad`. No library function applies JIT internally. When a
drawing sequence constructs a canvas, its dimensions and format must be known
at compile time. `show` creates matplotlib objects and runs outside JIT.

Images have shape `(height, width, channels)`; both meshes have shape
`(height, width)`. The default coordinates span `[0, 1]` in each direction,
with the origin at the lower left. Scalar colors broadcast to all channels;
vectors must contain exactly one value per channel.

Transitions use sigmoids with sharpness measured in mesh coordinates. Distances
at circle centers and polygon vertices are regularized at floating-point
precision to keep gradients finite. Polygon edge interiors use signed
perpendicular distances to preserve gradients at pixel-aligned boundaries.
Polygons use the even-odd interior rule and support concave shapes and either
vertex order.

Polygon gradients are piecewise where the nearest edge changes. Empty and full
arc sweeps also have branch boundaries. Apply `jax.grad` to parameters away from
these boundaries when a smooth parameter gradient is required.

## Canvas data

Pixel values in a (height, width, channels) image and two coordinate meshes.

```python
class Canvas(NamedTuple):
    image: Image
    mesh: Mesh
```

## show

```python
def show(c: Canvas) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
```

Display a canvas with matplotlib, returning the figure and axes.

## normalize

```python
def normalize(x: jnp.ndarray) -> jnp.ndarray
```

Normalize a vector, with a finite result and derivative at zero.

## rotmat

```python
def rotmat(angle: float) -> jnp.ndarray
```

Return the 2D rotation matrix for an angle in radians.

## canvas

```python
def canvas(width: int, height: Optional[int]=None, format: str='RGB', background: Optional[jnp.ndarray]=None) -> Canvas
```

Construct a white canvas over [0, 1] x [0, 1], with origin at lower left.

The image has shape (height, width, channels). Backgrounds and drawing colors
may be scalars or vectors with one value per channel.

## origin

```python
def origin(c: Canvas) -> Tuple[Canvas, Mesh]
```

Reset the mesh to [-1, 1] x [-1, 1]; return the canvas and previous mesh.

## scale

```python
def scale(c: Canvas, xscale: float, yscale: float) -> Tuple[Canvas, Mesh]
```

Scale the mesh, returning the canvas and previous mesh for restore.

## translate

```python
def translate(c: Canvas, dx: float, dy: float) -> Tuple[Canvas, Mesh]
```

Translate the mesh by (dx, dy); return the canvas and previous mesh.

## rotate

```python
def rotate(c: Canvas, angle: float) -> Tuple[Canvas, Mesh]
```

Rotate the mesh by angle radians; return the canvas and previous mesh.

## restore

```python
def restore(c: Canvas, mesh: Mesh) -> Canvas
```

Restore coordinates to an earlier mesh without changing pixel values.

## fill_rect

```python
def fill_rect(c: Canvas, x0: float, y0: float, x1: float, y1: float, color: jnp.ndarray, sharpness: float=100.0) -> Canvas
```

Fill an axis-aligned rectangle with lower and upper corners (x0,y0), (x1,y1).

## fill_poly

```python
def fill_poly(c: Canvas, ps: jnp.ndarray, color=0.0, sharpness: float=300.0) -> Canvas
```

Fill a convex or concave polygon with vertices in either order.

Use an (n, 2) array with at least three vertices. The even-odd rule determines
the interior; a sigmoid of signed distance to the nearest edge feathers it.

## draw_line

```python
def draw_line(c: Canvas, x0: float, y0: float, x1: float, y1: float, lineweight: float=0.01, color=0.0, sharpness: float=400.0) -> Canvas
```

Draw a line with square caps and half-width lineweight.

## fill_circle

```python
def fill_circle(c: Canvas, cx: float, cy: float, r: float, color: jnp.ndarray, sharpness: float=400.0) -> Canvas
```

Fill a circle of radius r centered at (cx, cy).

## draw_circle

```python
def draw_circle(c: Canvas, cx: float, cy: float, r: float, lineweight: float=0.01, color=0.0, sharpness: float=400.0) -> Canvas
```

Draw a circle of radius r with half-width lineweight.

## draw_arc

```python
def draw_arc(c: Canvas, cx: float, cy: float, r: float, a0: float, a1: float, lineweight: float=0.01, color=0.0, sharpness: float=400.0) -> Canvas
```

Draw a clockwise arc from a0 to a1 in radians, wrapping modulo 2*pi.

Equal angles draw nothing; an explicit whole turn draws a full circle.

## fill_arc

```python
def fill_arc(c: Canvas, cx: float, cy: float, r: float, a0: float, a1: float, color=0.0, sharpness: float=400.0) -> Canvas
```

Fill the convex hull of a clockwise arc, bounded by its chord and circle.

Angles wrap modulo 2*pi. Equal angles fill nothing; an explicit whole turn
fills the full disk.
