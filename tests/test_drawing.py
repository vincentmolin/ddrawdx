import ddrawdx as drx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest


def point_canvas(points, channels=3):
    points = jnp.asarray(points, dtype=jnp.float32)
    return drx.Canvas(
        jnp.ones((1, len(points), channels)),
        [points[None, :, 0], points[None, :, 1]],
    )


@pytest.mark.parametrize("width,height", [(17, 9), (9, 17)])
def test_rectangular_canvas_and_origin(execute, width, height):
    c = drx.canvas(width, height)
    assert c.image.shape == (height, width, 3)
    assert all(m.shape == (height, width) for m in c.mesh)
    np.testing.assert_allclose([c.mesh[0][0, 0], c.mesh[1][0, 0]], [0, 1])
    np.testing.assert_allclose([c.mesh[0][-1, -1], c.mesh[1][-1, -1]], [1, 0])

    filled = execute(drx.fill_rect)(c, 0.2, 0.2, 0.8, 0.8, drx.YELLOW)
    np.testing.assert_allclose(
        filled.image[height // 2, width // 2], drx.YELLOW, atol=1e-5
    )
    np.testing.assert_allclose(filled.image[0, 0], 1, atol=1e-5)

    centered, old_mesh = execute(drx.origin)(c)
    assert all(m.shape == (height, width) for m in centered.mesh)
    np.testing.assert_allclose(
        [centered.mesh[0][0, 0], centered.mesh[1][0, 0]], [-1, 1]
    )
    drawn = execute(drx.fill_circle)(centered, 0.0, 0.0, 0.5, drx.BLACK)
    np.testing.assert_allclose(drawn.image[height // 2, width // 2], 0, atol=1e-5)
    restored = execute(drx.restore)(drawn, old_mesh)
    np.testing.assert_array_equal(restored.image, drawn.image)
    for actual, expected in zip(restored.mesh, c.mesh):
        np.testing.assert_array_equal(actual, expected)


def test_composed_transforms_on_rectangular_mesh(execute):
    def transform(c):
        c, _ = drx.translate(c, 0.5, 0.5)
        c, _ = drx.scale(c, 0.5, 0.5)
        c, _ = drx.rotate(c, jnp.pi / 2)
        return drx.fill_rect(c, -0.2, -0.2, 0.2, 0.2, drx.BLACK)

    c = execute(transform)(drx.canvas(17, 9))
    assert c.image.shape == (9, 17, 3)
    np.testing.assert_allclose([c.mesh[0][0, 0], c.mesh[1][0, 0]], [-1, -1], atol=1e-6)
    np.testing.assert_allclose(c.image[4, 8], 0, atol=1e-5)


def test_line_color_sharpness_and_extent(execute):
    c = point_canvas([[0.5, 0.5], [0.5, 0.57], [0.1, 0.5], [0.22, 0.5]])
    color = jnp.array([0.2, 0.4, 0.6])
    draw = execute(
        lambda c, color, sharpness: drx.draw_line(
            c, 0.2, 0.5, 0.8, 0.5, lineweight=0.05, color=color, sharpness=sharpness
        )
    )
    sharp = draw(c, color, 1000.0)
    soft = draw(c, color, 100.0)
    np.testing.assert_allclose(
        sharp.image[0, [0, 3]], np.broadcast_to(color, (2, 3)), atol=1e-5
    )
    np.testing.assert_allclose(sharp.image[0, [1, 2]], 1, atol=1e-5)
    assert soft.image[0, 1, 0] < sharp.image[0, 1, 0] - 0.05
    default = execute(drx.draw_line)(c, 0.2, 0.5, 0.8, 0.5, 0.05)
    np.testing.assert_allclose(default.image[0, 0], 0, atol=1e-5)


CONCAVE = jnp.array(
    [[0.1, 0.1], [0.1, 0.9], [0.4, 0.9], [0.4, 0.4], [0.9, 0.4], [0.9, 0.1]]
)


@pytest.mark.parametrize("reverse", [False, True])
def test_concave_polygon_in_both_vertex_orders(execute, reverse):
    ps = CONCAVE[::-1] if reverse else CONCAVE
    c = point_canvas([[0.2, 0.7], [0.7, 0.2], [0.2, 0.2], [0.7, 0.7], [0.05, 0.5]])
    drawn = execute(drx.fill_poly)(c, ps)
    np.testing.assert_allclose(drawn.image[0, :3], 0, atol=1e-5)
    np.testing.assert_allclose(drawn.image[0, 3:], 1, atol=1e-5)


def test_polygon_edge_feathering_and_repeated_endpoint(execute):
    ps = jnp.array([[0.2, 0.2], [0.2, 0.8], [0.8, 0.8], [0.8, 0.2], [0.2, 0.2]])
    c = point_canvas([[0.5, 0.5], [0.2, 0.5], [0.1, 0.5], [0.2, 0.2]])
    drawn = execute(drx.fill_poly)(c, ps)
    np.testing.assert_allclose(drawn.image[0, :, 0], [0, 0.5, 1, 0.5], atol=1e-5)
    grad = execute(
        jax.grad(lambda vertices: drx.fill_poly(c, vertices).image.sum())
    )(ps)
    assert np.isfinite(grad).all()


def test_polygon_vertex_gradient_matches_finite_difference(execute):
    c = point_canvas([[0.3, 0.4], [0.65, 0.5], [0.85, 0.3]])
    ps = jnp.array([[0.2, 0.2], [0.25, 0.8], [0.8, 0.3]])

    def loss(dx):
        vertices = ps.at[0, 0].add(dx)
        return drx.fill_poly(c, vertices, sharpness=20.0).image.sum()

    gradient = execute(jax.grad(loss))(0.0)
    step = 0.001
    finite_difference = (loss(step) - loss(-step)) / (2 * step)
    assert abs(gradient) > 0.1
    np.testing.assert_allclose(gradient, finite_difference, rtol=0.005, atol=0.005)


CIRCULAR_PRIMITIVES = [
    pytest.param(
        lambda c, q: drx.fill_circle(c, q[0], q[1], 0.23, 0.0, sharpness=60.0),
        id="fill-circle",
    ),
    pytest.param(
        lambda c, q: drx.draw_circle(c, q[0], q[1], 0.23, sharpness=60.0),
        id="draw-circle",
    ),
    pytest.param(
        lambda c, q: drx.draw_arc(c, q[0], q[1], 0.23, 0.2, -1.2, sharpness=60.0),
        id="draw-arc",
    ),
    pytest.param(
        lambda c, q: drx.fill_arc(c, q[0], q[1], 0.23, 0.2, -1.2, sharpness=60.0),
        id="fill-arc",
    ),
]


@pytest.mark.parametrize("primitive", CIRCULAR_PRIMITIVES)
@pytest.mark.parametrize("center", [(0.0, 0.0), (0.5, 0.5), (0.51, 0.52)])
def test_circular_center_gradients_are_finite(execute, primitive, center):
    c = drx.canvas(17)
    q = jnp.array(center)
    image = execute(primitive)(c, q).image
    grad = execute(jax.grad(lambda q: primitive(c, q).image.sum()))(q)
    assert np.isfinite(image).all()
    assert np.isfinite(grad).all()


def test_circle_gradient_matches_finite_difference(execute):
    c = drx.canvas(17)
    weights = 0.2 + c.mesh[0] + 0.3 * c.mesh[1]

    def loss(q):
        opacity = 1 - drx.fill_circle(
            c, q[0], q[1], q[2], 0.0, sharpness=20.0
        ).image[..., 0]
        return (opacity * weights).sum()

    q = jnp.array([0.51, 0.52, 0.23])
    grad = execute(jax.grad(loss))(q)
    step = 0.001
    offsets = jnp.eye(3) * step
    finite_difference = jnp.array(
        [(loss(q + offset) - loss(q - offset)) / (2 * step) for offset in offsets]
    )
    assert np.all(np.abs(grad) > 0.1)
    np.testing.assert_allclose(grad, finite_difference, rtol=0.003, atol=0.003)


QUADRANTS = [jnp.pi / 4, 3 * jnp.pi / 4, 5 * jnp.pi / 4, 7 * jnp.pi / 4]
ARC_CASES = [
    pytest.param(
        0.0, -jnp.pi / 2, QUADRANTS, [False, False, False, True], id="quarter"
    ),
    pytest.param(
        0.0, jnp.pi / 2, QUADRANTS, [False, True, True, True], id="three-quarters"
    ),
    pytest.param(0.0, -jnp.pi, QUADRANTS, [False, False, True, True], id="semicircle"),
    pytest.param(
        jnp.pi / 4,
        7 * jnp.pi / 4,
        [0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2],
        [True, False, False, False],
        id="wraparound",
    ),
    pytest.param(
        2 * jnp.pi,
        3 * jnp.pi / 2,
        QUADRANTS,
        [False, False, False, True],
        id="shifted-angles",
    ),
    pytest.param(0.0, 0.0, QUADRANTS, [False] * 4, id="empty"),
    pytest.param(0.0, 2 * jnp.pi, QUADRANTS, [True] * 4, id="whole-turn"),
]


@pytest.mark.parametrize("primitive", [drx.draw_arc, drx.fill_arc])
@pytest.mark.parametrize("a0,a1,samples,expected", ARC_CASES)
def test_clockwise_arc_sweeps(execute, primitive, a0, a1, samples, expected):
    radius = 0.3 if primitive is drx.draw_arc else 0.29
    samples = jnp.array(samples)
    c = point_canvas(
        jnp.stack(
            [0.5 + radius * jnp.cos(samples), 0.5 + radius * jnp.sin(samples)],
            axis=-1,
        )
    )
    draw = execute(lambda c, angles: primitive(c, 0.5, 0.5, 0.3, angles[0], angles[1]))
    darkness = np.asarray(1 - draw(c, jnp.array([a0, a1])).image[0, :, 0])
    np.testing.assert_array_equal(darkness > 0.9, expected)
    assert np.all(darkness[~np.array(expected)] < 0.01)


def test_filled_arc_is_bounded_by_chord(execute):
    c = point_canvas([[0.55, 0.45], [0.7, 0.3], [0.7, 0.7]])
    drawn = execute(drx.fill_arc)(c, 0.5, 0.5, 0.3, 0.0, -jnp.pi / 2)
    np.testing.assert_allclose(drawn.image[0, :, 0], [1, 0, 1], atol=0.005)


@pytest.mark.parametrize(
    "arc,circle", [(drx.draw_arc, drx.draw_circle), (drx.fill_arc, drx.fill_circle)]
)
@pytest.mark.parametrize("start,turn", [(0.0, 2), (1.1, -2), (2.3, 2)])
def test_whole_turn_matches_circle(execute, arc, circle, start, turn):
    c = drx.canvas(13, 9)
    kwargs = {"color": 0.2, "sharpness": 40.0}
    actual = execute(
        lambda c, a0, a1: arc(c, 0.5, 0.5, 0.3, a0, a1, **kwargs)
    )(c, start, start + turn * jnp.pi)
    expected = circle(c, 0.5, 0.5, 0.3, **kwargs)
    np.testing.assert_allclose(actual.image, expected.image, atol=1e-6)


@pytest.mark.parametrize("primitive", [drx.draw_arc, drx.fill_arc])
def test_arc_parameter_gradients_are_finite(execute, primitive):
    c = drx.canvas(17)

    def loss(q):
        return primitive(c, q[0], q[1], q[2], q[3], q[4], sharpness=60.0).image.sum()

    q = jnp.array([0.5, 0.5, 0.3, 0.2, -1.2])
    grad = execute(jax.grad(loss))(q)
    assert np.isfinite(grad).all()
    assert np.all(np.abs(grad[2:]) > 0.01)


DEFAULT_DRAWINGS = [
    pytest.param(lambda c: drx.draw_line(c, 0.2, 0.5, 0.8, 0.5), id="line"),
    pytest.param(lambda c: drx.fill_poly(c, CONCAVE), id="polygon"),
    pytest.param(lambda c: drx.draw_circle(c, 0.5, 0.5, 0.25), id="circle"),
    pytest.param(lambda c: drx.draw_arc(c, 0.5, 0.5, 0.25, 0.0, -jnp.pi / 2), id="arc"),
    pytest.param(
        lambda c: drx.fill_arc(c, 0.5, 0.5, 0.3, 0.0, -jnp.pi / 2), id="filled-arc"
    ),
]


@pytest.mark.parametrize("draw", DEFAULT_DRAWINGS)
@pytest.mark.parametrize("format,channels", [("GRAY", 1), ("GREY", 1), ("RGB", 3)])
def test_default_colors_preserve_channels(execute, draw, format, channels):
    c = execute(draw)(drx.canvas(13, 9, format))
    assert c.image.shape == (9, 13, channels)
    assert np.isfinite(c.image).all()
    assert c.image.min() < 0.9


@pytest.mark.parametrize("format,channels", [("GRAY", 1), ("RGB", 3)])
def test_scalar_and_vector_colors(execute, format, channels):
    c = drx.canvas(9, format=format, background=0.8)
    np.testing.assert_allclose(c.image, 0.8)
    scalar = execute(drx.fill_rect)(c, 0.2, 0.2, 0.8, 0.8, 0.3)
    vector = execute(drx.fill_rect)(c, 0.2, 0.2, 0.8, 0.8, jnp.full((channels,), 0.3))
    np.testing.assert_array_equal(scalar.image, vector.image)
    np.testing.assert_allclose(scalar.image[4, 4], 0.3, atol=1e-5)


@pytest.mark.parametrize(
    "format,color",
    [("GRAY", [0.1, 0.2, 0.3]), ("RGB", [0.1, 0.2]), ("RGB", [[0.1, 0.2, 0.3]])],
)
def test_invalid_colors_raise(execute, format, color):
    c = drx.canvas(9, format=format)
    with pytest.raises(ValueError, match="Color must"):
        execute(drx.fill_rect)(c, 0.2, 0.2, 0.8, 0.8, jnp.array(color))
    with pytest.raises(ValueError, match="Color must"):
        drx.canvas(9, format=format, background=jnp.array(color))


@pytest.mark.parametrize("ps", [[], [[0, 0], [1, 1]], [[0, 0, 0]] * 3])
def test_invalid_polygon_shape_raises(execute, ps):
    with pytest.raises(ValueError, match="Polygon vertices"):
        execute(drx.fill_poly)(drx.canvas(5), jnp.array(ps))


@pytest.mark.parametrize("width,height", [(0, None), (5, 0), (-1, 5), (5, -1)])
def test_invalid_canvas_dimensions_raise(width, height):
    with pytest.raises(ValueError, match="dimensions must be positive"):
        drx.canvas(width, height)


def test_invalid_canvas_format_raises():
    with pytest.raises(ValueError, match="Format must"):
        drx.canvas(5, format="RGBA")


def test_zero_vector_and_degenerate_line_are_finite(execute):
    zero = jnp.zeros(2)
    np.testing.assert_array_equal(execute(drx.normalize)(zero), zero)
    assert np.isfinite(execute(jax.jacrev(drx.normalize))(zero)).all()
    c = drx.canvas(9)
    draw = lambda p: drx.draw_line(c, p[0], p[1], p[0], p[1]).image.sum()
    assert np.isfinite(execute(draw)(jnp.array([0.5, 0.5])))
    assert np.isfinite(execute(jax.grad(draw))(jnp.array([0.5, 0.5]))).all()


@pytest.mark.parametrize("format", ["GRAY", "RGB"])
def test_show_displays_rectangular_canvas(format):
    c = drx.canvas(7, 5, format)
    fig, ax = drx.show(c)
    try:
        image = ax.images[0].get_array()
        assert image.shape == ((5, 7) if format == "GRAY" else (5, 7, 3))
    finally:
        plt.close(fig)


def test_package_exports_only_public_api():
    for name in drx.__all__:
        assert hasattr(drx, name)
    for name in (
        "jax", "jnp", "plt", "matplotlib", "List", "NamedTuple", "Optional", "Tuple"
    ):
        assert not hasattr(drx, name)
