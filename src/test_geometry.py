"""Test akeyaa.geometry.py"""
import numpy as np

from geometry import Circle, Rectangle, Polygon


def test_circle():
    """Test geometry.circle"""
    circle_a = Circle((1, 2), 3)
    circle_b = Circle((1, 2), 3)
    circle_c = Circle((2, 1), 2)

    # __eq__(self, other)
    assert circle_a == circle_b
    assert circle_a != circle_c

    # boundary(self):
    bdry = circle_a.boundary()
    for vertex in bdry:
        assert np.isclose(np.linalg.norm(vertex-circle_a.center), circle_a.radius)

    # extent(self):
    assert circle_a.extent() == [-2, 4, -1, 5]

    # circumcircle(self):
    center, radius = circle_a.circumcircle()
    assert np.all(center == (1, 2))
    assert radius == 3

    # centroid(self):
    assert np.all(circle_a.centroid() == (1, 2))

    # area(self):
    assert np.isclose(circle_a.area(), np.pi*9)

    # perimeter(self):
    assert np.isclose(circle_a.perimeter(), 6*np.pi)

    # contains_point(self, point):
    assert circle_a.contains_point((1, 2))
    assert not circle_a.contains_point((10, 20))

    # contains_points(self, points):
    assert np.all(circle_a.contains_points([(1, 2), (2, 1), (1.5, 1.5)]))
    assert np.any(circle_a.contains_points([(1, 2), (10, 20)]))
    assert not np.all(circle_a.contains_points([(1, 2), (10, 20)]))

def test_rectangle():
    """Test geometry.rectangle"""
    rect_a = Rectangle(1, 3, 2, 3)
    rect_b = Rectangle(1, 3, 2, 3)
    rect_c = Rectangle(2, 3, 1, 3)

    # __eq__(self, other):
    assert rect_a == rect_b
    assert rect_a != rect_c

    # boundary(self):
    assert np.all(rect_a.boundary() == [(1, 2), (3, 2), (3, 3), (1, 3), (1, 2)])

    # extent(self):
    assert rect_a.extent() == [1, 3, 2, 3]

    # circumcircle(self):
    center, radius = rect_a.circumcircle()
    assert np.all(center == (2, 2.5))
    assert np.isclose(radius, np.sqrt(1.25))

    # centroid(self):
    assert np.all(rect_a.centroid() == (2, 2.5))

    # area(self):
    assert np.isclose(rect_a.area(), 2)

    # perimeter(self):
    assert np.isclose(rect_a.perimeter(), 6)

    # contains_point(self, point):
    assert rect_a.contains_point((2, 2.5))
    assert not rect_a.contains_point((20, 25))

    # contains_points(self, points):
    assert np.all(rect_a.contains_points([(2, 2.5), (2.5, 2.5), (1.5, 2.5)]))
    assert np.any(rect_a.contains_points([(2, 2.5), (10, 20)]))
    assert not np.all(rect_a.contains_points([(2, 2.5), (10, 20)]))


def test_polygon():
    """Test geometry.polygon"""
    poly_a = Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])
    poly_b = Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])
    poly_c = Polygon([(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)])

    # __eq__(self, other):
    assert poly_a == poly_b
    assert poly_a != poly_c

    # boundary(self):
    assert np.all(poly_a.boundary() == [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])

    # extent(self):
    assert poly_a.extent() == [1, 3, 1, 3]

    # circumcircle(self):
    center, radius = poly_a.circumcircle()
    assert np.all(center == (2, 2))
    assert np.isclose(radius, np.sqrt(2))

    # centroid(self):
    assert np.all(poly_a.centroid() == (2, 2))

    # area(self):
    assert np.isclose(poly_a.area(), 4)

    # perimeter(self):
    assert np.isclose(poly_a.perimeter(), 8)

    # contains_point(self, point):
    assert poly_a.contains_point((2, 2))
    assert not poly_a.contains_point((20, 20))

    # contains_points(self, points):
    assert np.all(poly_a.contains_points([(2, 2.5), (2.5, 2.5), (1.5, 2.5)]))
    assert np.any(poly_a.contains_points([(2, 2.5), (10, 20)]))
    assert not np.all(poly_a.contains_points([(2, 2.5), (10, 20)]))
