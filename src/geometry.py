"""Geometric shape classes: Circle, Rectangle, and Polygon.

Create three classes (Circle, Rectangle, Polygon) that satisfy all of the
geometric requirements needed by a venue.

Classes
-------
Shape(ABC)
    The abstract base class for all Shapes.

Circle(Shape)
    A circle.

Rectangle(Shape)
    An axis-aligned rectangle.

Polygon(Shape)
    A single, non-overlapping, but not necessarily convex or monotone, polygon.

See Also
--------
akeyaa.venues

"""
from abc import ABC, abstractmethod
import numpy as np
from matplotlib.path import Path


class Shape(ABC):
    """The abstract base class for all shapes.

    Methods
    -------
    boundary(self): -> ndarray, shape=(n, 2), dtype= float
        Return the boundary vertices -- domain to the left, and last vertex
        repeats the first vertex.

    extent(self) -> [float, float, float, float]
        Return [xmin, xmax, ymin, ymax] of the bounding axis-aligned rectangle.

    circumcircle(self) -> ((float, float), float)
        return ((x, y), radius) of the bounding circumcircle.

    centroid(self) -> ndarray, shape=(2,), dtype=float
        Return the centroid as a point.

    area(self) -> float
        Return the area [m^2].

    perimeter(self) -> float
        Return the perimeter [m].

    contains_point(self, point) -> bool
        Return True if the Shape contains the point and False otherwise.

    contains_points(self, points) -> bool
        Returns a bool array which is True if the Shape contains the point.

    """

    @abstractmethod
    def __eq__(self, other):
        """Return True if the two Shapes are equal."""
        raise NotImplementedError

    @abstractmethod
    def boundary(self):
        """Return the boundary vertices (domain to the left)."""
        raise NotImplementedError

    @abstractmethod
    def extent(self):
        """Return [xmin, xmax, ymin, ymax] of the bounding axis-aligned rectangle."""
        raise NotImplementedError

    @abstractmethod
    def circumcircle(self):
        """Return (xycenter, radius) of the bounding circumcircle."""
        raise NotImplementedError

    @abstractmethod
    def centroid(self):
        """Return the centroid as a point."""
        raise NotImplementedError

    @abstractmethod
    def area(self):
        """Return the area [m^2]."""
        raise NotImplementedError

    @abstractmethod
    def perimeter(self):
        """Return the length of the perimeter [m]."""
        raise NotImplementedError

    @abstractmethod
    def contains_point(self, point):
        """Return True if the Shape contains the point."""
        raise NotImplementedError

    @abstractmethod
    def contains_points(self, points):
        """Returns a bool array which is True if the Shape contains the point."""
        raise NotImplementedError


class Circle(Shape):
    """A circle.

    Attributes
    ----------
    center : ndarray, shape=(2,), dtype=float
        The [x, y] coordinates of the center point [m].

    radius : float
        The radius of the circle [m].

    """

    def __init__(self, xycenter, radius):
        self.center = np.array([xycenter[0], xycenter[1]], dtype=float)
        self.radius = float(radius)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"center = {self.center!r}, "
            f"radius = {self.radius!r})"
        )

    def __str__(self):
        return (
            f"Circle: " f"[{self.center[0]}, " f"{self.center[1]}], " f"{self.radius}"
        )

    def __eq__(self, other):
        """Return True if the two Circle are equal."""
        return (
            (self.__class__ == other.__class__)
            and (np.all(self.center == other.center))
            and (self.radius == other.radius)
        )

    def boundary(self):
        """Return the boundary vertices (domain to the left)."""
        unit_vertices = np.array(
            [
                (np.cos(theta), np.sin(theta))
                for theta in np.linspace(0, 2 * np.pi, 100)
            ],
            dtype=float,
        )
        return unit_vertices * self.radius + self.center

    def extent(self):
        """Return [xmin, xmax, ymin, ymax] of the bounding axis-aligned rectangle."""
        return [
            self.center[0] - self.radius,
            self.center[0] + self.radius,
            self.center[1] - self.radius,
            self.center[1] + self.radius,
        ]

    def circumcircle(self):
        """Return (xycenter, radius) of the bounding circumcircle."""
        return (self.center, self.radius)

    def centroid(self):
        """Return the centroid as a point (x, y)."""
        return self.center

    def area(self):
        """Return the area [m^2]."""
        return np.pi * self.radius ** 2

    def perimeter(self):
        """Return the length of the perimeter [m]."""
        return 2 * np.pi * self.radius

    def contains_point(self, point):
        """Return True if the Circle contains the point."""
        return (
            np.hypot(point[0] - self.center[0], point[1] - self.center[1]) < self.radius
        )

    def contains_points(self, points):
        """Returns a bool array which is True if the Circle contains the point."""
        return [np.hypot(point[0] - self.center[0], point[1] - self.center[1]) < self.radius
                for point in points]


class Rectangle(Shape):
    """An axis-aligned rectangle.

    Attributes
    ----------
    xmin : float
        The x coordinate of the left [m].

    xmax : float
        The x coordinate of the right [m].

    ymin : float
        The y coordinate of the bottom [m].

    ymax : float
        The y coordinate of the top [m].

    """

    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"xmin = {self.xmin!r}, "
            f"xmax = {self.xmax!r}, "
            f"ymin = {self.ymin!r}, "
            f"ymax = {self.ymax!r})"
        )

    def __str__(self):
        return f"Rectangle: " f"[{self.xmin}, {self.xmax}, {self.ymin}, {self.ymax}]"

    def __eq__(self, other):
        """Return True if the two Rectangles are equal."""
        return (
            (self.__class__ == other.__class__)
            and (self.xmin == other.xmin)
            and (self.xmax == other.xmax)
            and (self.ymin == other.ymin)
            and (self.ymax == other.ymax)
        )

    def boundary(self):
        """Return the boundary vertices (domain to the left)."""
        return np.array(
            [
                [self.xmin, self.ymin],
                [self.xmax, self.ymin],
                [self.xmax, self.ymax],
                [self.xmin, self.ymax],
                [self.xmin, self.ymin],
            ],
            dtype=float,
        )

    def extent(self):
        """Return [xmin, xmax, ymin, ymax] of the bounding axis-aligned rectangle."""
        return [self.xmin, self.xmax, self.ymin, self.ymax]

    def circumcircle(self):
        """Return (xycenter, radius) of the bounding circumcircle."""
        xcenter = (self.xmin + self.xmax) / 2
        ycenter = (self.ymin + self.ymax) / 2
        radius = np.hypot(self.xmax - self.xmin, self.ymax - self.ymin) / 2
        return (np.array([xcenter, ycenter], dtype=float), radius)

    def centroid(self):
        """Return the centroid as a point (x, y)."""
        return np.array(
            [(self.xmax + self.xmin) / 2, (self.ymax + self.ymin) / 2], dtype=float
        )

    def area(self):
        """Return the area [m^2]."""
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def perimeter(self):
        """Return the length of the perimeter [m]."""
        return 2 * (self.xmax - self.xmin) + 2 * (self.ymax - self.ymin)

    def contains_point(self, point):
        """Return True if the Rectangle contains the point."""
        return (self.xmin < point[0] < self.xmax) and (self.ymin < point[1] < self.ymax)

    def contains_points(self, points):
        """Returns a bool array which is True if the rectangle contains the point."""
        return [(self.xmin < point[0] < self.xmax) and (self.ymin < point[1] < self.ymax)
                for point in points]


class Polygon(Shape):
    """A single, non-overlapping, but not necessarily convex, polygon.

    Attributes
    ----------
    vertices : ndarray, shape=(n, 2), dtype=float
        An array of vertices; i.e. a 2D numpy array of (x, y) corredinates [m].
        The vertices are stored so that the domain is on the left, and the
        first vertex is repeated as the last vertex.

    """

    def __init__(self, vertices):
        self.vertices = np.array(vertices, dtype=float)
        if self.area() < 0:
            self.vertices = np.flipud(self.vertices)

        self.xmin, self.ymin = np.min(self.vertices, axis=0)
        self.xmax, self.ymax = np.max(self.vertices, axis=0)

    def __repr__(self):
        return f"{self.__class__.__name__}(vertices = {self.vertices})"

    def __str__(self):
        return "Polygon: [...]"

    def __eq__(self, other):
        """Return True if the two Polygons are equal."""
        return (self.__class__ == other.__class__) and np.all(
            self.vertices == other.vertices
        )

    def boundary(self):
        """Return the boundary vertices (domain to the left)."""
        return self.vertices

    def extent(self):
        """Return [xmin, xmax, ymin, ymax] of the bounding axis-aligned rectangle."""
        return [self.xmin, self.xmax, self.ymin, self.ymax]

    def circumcircle(self):
        """Return (xycenter, radius) of the bounding circumcircle."""
        xmin, xmax, ymin, ymax = self.extent()
        xcenter = (xmin + xmax) / 2
        ycenter = (ymin + ymax) / 2
        radius = np.hypot(xmax - xmin, ymax - ymin) / 2
        return (np.array([xcenter, ycenter], dtype=float), radius)

    def centroid(self):
        """Return the centroid as a point."""
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]

        sumx = 0.0
        sumy = 0.0

        for i in range(len(self.vertices) - 1):
            sumx += (x[i] + x[i + 1]) * (x[i] * y[i + 1] - x[i + 1] * y[i])
            sumy += (y[i] + y[i + 1]) * (x[i] * y[i + 1] - x[i + 1] * y[i])

        area = self.area()
        sumx /= 6 * area
        sumy /= 6 * area
        return np.array([sumx, sumy], dtype=float)

    def area(self):
        """Return the area [m^2]."""
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]

        area = 0.0
        for i in range(len(self.vertices) - 1):
            area += x[i] * y[i + 1] - x[i + 1] * y[i]
        area /= 2
        return area

    def perimeter(self):
        """Return the length of the perimeter [m]."""
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]

        length = 0.0
        for i in range(len(self.vertices) - 1):
            length += np.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
        return length

    def contains_point(self, point):
        """Return True if the Polygon contains the point."""
        path = Path(self.vertices)
        return path.contains_point(point)

    def contains_points(self, points):
        """Returns a bool array which is True if the rectangle contains the point."""
        path = Path(self.vertices)
        return path.contains_points(points)
