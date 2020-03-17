from Point3D import Point3D
from Vector import Vector

class Ray:
    """ A line that has a start and a direction. """
    def __init__(self,origin : Point3D, direction : Vector):
        self.origin = origin
        self.direction = direction

    