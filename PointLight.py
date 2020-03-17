from Light import Light
from Point3D import Point3D
from Vector import Vector

class PointLight(Light):
    """ A light that has a position in space but no direction. """
    def __init__(self, color:Vector, strength:float, position:Point3D):
        Light.__init__(self, color, strength)
        self.position = position