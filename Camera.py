from Point3D import Point3D
from Vector import Vector

class Camera:
    """ Abstraction of a Camera. Should not be directly instantiated."""
    def __init__(self, origin:Point3D, lookAt:Point3D, up:Vector, fov:float, backgroundColor:Vector):
       self.origin = origin
       self.lookAt = lookAt
       self.up = up
       self.fov = fov
       self.backgroundColor = backgroundColor