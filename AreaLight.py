from Light import Light
from Point3D import Point3D
from SceneObject import SceneObject
from Vector import Vector

class AreaLight(Light):
    """ A light attached to an object."""
    def __init__(self, color:Vector, strength:float, sceneObject:SceneObject):
        Light.__init__(self, color, strength)
        self.sceneObject = sceneObject