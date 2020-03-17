from SceneObject import SceneObject
from Point3D import Point3D
from Material import Material
from Ray import Ray
import math

class Sphere(SceneObject):
    """ A spherical scene object. """
    def __init__(self, material:Material, center:Point3D, radius:float):
        SceneObject.__init__(self, material)
        self.center = center
        self.radius = radius

    def intersect(self, ray:Ray):
        d = ray.direction
        a = d.dot(d)
        e = ray.origin.minus(self.center).toNormalized()
        b = 2 * d.dot(e)
        c = e.dot(e) - self.radius * self.radius
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            raise  "Result is imaginary"
        else:
            optionA = (-b - math.sqrt(discriminant))/(2*a)
            optionB = (-b + math.sqrt(discriminant))/(2*a)
            #What we do depends on whether they are both negative,
            #both positive,
            #or one of each
            if optionA < 0 and optionB < 0:
                return max(optionA, optionB)    #should just be optionB
            if optionA >= 0 and optionB >= 0:
                return min(optionA, optionB)    #should just be optionA
            return optionB                      #only optionB should be positive


