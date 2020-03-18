from Vector import Vector

class Material:
    """ Definition of the material found on an object. """
    def __init__(self, diffuseColor:Vector, specularColor:Vector, specularStrength:float, reflectivity:float):
        self.diffuseColor = diffuseColor
        self.specularColor = specularColor
        self.specularStrength  = specularStrength
        self.reflectivity = reflectivity