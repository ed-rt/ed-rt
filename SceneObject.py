from Material import Material 

class SceneObject:
    """ Abstraction of an object in the scene. Should not be directly instantiated. """
    def __init__(self, material:Material):
        self.material = material