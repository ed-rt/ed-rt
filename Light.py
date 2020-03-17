from Vector import Vector

class Light:
    """ Abstraction of a light in the scene. Should not be directly instantiated."""
    def __init__(self, color:Vector, strength:float):
       self.color = color
       self.strength = strength