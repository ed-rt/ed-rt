from Point3D import Point3D
from Camera import Camera
from Vector import Vector


class OrthographicCamera(Camera):
    """ A camera that is defined as two parallel planes. """
    def __init__(self, origin:Point3D, lookAt:Point3D, up:Point3D, backgroundColor:Vector, halfWidth:float, halfHeight:float):
        Camera.__init__(self, origin, lookAt, up, backgroundColor)
        self.halfWidth = halfWidth
        self.halfHeight = halfHeight
      