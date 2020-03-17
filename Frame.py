import math

class Frame:
    """ The virtual frame that we render to. Maps to the output image. """
    def __init__(self, width:int, height:int):
        self.buffer = [128 for x in range(width*height*3)]
        self.width = width
        self.height = height
        for h in range(height):
            for w in range(width):
                self.buffer[h*width*3 + w*3] = 255
                self.buffer[h*width*3 + w*3+1] = 128
                self.buffer[h*width*3 + w*3+2] = 255

    def set(self, x, y, vector):
        indexR = y * self.width*3 + x * 3 + 0
        indexG = y * self.width*3 + x * 3 + 1
        indexB = y * self.width*3 + x * 3 + 2

        #Make everything integers
        vector.x = math.floor(vector.x)
        vector.y = math.floor(vector.y)
        vector.z = math.floor(vector.z)

        #cap and clamp
        vector.x = max(0, min(255, vector.x))
        vector.y = max(0, min(255, vector.y))
        vector.z = max(0, min(255, vector.z))

        self.buffer[indexR] = vector.x
        self.buffer[indexG] = vector.y
        self.buffer[indexB] = vector.z