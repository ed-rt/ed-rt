""" 
To run this program do the following:

First, install the pip pacakages by running the following
python -m pip install -r requirements.txt

Then run this main file by running the following
python main.py

Note that this program is built to run in python 3

If you machine defaults to running python 2 when you run python
from your shell/command line, it won't work

You may need to run
python3 main.py to get the program to run


Written by B. Ricks, @bricksphd, 

Professor at unomaha.edu

MIT License, 2020

"""


import png
import math

from Frame import Frame
from Vector import Vector
from Point3D import Point3D
from Ray import Ray
from AreaLight import AreaLight
from Camera import Camera
from DirectionalLight import DirectionalLight
from Light import Light
from Material import Material
from OrthographicCamera import OrthographicCamera
from PerspectiveCamera import PerspectiveCamera
from PointLight import PointLight
from SceneObject import SceneObject
from Sphere import Sphere
from SpotLight import SpotLight





print("Starting our ray tracer")


# Minimum for a ray tracer
# A frame to render to
# A camera
# A light
# An object to render


frame = Frame(256, 256)

cameraOrigin = Point3D(0,0,1)
origin = Point3D(0,0,0)
cameraLookAt = origin
cameraUp = Vector(0,1,0)
cameraBackgroundColor = Vector(0,0,0)
fov = 45 / 360 * math.pi * 2 # convert 45 degrees to radians. Should result in pi/4 ~= .785

camera = Camera(cameraOrigin, cameraLookAt, cameraUp, fov, cameraBackgroundColor)

lightDirection = Vector(0,-1,0)
lightColor = Vector(255,255,255)

light = DirectionalLight(lightColor, 1, lightDirection)

sphereCenter = origin
sphereRadius = .5
sphereMaterialColor = Vector(255, 0, 0)
sphereMaterialSpecularColor = Vector(255,255,255)
sphereMaterialSpecularStrength = 1

sphereMaterial = Material(sphereMaterialColor, sphereMaterialSpecularColor, sphereMaterialSpecularStrength)

sphere = Sphere(sphereMaterial, sphereCenter, sphereRadius)

lights = [light]
objects = [sphere]

#Now loop over every pixel in our frame

#For every pixel
#Figure out where in camera space that pixel is
#Then figure out where in world space that pixel is
#Then shoot a ray from the world space origin of the camera to that world space location
#Then loop over each object in the scene
#For ever object that ray collides with
#Find out which one has the closest collission
#That's our hit
#If we don't have a hit, return the background color
#Then calculate the color based on the direction to the right

for y in range(frame.height):
    for x in range(frame.width):
        #Convert from screen space to camera space
        #Then from frame camera space to world space
        yPercent = -1 * (y / frame.height * 2 - 1) #-1 because images have y down
        xPercent = x / frame.width * 2 -1
        #yPercent and xPercent are now in [-1,1]
        #Now we multiply by the camera width and height at the lookAt point
        #To do that we first get the distance from the camera origin and the camera destination
        #This becomes the hyponetus for our triangle calculations

        toLookAt = camera.lookAt.minus(camera.origin)
        #toLookAt is a vector from the origin to the look at point.
        #We need this to calculate the camera right vector

        distance = toLookAt.length()
        toLookAtNormalized = toLookAt.toNormalized()
        width = math.cos(camera.fov) * distance
        height = math.cos(camera.fov) * distance
        #width and height should be the same unless we set different fovs for width and height
        cameraRight = toLookAt.cross(camera.up)
        rightWorld = cameraRight.toScaled(width * xPercent)
        upWorld =  camera.up.toScaled(height * yPercent)
        pixelLookAt = Point3D.fromVector(upWorld.plus(rightWorld))
        #We now have our world look at points
        #We need to generate our look at ray and NORMALIZE IT!!!
        ray = Ray(camera.origin, pixelLookAt.minus(camera.origin).toNormalized())

        for object in objects:
            try:
                t = object.intersect(ray)
                if t >= 0:
                    collisionPoint = Point3D.fromVector(ray.direction.toScaled(t).plus(camera.origin.vector))
                    normalDirection = collisionPoint.minus(object.center)
                    normal = normalDirection.toNormalized()

                    ambient = Vector(30, 30, 30)
                    diffuse = Vector(0,0,0)

                    for light in lights:
                        lightDiffuse = Vector(0,0,0)
                        toLight = light.direction.toScaled(-1)
                        product = toLight.dot(normal)
                        if product < 0:
                            product = 0
                        lightDiffuse = light.color.toScaled(product)
                        diffuse = diffuse.plus(lightDiffuse)                        

                    color = ambient.plus(diffuse)                    
                    
                    frame.set(x,y,color)
                else:
                    frame.set(x,y, Vector(0,0,0))
            except:
                frame.set(x,y, Vector(0,0,0))   


##Write the buffer out to a file

#Open the output file in binary mode
f = open('./saved.png', 'wb')

#Create a write object
w = png.Writer(frame.width, frame.height, greyscale=False)

#Write to the open file
w.write_array(f, frame.buffer)

#Close the file
f.close()

print("Finished rendering the file")