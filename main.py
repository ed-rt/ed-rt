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
import datetime
import random

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
startTime = datetime.datetime.now()


# Minimum for a ray tracer
# A frame to render to
# A camera
# A light
# An object to render


frame = Frame(256, 256)

cameraOrigin = Point3D(0, 0, 1)
origin = Point3D(0, 0, 0)
cameraLookAt = origin
cameraUp = Vector(0, 1, 0)
cameraBackgroundColor = Vector(0, 0, 0)
# convert 45 degrees to radians. Should result in pi/4 ~= .785
fov = 45 / 360 * math.pi * 2
raysPerPixel = 1
focusPoint = 0

camera = PerspectiveCamera(cameraOrigin, cameraLookAt, cameraUp,
                           cameraBackgroundColor, raysPerPixel, focusPoint, fov)

lightDirection = Vector(0, -1, 0)
lightColor = Vector(255, 255, 255)

light = DirectionalLight(lightColor, 1, lightDirection)
light2 = DirectionalLight(Vector(0, 0, 255), 1, Vector(1, 0, 0))

sphereCenter = origin
sphereRadius = .5
sphereMaterialColor = Vector(255, 255, 255)
sphereMaterialSpecularColor = Vector(255, 255, 255)
sphereMaterialSpecularStrength = 1

sphereMaterial = Material(
    sphereMaterialColor, sphereMaterialSpecularColor, sphereMaterialSpecularStrength)

sphereMaterial2 = Material(
    Vector(255, 0, 255), sphereMaterialSpecularColor, sphereMaterialSpecularStrength)

sphere = Sphere(sphereMaterial, sphereCenter, sphereRadius)
sphere2 = Sphere(sphereMaterial2, Point3D(.1, .1, .5), .1)
sphere3 = Sphere(sphereMaterial2, Point3D(-.1, .1, .5), .05)

lights = [light, light2]
objects = [sphere, sphere2, sphere3]

# Now loop over every pixel in our frame

# For every pixel
# Figure out where in camera space that pixel is
# Then figure out where in world space that pixel is
# Then shoot a ray from the world space origin of the camera to that world space location
# Then loop over each object in the scene
# For ever object that ray collides with
# Find out which one has the closest collission
# That's our hit
# If we don't have a hit, return the background color
# Then calculate the color based on the direction to the right


# There are constants we want to calculate once outside the double for loop
# to speed things up as much as possidle.
ONE_OVER_FRAME_WIDTH = 1/frame.width
ONE_OVER_FRAME_HEIGHT = 1/frame.height
TO_LOOK_AT = camera.lookAt.minus(camera.origin)
DISTANCE = TO_LOOK_AT.length()
TO_LOOK_AT_NORMALIZED = TO_LOOK_AT.toNormalized()
WIDTH = math.cos(camera.fov) * DISTANCE
HEIGHT = math.sin(camera.fov) * DISTANCE
CAMERA_RIGHT = TO_LOOK_AT_NORMALIZED.cross(camera.up)
PIXEL_WIDTH = ONE_OVER_FRAME_WIDTH * WIDTH
PIXEL_HEIGHT = ONE_OVER_FRAME_HEIGHT * HEIGHT
VECTOR_ZERO = Vector(0, 0, 0)
AMBIENT = Vector(10, 10, 10)
X_PERCENT_INC = ONE_OVER_FRAME_WIDTH * 2
Y_PERCENT_INC = ONE_OVER_FRAME_HEIGHT * 2


def castRay(x, y, pixelLookAt):
    # Now jitter the ray slightly

    if x == frame.width/2 and y == frame.height/2:
        print("here")

    # We now have our world look at points
    # We need to generate our look at ray and NORMALIZE IT!!!
    ray = Ray(camera.origin, pixelLookAt.minus(camera.origin).toNormalized())

    t = float("inf")
    i = -1
    for object in objects:

        temp = object.intersect(ray)
        if temp > 0 and temp < t:
            t = temp
            i = objects.index(object)

    if t >= 0 and not math.isinf(t):
        object = objects[i]
        collisionPoint = Point3D.fromVector(
            ray.direction.toScaled(t).plus(camera.origin.vector))
        normalDirection = collisionPoint.minus(object.center)
        normal = normalDirection.toNormalized()

        ambient = AMBIENT
        diffuse = VECTOR_ZERO

        for light in lights:
            # send out a ray and see if we actually get to the light
            toLight = light.direction.toScaled(-1)
            lightRay = Ray(collisionPoint, toLight)

            reject = False # reject goes true if we hit an object
            for lightObject in objects:
                if object !=lightObject:
                    temp = lightObject.intersect(lightRay)
                    if temp > 0:
                        reject = True
            if not reject:       
                lightDiffuse = VECTOR_ZERO
                product = toLight.dot(normal)
                if product < 0:
                    product = 0
                lightDiffuse = light.color.toScaled(product)
                materialColor = object.material.diffuseColor
                totalColor = lightDiffuse.pairwise(materialColor).toScaled(1/255)
                diffuse = diffuse.plus(totalColor)

        color = ambient.plus(diffuse)

        return color
    else:
        return VECTOR_ZERO


def renderRow(y):
    # -1 because images have y down
    yPercent = -1 * (y * Y_PERCENT_INC - 1)
    xPercent = -1
    SCALE_Y = HEIGHT * yPercent
    upWorld = camera.up.toScaled(SCALE_Y)

    for x in range(frame.width):
        # Convert from screen space to camera space
        # Then from frame camera space to world space
        #xPercent = x * ONE_OVER_FRAME_WIDTH * 2 -1
        # yPercent and xPercent are now in [-1,1]
        # Now we multiply by the camera width and height at the lookAt point
        # To do that we first get the distance from the camera origin and the camera destination
        # This becomes the hyponetus for our triangle calculations

        # width and height should be the same unless we set different fovs for width and height
        rightWorld = CAMERA_RIGHT.toScaled(WIDTH * xPercent)
        pixelLookAt = Point3D.fromVector(upWorld.plus(rightWorld))
        colorSum = VECTOR_ZERO
        for r in range(camera.raysPerPixel):
            pla = Point3D.fromVector(pixelLookAt.vector.clone())
            pla.vector.x += X_PERCENT_INC * WIDTH * (random.random() - .5)
            pla.vector.y += Y_PERCENT_INC * HEIGHT * (random.random() - .5)
            color = castRay(x, y, pla)
            colorSum = colorSum.plus(color)
        colorSum = colorSum.toScaled(1/camera.raysPerPixel)
        frame.set(x, y, colorSum)

        xPercent += X_PERCENT_INC


for y in range(frame.height):
    renderRow(y)

endCompute = datetime.datetime.now()
print(endCompute - startTime)

# Open the output file in binary mode
f = open('./saved.png', 'wb')

# Create a write object
w = png.Writer(frame.width, frame.height, greyscale=False)

# Write to the open file
w.write_array(f, frame.buffer)

# Close the file
f.close()

endTime = datetime.datetime.now()
print(endTime - startTime)

print("Finished rendering the file")
