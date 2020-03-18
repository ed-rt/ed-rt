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

background = png.Reader(filename="sky.png")

# https://stackoverflow.com/questions/138250/how-to-read-the-rgb-value-of-a-given-pixel-in-python/50894365
backgroundWidth, backgroundHeight, backrgoundRows, backgroundMeta =  background.read_flat()
backgroundPixelByteWidth = 4 if backgroundMeta['alpha'] else 3


frame = Frame(256, 256)

cameraOrigin = Point3D(0, 0, 1)
origin = Point3D(0, 0, 0)
cameraLookAt = Point3D(0, 0, 0)
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

sphereCenter = Point3D(0,0,-.2)
sphereRadius = .5
sphereMaterialColor = Vector(255, 255, 255)
sphereMaterialSpecularColor = Vector(255, 255, 255)
sphereMaterialSpecularStrength = 1

sphereMaterial = Material(
    sphereMaterialColor, sphereMaterialSpecularColor, sphereMaterialSpecularStrength,.2)

sphereMaterial2 = Material(
    Vector(255, 255, 255), sphereMaterialSpecularColor, sphereMaterialSpecularStrength, .5)

sphereMaterial3 = Material(
    Vector(0, 255, 0), sphereMaterialSpecularColor, sphereMaterialSpecularStrength, 0)

sphere = Sphere(sphereMaterial, sphereCenter, sphereRadius)
sphere2 = Sphere(sphereMaterial2, Point3D(.2, .2, .5), .1)
sphere3 = Sphere(sphereMaterial2, Point3D(-.2, -.1, .5), .1)
sphere4 = Sphere(sphereMaterial3, Point3D(0, -.1, .5), .1)

lights = [light]
objects = [sphere, sphere2, sphere3, sphere4]

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
HEIGHT = math.cos(camera.fov) * DISTANCE
CAMERA_RIGHT = TO_LOOK_AT_NORMALIZED.cross(camera.up)
PIXEL_WIDTH = ONE_OVER_FRAME_WIDTH * WIDTH
PIXEL_HEIGHT = ONE_OVER_FRAME_HEIGHT * HEIGHT
VECTOR_ZERO = Vector(0, 0, 0)
AMBIENT = Vector(10, 10, 10)
X_PERCENT_INC = ONE_OVER_FRAME_WIDTH * 2
Y_PERCENT_INC = ONE_OVER_FRAME_HEIGHT * 2


def castRay(ray, avoid, recursionDepth):
    if recursionDepth <= 0:
        return VECTOR_ZERO
    # Now jitter the ray slightly

    # We now have our world look at points
    # We need to generate our look at ray and NORMALIZE IT!!!
    #ray = Ray(camera.origin, pixelLookAt.minus(camera.origin).toNormalized())

    initialHit = hitDistance(ray, avoid)
    
    if initialHit[1] != -1:
        t = initialHit[0]
        i = initialHit[1]
        #if recursionDepth == 2 and i == 1:
        #   print("here")
        object = objects[i]
        material = object.material
        collisionPoint = Point3D.fromVector(ray.direction.toScaled(t).plus(camera.origin.vector))
        normalDirection = collisionPoint.minus(object.center)
        normal = normalDirection.toNormalized()

        ambient = AMBIENT
        diffuse = VECTOR_ZERO
        reflected = VECTOR_ZERO

        # Do the lighting calculations
        for light in lights:
            # send out a ray and see if we actually get to the light
            toLight = light.direction.toScaled(-1)
            shadowRay = Ray(collisionPoint, toLight)

            # Send out shadow-checking rays
            castResult = hitDistance(shadowRay, object)
            
            if castResult[1] == - 1:
                lightDiffuse = VECTOR_ZERO
                product = toLight.dot(normal)
                if product < 0:
                    product = 0
                lightDiffuse = light.color.toScaled(product)
                materialColor = material.diffuseColor
                totalColor = lightDiffuse.pairwise(
                    materialColor).toScaled(1/255)
                diffuse = diffuse.plus(totalColor)

        # Do the reflection calculations
        reflectionDirection = ray.direction.toScaled(-1).reflectAbout(normal)
        reflectionRay = Ray(collisionPoint, reflectionDirection)
        reflected = castRay(reflectionRay, object, recursionDepth -1)
        # hitDistance(reflectionRay, object)
        # if reflectionResult[1] == -1: # We didn't hit anything
        #     reflected = sampleBackground(reflectionDirection)
        # else:
        #     reflected = VECTOR_ZERO


        color = ambient.plus(diffuse.toScaled(1 - material.reflectivity)).plus(reflected.toScaled(material.reflectivity))

        return color
    else:
        return sampleBackground(ray.direction)
        # i = math.floor((math.atan2(ray.direction.z, ray.direction.x) + math.pi)/(2*math.pi)*backgroundWidth)
        # j = math.floor((math.atan2(ray.direction.y, ray.direction.x) + math.pi)/(2*math.pi)*backgroundHeight)
        # pixelPosition = i + j * backgroundWidth
        # backgroundColor = backrgoundRows[pixelPosition * backgroundPixelByteWidth:(pixelPosition+1)*backgroundPixelByteWidth]
        # return Vector(backgroundColor[0], backgroundColor[1], backgroundColor[2])
        # #return VECTOR_ZERO


def hitDistance(ray, originObject):
    closestHit = float("inf")
    closestIndex = -1
    for otherObject in objects:
        if originObject != otherObject:
            temp = otherObject.intersect(ray)
            if temp > 0:
                if temp < closestHit:
                    closestHit = temp
                    closestIndex = objects.index(otherObject)
    return [closestHit, closestIndex]


def sampleBackground(direction):
    i = math.floor((math.atan2(direction.z, direction.x) + math.pi)/(2*math.pi)*backgroundWidth)
    j = math.floor((math.atan2(direction.y, direction.x) + math.pi)/(2*math.pi)*backgroundHeight)
    pixelPosition = i + j * backgroundWidth
    backgroundColor = backrgoundRows[pixelPosition * backgroundPixelByteWidth:(pixelPosition+1)*backgroundPixelByteWidth]
    return Vector(backgroundColor[0], backgroundColor[1], backgroundColor[2])
    

def renderRow(y):
    # -1 because images have y down
    yPercent = -1 * (y * Y_PERCENT_INC - 1)
    xPercent = -1
    SCALE_Y = HEIGHT * yPercent
    upWorld = camera.up.toScaled(SCALE_Y)

    for x in range(frame.width):
        if x == 82 and y == 115:
            print("here")

        # Convert from screen space to camera space
        # Then from frame camera space to world space
        #xPercent = x * ONE_OVER_FRAME_WIDTH * 2 -1
        # yPercent and xPercent are now in [-1,1]
        # Now we multiply by the camera width and height at the lookAt point
        # To do that we first get the distance from the camera origin and the camera destination
        # This becomes the hyponetus for our triangle calculations

        # width and height should be the same unless we set different fovs for width and height
        rightWorld = camera.lookAt.plusVector(CAMERA_RIGHT.toScaled(WIDTH * xPercent))
        pixelLookAt = rightWorld.plusVector(upWorld)
        colorSum = VECTOR_ZERO
        for r in range(camera.raysPerPixel):
            pla = Point3D.fromVector(pixelLookAt.vector.clone())
            pla.vector.x += X_PERCENT_INC * WIDTH * (random.random() - .5)
            pla.vector.y += Y_PERCENT_INC * HEIGHT * (random.random() - .5)
            castDirection = pla.minus(camera.origin).toNormalized()
            color = castRay(Ray(camera.origin, castDirection), None,  4)
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
