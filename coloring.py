"""
Credits: 
	1. https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py
	2. http://richzhang.github.io/colorization/
	3. https://github.com/richzhang/colorization/
"""

# Import statements
import numpy as np
from argparse import ArgumentParser
import cv2 as cv
from os.path import join


"""
Download the model files: 
	1. colorization_deploy_v2.prototxt:    https://github.com/richzhang/colorization/tree/caffe/colorization/models
	2. pts_in_hull.npy:					   https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy
	3. colorization_release_v2.caffemodel: https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1

"""

# Paths to load the model
dir_path = r"C:\Users\nages\OneDrive\Desktop\photo"
prototxt = join(dir_path, r"model/colorization_deploy_v2.prototxt")
points = join(dir_path, r"model/pts_in_hull.npy")
model = join(dir_path, r"model/colorization_release_v2.caffemodel")

# ArgumentParser
ap = ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to input of black_and_white image")
args = vars(ap.parse_args())

# Load the model
print("Load model")
net = cv.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)

# Load centers for ab channel quantization used for rebalancing
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load the input image
image = cv.imread(args["image"])#frame
scaled = image.astype("float32") / 255.0
lab = cv.cvtColor(scaled, cv.COLOR_BGR2LAB)

# Resize and process the L channel
resized = cv.resize(lab, (224, 224))
L = cv.split(resized)[0]
L -= 50

# Colorize the image
print("Colorizing the image")
net.setInput(cv.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv.resize(ab, (image.shape[1], image.shape[0]))

# Combine the L channel with the ab channels
L = cv.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

# Convert back to BGR
colorized = cv.cvtColor(colorized, cv.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

# Rescale to 8-bit unsigned integer
colorized = (255 * colorized).astype("uint8")

# Display the images
cv.imshow("Original", image)
cv.imshow("Colorized", colorized)
cv.waitKey(0)