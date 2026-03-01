import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d
from scipy import ndimage
import os
from tqdm import tqdm
import math as math
import cv2
pie = math.pi # for convenience

"""
The purpose of this program is to implement hough transform to find circles in an image
first the image needs tobe processed to find edges, then those edges are passed to the hough transform algorithm


---> note how they are using polar coordinates instead of (x-a)^2 + (y-b)^2 = r^2
"""
def cot(x): # no built in cot() function...
    return np.cos(x) / np.sin(x)
def tan(x):
    return np.sin(x)/np.cos(x)

def sobelGradient(img_gray):
    # lazy many gradient using built in cv2 functions
    # Sobel is a good choice
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, dx=0, dy=1, ksize=3)

    magnitude = np.sqrt(sobelx**2 + sobely**2)
    angle     = np.arctan2(sobely, sobelx)          # radians, range [-π, π]

    # convert to degrees
    angle_deg = np.degrees(angle)

    return magnitude, angle  # or angle_deg


# use discrete gradient approximation
def genSuppressionMask(H, W, gradMags, gradAngs):
    out = np.zeros((H, W), dtype=bool)  # start with False, set True if local max

    # go through all the gradient magnituds and angles, approximate gradients for 4 general directions, and see if 
    # the current gradient is the local maximum, given the corresdponding angle
    for row in range(H):
        for col in range(W):
            mag = gradMags[row, col]
            if mag < 1e-6:
                continue

            ang = gradAngs[row, col]
            if ang < 0:
                ang += np.pi
            ang_deg = np.degrees(ang) % 180

            # Quantize to nearest 45° direction
            direction = round(ang_deg / 45) % 4

            n1 = n2 = 0.0

            if direction == 0 or direction == 4:      # ≈ horizontal
                if col > 0:          
                    n1 = gradMags[row, col-1]
                if col < W-1:        
                    n2 = gradMags[row, col+1]

            elif direction == 1:                      # ≈ 45°
                if row > 0 and col < W-1:   
                    n1 = gradMags[row-1, col+1]
                if row < H-1 and col > 0:   
                    n2 = gradMags[row+1, col-1]

            elif direction == 2:                      # ≈ vertical
                if row > 0:          
                    n1 = gradMags[row-1, col]
                if row < H-1:        
                    n2 = gradMags[row+1, col]

            elif direction == 3:                      # ≈ 135°
                if row > 0 and col > 0:     
                    n1 = gradMags[row-1, col-1]
                if row < H-1 and col < W-1: 
                    n2 = gradMags[row+1, col+1]

            if mag >= n1 and mag >= n2:
                out[row, col] = True

    return out



def houghCircleTranform(edgeMap, height, width):

    # perform the hough transform here, using x-y coordinates
    # apparently this is a brute force method and not practical, but polar coordinates (wikipedia) is much quicker, so i might try that 
    # after this is working.
    
    #set some limiters on the loops for practical reasons
    b_max = width
    a_max = height
    b_min = 0
    a_min = 0
    radiusMin = 1 
    radiusMax = 3 # make less than image size   
    # inc values
    da = 1
    db = 1
    dR = 1
    # define the search arrays
    R_values = np.arange(radiusMin,radiusMax, dR)
    a_values = np.arange(a_min,a_max, da)
    b_values = np.arange(b_min,b_max, db)
    print(a_values)
    # preallocate hough transform result vector
    houghCounter = np.zeros((height, width, len(R_values)), dtype=np.int32)
    # now do the loop thing
    for row, col in edgeMap:
        for r_idx, r in enumerate(R_values):
            for a in a_values:
                for b in b_values:
                    testRadius = math.sqrt((row - a)**2 + (col - b)**2)
                    if abs(testRadius-r)<= 0.5:    #strict, but loose enough to get the "corners"
                        houghCounter[a, b, r_idx] += 1   

    return houghCounter, R_values





##### part 1
# load in the provided homework image
fullPath = "hw3edges1.png"
image = Image.open(fullPath).convert("L") # it is black and white
numCols, numRows = image.size

# plot the steps of the image processing
fig, axes = plt.subplots(1, 5, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Input Image")


################################### dont perform this edge detection first. the  images provided in the hw "are already an edge map..."
# # perform gradient on image
# image = np.array(image) # cv2 functions want numpy arrays
# mags, angles = sobelGradient(image) # use gray image
# axes[1].imshow(mags, cmap='gray')
# axes[1].set_title("Gradient Magnitude")
# axes[1].imshow(angles, cmap='gray')
# axes[1].set_title("Gradient Angle")

# # now perform non maximum suppression using the gradient of the image
# gradientMask = genSuppressionMask(numRows, numCols, mags, angles)
# nonMaxSupprresedMags = mags * gradientMask # use the mask and the gradient magnitudes to perform the actual supprsion
# # perform thresholding:# result looks better if thresholding is done out here
# nonMaxSupprresedMags = nonMaxSupprresedMags > 35 # set value to something you lijke
# axes[2].imshow(nonMaxSupprresedMags, cmap='gray')
# axes[2].set_title("Gradient Mags after NMS")

# now time for hough transform
# define the parameters of searching during the hough tansform first for clarity
image = np.array(image)
edgePixels = np.argwhere(image > 0) # returns x,y list of the edge pixels in the image
print(edgePixels)
height, width = image.shape # output inage will be same shape as input image probably
# now we need to store the results of the transfor, and for circles, its 3 parameters (x center, y center, and the radius)
houghCounter, searchRadiusValues = houghCircleTranform(edgePixels, height, width)



max_votes = houghCounter.max()
if max_votes > 0:
    # Find all positions with max_votes
    positions = np.argwhere(houghCounter == max_votes)
    # positions is array of [x, y, r_idx]
    
    # use the mean of the points to determine the cnetroid of all the circles
    cent_x = np.mean(positions[:, 0])
    cent_y = np.mean(positions[:, 1])
    cent_r_idx = np.mean(positions[:, 2])  
    
    best_x = cent_x
    best_y = cent_y
    best_r = searchRadiusValues[int(round(cent_r_idx))]
    
    print(f"Centroid at (x={best_x:.2f}, y={best_y:.2f}), r={best_r}, based on {len(positions)} tied positions")

# now print the values for testing radius = 1
r1_idx = np.where(searchRadiusValues == 1)[0]
if len(r1_idx) == 1:
    r1_idx = r1_idx[0]
    print("\nHough votes for radius = 1 (slice shape:", houghCounter[:, :, r1_idx].shape, ")")
    print(houghCounter[:, :, r1_idx])
else:
    print("Radius 1 not found in searchRadiusValues")

fig2, ax2 = plt.subplots(figsize=(6,6))
r1_idx = np.where(searchRadiusValues == 1)[0][0]
ax2.imshow(houghCounter[:, :, r1_idx], cmap='hot')
ax2.set_title("Hough accumulator - radius = 1\n(brighter = more votes)")
ax2.set_xlabel("column (b)")
ax2.set_ylabel("row (a)")
plt.colorbar(ax2.imshow(houghCounter[:, :, r1_idx], cmap='hot'), ax=ax2, label='Vote count')
plt.show()