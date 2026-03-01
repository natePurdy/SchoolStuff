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
import sys
from matplotlib.patches import Circle
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



def houghCircleTranform(edgeMap, height, width, Rmax, Rmin):

    # perform the hough transform here, using x-y coordinates
    # apparently this is a brute force method and not practical, but polar coordinates (wikipedia) is much quicker, so i might try that 
    # after this is working.
    
    #set some limiters on the loops for practical reasons
    b_max = width
    a_max = height
    b_min = 0
    a_min = 0
    radiusMin = Rmin 
    radiusMax = Rmax # make less than image size   
    # inc values
    da = 1
    db = 1
    dR = 1
    # define the search arrays
    R_values = np.arange(radiusMin,radiusMax+1, dR)
    a_values = np.arange(a_min,a_max, da)
    b_values = np.arange(b_min,b_max, db)
    # print(a_values)
    # preallocate hough transform result vector
    houghCounter = np.zeros((height, width, len(R_values)), dtype=np.int32)
    # now do the loop thing
    for row, col in tqdm(edgeMap, desc="Processing edge pixels", unit="px"):
        for r_idx, r in enumerate(R_values):
            for a in a_values:
                for b in b_values:
                    testRadius = math.sqrt((row - a)**2 + (col - b)**2)
                    if abs(testRadius-r)<= 0.5:    #strict, but loose enough to get the "corners"
                        houghCounter[a, b, r_idx] += 1   

    return houghCounter, R_values

def houghCircleTranform_vectorized(edgeMap, height, width, Rmax, Rmin,radiusThreshold):
    """
    Vectorized brute-force version 
    """
    da = db = dR = 1
    R_values = np.arange(Rmin, Rmax + 1, dR)
    nr = len(R_values)

    a_grid = np.arange(0, height, da)          # shape (height,)
    b_grid = np.arange(0, width, db)           # shape (width,)

    # Meshgrid of ALL possible centers (y,x)
    A, B = np.meshgrid(a_grid, b_grid, indexing='ij')  # A=rows, B=cols

    houghCounter = np.zeros((height, width, nr), dtype=np.int32)

    for ey, ex in tqdm(edgeMap, desc="Processing edge pixels", unit="px"):
        # Distance from this pixel to EVERY center
        dist = np.sqrt((ey - A)**2 + (ex - B)**2)   # shape (height, width)

        for r_idx, r in enumerate(R_values):
            mask = np.abs(dist - r) <= radiusThreshold
            houghCounter[:, :, r_idx] += mask.astype(np.int32)

    return houghCounter, R_values

# def overlayCircles():



##### part 1
# load in the provided homework image
imageFile = sys.argv[1]  # user inpout the image...

# decide what do do based on the imaige you are loading in, to follow the hw flow
if imageFile == "hw3edges1.png":
    Rmax = 3
    Rmin = 1
    threshold = 0.5 # there is only one pixel in this example, so all  other circle drawing pixel locations will have a max of one single vote
    radiusThreshold = 0.5
elif imageFile == "hw3edges2.png":
    Rmax = 5
    Rmin = 1
    threshold = 3
    radiusThreshold = 0.5
elif imageFile == "hw3edges3.png":
    Rmax = 20
    Rmin = 8
    threshold = 70
    radiusThreshold = 1
image = Image.open(imageFile).convert("L") # it is black and white
original = image # make a copy for later
numCols, numRows = image.size



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
# print(edgePixels)
height, width = image.shape # output inage will be same shape as input image probably
# now we need to store the results of the transfor, and for circles, its 3 parameters (x center, y center, and the radius)
houghCounter, searchRadiusValues = houghCircleTranform_vectorized(edgePixels, height, width, Rmax, Rmin, radiusThreshold)



# now print the values for testing radius = 1,2,3
print("Accumulator slices for each radius:")
for r_idx, r in enumerate(searchRadiusValues):
    slice_2d = houghCounter[:, :, r_idx]
    print(f"\nRadius = {r:.1f}  (index {r_idx}, max votes in slice = {slice_2d.max()})")
    print(slice_2d)



# use local 3x3x3 neighborhood for thresholding
# threshold = max(1, int(houghCounter.max() * 0.25))
# Create mask for interior only (exclude 1-pixel border in all 3 dims)
interior_mask = np.zeros(houghCounter.shape, dtype=bool)
interior_mask[1:-1, 1:-1, 1:-1] = True # 3x3x3 mask of interior
local_max = (houghCounter == ndimage.maximum_filter(houghCounter, size=3)) & (houghCounter >= threshold)
peak_coords = np.argwhere(local_max)
print(f"\nLocal maxima (3x3x3 neighborhood, value ≥ {threshold}): {len(peak_coords)} found")
for i, (a, b, ridx) in enumerate(peak_coords):
    votes = houghCounter[a, b, ridx]
    r_val = searchRadiusValues[ridx]
    print(f"  #{i+1:3d} :  y={a:4d}, x={b:4d}, r={r_val:.1f} --> {votes} votes")

N = 100 # this is higher than will show, given the threshold is small enough.... only makes sense on images with actual circle shapes, not single pixels
if len(peak_coords) > 0:
    # Get indices sorted by vote descending
    sorted_idx = np.argsort([-houghCounter[tuple(p)] for p in peak_coords])
    top_peaks = peak_coords[sorted_idx[:N]]
    
    print(f"\nTop {min(N, len(peak_coords))} strongest local maxima:")
    for i, (a, b, ridx) in enumerate(top_peaks, 1):
        votes = houghCounter[a, b, ridx]
        r = searchRadiusValues[ridx]
        print(f"  #{i:2d} : y={a:4d}, x={b:4d}, r={r:.1f} → {votes} votes")


# overlay circles on the original imaige
fig_overlay, ax_overlay = plt.subplots(figsize=(10, 10))
ax_overlay.imshow(original, cmap='gray')
ax_overlay.set_title("Detected circles overlaid on original image")
ax_overlay.set_aspect('equal')

# How many of the top peaks to draw
N_draw = len(top_peaks)

for i in range(N_draw):
    y, x, ridx = top_peaks[i]           # y=row, x=col
    r = searchRadiusValues[ridx]
    votes = houghCounter[y, x, ridx]

    # Style: strongest → thick red, others → cyan dashed
    if i == 0:
        color = 'red'
        lw = 3.2
        alpha = 0.95
        ls = '-'
    else:
        color = 'cyan'
        lw = 1.6
        alpha = 0.65
        ls = '--'

    circ = Circle(
        (x, y),           # center = (column, row)
        r,
        edgecolor=color,
        facecolor='none',
        linewidth=lw,
        alpha=alpha,
        linestyle=ls
    )
    ax_overlay.add_patch(circ)

    # Small label with vote count
    ax_overlay.text(
        x + r + 8,
        y,
        f"{votes}",
        color='yellow',
        fontsize=10,
        va='center',
        bbox=dict(facecolor='black', alpha=0.45, edgecolor='none', pad=1.8)
    )

ax_overlay.axis('off')
plt.tight_layout()
plt.savefig(f"{imageFile.split(".")[0]}_overlay.png", dpi=150, bbox_inches='tight')
plt.show()




