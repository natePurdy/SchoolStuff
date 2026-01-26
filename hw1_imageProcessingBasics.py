import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image



def bruteForcePadding(H, W, kernelHeight, kernelWidth, inputImage):
    # pad amounts
    pad_h = int(np.floor(kernelHeight / 2))
    pad_w = int(np.floor(kernelWidth / 2))
    # make padded image init'd
    paddedImageCopy = np.zeros((H + 2*pad_h, W + 2*pad_w), dtype=inputImage.dtype) # zero out the array
    # apply padding to image manually to make sure you know whats going on under the hood
    # extract pixel values from original image and jam them into the padded array to copy it
    for inputRow in range(H):
        for inputCol in range(W):
            padImageRowIdx = inputRow + pad_h # explicitly do stuff
            padImageColIdx = inputCol + pad_w
            paddedImageCopy[padImageRowIdx, padImageColIdx] = inputImage[inputRow, inputCol]

    # top & bottom padding (replication, using raw images edge values and continuing them outward)
    for inputCol in range(W):
        # top of image
        paddedImageCopy[0, inputCol + pad_w] = inputImage[0, inputCol]
        # bottom of image
        paddedImageCopy[H + pad_h, inputCol + pad_w] = inputImage[H - 1, inputCol]

    # left & right padding (replication)
    for inputRow in range(H):
        # left column(s) of image
        paddedImageCopy[inputRow + pad_h, 0] = inputImage[inputRow, 0]
        #right column(s) of image
        paddedImageCopy[inputRow + pad_h, W + pad_w] = inputImage[inputRow, W - 1]


    # corner padding (replication) so i dont have if statements in those individual for loops

    ## dependent on the kernel of the alrgorithm being applied (2x2 means 1 courner pixel, larger kernals mean larger than 1 pixel needs to be filled...)
    # top-left corner area
    paddedImageCopy[0:pad_h, 0:pad_w] = inputImage[0, 0]
    # top-right corner
    paddedImageCopy[0:pad_h, W + pad_w:W + 2*pad_w] = inputImage[0, W - 1]
    # bottom-left corner
    paddedImageCopy[H + pad_h:H + 2*pad_h, 0:pad_w] = inputImage[H - 1, 0]
    # bottom-right corner
    paddedImageCopy[H + pad_h:H + 2*pad_h, W + pad_w:W + 2*pad_w] = inputImage[H - 1, W - 1]
    # now print out the padded copy to make sure it looks correct...
    return paddedImageCopy, pad_h, pad_w

    
def performSmoothing(H, W, padded_image, padAmountY, padAmountX, output_image, printStuff):
    for inputRow in range(H):
        for inputCol in range(W):

            # map input image indeces to the padded image indeces
            padIdxRow = inputRow + padAmountY
            padIdxCol = inputCol + padAmountX
            # current pixel of interest
            currentPixel = padded_image[padIdxRow, padIdxCol]
            #determine which are neighbors
            northernNeighbor = padded_image[padIdxRow - 1, padIdxCol]
            southernNeighbor = padded_image[padIdxRow + 1, padIdxCol] # index is from upper left corner
            westernNeighbor  = padded_image[padIdxRow, padIdxCol - 1]
            easternNeighbor  = padded_image[padIdxRow, padIdxCol + 1]
            # determine which neighbors are good for use for averaging
            neighbors = [northernNeighbor, southernNeighbor, westernNeighbor, easternNeighbor]
            neighborsAvg = np.mean(neighbors)
            goodNeighbors = []# init list for "good" neighbor values
            deltas = [] # for part a of part 1 of the hw description
            for neigh in neighbors:
                delta = abs(neigh-neighborsAvg)
                if inputRow == 0 and inputCol == 0 and printStuff ==True: # grab deltas for first part of assignment

                    deltas.append(delta)
                    
                outlierThreshold = 0.4*neighborsAvg
                if delta < outlierThreshold:
                    goodNeighbors.append(neigh)
            



            # now to determine the value of the output pixel
            if len(goodNeighbors) > 0:
                # use only good neighbors for average
                goodNeighborsAvg = np.mean(goodNeighbors)
                output_image[inputRow, inputCol] = int(round(goodNeighborsAvg))
            else:
                # if all neighbors happen to be outliers, use original pixel from input image
                output_image[inputRow, inputCol] = int(round(currentPixel))

            if inputRow ==0 and inputCol == 0 and printStuff:
                # a) Show the values of 𝑥̅,Δ𝑟, Δ𝑐, and 𝑔(𝑟,𝑐) for 𝑟 = 0,𝑐 = 0.  
                x_bar = neighborsAvg
                x_bar_good = goodNeighborsAvg
                print("Summary of first pixels operations - part1a - (Neighbor average, good neighbor average, and deltas between pixel of interest and its neighbors): \n")
                print(f"X bar: {x_bar}\n")
                print(f"X bar good neighbors: {x_bar_good}\n")
                print(f"and deltas: {deltas}\n")
                
    # now display the image


    # (b) Show the values of the output image, 𝑦𝑦(𝑟𝑟,𝑐𝑐) with one image row per line, starting at row 0. 
    if printStuff:
        for row in range(H):
            print(output_image[row, :])
    
    return output_image



############################################ MAIN routine here
# test image given in homework instructions (just an array of integer values)
inputImage = np.array([[135, 145, 140, 130],
             [130, 120, 50, 105],
             [45, 165, 105, 190],
             [155, 135, 115, 105]])


###### load in real image here...


# determine some information about the image
H, W = inputImage.shape   # keep output image shape ssame as  input image shape
# create the output image that will be the same size as the input image
output_image = np.zeros((H, W), dtype=inputImage.dtype) # zero out the array
# if we are only using one neighbor, we only need to pad one a row on top and bottom, and a column on left and right
kernelHeight, kernelWidth = (3,3) # for using nearest neighbor pixels (north sourth east and west)


# now pad the image the correct amount to run it over with the kernal size instantiated
padded_image, padAmountY, padAmountX = bruteForcePadding(H, W, kernelHeight, kernelWidth, inputImage)
# paddexImageHeight, paddedImageWidth = padded_image.shape





# now actually do the processing (smoothing/averaging)
printStuff = True
# double check it
# print(f"PAdded image: {padded_image}")
outputImage = performSmoothing(H, W, padded_image, padAmountY, padAmountX, output_image, printStuff)


# now for part two of the homework, read in the image and apply the same functions that were applied to the simple array
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
inputImage = Image.open("hw1noisy.png").convert("L") # expects it to be in same folder as the script, and is in grayscale
axes[0].imshow(inputImage, cmap='gray')
axes[0].set_title("Input Image")

inputImage = np.array(inputImage, dtype=np.uint8) # convert to regular numpy array for processing



#now do the same thing as was done before, but on a real image
H, W = inputImage.shape   # keep output image shape ssame as  input image shape
output_image = np.zeros((H, W), dtype=inputImage.dtype) # copy for the output image
# if we are only using one neighbor, we only need to pad one a row on top and bottom, and a column on left and right
kernelHeight, kernelWidth = (3,3) # for using nearest neighbor pixels (north sourth east and west) requires a 3x3 kernel in a sense

# now pad the image the correct amount to run it over with the kernal size instantiated
padded_image, padAmountY, padAmountX = bruteForcePadding(H, W, kernelHeight, kernelWidth, inputImage)

printStuff = False
# double check it
# print(f"PAdded image: {padded_image}")
output_image = performSmoothing(H, W, padded_image, padAmountY, padAmountX, output_image, printStuff)
axes[1].imshow(output_image, cmap='gray')
axes[1].set_title("Output Image")
plt.show()

