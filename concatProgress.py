#-------------------------------------------------------------------------
#                WE ASSUME PERFECT SQUARE (16x16, 32x32 etc.)
#-------------------------------------------------------------------------

# imports 
import cv2 
from PIL import Image
from scipy import misc
from math import sqrt

def countPowerOfNumber2(number):
    """
    Checking how many times we have to multiply by 2 in order to get the number passes as an argument to the function

    example: 8 = 2*2*2, meaning the number returned will be 3
    """

    # initializing counter
    count = 0

    # checking is it is divisible by 2 
    while(number % 2 == 0):
        
        # deviding by 2 in order to setup the next iteration, must be // for integer
        number //= 2

        # increasing counter
        count += 1

    if count == 0 or (count > 0 and number != 1):
        # number is not e ven
        return -1
    else:
        # number is valid
        return count





# number of total images we want to concatinate(tiles)
img_count=16    # !!! MUST be perfect square

# reading the first image, it will always appear in the result if it exists
result_img = cv2.imread('drawn_0.jpg')

# looping trough all tiles and concatinating
for tile_number in sqrt(img_count):

    

    # getting the tile
    tile = cv2.imread(f'drawn_{tile_number}.jpg') 

    # checking out the size
    tile_size = tile.size

    # width and height are the same since it is a perfect squared sized image
    width, height = tile.shape

    # Cut the image in half
    width_cutoff = width // 2
    

    s1 = tile[:, :width_cutoff] # first half of image
    # s2 = tile[:, width_cutoff:] -second half of image

    # concatinating based on the initial location
    if (tile_number + 1) % 4 == 0:
        cv2.vconcat([result_img, s1]) 
    else:
        cv2.hconcat([result_img, s1]) 

# 
# reading the first image, it will always appear in the result if it exists
result_img = cv2.imread('drawn_0.jpg')


"""
Taking the slices and creating each row by horizontally concatinating the slices:

Horizontal concatination of slices:
- take the first half of the first image
- take the even numbered images up to the last image (so second forth and so on)
"""

for row in sqrt(img_count):
    for col in range(1, sqrt(img_count), 2): 
        