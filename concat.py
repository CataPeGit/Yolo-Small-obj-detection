#-------------------------------------------------------------------------
#                WE ASSUME PERFECT SQUARE (16x16, 32x32 etc.)
#-------------------------------------------------------------------------

# imports 
import cv2 
from PIL import Image
from scipy import misc
from math import sqrt
import numpy as np

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


def cutImageInHalf(result_img, width):
    # Cut the image in half
    width_cutoff = width // 2

    result_img = result_img[:, :width_cutoff] 



# number of total images we want to concatinate(tiles)
img_count=16    # !!! MUST be perfect square
# width and height are the same since it is a perfect squared sized image

img = cv2.imread('drawn_0.jpg')
img_width, img_height, channels = img.shape
"""
    s1 = tile[:, :width_cutoff] # first half of image
    # s2 = tile[:, width_cutoff:] -second half of image

    # concatinating based on the initial location
    if (tile_number + 1) % 4 == 0:
        cv2.vconcat([result_img, s1]) 
    else:
        cv2.hconcat([result_img, s1]) 
"""
# 
# Reading the first image, it will always appear in the result if it exists.
# We only consider the first half
result_img = cutImageInHalf(img, img_width)


"""
Create each row by horizontally concatinating the slices:

Horizontal concatination of slices:
- take the first half of the first image
- concatinate horizontally the even numbered images up to the last image (so second forth and so on)

After all rows have a successfully created image representing them, we concatinate them vertically in pairs of 2.
We need to use pairs of two so that a third row would not be the size of the first two rows concatinated.
"""

current_img_number = 0

all_rows = []

for row in range(int(sqrt(img_count))):

    # Reading the first image in the row, it will always appear in the result if it exists.
    # We only consider the first half
    result_img = cutImageInHalf(cv2.imread(f'drawn_{current_img_number}.jpg'), img_width)

    for col in range(1, int(sqrt(img_count)), 2):
        # moving to the current image
        current_img_number += 2

        # getting the current photo
        current_tile = cv2.imread(f'drawn_{col}.jpg') 

        
        # concatinating tile
        cv2.hconcat([result_img, current_tile]) 
        print("UWUWUWUUWUWUWUWU")
    all_rows.append(result_img)


final_image = np.zeros((img_height,img_width,3), np.uint8)
for row in all_rows:
    cv2.vconcat([result_img, row]) 

cv2.imwrite('big_result.jpg', final_image)