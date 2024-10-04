#-------------------------------------------------------------------------
#                WE ASSUME PERFECT SQUARE (16x16, 32x32 etc.)
#-------------------------------------------------------------------------

# imports 
import cv2
import numpy as np
from math import sqrt

def concat_images_grid(img_count, overlap):
    """
    Concatenates the images into a 4x4 grid, handling overlap.
    
    Args:
        img_count (int): The number of images to concatenate (must be a perfect square).
        overlap (int): The number of pixels that overlap between slices.
        
    Returns:
        final_image: The concatenated image.
    """
    
    grid_size = int(sqrt(img_count))  # e.g., 4 for a 4x4 grid
    img = cv2.imread('drawn_0.jpg')  # Reading the first image to get dimensions
    img_height, img_width, channels = img.shape  # Get image dimensions
    
    # Final image dimensions
    final_height = (img_height - overlap) * grid_size + overlap
    final_width = (img_width - overlap) * grid_size + overlap
    
    # Create an empty array for the final concatenated image
    final_image = np.zeros((final_height, final_width, 3), dtype=np.uint8)
    
    current_img_number = 0  # Track the current image number being processed
    
    for row in range(grid_size):
        row_img = None  # To store the horizontally concatenated row

        for col in range(grid_size):
            # Load the current image
            current_tile = cv2.imread(f'drawn_{current_img_number}.jpg')
            
            if current_tile is None:
                raise ValueError(f"Image drawn_{current_img_number}.jpg could not be loaded.")
            
            # Handle overlap in the row (horizontal concatenation)
            if col == 0:
                # First image in the row (no overlap)
                row_img = current_tile
            else:
                # Concatenate with overlap removal horizontally
                row_img = np.hstack((row_img[:, :-overlap], current_tile[:, overlap:]))
            
            current_img_number += 1  # Move to the next image
        
        # Handle overlap between rows (vertical concatenation)
        y_start = row * (img_height - overlap)
        y_end = y_start + img_height
        
        if row == 0:
            # First row (no overlap)
            final_image[:img_height, :row_img.shape[1]] = row_img
        else:
            # Concatenate with overlap removal vertically
            final_image[y_start + overlap:y_end, :row_img.shape[1]] = row_img[overlap:, :]
    
    return final_image

# Example usage
img_count = 16  # We assume this is a perfect square (4x4 grid)
img = cv2.imread('drawn_0.jpg')  # Load the first image to get its dimensions
img_height, img_width, channels = img.shape

overlap = img_width // 2  # Assuming a 50% overlap

# Concatenate the images into a grid
final_image = concat_images_grid(img_count, overlap)

# Save the final image
cv2.imwrite('big_result.jpg', final_image)

