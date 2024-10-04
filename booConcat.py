import numpy as np
from PIL import Image

def get_image_list(prefix='drawn_', extension='.jpg', num_images=16):
    """
    Generate a list of image filenames in the current directory.
    
    Args:
        prefix (str): The prefix of the image filenames.
        extension (str): The extension of the image files (e.g., .jpg).
        num_images (int): The number of images to include in the list.
    
    Returns:
        list: List of image filenames.
    """
    
    image_list = [f"{prefix}{i}{extension}" for i in range(num_images)]
    return image_list

def concat_grid(slices, overlap, grid_size=(4, 4)):
    """
    Concatenates slices into a grid, removing overlaps.
    
    Args:
        slices (list of PIL.Image): List of image slices.
        overlap (int): Number of pixels that overlap between consecutive slices.
        grid_size (tuple): The grid dimensions (rows, columns).
    
    Returns:
        PIL.Image: Concatenated image in grid form without overlaps.
    """
    
    # Calculate dimensions of each slice and the final grid image
    slice_width, slice_height = slices[0].size
    grid_rows, grid_cols = grid_size
    
    # Calculate the size of the final image
    full_width = (slice_width - overlap) * grid_cols + overlap
    full_height = (slice_height - overlap) * grid_rows + overlap
    full_image = np.zeros((full_height, full_width, 3), dtype=np.uint8)
    
    # Iterate through the grid
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Get the current slice index
            idx = row * grid_cols + col
            slice_img = slices[idx]
            slice_array = np.array(slice_img)
            
            # Calculate the offsets
            y_offset = row * (slice_height - overlap)
            x_offset = col * (slice_width - overlap)
            
            # Place the slice into the correct position in the final image
            if row == 0 and col == 0:
                # Top-left slice (no overlap removal needed)
                full_image[y_offset:y_offset + slice_height, x_offset:x_offset + slice_width] = slice_array
            elif row == 0:
                # Top row (remove horizontal overlap)
                full_image[y_offset:y_offset + slice_height, x_offset + overlap:x_offset + slice_width] = slice_array[:, overlap:]
            elif col == 0:
                # First column (remove vertical overlap)
                full_image[y_offset + overlap:y_offset + slice_height, x_offset:x_offset + slice_width] = slice_array[overlap:, :]
            else:
                # Remove both horizontal and vertical overlap
                full_image[y_offset + overlap:y_offset + slice_height, x_offset + overlap:x_offset + slice_width] = slice_array[overlap:, overlap:]
    
    # Convert the numpy array back to a PIL image
    full_image_pil = Image.fromarray(full_image)
    
    return full_image_pil

# Example usage
image_list = get_image_list(prefix='drawn_', extension='.jpg', num_images=16)

# Load the images into the slices list
slices = [Image.open(image_name) for image_name in image_list]

# Assuming 50% overlap
overlap = slices[0].size[0] // 2  # Using the width for 50% overlap

# Concatenate the slices into a 4x4 grid
result_image = concat_grid(slices, overlap, grid_size=(4, 4))
result_image.show()  # To display the final image
