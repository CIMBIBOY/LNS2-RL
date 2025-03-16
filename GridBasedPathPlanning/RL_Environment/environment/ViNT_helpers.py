import numpy as np
from PIL import Image
import os

def save_numpy_to_txt(npy_path, txt_path):
    # Load the numpy array from the file
    array = np.load(npy_path)
    
    # Save the array to a text file
    np.savetxt(txt_path, array, fmt='%d', delimiter=' ')

def binary_img_gray_white(np_array, threshold=128, gray_value=128):
    # Ensure the numpy array is of type uint8
    np_array = np_array.astype(np.uint8)

    # Apply threshold: pixels >= threshold become 255 (white), others gray
    binary_array = np.where(np_array >= threshold, 255, gray_value).astype(np.uint8)

    # Convert binary array to a PIL image
    binary_image = Image.fromarray(binary_array)

    return binary_image

def binary_image_match_green(rgb_array):
    """
    Create a binary image where pixels matching [0, 255, 101] (green) are set to 192 (gray),
    and all others are set to 255 (white).

    Parameters:
    - rgb_array (numpy.ndarray): Input RGB array of shape (x, y, 3).

    Returns:
    - PIL.Image.Image: Output binary image.
    """
    # Create boolean masks for the two colors
    mask_green = (
        (rgb_array[:, :, 0] == 0) &
        (rgb_array[:, :, 1] == 255) &
        (rgb_array[:, :, 2] == 101)
    )

    mask_magenta = (
        (rgb_array[:, :, 0] == 203) &
        (rgb_array[:, :, 1] == 0) &
        (rgb_array[:, :, 2] == 255)
    )

    # Combine the two masks using element-wise OR
    mask = mask_green | mask_magenta
    mask2 = (rgb_array[:, :, 0] == 255) & (rgb_array[:, :, 1] == 0) & (rgb_array[:, :, 2] == 0)

    # Create an empty output array with all values set to white (255)
    output_array = np.full((rgb_array.shape[0], rgb_array.shape[1]), 255, dtype=np.uint8)

    # Set matching pixels to gray (192)
    output_array[mask] = 192
    output_array[mask2] = 128

    # Convert to PIL Image and return
    return Image.fromarray(output_array, mode='L')

def binary_image_match_black(rgb_array):
    """
    Create a binary image where pixels matching [0, 0, 0] (black) are set to 255 (black),
    and all others are set to 255 (white).

    Parameters:
    - rgb_array (numpy.ndarray): Input RGB array of shape (x, y, 3).

    Returns:
    - PIL.Image.Image: Output binary image.
    """
    # Create a boolean mask for black pixels ([0, 0, 0])
    mask = (rgb_array[:, :, 0] == 0) & (rgb_array[:, :, 1] == 0) & (rgb_array[:, :, 2] == 0)
    mask2 = (rgb_array[:, :, 0] == 255) & (rgb_array[:, :, 1] == 0) & (rgb_array[:, :, 2] == 0)

    # Create an empty output array with all values set to white (255)
    output_array = np.full((rgb_array.shape[0], rgb_array.shape[1]), 255, dtype=np.uint8)

    # Set matching pixels to black (0)
    output_array[mask] = 0
    output_array[mask2] = 128

    # Convert to PIL Image and return
    return Image.fromarray(output_array, mode='L')

def extract_and_print_colors(image):
    """
    Extracts and prints unique RGB colors from a numpy array representing an image.

    Parameters:
        image (numpy.ndarray): Input image of shape (h, w, 3) with RGB values.

    Returns:
        numpy.ndarray: Array of unique RGB values, shape (n_colors, 3).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input array must have shape (h, w, 3) representing an RGB image.")

    # Reshape the image into a list of RGB values
    reshaped_image = image.reshape(-1, 3)

    # Find unique RGB values
    unique_colors = np.unique(reshaped_image, axis=0)

    # Print the unique colors
    print("Unique RGB colors found in the image:")
    for color in unique_colors:
        print(f"  {color.tolist()}")  # Convert numpy array to a list for readability

    return unique_colors

def save_grid_with_name(grid_seq, save_dir, file_name):
    """
    Save the grid sequence as a .npy file with a specific name to the specified directory.

    Parameters:
        grid_seq (numpy.ndarray): The grid sequence to save.
        save_dir (str): The directory where the file will be saved.
        file_name (str): The name of the file (e.g., "grid_seq.npy").
    """
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Construct the full path
    save_path = os.path.join(save_dir, file_name)

    # Save the array
    np.save(save_path, grid_seq)
    print(f"Grid sequence saved to: {save_path}")