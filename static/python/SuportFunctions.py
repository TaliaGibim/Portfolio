# %% Max Polling

import numpy as np

def max_pooling(matrix, pool_size):
    """
    Perform max pooling on a 2D matrix using NumPy.

    Parameters:
        matrix (numpy.ndarray): Input 2D matrix.
        pool_size (integer): Size of the pooling window, the height and width must be the same

    Returns:
        numpy.ndarray: Resultant matrix after max pooling.
    """
    # Extract dimensions
    input_height = matrix.shape[0]
    input_width = matrix.shape[1]

    # Calculate output dimensions
    output_height = input_height // pool_size
    output_width = input_width // pool_size

    # Initialize output matrix
    pooled_matrix = np.zeros((output_height, output_width))

    # Perform max pooling
    for i in range(output_height):
        for j in range(output_width):
            start_row = i * pool_size
            end_row = (i + 1) * pool_size
            start_col  = j * pool_size
            end_col = (j + 1) * pool_size
            pooled_matrix[i, j] = np.max(matrix[start_row:end_row, start_col:end_col])

    return pooled_matrix

