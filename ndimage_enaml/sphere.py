import numpy as np
import matplotlib.pyplot as plt


def sphere(shape, radius):
    """
    Creates a 3D boolean NumPy array of a specified shape, containing a
    centered sphere.

    shape : int
        Size of each dimension of the 3D array
    radius : int or float
        The radius of the sphere

    Returns
    -------
    np.ndarray
        A 3D boolean array where values inside the sphere are True and values
        outside are False.

    Examples
    --------
    >>> print(sphere(3, 1))
    [[[False False False]
      [False  True False]
      [False False False]]
    <BLANKLINE>
     [[False  True False]
      [ True  True  True]
      [False  True False]]
    <BLANKLINE>
     [[False False False]
      [False  True False]
      [False False False]]]

    >>> print(sphere(5, 2))
    [[[False False False False False]
      [False False False False False]
      [False False  True False False]
      [False False False False False]
      [False False False False False]]
    <BLANKLINE>
     [[False False False False False]
      [False  True  True  True False]
      [False  True  True  True False]
      [False  True  True  True False]
      [False False False False False]]
    <BLANKLINE>
     [[False False  True False False]
      [False  True  True  True False]
      [ True  True  True  True  True]
      [False  True  True  True False]
      [False False  True False False]]
    <BLANKLINE>
     [[False False False False False]
      [False  True  True  True False]
      [False  True  True  True False]
      [False  True  True  True False]
      [False False False False False]]
    <BLANKLINE>
     [[False False False False False]
      [False False False False False]
      [False False  True False False]
      [False False False False False]
      [False False False False False]]]
    """
    d = h = w = shape

    # Define the center of the array. We use (dim - 1) / 2.0 to handle
    # both even and odd-sized dimensions, ensuring a true center.
    center_z = (d - 1) / 2.0
    center_y = (h - 1) / 2.0
    center_x = (w - 1) / 2.0

    # Create coordinate grids. np.ogrid is memory-efficient because it creates
    # open mesh-grids. These are broadcast-compatible arrays.
    z_grid, y_grid, x_grid = np.ogrid[:d, :h, :w]

    # The equation for a sphere is (x-c_x)² + (y-c_y)² + (z-c_z)² = r².
    # We calculate the squared distance from the center for each point in the grid.
    # Using squared distance is more computationally efficient as it avoids a
    # costly square root operation on every element in the array.
    dist_sq = (z_grid - center_z)**2 + (y_grid - center_y)**2 + (x_grid - center_x)**2

    # The sphere is the region where the squared distance from the center is
    # less than or equal to the squared radius. This comparison returns a
    # boolean array of the desired shape.
    sphere = dist_sq <= radius**2

    return sphere

# --- Example Usage ---
if __name__ == '__main__':
    # Define the shape of our 3D space
    array_shape = 3

    # Define the radius of the sphere
    sphere_radius = 1

    # Create the sphere using the function
    my_sphere = sphere(array_shape, sphere_radius)

    center_slice_index = array_shape // 2
    center_slice = my_sphere[center_slice_index, :, :]

    plt.figure()
    plt.imshow(center_slice, cmap='gray', interpolation='nearest')
    plt.title(f"Center Slice of the 3D Sphere")
    plt.show()
