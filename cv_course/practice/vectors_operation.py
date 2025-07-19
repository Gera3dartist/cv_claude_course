import numpy as np
from numpy import ndarray
"""
- Practice: Implement vector operations in NumPy
- Objective: To understand and apply vector operations using NumPy
Set of vector operations to be implemented:
- Vector addition
- Vector subtraction
- Scalar multiplication
- Dot product
- Cross product
- Vector normalization
- Vector magnitude
- Angle between two vectors
- Projection of one vector onto another
- Orthogonal projection of one vector onto another
- Vector reflection across another vector
- Vector distance
"""

def to_column_vector(a: ndarray) -> ndarray:
    return a.reshape(a.size, 1)


def vector_addition(a : ndarray, b: ndarray) -> ndarray:

    return a + b
    
def vector_subtraction(a: ndarray, b: ndarray) -> ndarray:
    return a - b

def scalar_multiplication(a: ndarray, scalar: int) -> ndarray:
    return  a * scalar

def dot_product(a: ndarray, b: ndarray) -> ndarray:
    return a @ b

def cross_product(a: ndarray, b:ndarray):
    return np.cross(a, b)

def vector_normalization(a):
    pass

def vector_magnitude(a):
    pass

def angle_between_vectors(a, b):
    pass

def projection_onto(a, b):
    pass

def orthogonal_projection(a, b):
    pass

def vector_reflection(a, b):
    pass

def vector_distance(a, b):
    pass