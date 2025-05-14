import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend instead (requires PyQt5)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg  # Agregamos esta importación para las funciones LU

# from linear_algebra import LinearAlgebra

from linear_algebra import LinearAlgebra

def main():
    """
    Example script demonstrating the usage of LinearAlgebra class
    """
    print("=== Linear Algebra Library Examples ===\n")
    
    # Vector operations
    print("--- Vector Operations ---")
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    
    print(f"Vectors: v1 = {v1}, v2 = {v2}")
    print(f"v1 + v2 = {LinearAlgebra.add_vectors(v1, v2)}")
    print(f"v1 · v2 (dot product) = {LinearAlgebra.dot_product(v1, v2)}")
    print(f"2 * v1 = {LinearAlgebra.scalar_multiply(2, v1)}")
    print(f"Magnitude of v1 = {LinearAlgebra.magnitude(v1)}")
    print(f"Normalized v1 = {LinearAlgebra.normalize(v1)}")
    print(f"Angle between v1 and v2 = {LinearAlgebra.angle_between(v1, v2):.2f} degrees\n")
    
    # 3D vectors
    v1_3d = [1, 0, 0]
    v2_3d = [0, 1, 0]
    print(f"3D vectors: v1 = {v1_3d}, v2 = {v2_3d}")
    print(f"Cross product v1 × v2 = {LinearAlgebra.cross_product(v1_3d, v2_3d)}\n")
    
    # Matrix operations
    print("--- Matrix Operations ---")
    m1 = [[1, 2], [3, 4]]
    m2 = [[5, 6], [7, 8]]
    
    print(f"Matrices: m1 = {m1}, m2 = {m2}")
    print(f"m1 + m2 = {LinearAlgebra.add_matrices(m1, m2)}")
    print(f"m1 * m2 = {LinearAlgebra.matrix_multiply(m1, m2)}")
    print(f"Transpose of m1 = {LinearAlgebra.transpose(m1)}")
    print(f"Determinant of m1 = {LinearAlgebra.determinant(m1)}")
    
    try:
        print(f"Inverse of m1 = {LinearAlgebra.inverse(m1)}\n")
    except ValueError as e:
        print(f"Error: {e}\n")
    
    # Solving linear systems
    print("--- Linear Systems ---")
    A = [[2, 1], [1, 1]]
    b = [3, 2]
    
    print(f"System of equations: {A}x = {b}")
    solution = LinearAlgebra.solve_linear_system(A, b)
    print(f"Solution: x = {solution}\n")
    
    # Eigenvalues and eigenvectors
    print("--- Eigenvalues and Eigenvectors ---")
    m = [[2, 1], [1, 2]]
    
    print(f"Matrix: {m}")
    eigenvalues = LinearAlgebra.eigenvalues(m)
    print(f"Eigenvalues: {eigenvalues}")
    
    values, vectors = LinearAlgebra.eigenvectors(m)
    print(f"Eigenvalues: {values}")
    print(f"Eigenvectors: {vectors}")
    
    # Show eigenspace for eigenvalue 3
    eigenspace = LinearAlgebra.eigenspace(m, 3)
    print(f"Eigenspace for λ=3: {eigenspace}\n")
    
    # Linear transformations
    print("--- Linear Transformations ---")
    rotation_90 = [[0, -1], [1, 0]]  # 90-degree rotation matrix
    vector = [1, 0]
    
    print(f"Transformation matrix (90° rotation): {rotation_90}")
    print(f"Original vector: {vector}")
    print(f"Transformed vector: {LinearAlgebra.apply_linear_transformation(rotation_90, vector)}\n")
    
    # Linear combinations
    print("--- Linear Combinations ---")
    basis = [[1, 0], [0, 1]]
    coeffs = [2, 3]
    
    print(f"Basis vectors: {basis}")
    print(f"Coefficients: {coeffs}")
    print(f"Linear combination: {LinearAlgebra.linear_combination(basis, coeffs)}\n")
    
    # Visualization examples (commented out for non-interactive environments)

    LinearAlgebra.plot_vectors_2d([[1, 0], [0, 1], [1, 1]])
    LinearAlgebra.plot_vectors_3d([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    LinearAlgebra.plot_vectors_3d([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    LinearAlgebra.plot_linear_transformation_2d([[0, -1], [1, 0]])
    
    # print("--- Visualization ---")
    # print("Uncomment the following lines to visualize vectors and transformations:")
    # print("# LinearAlgebra.plot_vectors_2d([[1, 0], [0, 1], [1, 1]])")
    # print("# LinearAlgebra.plot_vectors_3d([[1, 0, 0], [0, 1, 0], [0, 0, 1]])")
    # print("# LinearAlgebra.plot_linear_transformation_2d([[0, -1], [1, 0]])")

if __name__ == "__main__":
    main()