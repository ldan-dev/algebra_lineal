import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg  # Agregamos esta importación para las funciones LU

class LinearAlgebra:
    """
    A class providing static methods for linear algebra operations.
    All methods can be used directly without creating an instance of the class.
    
    Example:
        result = LinearAlgebra.dot_product([1, 2, 3], [4, 5, 6])
    """
    
    # ======================== VECTOR OPERATIONS ========================
    
    @staticmethod
    def add_vectors(v1, v2):
        """
        Add two vectors.
        
        Args:
            v1 (list or np.ndarray): First vector
            v2 (list or np.ndarray): Second vector with same dimension as v1
            
        Returns:
            np.ndarray: Sum of the two vectors
            
        Example:
            result = LinearAlgebra.add_vectors([1, 2, 3], [4, 5, 6])
            # Returns: [5, 7, 9]
        """
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        
        if v1_array.shape != v2_array.shape:
            raise ValueError("Vectors must have the same dimensions")
            
        return v1_array + v2_array
    
    @staticmethod
    def subtract_vectors(v1, v2):
        """
        Subtract the second vector from the first.
        
        Args:
            v1 (list or np.ndarray): First vector
            v2 (list or np.ndarray): Second vector with same dimension as v1
            
        Returns:
            np.ndarray: Result of v1 - v2
            
        Example:
            result = LinearAlgebra.subtract_vectors([5, 6, 7], [1, 2, 3])
            # Returns: [4, 4, 4]
        """
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        
        if v1_array.shape != v2_array.shape:
            raise ValueError("Vectors must have the same dimensions")
            
        return v1_array - v2_array
    
    @staticmethod
    def scalar_multiply(scalar, vector):
        """
        Multiply a vector by a scalar.
        
        Args:
            scalar (float): Scalar value
            vector (list or np.ndarray): Vector to multiply
            
        Returns:
            np.ndarray: Vector multiplied by scalar
            
        Example:
            result = LinearAlgebra.scalar_multiply(2, [1, 2, 3])
            # Returns: [2, 4, 6]
        """
        return np.array(vector) * scalar
    
    @staticmethod
    def dot_product(v1, v2):
        """
        Calculate the dot product (scalar product) of two vectors.
        
        Args:
            v1 (list or np.ndarray): First vector
            v2 (list or np.ndarray): Second vector with same dimension as v1
            
        Returns:
            float: Dot product of the two vectors
            
        Example:
            result = LinearAlgebra.dot_product([1, 2, 3], [4, 5, 6])
            # Returns: 32 (1*4 + 2*5 + 3*6)
        """
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        
        if v1_array.shape != v2_array.shape:
            raise ValueError("Vectors must have the same dimensions")
            
        return np.dot(v1_array, v2_array)
    
    @staticmethod
    def cross_product(v1, v2):
        """
        Calculate the cross product of two 3D vectors.
        
        Args:
            v1 (list or np.ndarray): First 3D vector
            v2 (list or np.ndarray): Second 3D vector
            
        Returns:
            np.ndarray: Cross product of the two vectors
            
        Example:
            result = LinearAlgebra.cross_product([1, 0, 0], [0, 1, 0])
            # Returns: [0, 0, 1]
        """
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        
        if v1_array.shape != (3,) or v2_array.shape != (3,):
            raise ValueError("Cross product requires two 3D vectors")
            
        return np.cross(v1_array, v2_array)
    
    @staticmethod
    def magnitude(vector):
        """
        Calculate the magnitude (length) of a vector.
        
        Args:
            vector (list or np.ndarray): Input vector
            
        Returns:
            float: Magnitude of the vector
            
        Example:
            result = LinearAlgebra.magnitude([3, 4])
            # Returns: 5.0
        """
        return np.linalg.norm(vector)
    
    @staticmethod
    def normalize(vector):
        """
        Normalize a vector to unit length.
        
        Args:
            vector (list or np.ndarray): Input vector
            
        Returns:
            np.ndarray: Normalized vector
            
        Example:
            result = LinearAlgebra.normalize([3, 4])
            # Returns: [0.6, 0.8]
        """
        v_array = np.array(vector)
        norm = np.linalg.norm(v_array)
        
        if norm == 0:
            raise ValueError("Cannot normalize the zero vector")
            
        return v_array / norm
    
    @staticmethod
    def angle_between(v1, v2, in_degrees=True):
        """
        Calculate the angle between two vectors.
        
        Args:
            v1 (list or np.ndarray): First vector
            v2 (list or np.ndarray): Second vector
            in_degrees (bool): If True, return angle in degrees, otherwise in radians
            
        Returns:
            float: Angle between vectors
            
        Example:
            result = LinearAlgebra.angle_between([1, 0], [0, 1])
            # Returns: 90.0 (degrees)
        """
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        
        dot = np.dot(v1_array, v2_array)
        mag1 = np.linalg.norm(v1_array)
        mag2 = np.linalg.norm(v2_array)
        
        if mag1 == 0 or mag2 == 0:
            raise ValueError("Cannot calculate angle with zero vector")
            
        cos_angle = dot / (mag1 * mag2)
        # Handle numerical errors that might make |cos_angle| slightly > 1
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle = np.arccos(cos_angle)
        
        if in_degrees:
            return np.degrees(angle)
        return angle
    
    @staticmethod
    def project_vector(v, onto):
        """
        Project a vector onto another vector.
        
        Args:
            v (list or np.ndarray): Vector to project
            onto (list or np.ndarray): Vector to project onto
            
        Returns:
            np.ndarray: Projection of v onto 'onto'
            
        Example:
            result = LinearAlgebra.project_vector([3, 3], [0, 1])
            # Returns: [0, 3]
        """
        v_array = np.array(v)
        onto_array = np.array(onto)
        
        onto_magnitude = np.linalg.norm(onto_array)
        if onto_magnitude == 0:
            raise ValueError("Cannot project onto zero vector")
            
        projection_scalar = np.dot(v_array, onto_array) / onto_magnitude**2
        return projection_scalar * onto_array
    
    # ======================== ADDITIONAL VECTOR OPERATIONS ========================
    
    @staticmethod
    def suma_n_vectores(*vectors):
        """
        Add multiple vectors.
        
        Args:
            *vectors: Variable number of vectors to add
            
        Returns:
            np.ndarray: Sum of all vectors
            
        Example:
            result = LinearAlgebra.suma_n_vectores([1, 2], [3, 4], [5, 6])
            # Returns: [9, 12]
        """
        vectors_array = [np.array(v) for v in vectors]
        return np.sum(vectors_array, axis=0)
    
    @staticmethod
    def resta_n_vectores(*vectors):
        """
        Subtract multiple vectors from the first one.
        
        Args:
            *vectors: Variable number of vectors, first one is subtracted by all others
            
        Returns:
            np.ndarray: Result of v1 - v2 - v3 - ...
            
        Example:
            result = LinearAlgebra.resta_n_vectores([10, 10], [1, 2], [3, 4])
            # Returns: [6, 4]
        """
        if len(vectors) < 2:
            raise ValueError("Need at least two vectors to perform subtraction")
        
        vectors_array = [np.array(v) for v in vectors]
        return vectors_array[0] - np.sum(vectors_array[1:], axis=0)
    
    @staticmethod
    def triple_producto_cruz(v1, v2, v3):
        """
        Calculate the scalar triple product: v1 · (v2 × v3)
        
        Args:
            v1 (list or np.ndarray): First 3D vector
            v2 (list or np.ndarray): Second 3D vector
            v3 (list or np.ndarray): Third 3D vector
            
        Returns:
            float: Scalar triple product
            
        Example:
            result = LinearAlgebra.triple_producto_cruz([1, 0, 0], [0, 1, 0], [0, 0, 1])
            # Returns: 1.0
        """
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        v3_array = np.array(v3)
        
        return np.dot(v1_array, np.cross(v2_array, v3_array))
    
    @staticmethod
    def cuatriple_producto_cruz(v1, v2, v3, v4):
        """
        Calculate the quadruple product: (v1 × v2) · (v3 × v4)
        
        Args:
            v1 (list or np.ndarray): First 3D vector
            v2 (list or np.ndarray): Second 3D vector
            v3 (list or np.ndarray): Third 3D vector
            v4 (list or np.ndarray): Fourth 3D vector
            
        Returns:
            float: Quadruple product
            
        Example:
            result = LinearAlgebra.cuatriple_producto_cruz([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0])
            # Returns: 0.0
        """
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        v3_array = np.array(v3)
        v4_array = np.array(v4)
        
        return np.dot(np.cross(v1_array, v2_array), np.cross(v3_array, v4_array))
    
    @staticmethod
    def ortogonalizar(v, u):
        """
        Orthogonalize vector v with respect to vector u.
        
        Args:
            v (list or np.ndarray): Vector to orthogonalize
            u (list or np.ndarray): Reference vector
            
        Returns:
            np.ndarray: Vector v orthogonalized with respect to u
            
        Example:
            result = LinearAlgebra.ortogonalizar([3, 1], [1, 0])
            # Returns: [0, 1]
        """
        v_array = np.array(v)
        u_array = np.array(u)
        
        projection = LinearAlgebra.project_vector(v_array, u_array)
        return v_array - projection
    
    # ======================== MATRIX OPERATIONS ========================
    
    @staticmethod
    def add_matrices(m1, m2):
        """
        Add two matrices.
        
        Args:
            m1 (list or np.ndarray): First matrix
            m2 (list or np.ndarray): Second matrix with same dimensions as m1
            
        Returns:
            np.ndarray: Sum of the two matrices
            
        Example:
            result = LinearAlgebra.add_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]])
            # Returns: [[6, 8], [10, 12]]
        """
        m1_array = np.array(m1)
        m2_array = np.array(m2)
        
        if m1_array.shape != m2_array.shape:
            raise ValueError("Matrices must have the same dimensions")
            
        return m1_array + m2_array
    
    @staticmethod
    def subtract_matrices(m1, m2):
        """
        Subtract the second matrix from the first.
        
        Args:
            m1 (list or np.ndarray): First matrix
            m2 (list or np.ndarray): Second matrix with same dimensions as m1
            
        Returns:
            np.ndarray: Result of m1 - m2
            
        Example:
            result = LinearAlgebra.subtract_matrices([[6, 7], [8, 9]], [[1, 2], [3, 4]])
            # Returns: [[5, 5], [5, 5]]
        """
        m1_array = np.array(m1)
        m2_array = np.array(m2)
        
        if m1_array.shape != m2_array.shape:
            raise ValueError("Matrices must have the same dimensions")
            
        return m1_array - m2_array
    
    @staticmethod
    def scalar_matrix_multiply(scalar, matrix):
        """
        Multiply a matrix by a scalar.
        
        Args:
            scalar (float): Scalar value
            matrix (list or np.ndarray): Matrix to multiply
            
        Returns:
            np.ndarray: Matrix multiplied by scalar
            
        Example:
            result = LinearAlgebra.scalar_matrix_multiply(2, [[1, 2], [3, 4]])
            # Returns: [[2, 4], [6, 8]]
        """
        return np.array(matrix) * scalar
    
    @staticmethod
    def matrix_multiply(m1, m2):
        """
        Multiply two matrices.
        
        Args:
            m1 (list or np.ndarray): First matrix
            m2 (list or np.ndarray): Second matrix where m1 columns = m2 rows
            
        Returns:
            np.ndarray: Matrix product m1 * m2
            
        Example:
            result = LinearAlgebra.matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
            # Returns: [[19, 22], [43, 50]]
        """
        m1_array = np.array(m1)
        m2_array = np.array(m2)
        
        if m1_array.ndim == 1:
            m1_array = m1_array.reshape(1, -1)
        if m2_array.ndim == 1:
            m2_array = m2_array.reshape(-1, 1)
            
        if m1_array.shape[1] != m2_array.shape[0]:
            raise ValueError(f"Matrix dimensions incompatible for multiplication: {m1_array.shape} and {m2_array.shape}")
            
        return np.matmul(m1_array, m2_array)
    
    @staticmethod
    def transpose(matrix):
        """
        Calculate the transpose of a matrix.
        
        Args:
            matrix (list or np.ndarray): Input matrix
            
        Returns:
            np.ndarray: Transposed matrix
            
        Example:
            result = LinearAlgebra.transpose([[1, 2, 3], [4, 5, 6]])
            # Returns: [[1, 4], [2, 5], [3, 6]]
        """
        return np.array(matrix).T
    
    @staticmethod
    def inverse(matrix):
        """
        Calculate the inverse of a square matrix.
        
        Args:
            matrix (list or np.ndarray): Square matrix
            
        Returns:
            np.ndarray: Inverse of the matrix
            
        Raises:
            ValueError: If the matrix is singular (not invertible)
            
        Example:
            result = LinearAlgebra.inverse([[1, 0], [0, 1]])
            # Returns: [[1, 0], [0, 1]]
        """
        m_array = np.array(matrix)
        
        if m_array.shape[0] != m_array.shape[1]:
            raise ValueError("Matrix must be square")
            
        try:
            return np.linalg.inv(m_array)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is singular and cannot be inverted")
    
    @staticmethod
    def determinant(matrix):
        """
        Calculate the determinant of a square matrix.
        
        Args:
            matrix (list or np.ndarray): Square matrix
            
        Returns:
            float: Determinant of the matrix
            
        Example:
            result = LinearAlgebra.determinant([[1, 2], [3, 4]])
            # Returns: -2.0
        """
        m_array = np.array(matrix)
        
        if m_array.shape[0] != m_array.shape[1]:
            raise ValueError("Matrix must be square")
            
        return np.linalg.det(m_array)
    
    @staticmethod
    def trace(matrix):
        """
        Calculate the trace of a square matrix (sum of diagonal elements).
        
        Args:
            matrix (list or np.ndarray): Square matrix
            
        Returns:
            float: Trace of the matrix
            
        Example:
            result = LinearAlgebra.trace([[1, 2], [3, 4]])
            # Returns: 5
        """
        m_array = np.array(matrix)
        
        if m_array.shape[0] != m_array.shape[1]:
            raise ValueError("Matrix must be square")
            
        return np.trace(m_array)
    
    @staticmethod
    def rank(matrix):
        """
        Calculate the rank of a matrix.
        
        Args:
            matrix (list or np.ndarray): Input matrix
            
        Returns:
            int: Rank of the matrix
            
        Example:
            result = LinearAlgebra.rank([[1, 2], [2, 4]])
            # Returns: 1
        """
        return np.linalg.matrix_rank(matrix)
    
    # ======================== SYSTEMS OF EQUATIONS ========================
    
    @staticmethod
    def solve_linear_system(A, b):
        """
        Solve the linear system Ax = b.
        
        Args:
            A (list or np.ndarray): Coefficient matrix
            b (list or np.ndarray): Right-hand side vector
            
        Returns:
            np.ndarray: Solution vector x
            
        Example:
            result = LinearAlgebra.solve_linear_system([[2, 1], [1, 1]], [3, 2])
            # Returns: [1, 1]
        """
        A_array = np.array(A)
        b_array = np.array(b)
        
        try:
            return np.linalg.solve(A_array, b_array)
        except np.linalg.LinAlgError:
            # If direct solution fails, use least squares method
            return np.linalg.lstsq(A_array, b_array, rcond=None)[0]
    
    @staticmethod
    def solve_linear_system_lu(A, b):
        """
        Solve a linear system using LU decomposition.
        
        Args:
            A (list or np.ndarray): Coefficient matrix
            b (list or np.ndarray): Right-hand side vector
            
        Returns:
            np.ndarray: Solution vector x
            
        Example:
            result = LinearAlgebra.solve_linear_system_lu([[2, 1], [1, 1]], [3, 2])
            # Returns: [1, 1]
        """
        A_array = np.array(A)
        b_array = np.array(b)
        
        lu, piv = LinearAlgebra.lu_decomposition(A_array)
        return LinearAlgebra.lu_solve(lu, piv, b_array)
    
    @staticmethod
    def lu_decomposition(A):
        """
        Compute the LU decomposition of a matrix.
        
        Args:
            A (list or np.ndarray): Input matrix
            
        Returns:
            tuple: (LU matrix, pivot indices)
        """
        A_array = np.array(A, dtype=float)
        return scipy.linalg.lu_factor(A_array)
    
    @staticmethod
    def lu_solve(lu, piv, b):
        """
        Solve a linear system using the LU decomposition.
        
        Args:
            lu: LU decomposition from lu_decomposition
            piv: Pivot indices from lu_decomposition
            b (list or np.ndarray): Right-hand side vector
            
        Returns:
            np.ndarray: Solution vector x
        """
        b_array = np.array(b, dtype=float)
        return scipy.linalg.lu_solve((lu, piv), b_array)
    
    # ======================== LINEAR COMBINATIONS & TRANSFORMATIONS ========================
    
    @staticmethod
    def linear_combination(vectors, coefficients):
        """
        Compute a linear combination of vectors.
        
        Args:
            vectors (list): List of vectors
            coefficients (list): List of coefficients
            
        Returns:
            np.ndarray: Result of the linear combination
            
        Example:
            result = LinearAlgebra.linear_combination([[1, 0], [0, 1]], [2, 3])
            # Returns: [2, 3]
        """
        if len(vectors) != len(coefficients):
            raise ValueError("Number of vectors and coefficients must match")
            
        result = np.zeros_like(np.array(vectors[0]), dtype=float)
        
        for vector, coeff in zip(vectors, coefficients):
            result += np.array(vector) * coeff
            
        return result
    
    @staticmethod
    def check_linear_independence(vectors):
        """
        Check if a set of vectors is linearly independent.
        
        Args:
            vectors (list): List of vectors
            
        Returns:
            bool: True if vectors are linearly independent, False otherwise
            
        Example:
            result = LinearAlgebra.check_linear_independence([[1, 0], [0, 1]])
            # Returns: True
        """
        matrix = np.array(vectors)
        rank = np.linalg.matrix_rank(matrix)
        
        return rank == len(vectors)
    
    @staticmethod
    def apply_linear_transformation(matrix, vector):
        """
        Apply a linear transformation to a vector.
        
        Args:
            matrix (list or np.ndarray): Transformation matrix
            vector (list or np.ndarray): Vector to transform
            
        Returns:
            np.ndarray: Transformed vector
            
        Example:
            result = LinearAlgebra.apply_linear_transformation([[0, -1], [1, 0]], [1, 0])
            # Returns: [0, 1]
        """
        return np.matmul(np.array(matrix), np.array(vector))
    
    @staticmethod
    def transformation_matrix(domain_basis, codomain_vectors):
        """
        Find the matrix of a linear transformation given the images of basis vectors.
        
        Args:
            domain_basis (list): Basis of the domain
            codomain_vectors (list): Images of the basis vectors
            
        Returns:
            np.ndarray: Transformation matrix
            
        Example:
            result = LinearAlgebra.transformation_matrix([[1, 0], [0, 1]], [[0, 1], [-1, 0]])
            # Returns: [[0, -1], [1, 0]]
        """
        if len(domain_basis) != len(codomain_vectors):
            raise ValueError("Number of basis vectors and their images must match")
            
        codomain_matrix = np.array(codomain_vectors).T
        
        if np.linalg.matrix_rank(np.array(domain_basis)) < len(domain_basis):
            raise ValueError("Domain basis vectors are not linearly independent")
            
        domain_matrix = np.array(domain_basis).T
        
        return np.matmul(codomain_matrix, np.linalg.inv(domain_matrix))
    
    # ======================== EIGENVALUES & EIGENVECTORS ========================
    
    @staticmethod
    def eigenvalues(matrix):
        """
        Calculate the eigenvalues of a square matrix.
        
        Args:
            matrix (list or np.ndarray): Square matrix
            
        Returns:
            np.ndarray: Array of eigenvalues
            
        Example:
            result = LinearAlgebra.eigenvalues([[2, 1], [1, 2]])
            # Returns: [3, 1]
        """
        m_array = np.array(matrix)
        
        if m_array.shape[0] != m_array.shape[1]:
            raise ValueError("Matrix must be square")
            
        return np.linalg.eigvals(m_array)
    
    @staticmethod
    def eigenvectors(matrix):
        """
        Calculate the eigenvalues and right eigenvectors of a square matrix.
        
        Args:
            matrix (list or np.ndarray): Square matrix
            
        Returns:
            tuple: (eigenvalues, eigenvectors) where eigenvectors[:,i] is the eigenvector 
                  corresponding to eigenvalues[i]
            
        Example:
            values, vectors = LinearAlgebra.eigenvectors([[2, 1], [1, 2]])
            # eigenvalues: [3, 1], eigenvectors: columns correspond to each eigenvalue
        """
        m_array = np.array(matrix)
        
        if m_array.shape[0] != m_array.shape[1]:
            raise ValueError("Matrix must be square")
            
        return np.linalg.eig(m_array)
    
    @staticmethod
    def eigenspace(matrix, eigenvalue, tol=1e-10):
        """
        Find a basis for the eigenspace of a matrix corresponding to a given eigenvalue.
        
        Args:
            matrix (list or np.ndarray): Square matrix
            eigenvalue (float): Eigenvalue to find the eigenspace for
            tol (float): Tolerance for numerical comparisons
            
        Returns:
            np.ndarray: Matrix whose rows form a basis of the eigenspace
            
        Example:
            result = LinearAlgebra.eigenspace([[2, 0], [0, 2]], 2)
            # Returns a basis for the eigenspace, which is the entire 2D space
        """
        m_array = np.array(matrix, dtype=float)
        
        if m_array.shape[0] != m_array.shape[1]:
            raise ValueError("Matrix must be square")
            
        # Compute A - λI
        A_minus_lambda_I = m_array - eigenvalue * np.eye(m_array.shape[0])
        
        # Find the null space (kernel)
        u, s, vh = np.linalg.svd(A_minus_lambda_I)
        
        # Vectors corresponding to singular values close to zero
        null_mask = (s <= tol)
        null_space = vh[null_mask]
        
        if null_space.size == 0:
            return np.array([])
        
        return null_space
    
    # ======================== VISUALIZATION ========================
    
    @staticmethod
    def plot_vectors_2d(vectors, figsize=(8, 8)):
        """
        Plot vectors in a 2D coordinate system
        
        Parameters:
        -----------
        vectors : list of numpy arrays
            List of vectors to plot
        figsize : tuple, optional
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Colores para los vectores
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
        color_list = [colors[i % len(colors)] for i in range(len(vectors))]
        
        # Graficar cada vector con su color y etiqueta
        for i, (v, c) in enumerate(zip(vectors, color_list)):
            ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=c, label=f"v{i+1}")
            ax.text(v[0]*1.1, v[1]*1.1, f"v{i+1}=({v[0]}, {v[1]})", color=c)
        
        # Ajustar límites automáticamente según los vectores
        max_coord = max([max(abs(v[0]), abs(v[1])) for v in vectors]) * 1.2
        ax.set_xlim([-max_coord, max_coord])
        ax.set_ylim([-max_coord, max_coord])
        
        ax.grid(True)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_title("Vectors in 2D")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        
        return fig, ax

    @staticmethod
    def plot_vectors_3d(vectors, figsize=(8, 8)):
        """
        Plot vectors in a 3D coordinate system
        
        Parameters:
        -----------
        vectors : list of numpy arrays
            List of vectors to plot
        figsize : tuple, optional
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Colores para los vectores
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
        color_list = [colors[i % len(colors)] for i in range(len(vectors))]
        
        # Graficar cada vector con su color y etiqueta
        for i, (v, c) in enumerate(zip(vectors, color_list)):
            ax.quiver(0, 0, 0, v[0], v[1], v[2], color=c, label=f"v{i+1}")
            ax.text(v[0], v[1], v[2], f"v{i+1}=({v[0]}, {v[1]}, {v[2]})", color=c)
        
        # Ajustar límites automáticamente según los vectores
        max_coord = max([max(abs(v[0]), abs(v[1]), abs(v[2])) for v in vectors]) * 1.2
        ax.set_xlim([-max_coord, max_coord])
        ax.set_ylim([-max_coord, max_coord])
        ax.set_zlim([-max_coord, max_coord])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Vectors in 3D")
        ax.legend()
        
        return fig, ax
    
    @staticmethod
    def plot_linear_transformation_2d(matrix, grid_lines=10, figsize=(15, 8)):
        """
        Visualize a 2D linear transformation.
        
        Args:
            matrix (list or np.ndarray): 2x2 transformation matrix
            grid_lines (int, optional): Number of grid lines
            figsize (tuple, optional): Figure size
            
        Returns:
            tuple: (fig, axes) matplotlib figure and axes objects
        """
        matrix = np.array(matrix)
        if matrix.shape != (2, 2):
            raise ValueError("Transformation matrix must be 2x2")
            
        # Create a grid of points
        grid_points = np.linspace(-1, 1, grid_lines)
        x, y = np.meshgrid(grid_points, grid_points)
        
        # Stack coordinates into vectors
        vectors = np.stack([x.flatten(), y.flatten()])
        
        # Apply the linear transformation
        transformed = np.dot(matrix, vectors)
        
        # Reshape back to grid
        x_transformed = transformed[0, :].reshape(grid_lines, grid_lines)
        y_transformed = transformed[1, :].reshape(grid_lines, grid_lines)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Original grid
        for i in range(grid_lines):
            ax1.plot(x[i, :], y[i, :], 'b', alpha=0.5)
            ax1.plot(x[:, i], y[:, i], 'b', alpha=0.5)
            
        # Transformed grid
        for i in range(grid_lines):
            ax2.plot(x_transformed[i, :], y_transformed[i, :], 'r', alpha=0.5)
            ax2.plot(x_transformed[:, i], y_transformed[:, i], 'r', alpha=0.5)
            
        # Add unit vectors and their transformations
        e1 = np.array([1, 0])
        e2 = np.array([0, 1])
        
        e1_transformed = matrix @ e1
        e2_transformed = matrix @ e2
        
        ax1.arrow(0, 0, e1[0], e1[1], head_width=0.05, head_length=0.1, fc='blue', ec='blue', label='e1=(1,0)')
        ax1.arrow(0, 0, e2[0], e2[1], head_width=0.05, head_length=0.1, fc='green', ec='green', label='e2=(0,1)')
        
        ax2.arrow(0, 0, e1_transformed[0], e1_transformed[1], head_width=0.05, head_length=0.1, 
                 fc='blue', ec='blue', label=f'T(e1)={tuple(e1_transformed)}')
        ax2.arrow(0, 0, e2_transformed[0], e2_transformed[1], head_width=0.05, head_length=0.1, 
                 fc='green', ec='green', label=f'T(e2)={tuple(e2_transformed)}')
        
        # Set limits and labels
        max_limit = max(abs(e1_transformed.max()), abs(e2_transformed.max()), 
                        abs(e1_transformed.min()), abs(e2_transformed.min()), 1.5) * 1.2
        
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.grid(True)
        ax1.set_aspect('equal')
        ax1.set_title('Original Space')
        ax1.legend()
        
        ax2.set_xlim(-max_limit, max_limit)
        ax2.set_ylim(-max_limit, max_limit)
        ax2.grid(True)
        ax2.set_aspect('equal')
        ax2.set_title(f'Transformed Space - Matrix {matrix.tolist()}')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig, (ax1, ax2)
    
    # ======================== MATRIX TYPES & PROPERTIES ========================
    
    @staticmethod
    def is_hermitian(matrix):
        """
        Check if a matrix is Hermitian (equal to its conjugate transpose).
        
        Args:
            matrix (list or np.ndarray): Square matrix to check
            
        Returns:
            bool: True if the matrix is Hermitian, False otherwise
            
        Example:
            result = LinearAlgebra.is_hermitian([[1, 2], [2, 3]])
            # Returns: True
        """
        m_array = np.array(matrix, dtype=complex)
        return np.allclose(m_array, np.conj(m_array.T))
    
    @staticmethod
    def is_unitary(matrix):
        """
        Check if a matrix is unitary (its conjugate transpose is its inverse).
        
        Args:
            matrix (list or np.ndarray): Square matrix to check
            
        Returns:
            bool: True if the matrix is unitary, False otherwise
            
        Example:
            result = LinearAlgebra.is_unitary([[0, 1], [1, 0]])
            # Returns: True
        """
        m_array = np.array(matrix, dtype=complex)
        identity = np.eye(m_array.shape[0], dtype=complex)
        return np.allclose(np.dot(m_array, np.conj(m_array.T)), identity)
    
    @staticmethod
    def is_orthogonal(matrix):
        """
        Check if a matrix is orthogonal (its transpose is its inverse).
        
        Args:
            matrix (list or np.ndarray): Square matrix to check
            
        Returns:
            bool: True if the matrix is orthogonal, False otherwise
            
        Example:
            result = LinearAlgebra.is_orthogonal([[0, 1], [1, 0]])
            # Returns: True
        """
        m_array = np.array(matrix, dtype=float)
        identity = np.eye(m_array.shape[0])
        return np.allclose(np.dot(m_array, m_array.T), identity)
    
    @staticmethod
    def is_symmetric(matrix):
        """
        Check if a matrix is symmetric (equal to its transpose).
        
        Args:
            matrix (list or np.ndarray): Square matrix to check
            
        Returns:
            bool: True if the matrix is symmetric, False otherwise
            
        Example:
            result = LinearAlgebra.is_symmetric([[1, 2], [2, 3]])
            # Returns: True
        """
        m_array = np.array(matrix)
        return np.allclose(m_array, m_array.T)
    
    @staticmethod
    def is_diagonal(matrix):
        """
        Check if a matrix is diagonal.
        
        Args:
            matrix (list or np.ndarray): Square matrix to check
            
        Returns:
            bool: True if the matrix is diagonal, False otherwise
            
        Example:
            result = LinearAlgebra.is_diagonal([[1, 0], [0, 2]])
            # Returns: True
        """
        m_array = np.array(matrix)
        return np.allclose(m_array, np.diag(np.diag(m_array)))
    
    @staticmethod
    def is_tridiagonal(matrix):
        """
        Check if a matrix is tridiagonal.
        
        Args:
            matrix (list or np.ndarray): Square matrix to check
            
        Returns:
            bool: True if the matrix is tridiagonal, False otherwise
            
        Example:
            result = LinearAlgebra.is_tridiagonal([[1, 2, 0], [3, 4, 5], [0, 6, 7]])
            # Returns: True
        """
        m_array = np.array(matrix)
        return np.allclose(m_array, np.triu(np.tril(m_array, k=1), k=-1))
    
    @staticmethod
    def create_identity(n):
        """
        Create an identity matrix of size n×n.
        
        Args:
            n (int): Size of the matrix
            
        Returns:
            np.ndarray: Identity matrix of size n×n
            
        Example:
            result = LinearAlgebra.create_identity(3)
            # Returns: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        """
        return np.eye(n)
    
    @staticmethod
    def create_zeros(n, m=None):
        """
        Create a matrix of zeros.
        
        Args:
            n (int): Number of rows
            m (int, optional): Number of columns. If None, a square matrix is created.
            
        Returns:
            np.ndarray: Matrix of zeros
            
        Example:
            result = LinearAlgebra.create_zeros(2, 3)
            # Returns: [[0, 0, 0], [0, 0, 0]]
        """
        if m is None:
            m = n
        return np.zeros((n, m))
    
    @staticmethod
    def create_ones(n, m=None):
        """
        Create a matrix of ones.
        
        Args:
            n (int): Number of rows
            m (int, optional): Number of columns. If None, a square matrix is created.
            
        Returns:
            np.ndarray: Matrix of ones
            
        Example:
            result = LinearAlgebra.create_ones(2, 2)
            # Returns: [[1, 1], [1, 1]]
        """
        if m is None:
            m = n
        return np.ones((n, m))
    
    @staticmethod
    def create_random(n, m=None):
        """
        Create a matrix with random values between 0 and 1.
        
        Args:
            n (int): Number of rows
            m (int, optional): Number of columns. If None, a square matrix is created.
            
        Returns:
            np.ndarray: Matrix with random values
            
        Example:
            result = LinearAlgebra.create_random(2, 2)
            # Returns a 2×2 matrix with random values
        """
        if m is None:
            m = n
        return np.random.rand(n, m)