import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Union

class AlgebraLineal:
    def __init__(self):
        self.fig = None
        self.ax = None

    # ======================
    # Operaciones vectoriales
    # ======================
    def suma_vectores(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Suma dos vectores de igual dimensión"""
        return v1 + v2
    
    # suma de n vectores, sin saber cuantos son, usando *args
    def suma_n_vectores(self, *args: np.ndarray) -> np.ndarray:
        """Suma n vectores de igual dimensión"""
        return np.sum(args, axis=0)

    def producto_escalar(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Producto punto entre dos vectores"""
        return np.dot(v1, v2)

    def producto_vectorial(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Producto cruz (solo para R3)"""
        return np.cross(v1, v2)

    def triple_producto_escalar(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
        """Producto mixto: v1 · (v2 × v3)"""
        return np.dot(v1, np.cross(v2, v3))

    # ===================
    # Operaciones matriciales
    # ===================
    def inversa_matriz(self, matriz: np.ndarray) -> Union[np.ndarray, None]:
        """Calcula la inversa usando numpy"""
        try:
            return np.linalg.inv(matriz)
        except np.linalg.LinAlgError:
            print("Matriz singular, no invertible")
            return None

    def determinante(self, matriz: np.ndarray) -> float:
        """Calcula el determinante"""
        return np.linalg.det(matriz)

    def resolver_sistema(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Resuelve Ax = b usando mínimos cuadrados"""
        return np.linalg.lstsq(A, b, rcond=None)[0]

    # ==============
    # Visualización
    # ==============
    def graficar_2d(self, vectores: List[np.ndarray], colores: List[str]):
        """Grafica vectores en 2D"""
        plt.figure()
        for v, c in zip(vectores, colores):
            plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=c)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.grid()
        plt.show()

    def graficar_3d(self, vectores: List[np.ndarray], colores: List[str]):
        """Grafica vectores en 3D"""
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        for v, c in zip(vectores, colores):
            self.ax.quiver(0, 0, 0, v[0], v[1], v[2], color=c)
        
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([-5, 5])
        plt.show()

    # =======================
    # Espacios vectoriales
    # =======================
    def es_base(self, vectores: List[np.ndarray]) -> bool:
        """Determina si un conjunto de vectores forma una base"""
        matriz = np.array(vectores).T
        return np.linalg.matrix_rank(matriz) == len(vectores)

    def dimension_espacio(self, matriz: np.ndarray) -> int:
        """Calcula la dimensión del espacio columna"""
        return np.linalg.matrix_rank(matriz)

    # =====================
    # Valores y vectores propios
    # =====================
    def eigenvalores(self, matriz: np.ndarray) -> np.ndarray:
        """Calcula valores propios"""
        return np.linalg.eigvals(matriz)

    def eigenvectores(self, matriz: np.ndarray) -> tuple:
        """Calcula valores y vectores propios"""
        return np.linalg.eig(matriz)

    # ====================
    # Métodos adicionales
    # ====================
    def producto_tensorial(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Producto tensorial entre dos vectores"""
        return np.outer(v1, v2)

    def norma_vector(self, vector: np.ndarray) -> float:
        """Calcula la norma euclidiana"""
        return np.linalg.norm(vector)

    def volumen_paralelepipedo(self, vectores: List[np.ndarray]) -> float:
        """Calcula el volumen mediante determinante"""
        return abs(np.linalg.det(np.array(vectores).T))