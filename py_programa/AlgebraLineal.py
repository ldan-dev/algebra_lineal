# LEONARDO DANIEL AVIÑA NERI
# Fecha: 05/03/2025
# CARRERA: LIDIA
# Universidad de Guanajuato - Campus Irapuato-Salamanca
# Correo: ld.avinaneri@ugto.mx
# UDA: 
# DESCRIPCION: 

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

    def magnitud_vector(self, vector: np.ndarray) -> float:
        """Calcula la magnitud de un vector"""
        return np.linalg.norm(vector)

    def magnitd_vector_3d(self, vector: np.ndarray) -> float:
        """Calcula la magnitud de un vector en R3"""
        return np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

    def magnitud_vector_Rn(self, vector: np.ndarray) -> float:
        """Calcula la magnitud de un vector en R^n"""
        return np.sqrt(np.sum(vector**2))

    def vector_unitario(self, vector: np.ndarray) -> np.ndarray:
        """Calcula el vector unitario"""
        return vector / np.linalg.norm(vector)

    def normalizar_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normaliza un vector (lo convierte en vector unitario)"""
        return vector / np.linalg.norm(vector)

    def suma_vectores(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Suma dos vectores de igual dimensión"""
        return v1 + v2
    
    def resta_vectores(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Resta dos vectores de igual dimensión"""
        return v1 - v2
    
    # suma de n vectores, sin saber cuantos son, usando *args
    def suma_n_vectores(self, *args: np.ndarray) -> np.ndarray:
        """Suma n vectores de igual dimensión"""
        return np.sum(args, axis=0)
    
    def resta_n_vectores(self, *args: np.ndarray) -> np.ndarray:
        """Resta n vectores de igual dimensión"""
        return np.subtract(args[0], np.sum(args[1:], axis=0))

    def angulo_vectores(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Ángulo entre dos vectores"""
        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    # =================== 
    # punto
    # ===================

    def producto_punto(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Producto punto entre dos vectores"""
        return np.dot(v1, v2)
    
    def producto_punto_n(self, *args: np.ndarray) -> float:
        """Producto punto entre n vectores"""
        return np.dot(args[0], np.sum(args[1:], axis=0))

    def angulo_producto_punto(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Ángulo entre dos vectores usando el producto punto"""
        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    # ===================
    # Producto cruz
    # ===================

    def producto_cruz(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Producto cruz (solo para R3)"""
        return np.cross(v1, v2)

    def triple_producto_cruz(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
        """Producto mixto: v1 · (v2 × v3)"""
        return np.dot(v1, np.cross(v2, v3))
    
    def cuatriple_producto_cruz(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, v4: np.ndarray) -> float:
        """Producto mixto: v1 · (v2 × v3) × v4"""
        return np.dot(np.cross(v1, np.cross(v2, v3)), v4)

    def n_producto_cruz(self, *args: np.ndarray) -> float:
        """Producto mixto: v1 · (v2 × v3) × v4 × ..."""
        return np.dot(np.cross(args[0], np.cross(args[1], args[2])), np.sum(args[3:], axis=0))

    def angulo_producto_cruz(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Ángulo entre dos vectores usando el producto cruz"""
        return np.arcsin(np.linalg.norm(np.cross(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    # =================== 
    # Operaciones con escalares
    def proyeccion_vector(self, v: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Proyección de v sobre u"""
        return np.dot(v, u) / np.dot(u, u) * u
    
    # ortogonalizar un vector respecto a otro vector en R^n (n-dimensiones)
    def ortogonalizar(self, v: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Ortogonaliza v respecto a u"""
        return v - self.proyeccion_vector(v, u)
    

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

    # ecuaciones lineales
    def resolver_sistema(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Resuelve Ax = b usando mínimos cuadrados"""
        return np.linalg.lstsq(A, b, rcond=None)[0]

    def matriz_transpuesta(self, matriz: np.ndarray) -> np.ndarray:
        """Calcula la transpuesta"""
        return matriz.T

    def matriz_adjunta(self, matriz: np.ndarray) -> np.ndarray:
        """Calcula la adjunta (conjugada transpuesta)"""
        return np.conj(matriz.T)

    def matriz_hermitiana(self, matriz: np.ndarray) -> bool:
        """Determina si una matriz es hermitiana"""
        return np.allclose(matriz, self.matriz_adjunta(matriz))

    def matriz_unitaria(self, matriz: np.ndarray) -> bool:
        """Determina si una matriz es unitaria"""
        return np.allclose(np.eye(matriz.shape[0]), np.dot(matriz, self.matriz_adjunta(matriz)))

    def matriz_ortogonal(self, matriz: np.ndarray) -> bool:
        """Determina si una matriz es ortogonal"""
        return np.allclose(np.eye(matriz.shape[0]), np.dot(matriz, matriz.T))

    def matriz_simetrica(self, matriz: np.ndarray) -> bool:
        """Determina si una matriz es simétrica"""
        return np.allclose(matriz, matriz.T)

    def matriz_diagonal(self, matriz: np.ndarray) -> bool:
        """Determina si una matriz es diagonal"""
        return np.allclose(matriz, np.diag(np.diag(matriz)))

    def matriz_tridiagonal(self, matriz: np.ndarray) -> bool:
        """Determina si una matriz es tridiagonal"""
        return np.allclose(matriz, np.triu(np.tril(matriz, k=1), k=-1))

    def matriz_identidad(self, n: int) -> np.ndarray:
        """Crea una matriz identidad de nxn"""
        return np.eye(n)

    def matriz_ceros(self, n: int, m: int) -> np.ndarray:
        """Crea una matriz de ceros de nxm"""
        return np.zeros((n, m))

    def matriz_unos(self, n: int, m: int) -> np.ndarray:
        """Crea una matriz de unos de nxm"""
        return np.ones((n, m))

    def matriz_random(self, n: int, m: int) -> np.ndarray:
        """Crea una matriz aleatoria de nxm"""
        return np.random.rand(n, m)

    # ==============
    # Visualización
    # ==============
    def graficar_2d(self, vectores: List[np.ndarray], colores: List[str] = None, etiquetas: List[str] = None):
        """
        Grafica vectores en 2D
        
        Parameters:
        -----------
        vectores: Lista de vectores a graficar
        colores: Lista de colores para cada vector (opcional)
        etiquetas: Lista de etiquetas para cada vector (opcional)
        """
        plt.figure(figsize=(10, 8))  # crea una nueva figura con tamaño personalizado
        
        # Si no se proporcionan colores, crear una lista de colores predeterminados
        if colores is None:
            colores_base = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
            colores = [colores_base[i % len(colores_base)] for i in range(len(vectores))]
        
        # Si no se proporcionan etiquetas, crear etiquetas predeterminadas (v1, v2, ...)
        if etiquetas is None:
            etiquetas = [f'v{i+1}' for i in range(len(vectores))]
            
        # Graficar cada vector con su color y etiqueta
        for i, (v, c, etq) in enumerate(zip(vectores, colores, etiquetas)):
            plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=c, label=etq)
            plt.text(v[0] * 1.1, v[1] * 1.1, f'{etq}=({v[0]}, {v[1]})', color=c, fontsize=10)
        
        # Ajustar límites automáticamente según los vectores
        max_coord = max([max(abs(v[0]), abs(v[1])) for v in vectores]) * 1.2
        plt.xlim(-max_coord, max_coord)
        plt.ylim(-max_coord, max_coord)
        
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.title('Representación de Vectores en 2D', fontsize=14)
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.legend()
        plt.show()

    def graficar_3d(self, vectores: List[np.ndarray], colores: List[str] = None, etiquetas: List[str] = None):
        """
        Grafica vectores en 3D usando matplotlib
        
        Parameters:
        -----------
        vectores: Lista de vectores a graficar
        colores: Lista de colores para cada vector (opcional)
        etiquetas: Lista de etiquetas para cada vector (opcional)
        """
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Si no se proporcionan colores, crear una lista de colores predeterminados
        if colores is None:
            colores_base = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
            colores = [colores_base[i % len(colores_base)] for i in range(len(vectores))]
        
        # Si no se proporcionan etiquetas, crear etiquetas predeterminadas (v1, v2, ...)
        if etiquetas is None:
            etiquetas = [f'v{i+1}' for i in range(len(vectores))]
        
        # Graficar cada vector con su color y etiqueta
        for i, (v, c, etq) in enumerate(zip(vectores, colores, etiquetas)):
            self.ax.quiver(0, 0, 0, v[0], v[1], v[2], color=c, label=etq, arrow_length_ratio=0.1)
            self.ax.text(v[0] * 1.1, v[1] * 1.1, v[2] * 1.1, f'{etq}=({v[0]}, {v[1]}, {v[2]})', color=c, fontsize=10)
        
        # Ajustar límites automáticamente según los vectores
        max_coord = max([max(abs(v[0]), abs(v[1]), abs(v[2])) for v in vectores]) * 1.2
        self.ax.set_xlim([-max_coord, max_coord])
        self.ax.set_ylim([-max_coord, max_coord])
        self.ax.set_zlim([-max_coord, max_coord])
        
        self.ax.set_xlabel('X', fontsize=12)
        self.ax.set_ylabel('Y', fontsize=12)
        self.ax.set_zlabel('Z', fontsize=12)
        self.ax.set_title('Representación de Vectores en 3D', fontsize=14)
        self.ax.legend()
        
        # Agregar una cuadrícula y ejes
        self.ax.grid(True)
        
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

    def area_paralelogramo(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calcula el área de un paralelogramo"""
        return np.linalg.norm(np.cross(v1, v2))

    def area_triangulo(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calcula el área de un triángulo"""
        return 0.5 * np.linalg.norm(np.cross(v1, v2))

    def distancia_punto_plano(self, punto: np.ndarray, plano: np.ndarray) -> float:
        """Calcula la distancia de un punto a un plano (en R3)"""
        return np.dot(punto, plano[:-1]) + plano[-1] / np.linalg.norm(plano[:-1])

    def distancia_entre_puntos(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calcula la distancia entre dos puntos"""
        return np.linalg.norm(p1 - p2)

    def distancia_entre_punto_recta(self, punto: np.ndarray, recta: np.ndarray) -> float:
        """Calcula la distancia de un punto a una recta (en R3)"""
        return np.linalg.norm(np.cross(punto - recta[:3], recta[3:])) / np.linalg.norm(recta[3:])

    def distancia_entre_rectas(self, r1: np.ndarray, r2: np.ndarray) -> float:
        """Calcula la distancia entre dos rectas (en R3)"""
        n = np.cross(r1[3:], r2[3:])
        return abs(np.dot(r1[:3] - r2[:3], n)) / np.linalg.norm(n)

    def interseccion_rectas(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """Calcula la intersección de dos rectas (en R3)"""
        n = np.cross(r1[3:], r2[3:])
        return np.cross(np.cross(r1[:3], r2[3:]) - np.cross(r1[3:], r2[:3]), n) / np.linalg.norm(n)

    def interseccion_plano_recta(self, plano: np.ndarray, recta: np.ndarray) -> np.ndarray:
        """Calcula la intersección de un plano y una recta (en R3)"""
        n = plano[:3]
        return recta[:3] + np.dot(plano[:3], plano[3] - recta[:3]) / np.dot(recta[3:], plano[:3]) * recta[3:]

    def interseccion_plano_plano(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Calcula la intersección de dos planos (en R3)"""
        n1, n2 = p1[:3], p2[:3]
        n = np.cross(n1, n2)
        return np.cross(p1[3] * n2 - p2[3] * n1, n) / np.linalg.norm(n)

    def recta_dos_puntos(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Calcula la recta que pasa por dos puntos"""
        return np.concatenate([p1, p2 - p1])

    def plano_tres_puntos(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        """Calcula el plano que pasa por tres puntos"""
        n = np.cross(p2 - p1, p3 - p1)
        return np.concatenate([n, np.dot(n, p1)])

    def plano_normal_vector(self, v: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Calcula el plano normal a un vector que pasa por un punto"""
        return np.concatenate([v, np.dot(v, p)])

    def recta_paralela_plano(self, v: np.ndarray, p: np.ndarray, p0: np.ndarray) -> np.ndarray:
        """Calcula la recta paralela a un plano que pasa por un punto"""
        return np.concatenate([p0, v])

    
# ejemplo de uso para graficar vector 3d:
v1 = np.array([1, 2, 3])
v2 = np.array([3, 2, 1])
al = AlgebraLineal()
al.graficar_3d([v1, v2], ['b', 'r'])

# ejemplo de uso para graficar vector 2d:
v1 = np.array([1, 2])
v2 = np.array([3, 2])
al.graficar_2d([v1, v2], ['b', 'r'])
