import numpy as np
from AlgebraLineal import AlgebraLineal

class Maths:
    def __init__(self):
        self.algebra = AlgebraLineal()

    def suma_vectores(self, v1, v2):
        return self.algebra.suma_vectores(np.array(v1), np.array(v2))

    def suma_n_vectores(self, *vectores):
        return self.algebra.suma_n_vectores(*[np.array(v) for v in vectores])

    def producto_escalar(self, v1, v2):
        return self.algebra.producto_escalar(np.array(v1), np.array(v2))

    def producto_vectorial(self, v1, v2):
        return self.algebra.producto_vectorial(np.array(v1), np.array(v2))

    def inversa_matriz(self, matriz):
        return self.algebra.inversa_matriz(np.array(matriz))

    def determinante(self, matriz):
        return self.algebra.determinante(np.array(matriz))

    def resolver_sistema(self, A, b):
        return self.algebra.resolver_sistema(np.array(A), np.array(b))

    def graficar_2d(self, vectores, colores):
        self.algebra.graficar_2d([np.array(v) for v in vectores], colores)

    def graficar_3d(self, vectores, colores):
        self.algebra.graficar_3d([np.array(v) for v in vectores], colores)

    def es_base(self, vectores):
        return self.algebra.es_base([np.array(v) for v in vectores])

    def dimension_espacio(self, matriz):
        return self.algebra.dimension_espacio(np.array(matriz))

    def eigenvalores(self, matriz):
        return self.algebra.eigenvalores(np.array(matriz))

    def eigenvectores(self, matriz):
        return self.algebra.eigenvectores(np.array(matriz))

    def producto_tensorial(self, v1, v2):
        return self.algebra.producto_tensorial(np.array(v1), np.array(v2))

    def norma_vector(self, vector):
        return self.algebra.norma_vector(np.array(vector))

    def volumen_paralelepipedo(self, vectores):
        return self.algebra.volumen_paralelepipedo([np.array(v) for v in vectores])