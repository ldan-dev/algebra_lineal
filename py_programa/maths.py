import numpy as np
from AlgebraLineal import AlgebraLineal
# from algebra_lineal import AlgebraLineal

class MathOperations:
    def __init__(self):
        self.algebra = AlgebraLineal()

    def calcular_determinante(self, matriz):
        return self.algebra.determinante(matriz)

    def graficar_2d(self, vectores, colores):
        self.algebra.graficar_2d(vectores, colores)