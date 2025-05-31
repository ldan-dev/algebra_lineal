"""
Test script for the logarithmic functions in AlgebraLineal.py
"""

from AlgebraLineal import AlgebraLineal
import matplotlib.pyplot as plt

# Test graficar_funciones_logaritmicas method
print("Testing logarithmic functions...")

# Test 1: Basic logarithmic functions (ln, log10, log2)
print("\nTest 1: Basic logarithmic functions")
AlgebraLineal.graficar_funciones_logaritmicas(
    funciones=["ln", "log10", "log2"],
    x_min=0.1,
    x_max=10,
    titulo="Funciones Logarítmicas Básicas",
    mostrar_puntos_destacados=True
)

# Test 2: Custom base logarithmic functions
print("\nTest 2: Custom base logarithmic functions")
AlgebraLineal.graficar_funciones_logaritmicas(
    funciones=["log3", "log5"],
    bases_personalizadas={"log3": 3, "log5": 5},
    x_min=0.5,
    x_max=20,
    titulo="Logaritmos con Bases Personalizadas",
    etiquetas=["log₃(x)", "log₅(x)"],
    mostrar_puntos_destacados=True
)

# Test 3: All types of logarithmic functions
print("\nTest 3: All types of logarithmic functions")
AlgebraLineal.graficar_funciones_logaritmicas(
    funciones=["ln", "log10", "log2", "log3", "log7"],
    bases_personalizadas={"log3": 3, "log7": 7},
    x_min=0.1,
    x_max=15,
    titulo="Comparación de Funciones Logarítmicas",
    mostrar_puntos_destacados=True
)

print("\nTests completed. Check the graphs.")
