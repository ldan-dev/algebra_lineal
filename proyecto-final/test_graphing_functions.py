"""
Test script for all graphing functions in AlgebraLineal.py
"""

from AlgebraLineal import AlgebraLineal
import matplotlib.pyplot as plt
import math
import numpy as np

def test_graficar_funcion():
    """Test the basic function graphing method"""
    print("\nTesting graficar_funcion...")
    AlgebraLineal.graficar_funcion(
        lambda x: x**2,
        x_min=-5,
        x_max=5,
        titulo="Función Cuadrática",
        etiqueta_x="x",
        etiqueta_y="f(x) = x²",
        mostrar_puntos_destacados=True
    )

def test_graficar_funciones():
    """Test multiple function graphing method"""
    print("\nTesting graficar_funciones...")
    funciones = [
        lambda x: x**2,
        lambda x: math.sin(x),
        lambda x: x**3 - 2*x
    ]
    etiquetas = ["f(x) = x²", "g(x) = sin(x)", "h(x) = x³ - 2x"]
    
    AlgebraLineal.graficar_funciones(
        funciones,
        etiquetas=etiquetas,
        x_min=-5,
        x_max=5,
        titulo="Múltiples Funciones",
        mostrar_puntos_destacados=True
    )

def test_graficar_funciones_trigonometricas():
    """Test trigonometric function graphing method"""
    print("\nTesting graficar_funciones_trigonometricas...")
    AlgebraLineal.graficar_funciones_trigonometricas(
        funciones=["sin", "cos", "tan"],
        x_min=-2*math.pi,
        x_max=2*math.pi,
        titulo="Funciones Trigonométricas"
    )

def test_graficar_funciones_logaritmicas():
    """Test logarithmic function graphing method"""
    print("\nTesting graficar_funciones_logaritmicas...")
    AlgebraLineal.graficar_funciones_logaritmicas(
        funciones=["ln", "log10", "log2"],
        x_min=0.1,
        x_max=10,
        titulo="Funciones Logarítmicas",
        mostrar_puntos_destacados=True
    )
    
    # Test with custom bases
    print("Testing logarithmic functions with custom bases...")
    AlgebraLineal.graficar_funciones_logaritmicas(
        funciones=["ln", "log3", "log5"],
        bases_personalizadas={"log3": 3, "log5": 5},
        x_min=0.1,
        x_max=10,
        titulo="Logaritmos con Bases Personalizadas",
        mostrar_puntos_destacados=True
    )

def test_graficar_funciones_inversas():
    """Test inverse function graphing method"""
    print("\nTesting graficar_funciones_inversas...")
    # Test with quadratic function (restricting to positive domain for injectivity)
    AlgebraLineal.graficar_funciones_inversas(
        lambda x: x**2,
        x_min=0,
        x_max=5,
        titulo="Función Cuadrática y su Inversa",
        etiquetas=["f(x) = x²", "f⁻¹(x) = √x"],
        mostrar_puntos_destacados=True
    )

def test_graficar_vectores():
    """Test vector graphing method"""
    print("\nTesting graficar_vectores...")
    vectores = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ]
    
    AlgebraLineal.graficar_vectores(
        vectores,
        etiquetas=["i", "j", "k", "v"],
        origen=[0, 0, 0],
        titulo="Vectores en 3D",
        mostrar_ejes=True
    )

def main():
    """Run all tests"""
    print("Testing all graphing methods in AlgebraLineal.py")
    
    # Test each graphing function
    test_graficar_funcion()
    test_graficar_funciones()
    test_graficar_funciones_trigonometricas()
    test_graficar_funciones_logaritmicas()
    test_graficar_funciones_inversas()
    test_graficar_vectores()
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()
