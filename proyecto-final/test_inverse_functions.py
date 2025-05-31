"""
Simple test for inverse functions
"""

from AlgebraLineal import AlgebraLineal
import math

def main():
    """Test the inverse function graphing"""
    # Test 1: Quadratic function and its inverse (in positive domain)
    AlgebraLineal.graficar_funciones_inversas(
        lambda x: x**2,
        x_min=0,  # Limit domain to positive values for injectivity
        x_max=5,
        titulo="Función Cuadrática y su Inversa",
        etiquetas=["f(x) = x²", "f⁻¹(x) = √x"],
        mostrar_puntos_destacados=True
    )
    
    # Wait for user to close the first plot
    input("Press Enter to continue to the next test...")
    
    # Test 2: Exponential function and its inverse (logarithm)
    AlgebraLineal.graficar_funciones_inversas(
        lambda x: math.exp(x),
        x_min=-2,
        x_max=2,
        titulo="Función Exponencial y su Inversa",
        etiquetas=["f(x) = e^x", "f⁻¹(x) = ln(x)"],
        mostrar_puntos_destacados=True
    )
    
    # Wait for user to close the second plot
    input("Press Enter to continue to the next test...")
    
    # Test 3: Linear function and its inverse
    AlgebraLineal.graficar_funciones_inversas(
        lambda x: 2*x + 3,
        x_min=-5,
        x_max=5,
        titulo="Función Lineal y su Inversa",
        etiquetas=["f(x) = 2x + 3", "f⁻¹(x) = (x-3)/2"],
        mostrar_puntos_destacados=True
    )

if __name__ == "__main__":
    main()
