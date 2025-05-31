"""
Simple test for multiple function graphing
"""

from AlgebraLineal import AlgebraLineal
import math

def main():
    """Test the multiple function graphing"""
    # Test with various types of functions
    functions = [
        lambda x: x**2,                  # Quadratic
        lambda x: math.sin(x),           # Sine
        lambda x: x**3 - 2*x,            # Cubic
        lambda x: math.exp(x/2),         # Exponential
        lambda x: math.log(abs(x) + 0.1) # Logarithmic (with adjustment for x near 0)
    ]
    
    labels = [
        "f₁(x) = x²",
        "f₂(x) = sin(x)",
        "f₃(x) = x³ - 2x",
        "f₄(x) = e^(x/2)",
        "f₅(x) = ln(|x| + 0.1)"
    ]
    
    AlgebraLineal.graficar_funciones(
        functions,
        etiquetas=labels,
        x_min=-5,
        x_max=5,
        puntos=200,
        titulo="Múltiples Funciones en una Gráfica",
        etiqueta_x="x",
        etiqueta_y="f(x)",
        mostrar_puntos_destacados=True
    )

if __name__ == "__main__":
    main()
