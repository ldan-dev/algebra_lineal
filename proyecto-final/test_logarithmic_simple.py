"""
Simple test for logarithmic functions
"""

from AlgebraLineal import AlgebraLineal

def main():
    """Test the logarithmic functions"""
    # Test 1: Basic logarithmic functions
    AlgebraLineal.graficar_funciones_logaritmicas(
        funciones=["ln", "log10", "log2"],
        x_min=0.1,
        x_max=10,
        titulo="Funciones Logarítmicas Básicas",
        mostrar_puntos_destacados=True
    )
    
    # Wait for user to close the first plot
    input("Press Enter to continue to the next test...")
    
    # Test 2: Custom base logarithmic functions
    AlgebraLineal.graficar_funciones_logaritmicas(
        funciones=["ln", "log3", "log5"],
        bases_personalizadas={"log3": 3, "log5": 5},
        x_min=0.5,
        x_max=20,
        titulo="Logaritmos con Bases Personalizadas",
        etiquetas=["ln(x)", "log₃(x)", "log₅(x)"],
        mostrar_puntos_destacados=True
    )

if __name__ == "__main__":
    main()
