"""
Test script to verify the Fraction implementation in AlgebraLineal class.
"""

from AlgebraLineal import AlgebraLineal
from fractions import Fraction

def main():
    # Test transformacion_lineal with fractions
    print("Testing transformacion_lineal with fractions:")
    base_dominio = [
        [Fraction(1), Fraction(0)],
        [Fraction(0), Fraction(1)]
    ]
    base_codominio = [
        [Fraction(2), Fraction(1)],
        [Fraction(0), Fraction(3)]
    ]
    
    result = AlgebraLineal.transformacion_lineal(base_dominio, base_codominio)
    print("Result matrix:")
    for row in result:
        print([str(val) for val in row])
    
    # Test gauss_jordan with fractions
    print("\nTesting gauss_jordan with fractions:")
    matriz = [
        [Fraction(1, 2), Fraction(1, 3), Fraction(1)],
        [Fraction(1), Fraction(-1), Fraction(2)]
    ]
    
    solution, solution_type = AlgebraLineal.gauss_jordan(matriz)
    print(f"Solution type: {solution_type}")
    print("Solution:", [str(val) for val in solution])
    
    # Test gauss with fractions
    print("\nTesting gauss with fractions:")
    solution, solution_type = AlgebraLineal.gauss(matriz, verbose=True)
    print(f"Solution type: {solution_type}")
    if solution:
        print("Solution:", [str(val) for val in solution])

if __name__ == "__main__":
    main()
