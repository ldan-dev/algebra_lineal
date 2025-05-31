from fractions import Fraction

class TestAlgebraLineal:
    @staticmethod
    def combinacion_lineal(vectores, coeficientes):
        if len(vectores) != len(coeficientes):
            raise ValueError("El número de vectores y coeficientes debe ser igual")
        
        if not vectores:
            return []
        
        # Verificar que todos los vectores tengan la misma dimensión
        dimension = len(vectores[0])
        for i, v in enumerate(vectores):
            if len(v) != dimension:
                raise ValueError(f"Todos los vectores deben tener la misma dimensión. El vector {i+1} tiene dimensión {len(v)}, pero se esperaba {dimension}")
            
        # Inicializar el resultado con ceros
        resultado = [Fraction(0) for _ in range(dimension)]
        
        # Convertir vectores y coeficientes a Fraction para cálculos precisos
        vectores_frac = [[Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in v] for v in vectores]
        coefs_frac = [Fraction(c) if not isinstance(c, Fraction) else c for c in coeficientes]
        
        # Sumar cada vector multiplicado por su coeficiente
        for j in range(len(vectores_frac)):
            vector = vectores_frac[j]
            coef = coefs_frac[j]
            for i in range(dimension):
                resultado[i] += coef * vector[i]
        
        return resultado

def main():
    # Test case 1: Simple linear combination
    vectors = [[1, 0], [0, 1]]
    coeffs = [2, 3]
    result = TestAlgebraLineal.combinacion_lineal(vectors, coeffs)
    print(f"Linear combination of {vectors} with coefficients {coeffs} = {result}")
    
    # Test case 2: Different dimensions
    try:
        vectors = [[1, 0], [0, 1, 2]]
        coeffs = [2, 3]
        result = TestAlgebraLineal.combinacion_lineal(vectors, coeffs)
        print(f"Linear combination of {vectors} with coefficients {coeffs} = {result}")
    except ValueError as e:
        print(f"Error correctly caught: {e}")

if __name__ == "__main__":
    main()
