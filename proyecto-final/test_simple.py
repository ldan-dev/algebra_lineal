from AlgebraLineal import AlgebraLineal

def main():
    print("=== Test de combinacion_lineal ===")
    
    # Caso de prueba 1: Combinación lineal simple
    vectores = [[1, 0], [0, 1]]
    coeficientes = [2, 3]
    resultado = AlgebraLineal.combinacion_lineal(vectores, coeficientes)
    print(f"1. {vectores} con coeficientes {coeficientes} = {resultado}")
    
    # Caso de prueba 2: Vector que no está en la dimensión correcta
    try:
        vectores = [[1, 0], [0, 1, 2]]
        coeficientes = [2, 3]
        resultado = AlgebraLineal.combinacion_lineal(vectores, coeficientes)
        print(f"2. {vectores} con coeficientes {coeficientes} = {resultado}")
    except ValueError as e:
        print(f"2. Error correctamente capturado: {e}")
        
    print("\n=== Test completo ===")

if __name__ == "__main__":
    main()
