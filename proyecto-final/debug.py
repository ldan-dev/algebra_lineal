import sys
print(f"Python version: {sys.version}")
print("Importing AlgebraLineal...")

try:
    from AlgebraLineal import AlgebraLineal
    print("Import successful!")
    
    print("\nTesting combinacion_lineal method...")
    vectores = [[1, 0], [0, 1]]
    coeficientes = [2, 3]
    print(f"Computing linear combination of {vectores} with coefficients {coeficientes}")
    
    try:
        resultado = AlgebraLineal.combinacion_lineal(vectores, coeficientes)
        print(f"Result: {resultado}")
        print("Test successful!")
    except Exception as e:
        print(f"Error in combinacion_lineal: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"Error importing AlgebraLineal: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
