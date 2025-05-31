import sys
print(f"Python version: {sys.version}")
print("Importing AlgebraLineal...")

try:
    from AlgebraLineal import AlgebraLineal
    print("Import successful!")
    
    print("\nTesting es_combinacion_lineal method...")
    vector = [3, 3]
    conjunto_vectores = [[1, 0], [0, 1]]
    print(f"Checking if {vector} is a linear combination of {conjunto_vectores}")
    
    try:
        es_cl, coefs = AlgebraLineal.es_combinacion_lineal(vector, conjunto_vectores)
        print(f"Result: {es_cl}, Coefficients: {coefs}")
        print("Test successful!")
    except Exception as e:
        print(f"Error in es_combinacion_lineal: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"Error importing AlgebraLineal: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
