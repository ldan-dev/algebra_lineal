print("Testing AlgebraLineal module")

# Import the module
from AlgebraLineal import AlgebraLineal

# Simple test
print("Module imported successfully")

# Test a simple linear combination
vectors = [[1, 0], [0, 1]]
coeffs = [2, 3]
print(f"Computing linear combination of {vectors} with coefficients {coeffs}")
result = AlgebraLineal.combinacion_lineal(vectors, coeffs)
print(f"Result: {result}")

print("Test completed")
