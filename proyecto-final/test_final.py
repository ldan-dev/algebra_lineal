from AlgebraLineal import AlgebraLineal

print("=== Testing combinacion_lineal method ===")

# Test case 1: Basic linear combination
vectors = [[1, 0], [0, 1]]
coeffs = [2, 3]
result = AlgebraLineal.combinacion_lineal(vectors, coeffs)
print(f"1. Linear combination of {vectors} with coefficients {coeffs} = {result}")

# Test case 2: More complex example
vectors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
coeffs = [2, -1, 3]
result = AlgebraLineal.combinacion_lineal(vectors, coeffs)
print(f"2. Linear combination of vectors with coefficients = {result}")

# Test case 3: Handling vectors with different dimensions
try:
    vectors = [[1, 0], [0, 1, 2]]
    coeffs = [2, 3]
    result = AlgebraLineal.combinacion_lineal(vectors, coeffs)
    print(f"3. Should not reach here: {result}")
except ValueError as e:
    print(f"3. Correctly caught error: {e}")

print("\n=== Testing es_combinacion_lineal method ===")

# Test case 4: Vector that is a linear combination
vector = [3, 3]
vectors_set = [[1, 0], [0, 1]]
is_lc, coeffs = AlgebraLineal.es_combinacion_lineal(vector, vectors_set)
print(f"4. Is {vector} a linear combination of {vectors_set}? {is_lc}, Coefficients: {coeffs}")

# Test case 5: Vector that is not a linear combination
vector = [1, 2, 3]
vectors_set = [[1, 0, 0], [0, 1, 0]]
is_lc, message = AlgebraLineal.es_combinacion_lineal(vector, vectors_set)
print(f"5. Is {vector} a linear combination of {vectors_set}? {is_lc}, Message: {message}")

print("=== All tests completed ===")
