from AlgebraLineal import AlgebraLineal

# Test combinacion_lineal
print("Test de combinación lineal:")
print("-" * 40)

# Test case 1: Simple linear combination
vectors = [[1, 0], [0, 1]]
coeffs = [2, 3]
result = AlgebraLineal.combinacion_lineal(vectors, coeffs)
print(f"1. combinacion_lineal({vectors}, {coeffs}) = {result}")

# Test case 2: More complex example
vectors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
coeffs = [2, -1, 3]
result = AlgebraLineal.combinacion_lineal(vectors, coeffs)
print(f"2. combinacion_lineal({vectors}, {coeffs}) = {result}")

# Test case 3: Using fractions
vectors = [[1, 1/2], [1/3, 1/4]]
coeffs = [1/2, 1/3]
result = AlgebraLineal.combinacion_lineal(vectors, coeffs)
print(f"3. combinacion_lineal({vectors}, {coeffs}) = {result}")

print("\nTest de es_combinacion_lineal:")
print("-" * 40)

# Test case 4: Vector is a linear combination
vector = [3, 3]
vectors = [[1, 0], [0, 1]]
is_lc, coefs = AlgebraLineal.es_combinacion_lineal(vector, vectors)
print(f"4. es_combinacion_lineal({vector}, {vectors}) = {is_lc}, {coefs}")

# Test case 5: Vector is not a linear combination
vector = [1, 2, 3]
vectors = [[1, 0, 0], [0, 1, 0]]
is_lc, coefs = AlgebraLineal.es_combinacion_lineal(vector, vectors)
print(f"5. es_combinacion_lineal({vector}, {vectors}) = {is_lc}, {coefs}")

# Try to catch the error case with different dimensions
try:
    vectors = [[1, 0], [0, 1, 2]]
    coeffs = [2, 3]
    result = AlgebraLineal.combinacion_lineal(vectors, coeffs)
    print(f"6. combinacion_lineal({vectors}, {coeffs}) = {result}")
except ValueError as e:
    print(f"6. Error capturado correctamente: {e}")
