from AlgebraLineal import AlgebraLineal
from fractions import Fraction

def test_combinacion_lineal():
    results = []
    results.append("=== Testing combinacion_lineal method ===")

    # Test case 1: Basic linear combination
    vectors = [[1, 0], [0, 1]]
    coeffs = [2, 3]
    result = AlgebraLineal.combinacion_lineal(vectors, coeffs)
    results.append(f"1. Linear combination of {vectors} with coefficients {coeffs} = {result}")
    expected = [Fraction(2, 1), Fraction(3, 1)]
    results.append(f"   Expected: {expected}")
    results.append(f"   Test passed: {result == expected}")

    # Test case 2: More complex example
    vectors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    coeffs = [2, -1, 3]
    result = AlgebraLineal.combinacion_lineal(vectors, coeffs)
    results.append(f"2. Linear combination of {vectors} with coefficients {coeffs} = {result}")
    expected = [Fraction(19, 1), Fraction(23, 1), Fraction(27, 1)]
    results.append(f"   Expected: {expected}")
    results.append(f"   Test passed: {result == expected}")

    # Test case 3: Handling vectors with different dimensions
    try:
        vectors = [[1, 0], [0, 1, 2]]
        coeffs = [2, 3]
        result = AlgebraLineal.combinacion_lineal(vectors, coeffs)
        results.append(f"3. Should not reach here: {result}")
        results.append(f"   Test failed: Expected ValueError")
    except ValueError as e:
        results.append(f"3. Correctly caught error: {e}")
        results.append(f"   Test passed: ValueError was raised")

    return "\n".join(results)

def test_es_combinacion_lineal():
    results = []
    results.append("\n=== Testing es_combinacion_lineal method ===")

    # Test case 4: Vector that is a linear combination
    vector = [3, 3]
    vectors_set = [[1, 0], [0, 1]]
    is_lc, coeffs = AlgebraLineal.es_combinacion_lineal(vector, vectors_set)
    results.append(f"4. Is {vector} a linear combination of {vectors_set}? {is_lc}, Coefficients: {coeffs}")
    expected = (True, [Fraction(3, 1), Fraction(3, 1)])
    results.append(f"   Expected: {expected}")
    results.append(f"   Test passed: {(is_lc, coeffs) == expected}")

    # Test case 5: Vector that is not a linear combination
    vector = [1, 2, 3]
    vectors_set = [[1, 0, 0], [0, 1, 0]]
    is_lc, message = AlgebraLineal.es_combinacion_lineal(vector, vectors_set)
    results.append(f"5. Is {vector} a linear combination of {vectors_set}? {is_lc}, Message: {message}")
    results.append(f"   Expected: False, <message>")
    results.append(f"   Test passed: {is_lc == False}")

    return "\n".join(results)

def main():
    # Run all tests
    test_results = []
    test_results.append(test_combinacion_lineal())
    test_results.append(test_es_combinacion_lineal())
    test_results.append("\n=== All tests completed ===")
    
    # Write results to file
    with open("test_results.txt", "w") as f:
        f.write("\n".join(test_results))
    
    # Also print to console
    print("\n".join(test_results))

if __name__ == "__main__":
    main()
