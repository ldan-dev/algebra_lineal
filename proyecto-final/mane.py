NV = 100

while True:
    try:
        n = int(input("Ingrese el numero de variables: "))
        if 1 <= n <= NV:
            break
        else:
            print(f"Por favor ingrese un número entre 1 y {NV}.")
    except ValueError:
        print("Entrada inválida. Por favor ingrese un número entero.")

A = [[0.0] * n for _ in range(n)]
b = [0.0] * n
x = [0.0] * n

for i in range(n):
    for j in range(n):
        A[i][j] = float(input(f"A[{i+1}][{j+1}] = "))
    b[i] = float(input(f"b[{i+1}] = "))

print("---")
for i in range(n):
    for j in range(n):
        print(f"{A[i][j]:.2f}\t", end="")
    print(f"{b[i]:.2f}")

for k in range(n):
    max_val = abs(A[k][k])
    max_row = k
    for i in range(k + 1, n):
        if abs(A[i][k]) > max_val:
            max_val = abs(A[i][k])
            max_row = i

    if max_row != k:
        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]

    if A[k][k] == 0:
        print(f"Error: El sistema no tiene solución única o el pivote en A[{k+1}][{k+1}] es cero después del pivoteo. La eliminación gaussiana no puede continuar.")
        exit()

    for i in range(k + 1, n):
        fct = A[i][k] / A[k][k]
        for j in range(k, n):
            A[i][j] -= (fct * A[k][j])
        b[i] -= fct * b[k]

    print(f"--- (Después de la eliminación de la columna {k+1})")
    for i_print in range(n):
        for j_print in range(n):
            print(f"{A[i_print][j_print]:.2f}\t", end="")
        print(f"{b[i_print]:.2f}")

for i in range(n - 1, -1, -1):
    x[i] = b[i]
    for j in range(i + 1, n):
        x[i] -= A[i][j] * x[j]

    if A[i][i] == 0:
        print(f"Error: División por cero en A[{i+1}][{i+1}]. El sistema no tiene solución única o es singular.")
        exit()

    x[i] /= A[i][i]

print("--- Soluciones ---")
for i in range(n):
    print(f"x[{i+1}] = {x[i]:.6f}")