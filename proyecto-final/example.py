"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 28/04/2025 (dd/mm/aaaa)
CARRERA: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Correo: ld.avinaneri@ugto.mx
UDA: Álgebra Lineal
DESCRIPCION: Ejemplo de uso de las funciones del módulo AlgebraLineal.py
"""
from AlgebraLineal import AlgebraLineal as AL
import math


def main():
    """Ejemplo de uso de todas las funciones de AlgebraLineal.py"""
    
    print("===== EJEMPLOS DE ÁLGEBRA LINEAL =====\n")
    
    # Crear matrices para los ejemplos
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    v = [1, 2, 3]
    
    print("Matriz A:")
    for fila in A:
        print(fila)
    print("\nMatriz B:")
    for fila in B:
        print(fila)
    print("\nVector v:")
    print(v)
    
    # Operaciones con matrices
    print("\n1. Suma de matrices:")
    result = AL.suma_matrices(A, B)
    for fila in result:
        print(fila)
    
    print("\n2. Resta de matrices:")
    result = AL.resta_matrices(A, B)
    for fila in result:
        print(fila)
    
    print("\n3. Multiplicación de matrices:")
    result = AL.mult_matrices(A, B)
    for fila in result:
        print(fila)
    
    print("\n4. Determinante:")
    result = AL.determinante(A)
    print(f"Determinante de A: {result}")
    
    print("\n5. Transpuesta:")
    result = AL.transpuesta(A)
    for fila in result:
        print(fila)
    
    print("\n6. Matriz inversa:")
    C = [[1, 2], [3, 4]]  # Usar matriz invertible
    print("Matriz C:")
    for fila in C:
        print(fila)
    
    result = AL.inversa(C)
    print("Inversa de C:")
    for fila in result:
        print(fila)
    
    print("\n7. Producto matriz-vector:")
    # Convertimos a matriz-vector mediante multiplicación de matrices
    v_col = [[v[i]] for i in range(len(v))]
    result = AL.mult_matrices(A, v_col)
    print("Como columna:")
    for fila in result:
        print(fila)
    
    print("\n8. Resolver sistema de ecuaciones:")
    b = [6, 15, 24]
    print("Vector b:", b)
    result, tipo = AL.resolver_sistema(A, b)
    print(f"Solución del sistema Ax = b (tipo: {tipo}):")
    print(result)
    
    print("\n9. Norma de un vector:")
    result = AL.norma(v)
    print(f"Norma del vector v: {result}")
    
    print("\n10. Producto Escalar:")
    result = AL.producto_escalar(v, [4, 5, 6])
    print(f"Producto escalar de v y [4, 5, 6]: {result}")
    
    print("\n11. Producto Vectorial:")
    result = AL.producto_vectorial(v, [4, 5, 6])
    print(f"Producto vectorial de v y [4, 5, 6]: {result}")
    
    print("\n12. Independencia Lineal:")
    vectores = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    print("Vectores:")
    for vec in vectores:
        print(vec)
    es_li, justificacion = AL.es_linealmente_independiente(vectores)
    print(f"¿Son linealmente independientes?: {es_li}")
    print(f"Justificación: {justificacion}")
    
    print("\n13. Combinación Lineal:")
    vector_combinacion = AL.combinacion_lineal(vectores, [2, 3, 4])
    print(f"Combinación lineal con coeficientes [2, 3, 4]: {vector_combinacion}")
    
    print("\n14. Proyección de vectores:")
    v1 = [3, 4, 0]
    v2 = [0, 1, 0]
    result = AL.proyeccion(v1, v2)
    print(f"Proyección de {v1} sobre {v2}: {result}")
    
    print("\n15. Ángulo entre vectores:")
    angle = AL.angulo_entre_vectores(v1, v2)
    print(f"Ángulo entre {v1} y {v2}: {angle} radianes")
    print(f"Ángulo en grados: {math.degrees(angle)} grados")
    
    print("\n16. Visualización:")
    print("Mostrando gráficas en ventanas separadas...")
    
    # Graficar función
    AL.graficar_funcion(lambda x: x**2, -5, 5, titulo="Parábola: f(x) = x^2")
    
    # Graficar vectores 2D
    AL.graficar_vectores([[1, 0], [0, 1], [1, 1]], 
                        etiquetas=["Vector i", "Vector j", "Vector i+j"],
                        titulo="Vectores en 2D")
    
    # Graficar vectores 3D si hay tiempo/interés
    AL.graficar_vectores([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                        etiquetas=["Vector i", "Vector j", "Vector k", "Vector i+j+k"],
                        titulo="Vectores en 3D")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()