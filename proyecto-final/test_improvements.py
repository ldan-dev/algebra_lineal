"""
Test para las mejoras implementadas en la clase AlgebraLineal
"""
from AlgebraLineal import AlgebraLineal as AL
import time

def test_producto_vectorial():
    """
    Prueba el método producto_vectorial para diferentes dimensiones
    """
    print("\n=== Test de producto_vectorial ===")
    
    # Caso 2D
    v1_2d = [2, 3]
    v2_2d = [4, 5]
    result_2d = AL.producto_vectorial(v1_2d, v2_2d)
    print(f"Producto vectorial en 2D: {v1_2d} × {v2_2d} = {result_2d}")
    
    # Caso 3D con vectores conocidos
    v1_3d = [1, 0, 0]  # vector i
    v2_3d = [0, 1, 0]  # vector j
    result_3d = AL.producto_vectorial(v1_3d, v2_3d)
    print(f"Producto vectorial en 3D: {v1_3d} × {v2_3d} = {result_3d}")  # Debería ser [0, 0, 1] (vector k)
    
    # Caso 4D
    v1_4d = [1, 2, 3, 4]
    v2_4d = [5, 6, 7, 8]
    result_4d = AL.producto_vectorial(v1_4d, v2_4d)
    print(f"Producto vectorial en 4D: {v1_4d} × {v2_4d} = {result_4d}")
    
    # Caso 5D
    v1_5d = [1, 2, 3, 4, 5]
    v2_5d = [6, 7, 8, 9, 10]
    result_5d = AL.producto_vectorial(v1_5d, v2_5d)
    print(f"Producto vectorial en 5D: {v1_5d} × {v2_5d} = {result_5d}")

def test_determinante():
    """
    Prueba el rendimiento del método determinante optimizado
    """
    print("\n=== Test de rendimiento de determinante ===")
    
    # Matrices pequeñas
    matriz_2x2 = [[1, 2], [3, 4]]
    det_2x2 = AL.determinante(matriz_2x2)
    print(f"Determinante matriz 2x2: {det_2x2}")
    
    matriz_3x3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    det_3x3 = AL.determinante(matriz_3x3)
    print(f"Determinante matriz 3x3: {det_3x3}")
    
    # Matrices medianas
    n = 5
    matriz_5x5 = [[i+j for j in range(n)] for i in range(n)]
    start_time = time.time()
    det_5x5 = AL.determinante(matriz_5x5)
    tiempo_5x5 = time.time() - start_time
    print(f"Determinante matriz 5x5: {det_5x5}, tiempo: {tiempo_5x5:.6f} segundos")
    
    # Matrices grandes
    n = 10
    matriz_10x10 = [[i+j for j in range(n)] for i in range(n)]
    start_time = time.time()
    det_10x10 = AL.determinante(matriz_10x10)
    tiempo_10x10 = time.time() - start_time
    print(f"Determinante matriz 10x10: {det_10x10}, tiempo: {tiempo_10x10:.6f} segundos")
    
    n = 15
    matriz_15x15 = [[i+j for j in range(n)] for i in range(n)]
    start_time = time.time()
    det_15x15 = AL.determinante(matriz_15x15)
    tiempo_15x15 = time.time() - start_time
    print(f"Determinante matriz 15x15: {det_15x15}, tiempo: {tiempo_15x15:.6f} segundos")

def test_tolerancia():
    """
    Prueba que todas las comparaciones usen la misma constante de tolerancia
    """
    print("\n=== Test de tolerancia ===")
    
    # Crear vectores casi paralelos
    v1 = [1, 0, 0]
    v2 = [1, 1e-11, 0]  # Vector que difiere en menos que TOLERANCIA
    
    print(f"Constante de tolerancia: {AL.TOLERANCIA}")
    
    # Probar producto escalar
    dot = AL.producto_escalar(v1, v2)
    print(f"Producto escalar: {dot}")
    
    # Probar ángulo entre vectores
    angle = AL.angulo_entre_vectores(v1, v2)
    print(f"Ángulo (radianes): {angle}")
    
    # Crear matriz casi singular
    matriz_casi_singular = [
        [1, 1, 1],
        [1, 1, 1 + 1e-11],
        [1, 1 + 1e-11, 1]
    ]
    
    det = AL.determinante(matriz_casi_singular)
    print(f"Determinante de matriz casi singular: {det}")
    
    # Probar independencia lineal de vectores casi dependientes
    vectores = [[1, 0], [1, 1e-11]]
    es_li, justificacion = AL.es_linealmente_independiente(vectores)
    print(f"¿Vectores casi dependientes son LI?: {es_li}")
    print(f"Justificación: {justificacion}")

def test_rank_linearly_independent():
    """
    Prueba del cálculo mejorado de rango para independencia lineal
    """
    print("\n=== Test de independencia lineal ===")
    
    # Conjunto linealmente independiente
    vectores_li = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    es_li, justificacion = AL.es_linealmente_independiente(vectores_li)
    print(f"¿Vectores base canónica son LI?: {es_li}")
    print(f"Justificación: {justificacion}")
    
    # Conjunto linealmente dependiente
    vectores_ld = [[1, 0, 0], [0, 1, 0], [1, 1, 0]]
    es_li, justificacion = AL.es_linealmente_independiente(vectores_ld)
    print(f"¿Vectores dependientes son LI?: {es_li}")
    print(f"Justificación: {justificacion}")
    
    # Probar caso borde: más vectores que dimensión
    vectores_muchos = [[1, 0], [0, 1], [1, 1], [2, 2]]
    es_li, justificacion = AL.es_linealmente_independiente(vectores_muchos)
    print(f"¿Conjunto con más vectores que dimensión es LI?: {es_li}")
    print(f"Justificación: {justificacion}")

def main():
    print("PRUEBAS DE LAS MEJORAS EN ALGEBRALINEAL")
    
    test_producto_vectorial()
    test_determinante()
    test_tolerancia()
    test_rank_linearly_independent()
    
    print("\nTodas las pruebas completadas.")

if __name__ == "__main__":
    main()
