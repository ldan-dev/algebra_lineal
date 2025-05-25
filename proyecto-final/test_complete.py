"""
Script completo para probar los métodos 'transformacion_lineal' y 'visualizar_transformacion_lineal'
tanto en casos 2D como 3D.
"""

from AlgebraLineal import AlgebraLineal
import numpy as np

def test_transformacion_lineal_2d():
    """Prueba el método transformacion_lineal con bases 2D"""
    print("=== Prueba de transformación lineal 2D ===")
    
    # Definir la base del dominio (R²)
    base_dominio = [
        [1, 0],  # Vector (1,0)
        [0, 1]   # Vector (0,1)
    ]
    
    # Definir la base del codominio - Matriz de rotación 90° antihorario
    base_codominio = [
        [0, -1],  # Imagen de (1,0) -> (0,-1)
        [1, 0]    # Imagen de (0,1) -> (1,0)
    ]
    
    # Calcular la matriz de transformación
    matriz = AlgebraLineal.transformacion_lineal(base_dominio, base_codominio)
    print("Matriz de transformación calculada:")
    for fila in matriz:
        print(fila)
    
    # Visualizar la transformación
    AlgebraLineal.visualizar_transformacion_lineal(matriz, "Rotación 90° Antihorario")
    
    # Verificar la transformación aplicándola a un vector
    vector_original = [2, 3]  # Vector (2,3)
    vector_transformado = []
    for i in range(len(matriz)):
        valor = 0
        for j in range(len(vector_original)):
            valor += matriz[i][j] * vector_original[j]
        vector_transformado.append(valor)
    
    print(f"Vector original: {vector_original}")
    print(f"Vector transformado: {vector_transformado}")
    # Debería ser [-3, 2]

def test_transformacion_lineal_3d():
    """Prueba el método transformacion_lineal con bases 3D"""
    print("\n=== Prueba de transformación lineal 3D ===")
    
    # Definir la base del dominio (R³)
    base_dominio = [
        [1, 0, 0],  # Vector (1,0,0)
        [0, 1, 0],  # Vector (0,1,0)
        [0, 0, 1]   # Vector (0,0,1)
    ]
    
    # Definir la base del codominio - Reflexión sobre el plano xy
    base_codominio = [
        [1, 0, 0],    # Imagen de (1,0,0) -> (1,0,0)
        [0, 1, 0],    # Imagen de (0,1,0) -> (0,1,0)
        [0, 0, -1]    # Imagen de (0,0,1) -> (0,0,-1)
    ]
    
    # Calcular la matriz de transformación
    matriz = AlgebraLineal.transformacion_lineal(base_dominio, base_codominio)
    print("Matriz de transformación calculada:")
    for fila in matriz:
        print(fila)
    
    # Visualizar la transformación
    AlgebraLineal.visualizar_transformacion_lineal(matriz, "Reflexión sobre el plano XY")

def test_visualizar_transformacion_lineal_2d():
    """Prueba el método visualizar_transformacion_lineal con diferentes matrices 2D"""
    print("\n=== Prueba de visualización de transformaciones 2D ===")
    
    # 1. Rotación
    print("\n1. Rotación de 45 grados")
    angulo = np.pi/4  # 45 grados en radianes
    matriz_rotacion = [
        [np.cos(angulo), -np.sin(angulo)],
        [np.sin(angulo), np.cos(angulo)]
    ]
    AlgebraLineal.visualizar_transformacion_lineal(matriz_rotacion, "Rotación 45°")
    
    # 2. Escalamiento
    print("\n2. Escalamiento (2x, 0.5y)")
    matriz_escalamiento = [
        [2, 0],
        [0, 0.5]
    ]
    AlgebraLineal.visualizar_transformacion_lineal(matriz_escalamiento, "Escalamiento (2x, 0.5y)")
    
    # 3. Cizalladura
    print("\n3. Cizalladura Horizontal")
    matriz_cizalladura = [
        [1, 1],
        [0, 1]
    ]
    AlgebraLineal.visualizar_transformacion_lineal(matriz_cizalladura, "Cizalladura Horizontal")
    
    # 4. Proyección
    print("\n4. Proyección sobre el eje X")
    matriz_proyeccion = [
        [1, 0],
        [0, 0]
    ]
    AlgebraLineal.visualizar_transformacion_lineal(matriz_proyeccion, "Proyección sobre el eje X")

def test_visualizar_transformacion_lineal_3d():
    """Prueba el método visualizar_transformacion_lineal con diferentes matrices 3D"""
    print("\n=== Prueba de visualización de transformaciones 3D ===")
    
    # 1. Rotación alrededor del eje Z
    print("\n1. Rotación alrededor del eje Z")
    angulo = np.pi/4  # 45 grados en radianes
    matriz_rotacion_z = [
        [np.cos(angulo), -np.sin(angulo), 0],
        [np.sin(angulo), np.cos(angulo), 0],
        [0, 0, 1]
    ]
    AlgebraLineal.visualizar_transformacion_lineal(matriz_rotacion_z, "Rotación 3D alrededor del eje Z")
    
    # 2. Escalamiento 3D
    print("\n2. Escalamiento 3D (2x, 2y, 0.5z)")
    matriz_escalamiento_3d = [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 0.5]
    ]
    AlgebraLineal.visualizar_transformacion_lineal(matriz_escalamiento_3d, "Escalamiento 3D (2x, 2y, 0.5z)")
    
    # 3. Reflexión sobre el plano XY
    print("\n3. Reflexión sobre el plano XY")
    matriz_reflexion_xy = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ]
    AlgebraLineal.visualizar_transformacion_lineal(matriz_reflexion_xy, "Reflexión sobre el plano XY")

if __name__ == "__main__":
    print("Ejecutando pruebas completas de transformación lineal...\n")
    
    # Descomentar las pruebas que quieras ejecutar
    test_transformacion_lineal_2d()
    # test_transformacion_lineal_3d()
    # test_visualizar_transformacion_lineal_2d()
    # test_visualizar_transformacion_lineal_3d()
    
    print("\nPruebas completas ejecutadas con éxito.")
