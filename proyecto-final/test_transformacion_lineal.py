"""
Script para probar la interfaz de los métodos 'transformacion_lineal' y 'visualizar_transformacion_lineal'
"""

from AlgebraLineal import AlgebraLineal
import numpy as np

def test_transformacion_lineal():
    """Prueba el método transformacion_lineal"""
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
    
    # Debería ser [[0, -1], [1, 0]]
    
    # Visualizar la transformación
    AlgebraLineal.visualizar_transformacion_lineal(matriz, "Rotación 90° Antihorario")

def test_visualizar_transformacion_lineal_2d():
    """Prueba el método visualizar_transformacion_lineal con una matriz 2x2"""
    # Matriz de transformación para una rotación de 45 grados
    angulo = np.pi/4  # 45 grados en radianes
    matriz = [
        [np.cos(angulo), -np.sin(angulo)],
        [np.sin(angulo), np.cos(angulo)]
    ]
    
    AlgebraLineal.visualizar_transformacion_lineal(matriz, "Rotación 45°")

def test_visualizar_transformacion_lineal_3d():
    """Prueba el método visualizar_transformacion_lineal con una matriz 3x3"""
    # Matriz de transformación para una rotación alrededor del eje Z
    angulo = np.pi/4  # 45 grados en radianes
    matriz = [
        [np.cos(angulo), -np.sin(angulo), 0],
        [np.sin(angulo), np.cos(angulo), 0],
        [0, 0, 1]
    ]
    
    AlgebraLineal.visualizar_transformacion_lineal(matriz, "Rotación 3D alrededor del eje Z")

if __name__ == "__main__":
    print("Ejecutando pruebas de transformación lineal...")
    
    # Descomentar la prueba que quieras ejecutar
    # test_transformacion_lineal()
    # test_visualizar_transformacion_lineal_2d()
    test_visualizar_transformacion_lineal_3d()
