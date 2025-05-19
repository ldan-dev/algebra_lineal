"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 28/04/2025 (dd/mm/aaaa)
CARRERA: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
UDA: Álgebra Lineal
DESCRIPCION: Implementación de una clase de Álgebra Lineal que proporciona
             operaciones comunes como productos escalares y vectoriales,
             operaciones matriciales, solución de sistemas de ecuaciones,
             análisis de independencia lineal, cálculo de determinantes, y
             visualización de vectores y funciones.
"""

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class AlgebraLineal:
    """
    Clase que proporciona operaciones de álgebra lineal implementadas desde cero
    sin utilizar bibliotecas como NumPy o similares.
    
    Esta clase incluye:
    - Operaciones con vectores (producto escalar, producto vectorial, etc.)
    - Operaciones con matrices (suma, multiplicación, inversa, etc.)
    - Visualización de funciones y vectores
    - Resolución de sistemas de ecuaciones lineales
    - Análisis de combinaciones lineales e independencia lineal
    - Cálculo de determinantes
    """
    
    # ======================== OPERACIONES CON VECTORES ========================
    @staticmethod
    def producto_escalar(v1, v2):
        """
        Calcula el producto escalar (producto punto) entre dos vectores.
        
        Args:
            v1 (list): Primer vector
            v2 (list): Segundo vector con la misma dimensión que v1
            
        Returns:
            float: Resultado del producto escalar
            
        Raises:
            ValueError: Si los vectores tienen dimensiones diferentes
            
        Example:
            resultado = AlgebraLineal.producto_escalar([1, 2, 3], [4, 5, 6])
            # Retorna: 32 (1*4 + 2*5 + 3*6)
        """
        if len(v1) != len(v2):
            raise ValueError("Los vectores deben tener la misma dimensión")
        
        resultado = 0
        for i in range(len(v1)):
            resultado += v1[i] * v2[i]
        return resultado
    @staticmethod
    def producto_vectorial(v1, v2):
        """
        Calcula el producto vectorial (producto cruz) entre dos vectores. 
        Funciona para vectores de cualquier dimensión, aunque el producto vectorial
        está únicamente definido de forma estándar para vectores en R³.
        
        - Para vectores 3D (R³): Implementa el producto vectorial estándar.
        - Para vectores 2D (R²): Los trata como vectores 3D con componente z=0, 
          retornando un vector en R³ perpendicular al plano que contiene v1 y v2.
        - Para vectores de dimensión n>3: Implementa una generalización del producto
          vectorial usando determinantes de submatrices.
        
        Args:
            v1 (list): Primer vector
            v2 (list): Segundo vector con la misma dimensión que v1
            
        Returns:
            list: Resultado del producto vectorial
            
        Raises:
            ValueError: Si los vectores tienen dimensiones diferentes
            
        Example:
            resultado = AlgebraLineal.producto_vectorial([1, 2, 3], [4, 5, 6])
            # Retorna: [-3, 6, -3]
        """
        if len(v1) != len(v2):
            raise ValueError("Los vectores deben tener la misma dimensión")
        
        dimension = len(v1)
        
        # Caso especial para vectores de dimensión 2
        if dimension == 2:
            # Tratar como vectores 3D con z=0
            v1_3d = v1 + [0]
            v2_3d = v2 + [0]
            return AlgebraLineal.producto_vectorial(v1_3d, v2_3d)
        
        # Caso para vectores 3D (fórmula estándar)
        if dimension == 3:
            return [
                v1[1] * v2[2] - v1[2] * v2[1],  # i componente
                v1[2] * v2[0] - v1[0] * v2[2],  # j componente
                v1[0] * v2[1] - v1[1] * v2[0]   # k componente
            ]
        
        # Caso general para n dimensiones
        if dimension < 2:
            raise ValueError("El producto vectorial requiere vectores de al menos dimensión 2")
        
        # Para n > 3, utilizamos una generalización basada en determinantes
        resultado = []
        base_estandar = AlgebraLineal.crear_matriz_identidad(dimension)
        
        for i in range(dimension):
            # Crear una matriz donde la primera fila es el i-ésimo vector de la base estándar
            # la segunda fila es v1, y la tercera es v2
            matriz = [base_estandar[i], v1, v2]
            
            # Calcular el determinante de esta matriz 3xn expandiendo por la primera fila
            det = 0
            signo = 1
            
            # Para cada elemento del vector base, calculamos su menor y lo multiplicamos
            for j in range(dimension):
                # Construir submatriz excluyendo fila 0 y columna j
                submatriz = []
                for fila in range(1, 3):  # Solo filas 1 y 2 (v1 y v2)
                    submatriz_fila = []
                    for col in range(dimension):
                        if col != j:
                            submatriz_fila.append(matriz[fila][col])
                    submatriz.append(submatriz_fila)
                
                # Si el vector base tiene un 1 en posición j, contribuye al determinante
                if matriz[0][j] == 1:
                    # Determinante de matriz 2x2
                    det_submatriz = 0
                    for k in range(dimension - 1):
                        for l in range(dimension - 1):
                            if k != l:  # Solo consideramos elementos fuera de la diagonal
                                det_submatriz += submatriz[0][k] * submatriz[1][l] * ((-1) ** (k + l))
                    
                    det += signo * det_submatriz
                
                signo = -signo
            
            resultado.append(det)
        
        return resultado
    
    @staticmethod
    def suma_vectores(v1, v2):
        """
        Suma dos vectores elemento a elemento.
        
        Args:
            v1 (list): Primer vector
            v2 (list): Segundo vector con la misma dimensión que v1
            
        Returns:
            list: Vector resultante de la suma
            
        Raises:
            ValueError: Si los vectores tienen dimensiones diferentes
            
        Example:
            resultado = AlgebraLineal.suma_vectores([1, 2, 3], [4, 5, 6])
            # Retorna: [5, 7, 9]
        """
        if len(v1) != len(v2):
            raise ValueError("Los vectores deben tener la misma dimensión")
        
        resultado = []
        for i in range(len(v1)):
            resultado.append(v1[i] + v2[i])
        return resultado
    
    @staticmethod
    def resta_vectores(v1, v2):
        """
        Resta el segundo vector del primero elemento a elemento.
        
        Args:
            v1 (list): Primer vector
            v2 (list): Segundo vector con la misma dimensión que v1
            
        Returns:
            list: Vector resultante de la resta
            
        Raises:
            ValueError: Si los vectores tienen dimensiones diferentes
            
        Example:
            resultado = AlgebraLineal.resta_vectores([5, 7, 9], [1, 2, 3])
            # Retorna: [4, 5, 6]
        """
        if len(v1) != len(v2):
            raise ValueError("Los vectores deben tener la misma dimensión")
        
        resultado = []
        for i in range(len(v1)):
            resultado.append(v1[i] - v2[i])
        return resultado
    
    @staticmethod
    def escalar_por_vector(escalar, vector):
        """
        Multiplica un vector por un escalar.
        
        Args:
            escalar (float): Valor escalar para multiplicar
            vector (list): Vector a multiplicar
            
        Returns:
            list: Vector resultante de la multiplicación
            
        Example:
            resultado = AlgebraLineal.escalar_por_vector(2, [1, 2, 3])
            # Retorna: [2, 4, 6]
        """
        return [escalar * componente for componente in vector]
    
    @staticmethod
    def norma(vector):
        """
        Calcula la norma (magnitud) de un vector.
        
        Args:
            vector (list): Vector de entrada
            
        Returns:
            float: Norma del vector
            
        Example:
            resultado = AlgebraLineal.norma([3, 4])
            # Retorna: 5.0
        """
        return math.sqrt(sum(x * x for x in vector))
    
    @staticmethod
    def normalizar(vector):
        """
        Normaliza un vector (convierte a un vector unitario).
        
        Args:
            vector (list): Vector a normalizar
            
        Returns:
            list: Vector normalizado
            
        Raises:
            ValueError: Si el vector es el vector cero
            
        Example:
            resultado = AlgebraLineal.normalizar([3, 0, 0])
            # Retorna: [1, 0, 0]
        """
        norma = AlgebraLineal.norma(vector)
        if norma == 0:
            raise ValueError("No se puede normalizar el vector cero")
        
        return [componente / norma for componente in vector]
    
    @staticmethod
    def angulo_entre_vectores(v1, v2):
        """
        Calcula el ángulo entre dos vectores en radianes.
        
        Args:
            v1 (list): Primer vector
            v2 (list): Segundo vector con la misma dimensión que v1
            
        Returns:
            float: Ángulo en radianes
            
        Raises:
            ValueError: Si los vectores tienen dimensiones diferentes o si alguno es el vector cero
            
        Example:
            resultado = AlgebraLineal.angulo_entre_vectores([1, 0], [0, 1])
            # Retorna: pi/2 (90 grados en radianes)
        """
        if len(v1) != len(v2):
            raise ValueError("Los vectores deben tener la misma dimensión")
        
        norma_v1 = AlgebraLineal.norma(v1)
        norma_v2 = AlgebraLineal.norma(v2)
        
        if norma_v1 == 0 or norma_v2 == 0:
            raise ValueError("El ángulo no está definido para el vector cero")
        
        producto = AlgebraLineal.producto_escalar(v1, v2)
        cos_angulo = producto / (norma_v1 * norma_v2)
        
        # Corregir errores de precisión que pueden ocurrir
        if cos_angulo > 1:
            cos_angulo = 1
        elif cos_angulo < -1:
            cos_angulo = -1
            
        return math.acos(cos_angulo)
    
    @staticmethod
    def proyeccion(v1, v2):
        """
        Calcula la proyección del vector v1 sobre el vector v2.
        
        Args:
            v1 (list): Vector a proyectar
            v2 (list): Vector sobre el cual proyectar
            
        Returns:
            list: Vector proyección
            
        Raises:
            ValueError: Si v2 es el vector cero o si los vectores tienen dimensiones diferentes
            
        Example:
            resultado = AlgebraLineal.proyeccion([3, 3], [0, 1])
            # Retorna: [0, 3]
        """
        if len(v1) != len(v2):
            raise ValueError("Los vectores deben tener la misma dimensión")
        
        norma_v2_cuadrado = sum(x * x for x in v2)
        if norma_v2_cuadrado == 0:
            raise ValueError("No se puede proyectar sobre el vector cero")
        
        escalar = AlgebraLineal.producto_escalar(v1, v2) / norma_v2_cuadrado
        return AlgebraLineal.escalar_por_vector(escalar, v2)
    
    # ======================== OPERACIONES CON MATRICES ========================
    
    @staticmethod
    def crear_matriz(filas, columnas, valor_inicial=0):
        """
        Crea una matriz con dimensiones dadas y un valor inicial.
        
        Args:
            filas (int): Número de filas
            columnas (int): Número de columnas
            valor_inicial (float, optional): Valor inicial para todos los elementos. Por defecto es 0.
            
        Returns:
            list: Matriz creada
            
        Example:
            matriz = AlgebraLineal.crear_matriz(2, 3, 1)
            # Retorna: [[1, 1, 1], [1, 1, 1]]
        """
        return [[valor_inicial for _ in range(columnas)] for _ in range(filas)]
    
    @staticmethod
    def crear_matriz_identidad(n):
        """
        Crea una matriz identidad de tamaño n x n.
        
        Args:
            n (int): Tamaño de la matriz
            
        Returns:
            list: Matriz identidad
            
        Example:
            matriz = AlgebraLineal.crear_matriz_identidad(3)
            # Retorna: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        """
        identidad = AlgebraLineal.crear_matriz(n, n)
        for i in range(n):
            identidad[i][i] = 1
        return identidad
    
    @staticmethod
    def suma_matrices(m1, m2):
        """
        Suma dos matrices elemento a elemento.
        
        Args:
            m1 (list): Primera matriz
            m2 (list): Segunda matriz con las mismas dimensiones que m1
            
        Returns:
            list: Matriz resultante de la suma
            
        Raises:
            ValueError: Si las matrices tienen dimensiones diferentes
            
        Example:
            resultado = AlgebraLineal.suma_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]])
            # Retorna: [[6, 8], [10, 12]]
        """
        # Comprobar si las matrices tienen las mismas dimensiones
        if len(m1) != len(m2):
            raise ValueError("Las matrices deben tener las mismas dimensiones")
        
        for i in range(len(m1)):
            if len(m1[i]) != len(m2[i]):
                raise ValueError("Las matrices deben tener las mismas dimensiones")
        
        filas = len(m1)
        columnas = len(m1[0])
        resultado = AlgebraLineal.crear_matriz(filas, columnas)
        
        for i in range(filas):
            for j in range(columnas):
                resultado[i][j] = m1[i][j] + m2[i][j]
                
        return resultado
    
    
    @staticmethod
    def resta_matrices(m1, m2):
        """
        Resta la segunda matriz de la primera elemento a elemento.
        
        Args:
            m1 (list): Primera matriz
            m2 (list): Segunda matriz con las mismas dimensiones que m1
            
        Returns:
            list: Matriz resultante de la resta
            
        Raises:
            ValueError: Si las matrices tienen dimensiones diferentes
            
        Example:
            resultado = AlgebraLineal.resta_matrices([[6, 8], [10, 12]], [[1, 2], [3, 4]])
            # Retorna: [[5, 6], [7, 8]]
        """
        # Comprobar si las matrices tienen las mismas dimensiones
        if len(m1) != len(m2):
            raise ValueError("Las matrices deben tener las mismas dimensiones")
        
        for i in range(len(m1)):
            if len(m1[i]) != len(m2[i]):
                raise ValueError("Las matrices deben tener las mismas dimensiones")
        
        filas = len(m1)
        columnas = len(m1[0])
        resultado = AlgebraLineal.crear_matriz(filas, columnas)
        
        for i in range(filas):
            for j in range(columnas):
                resultado[i][j] = m1[i][j] - m2[i][j]
                
        return resultado
    
    @staticmethod
    def escalar_por_matriz(escalar, matriz):
        """
        Multiplica una matriz por un escalar.
        
        Args:
            escalar (float): Valor escalar para multiplicar
            matriz (list): Matriz a multiplicar
            
        Returns:
            list: Matriz resultante de la multiplicación
            
        Example:
            resultado = AlgebraLineal.escalar_por_matriz(2, [[1, 2], [3, 4]])
            # Retorna: [[2, 4], [6, 8]]
        """
        filas = len(matriz)
        columnas = len(matriz[0])
        resultado = AlgebraLineal.crear_matriz(filas, columnas)
        
        for i in range(filas):
            for j in range(columnas):
                resultado[i][j] = escalar * matriz[i][j]
                
        return resultado
    
    @staticmethod
    def mult_matrices(m1, m2):
        """
        Multiplica dos matrices.
        
        Args:
            m1 (list): Primera matriz
            m2 (list): Segunda matriz donde el número de columnas de m1 igual al número de filas de m2
            
        Returns:
            list: Matriz resultante de la multiplicación
            
        Raises:
            ValueError: Si las dimensiones no son compatibles para la multiplicación
            
        Example:
            resultado = AlgebraLineal.mult_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]])
            # Retorna: [[19, 22], [43, 50]]
        """
        filas_m1 = len(m1)
        columnas_m1 = len(m1[0])
        filas_m2 = len(m2)
        
        if columnas_m1 != filas_m2:
            raise ValueError("Las dimensiones de las matrices no son compatibles para la multiplicación")
        
        columnas_m2 = len(m2[0])
        resultado = AlgebraLineal.crear_matriz(filas_m1, columnas_m2)
        
        for i in range(filas_m1):
            for j in range(columnas_m2):
                for k in range(columnas_m1):
                    resultado[i][j] += m1[i][k] * m2[k][j]
                    
        return resultado
    
    @staticmethod
    def transpuesta(matriz):
        """
        Calcula la transpuesta de una matriz.
        
        Args:
            matriz (list): Matriz de entrada
            
        Returns:
            list: Matriz transpuesta
            
        Example:
            resultado = AlgebraLineal.transpuesta([[1, 2, 3], [4, 5, 6]])
            # Retorna: [[1, 4], [2, 5], [3, 6]]
        """
        filas = len(matriz)
        columnas = len(matriz[0])
        
        transpuesta = AlgebraLineal.crear_matriz(columnas, filas)
        
        for i in range(filas):
            for j in range(columnas):
                transpuesta[j][i] = matriz[i][j]
                
        return transpuesta
    
    @staticmethod
    def submatriz(matriz, fila_excluida, columna_excluida):
        """
        Crea una submatriz excluyendo una fila y una columna específicas.
        
        Args:
            matriz (list): Matriz original
            fila_excluida (int): Índice de la fila a excluir
            columna_excluida (int): Índice de la columna a excluir
            
        Returns:
            list: Submatriz resultante
            
        Example:
            resultado = AlgebraLineal.submatriz([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 0, 0)
            # Retorna: [[5, 6], [8, 9]]
        """
        return [[matriz[i][j] for j in range(len(matriz[0])) if j != columna_excluida]
                for i in range(len(matriz)) if i != fila_excluida]
    
    @staticmethod
    def determinante(matriz):
        """
        Calcula el determinante de una matriz usando expansión por cofactores.
        
        Args:
            matriz (list): Matriz cuadrada
            
        Returns:
            float: Determinante de la matriz
            
        Raises:
            ValueError: Si la matriz no es cuadrada
            
        Example:
            resultado = AlgebraLineal.determinante([[1, 2], [3, 4]])
            # Retorna: -2
        """
        filas = len(matriz)
        
        # Comprobar si es una matriz cuadrada
        if any(len(fila) != filas for fila in matriz):
            raise ValueError("La matriz debe ser cuadrada para calcular su determinante")
        
        # Caso base: matriz 1x1
        if filas == 1:
            return matriz[0][0]
        
        # Caso base: matriz 2x2
        if filas == 2:
            return matriz[0][0] * matriz[1][1] - matriz[0][1] * matriz[1][0]
        
        # Expansión por cofactores a lo largo de la primera fila
        det = 0
        for j in range(filas):
            cofactor = matriz[0][j] * AlgebraLineal.determinante(AlgebraLineal.submatriz(matriz, 0, j))
            det += cofactor if j % 2 == 0 else -cofactor
            
        return det
    
    @staticmethod
    def inversa(matriz):
        """
        Calcula la matriz inversa usando el método de la matriz adjunta.
        
        Args:
            matriz (list): Matriz cuadrada
            
        Returns:
            list: Matriz inversa
            
        Raises:
            ValueError: Si la matriz no es cuadrada o si su determinante es cero
            
        Example:
            resultado = AlgebraLineal.inversa([[1, 2], [3, 4]])
            # Retorna aproximadamente: [[-2, 1], [1.5, -0.5]]
        """
        filas = len(matriz)
        
        # Comprobar si es una matriz cuadrada
        if any(len(fila) != filas for fila in matriz):
            raise ValueError("La matriz debe ser cuadrada para calcular su inversa")
        
        # Calcular el determinante
        det = AlgebraLineal.determinante(matriz)
        
        if abs(det) < 1e-10:  # Usar una tolerancia pequeña para evitar problemas de precisión
            raise ValueError("La matriz no es invertible (determinante = 0)")
        
        # Para una matriz 1x1, la inversa es trivial
        if filas == 1:
            return [[1 / matriz[0][0]]]
        
        # Calcular la matriz de cofactores
        cofactores = AlgebraLineal.crear_matriz(filas, filas)
        for i in range(filas):
            for j in range(filas):
                # Calcular el cofactor
                menor = AlgebraLineal.submatriz(matriz, i, j)
                cofactor = AlgebraLineal.determinante(menor)
                # Aplicar el signo correcto
                cofactores[i][j] = cofactor if (i + j) % 2 == 0 else -cofactor
        
        # La adjunta es la transpuesta de la matriz de cofactores
        adjunta = AlgebraLineal.transpuesta(cofactores)
        
        # La inversa es la adjunta dividida por el determinante
        inversa = AlgebraLineal.crear_matriz(filas, filas)
        for i in range(filas):
            for j in range(filas):
                inversa[i][j] = adjunta[i][j] / det
                
        return inversa
    
    # ======================== VISUALIZACIÓN ========================
    
    @staticmethod
    def graficar_funcion(f, x_min, x_max, puntos=100, titulo="Gráfica de función", etiqueta_x="x", etiqueta_y="f(x)"):
        """
        Grafica una función en un intervalo dado.
        
        Args:
            f (function): Función a graficar que toma un valor x y devuelve un valor y
            x_min (float): Valor mínimo de x
            x_max (float): Valor máximo de x
            puntos (int, optional): Número de puntos para graficar. Por defecto es 100.
            titulo (str, optional): Título de la gráfica. Por defecto es "Gráfica de función".
            etiqueta_x (str, optional): Etiqueta del eje x. Por defecto es "x".
            etiqueta_y (str, optional): Etiqueta del eje y. Por defecto es "f(x)".
            
        Returns:
            None: Muestra la gráfica
            
        Example:
            AlgebraLineal.graficar_funcion(lambda x: x**2, -5, 5, titulo="Parábola")
        """
        x = [x_min + i * (x_max - x_min) / (puntos - 1) for i in range(puntos)]
        y = [f(valor_x) for valor_x in x]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y)
        plt.title(titulo)
        plt.xlabel(etiqueta_x)
        plt.ylabel(etiqueta_y)
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.show()
    
    @staticmethod
    def graficar_vectores(vectores, etiquetas=None, origen=None, titulo="Vectores", mostrar_ejes=True):
        """
        Grafica vectores en 2D o 3D.
        
        Args:
            vectores (list): Lista de vectores a graficar
            etiquetas (list, optional): Lista de etiquetas para los vectores. Por defecto es None.
            origen (list, optional): Punto de origen para los vectores. Por defecto es el origen (0,0) o (0,0,0).
            titulo (str, optional): Título de la gráfica. Por defecto es "Vectores".
            mostrar_ejes (bool, optional): Mostrar líneas de ejes. Por defecto es True.
            
        Returns:
            None: Muestra la gráfica
            
        Raises:
            ValueError: Si los vectores no tienen la misma dimensión o si la dimensión no es 2 o 3
            
        Example:
            AlgebraLineal.graficar_vectores([[1, 0], [0, 1]], etiquetas=["i", "j"])
        """
        if not vectores:
            raise ValueError("Debe proporcionar al menos un vector")
            
        # Determinar la dimensión
        dim = len(vectores[0])
        
        if dim not in [2, 3]:
            raise ValueError("Solo se admiten vectores 2D o 3D para graficar")
            
        if any(len(v) != dim for v in vectores):
            raise ValueError("Todos los vectores deben tener la misma dimensión")
            
        # Establecer el origen si no se proporciona
        if origen is None:
            origen = [0] * dim
            
        if etiquetas is None:
            # Generar etiquetas que muestren los elementos de los vectores
            etiquetas = []
            for i, v in enumerate(vectores):
                # Formatear vector como (v1, v2, v3, ...) para la etiqueta
                components_str = ", ".join(str(round(component, 2)) for component in v)
                etiquetas.append(f"v{i+1}=({components_str})")
            
        # Preparar colores
        colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        
        if dim == 2:
            plt.figure(figsize=(10, 8))
            
            # Encontrar límites
            max_val = max([abs(comp) for v in vectores for comp in v] + [abs(comp) for comp in origen])
            limit = max_val * 1.2  # Añadir un margen
            
            if mostrar_ejes:
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            for i, v in enumerate(vectores):
                color = colores[i % len(colores)]
                plt.quiver(origen[0], origen[1], v[0], v[1], 
                         angles='xy', scale_units='xy', scale=1, color=color, label=etiquetas[i])
                
            plt.xlim(-limit, limit)
            plt.ylim(-limit, limit)
            plt.grid(True)
            plt.legend()
            plt.title(titulo)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.axis('equal')
            plt.show()
        else:  # dim == 3
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Encontrar límites
            max_val = max([abs(comp) for v in vectores for comp in v] + [abs(comp) for comp in origen])
            limit = max_val * 1.2  # Añadir un margen
            
            for i, v in enumerate(vectores):
                color = colores[i % len(colores)]
                ax.quiver(origen[0], origen[1], origen[2], v[0], v[1], v[2], color=color, label=etiquetas[i])
                
            if mostrar_ejes:
                # Dibujar ejes
                ax.plot([-limit, limit], [0, 0], [0, 0], 'k-', alpha=0.3)  # Eje X
                ax.plot([0, 0], [-limit, limit], [0, 0], 'k-', alpha=0.3)  # Eje Y
                ax.plot([0, 0], [0, 0], [-limit, limit], 'k-', alpha=0.3)  # Eje Z
            
            ax.set_xlim([-limit, limit])
            ax.set_ylim([-limit, limit])
            ax.set_zlim([-limit, limit])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            ax.set_title(titulo)
            plt.show()
    
    # ======================== SISTEMAS DE ECUACIONES LINEALES ========================
    
    @staticmethod
    def gauss_jordan(matriz_aumentada):
        """
        Resuelve un sistema de ecuaciones lineales usando el método de Gauss-Jordan.
        
        Args:
            matriz_aumentada (list): Matriz aumentada [A|b] donde A es la matriz de coeficientes
                                    y b es el vector de términos independientes
            
        Returns:
            tuple: (solucion, tipo_solucion)
                  - solucion: Lista con la solución única, ejemplo de solución (sistema compatible indeterminado),
                             o None (sistema incompatible)
                  - tipo_solucion: String indicando el tipo de solución ("unica", "infinitas", "incompatible")
            
        Example:
            resultado, tipo = AlgebraLineal.gauss_jordan([[1, 1, 1, 6], [2, -1, 3, 9], [3, 2, -4, 3]])
            # Para el sistema: x + y + z = 6, 2x - y + 3z = 9, 3x + 2y - 4z = 3
        """
        # Hacer una copia profunda de la matriz para no modificar la original
        matriz = [fila[:] for fila in matriz_aumentada]
        
        filas = len(matriz)
        columnas = len(matriz[0])
        
        # El número de variables es el número de columnas menos 1 (para la columna de términos independientes)
        n_variables = columnas - 1
        
        # Variable para rastrear la fila y columna actual durante la eliminación
        fila_actual = 0
        for columna_actual in range(n_variables):
            # Si estamos al final de las filas, terminamos
            if fila_actual >= filas:
                break
                
            # Encontrar el pivote máximo en esta columna (por estabilidad numérica)
            max_fila = fila_actual
            max_valor = abs(matriz[fila_actual][columna_actual])
            
            for i in range(fila_actual + 1, filas):
                if abs(matriz[i][columna_actual]) > max_valor:
                    max_fila = i
                    max_valor = abs(matriz[i][columna_actual])
            
            # Si el máximo es cero, esta columna ya está reducida
            if max_valor < 1e-10:  # Usar tolerancia para errores de punto flotante
                continue
                
            # Intercambiar filas si es necesario
            if max_fila != fila_actual:
                matriz[fila_actual], matriz[max_fila] = matriz[max_fila], matriz[fila_actual]
                
            # Normalizar la fila del pivote
            pivote = matriz[fila_actual][columna_actual]
            for j in range(columna_actual, columnas):
                matriz[fila_actual][j] /= pivote
                
            # Eliminar esta variable de las otras filas
            for i in range(filas):
                if i != fila_actual:
                    factor = matriz[i][columna_actual]
                    for j in range(columna_actual, columnas):
                        matriz[i][j] -= factor * matriz[fila_actual][j]
            
            fila_actual += 1
        
        # Verificar si el sistema es incompatible
        for i in range(fila_actual, filas):
            # Si hay una fila con todos ceros excepto el término independiente
            if all(abs(matriz[i][j]) < 1e-10 for j in range(n_variables)) and abs(matriz[i][n_variables]) > 1e-10:
                return None, "incompatible"
        
        # Verificar si el sistema tiene infinitas soluciones
        if fila_actual < n_variables:
            return [matriz[i][n_variables] if i < fila_actual else 0 for i in range(n_variables)], "infinitas"
        
        # El sistema tiene solución única
        return [matriz[i][n_variables] for i in range(n_variables)], "unica"
    

    @staticmethod
    def gauss(matriz_aumentada, verbose=False):
        """
        Resuelve un sistema de ecuaciones lineales usando el método de eliminación Gaussiana.
        
        Args:
            matriz_aumentada (list): Matriz aumentada [A|b] donde A es la matriz de coeficientes
                                    y b es el vector de términos independientes
            verbose (bool, optional): Si es True, muestra información detallada del proceso. Por defecto es False.
        
        Returns:
            tuple: (solucion, tipo_solucion)
                  - solucion: Lista con la solución si es única, None en caso contrario
                  - tipo_solucion: String indicando el tipo de solución ("unica", "infinitas", "incompatible")
        
        Example:
            solucion, tipo = AlgebraLineal.gauss([[1, 1, 1, 6], [2, -1, 3, 9], [3, 2, -4, 3]])
            # Para el sistema: x + y + z = 6, 2x - y + 3z = 9, 3x + 2y - 4z = 3
        """
        # Hacer una copia profunda de la matriz para no modificar la original
        amat = [fila[:] for fila in matriz_aumentada]
        
        n_eq = len(amat)  # Número de ecuaciones (filas)
        n_var = len(amat[0]) - 1  # Número de variables (columnas - 1)
        
        if verbose:
            print("\n--- Sistema ingresado ---")
            for i in range(n_eq):
                for j in range(n_var):
                    print(f"{amat[i][j]:.2f}x_{j+1}", end="")
                    if j < n_var - 1:
                        print(" + ", end="")
                print(f" = {amat[i][n_var]:.2f}")
        
        num_piv = min(n_eq, n_var)
        is_sngl = False
        
        # Eliminación Gaussiana hacia adelante
        for k in range(num_piv):
            # Pivoteo parcial: encontrar el valor máximo en la columna actual
            max_val = abs(amat[k][k])
            max_fil = k
            for i in range(k + 1, n_eq):
                if abs(amat[i][k]) > max_val:
                    max_val = abs(amat[i][k])
                    max_fil = i
            
            # Intercambiar filas si es necesario
            if max_fil != k:
                amat[k], amat[max_fil] = amat[max_fil], amat[k]
            
            # Si el pivote es casi cero, marcar como singular y continuar
            if abs(amat[k][k]) < 1e-9:
                is_sngl = True
                continue
            
            # Eliminar la variable actual de las ecuaciones siguientes
            for i in range(k + 1, n_eq):
                if abs(amat[i][k]) > 1e-9 and abs(amat[k][k]) > 1e-9:
                    fact = amat[i][k] / amat[k][k]
                    for j in range(k, n_var + 1):
                        amat[i][j] -= fact * amat[k][j]
        
        if verbose:
            print("\n--- Matriz después de la Eliminación Gaussiana ---")
            for i in range(n_eq):
                for j in range(n_var + 1):
                    print(f"{amat[i][j]:.4f}\t", end="")
                print()
        
        # Verificar si el sistema es incompatible
        is_incn = False
        for i in range(num_piv, n_eq):
            all_z_cof = True
            for j in range(n_var):
                if abs(amat[i][j]) > 1e-9:
                    all_z_cof = False
                    break
            
            if all_z_cof and abs(amat[i][n_var]) > 1e-9:
                is_incn = True
                break
        
        if is_incn:
            if verbose:
                print("\nEl sistema es incompatible.")
            return None, "incompatible"
        elif is_sngl or n_eq < n_var:
            if verbose:
                print("\nEl sistema tiene infinitas soluciones.")
            return None, "infinitas"
        else:
            # Sustitución hacia atrás
            sol = [0.0] * n_var
            try:
                for i in range(n_var - 1, -1, -1):
                    curr_sum = 0.0
                    for j in range(i + 1, n_var):
                        curr_sum += amat[i][j] * sol[j]
                    
                    if abs(amat[i][i]) < 1e-9:
                        if verbose:
                            print("\nEl sistema tiene infinitas soluciones o es incompatible.")
                        return None, "infinitas"
                    
                    sol[i] = (amat[i][n_var] - curr_sum) / amat[i][i]
                
                if verbose:
                    print("\n--- Soluciones ---")
                    for i in range(n_var):
                        print(f"Solución x_{i+1} = {sol[i]:.6f}")
                
                return sol, "unica"
            
            except Exception:
                if verbose:
                    print("\nError en el cálculo: El sistema tiene infinitas soluciones o es incompatible.")
                return None, "indeterminado"
    
    @staticmethod
    def resolver_sistema(coeficientes, terminos_independientes):
        """
        Resuelve un sistema de ecuaciones lineales Ax = b.
        
        Args:
            coeficientes (list): Matriz A de coeficientes
            terminos_independientes (list): Vector b de términos independientes
            
        Returns:
            tuple: (solucion, tipo_solucion) como en gauss_jordan
            
        Example:
            solucion, tipo = AlgebraLineal.resolver_sistema([[1, 1, 1], [2, -1, 3], [3, 2, -4]], [6, 9, 3])
            # Para el sistema: x + y + z = 6, 2x - y + 3z = 9, 3x + 2y - 4z = 3
        """
        # Crear la matriz aumentada [A|b]
        matriz_aumentada = []
        for i in range(len(coeficientes)):
            fila = coeficientes[i].copy()
            fila.append(terminos_independientes[i])
            matriz_aumentada.append(fila)
            
        return AlgebraLineal.gauss_jordan(matriz_aumentada)
    
    # ======================== ANÁLISIS DE INDEPENDENCIA LINEAL ========================
    
    @staticmethod
    def es_linealmente_independiente(vectores):
        """
        Determina si un conjunto de vectores es linealmente independiente.
        
        Args:
            vectores (list): Lista de vectores
            
        Returns:
            tuple: (bool, str) - (True/False, justificación)
            
        Example:
            resultado, justificacion = AlgebraLineal.es_linealmente_independiente([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            # Retorna: (True, "Los vectores son linealmente independientes...")
        """
        if not vectores:
            return False, "El conjunto vacío no es linealmente independiente."
        
        n_vectores = len(vectores)
        dimension = len(vectores[0])
        
        # Si hay más vectores que la dimensión, no pueden ser l.i.
        if n_vectores > dimension:
            return False, f"Hay {n_vectores} vectores en un espacio de dimensión {dimension}. Por el teorema de extensión lineal, estos vectores no pueden ser linealmente independientes."
        
        # Comprobar que todos los vectores tienen la misma dimensión
        if any(len(v) != dimension for v in vectores):
            return False, "Todos los vectores deben tener la misma dimensión."
        
        # Caso especial: si algún vector es el vector cero
        for i, v in enumerate(vectores):
            if all(abs(comp) < 1e-10 for comp in v):
                return False, f"El vector {i+1} es el vector cero, por lo que el conjunto no es linealmente independiente."
        
        # Caso especial: si hay un solo vector no nulo
        if n_vectores == 1:
            return True, "Un conjunto con un solo vector no nulo es linealmente independiente."
        
        # Formar la matriz con los vectores como columnas
        matriz = AlgebraLineal.transpuesta(vectores)
        
        # Calcular el rango de la matriz
        # Usamos nuestro método gauss_jordan 
        # Primero creamos una matriz aumentada con una columna de ceros
        matriz_aumentada = [fila + [0] for fila in matriz]
        solucion, tipo = AlgebraLineal.gauss_jordan(matriz_aumentada)
        
        # El rango es igual al número de filas no nulas después de la eliminación
        # Esto se refleja en el número de elementos de la solución que no son arbitrarios
        rango = sum(1 for i in range(min(len(matriz), len(matriz[0]))) 
                   if any(abs(matriz_aumentada[i][j]) > 1e-10 for j in range(len(matriz[0]))))
        
        if rango == n_vectores:
            return True, f"Los vectores son linealmente independientes porque el rango de la matriz formada por ellos es {rango}, igual al número de vectores."
        else:
            return False, f"Los vectores son linealmente dependientes porque el rango de la matriz formada por ellos es {rango}, menor que el número de vectores ({n_vectores})."
    
    @staticmethod
    def combinacion_lineal(vectores, coeficientes):
        """
        Calcula la combinación lineal de vectores con los coeficientes dados.
        
        Args:
            vectores (list): Lista de vectores
            coeficientes (list): Lista de coeficientes
            
        Returns:
            list: Vector resultante de la combinación lineal
            
        Raises:
            ValueError: Si el número de vectores y coeficientes no coincide
            
        Example:
            resultado = AlgebraLineal.combinacion_lineal([[1, 0], [0, 1]], [2, 3])
            # Retorna: [2, 3]
        """
        if len(vectores) != len(coeficientes):
            raise ValueError("El número de vectores y coeficientes debe ser igual")
        
        if not vectores:
            return []
        
        # Inicializar el resultado con ceros
        dimension = len(vectores[0])
        resultado = [0] * dimension
          # Sumar cada vector multiplicado por su coeficiente
        for j in range(len(vectores)):
            vector = vectores[j]
            coef = coeficientes[j]
            for i in range(dimension):
                resultado[i] += coef * vector[i]
                
        return resultado
    


    @staticmethod
    def es_combinacion_lineal(vector, conjunto_vectores):
        """
        Determina si un vector es combinación lineal de un conjunto de vectores.
        
        Args:
            vector (list): Vector a comprobar
            conjunto_vectores (list): Conjunto de vectores
            
        Returns:
            tuple: (bool, list/str) - (True/False, coeficientes o mensaje)
            
        Example:
            es_cl, coefs = AlgebraLineal.es_combinacion_lineal([3, 3], [[1, 0], [0, 1]])
            # Retorna: (True, [3, 3])
        """
        if not conjunto_vectores:
            return False, "El conjunto de vectores está vacío."
        
        dimension = len(vector)
        
        # Comprobar que todos los vectores tienen la misma dimensión
        if any(len(v) != dimension for v in conjunto_vectores):
            return False, "El vector y los vectores del conjunto deben tener la misma dimensión."
        
        # Crear la matriz de coeficientes y el vector de términos independientes
        # La ecuación a resolver es: a1*v1 + a2*v2 + ... + an*vn = vector
        coeficientes = AlgebraLineal.transpuesta(conjunto_vectores)
        terminos_independientes = vector
        
        # Resolver el sistema
        solucion, tipo = AlgebraLineal.resolver_sistema(coeficientes, terminos_independientes)
        
        if tipo == "incompatible":
            return False, "El vector no es combinación lineal del conjunto dado."
        
        # Verificar la solución reconstruyendo el vector
        reconstruccion = AlgebraLineal.combinacion_lineal(conjunto_vectores, solucion)
        
        # Comparar con cierta tolerancia debido a errores de punto flotante
        es_igual = all(abs(reconstruccion[i] - vector[i]) < 1e-10 for i in range(dimension))
        
        if es_igual:
            return True, solucion
        else:
            return False, "El vector no es combinación lineal del conjunto dado."
        

    
