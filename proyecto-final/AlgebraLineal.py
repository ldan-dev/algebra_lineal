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
import numpy as np
from fractions import Fraction

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
    
    # Constante de tolerancia para comparaciones numéricas (detectar valores "casi cero")
    TOLERANCIA = 1e-10
    
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
        
        # Convertir a Fraction para mantener precisión exacta
        v1_frac = [Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in v1]
        v2_frac = [Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in v2]
        
        resultado = Fraction(0)
        for i in range(len(v1_frac)):
            resultado += v1_frac[i] * v2_frac[i]
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
            
            # Calcular el determinante del menor correspondiente
            det = 0
            signo = 1
            
            # Para cada elemento del vector base, calculamos su menor y lo multiplicamos
            for j in range(dimension):
                # Si el vector base tiene un 1 en posición j, contribuye al determinante
                if matriz[0][j] == 1:
                    # Construir submatriz 2x(n-1) excluyendo fila 0 y columna j
                    submatriz = []
                    for fila in range(1, 3):  # Solo filas 1 y 2 (v1 y v2)
                        submatriz_fila = []
                        for col in range(dimension):
                            if col != j:
                                submatriz_fila.append(matriz[fila][col])
                        submatriz.append(submatriz_fila)
                    
                    # Calcular determinante de la submatriz 2x(n-1)
                    # Para una matriz 2x2, el determinante es simple
                    if dimension == 3:  # Caso especial para 3D
                        det_submatriz = submatriz[0][0] * submatriz[1][1] - submatriz[0][1] * submatriz[1][0]
                        det += signo * det_submatriz
                    else:
                        # Para matrices 2x(n-1) con n>3, seleccionamos todas las combinaciones posibles
                        # de 2 columnas y calculamos determinantes de las submatrices 2x2 resultantes
                        det_submatriz = 0
                        for k in range(dimension - 2):  # n-2 porque ya excluimos una columna
                            for l in range(k + 1, dimension - 1):
                                # Determinante de la submatriz 2x2
                                minor_det = submatriz[0][k] * submatriz[1][l] - submatriz[0][l] * submatriz[1][k]
                                # Ajustar el signo según la posición de las columnas
                                det_submatriz += minor_det * ((-1) ** (k + l))
                                
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
        
        # Convertir a Fraction para mantener precisión exacta
        v1_frac = [Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in v1]
        v2_frac = [Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in v2]
        
        resultado = []
        for i in range(len(v1_frac)):
            resultado.append(v1_frac[i] + v2_frac[i])
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
        
        # Convertir a Fraction para mantener precisión exacta
        v1_frac = [Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in v1]
        v2_frac = [Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in v2]
        
        resultado = []
        for i in range(len(v1_frac)):
            resultado.append(v1_frac[i] - v2_frac[i])
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
        # Convertir a Fraction para mantener precisión exacta
        escalar_frac = Fraction(escalar) if not isinstance(escalar, Fraction) else escalar
        vector_frac = [Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in vector]
        
        return [escalar_frac * componente for componente in vector_frac]
    
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
        # Convertir a Fraction para cálculos internos precisos
        vector_frac = [Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in vector]
        
        # Sumar los cuadrados
        suma_cuadrados = sum(x * x for x in vector_frac)
        
        # Para la raíz cuadrada necesitamos convertir a float
        return math.sqrt(float(suma_cuadrados))
    
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
    
    @staticmethod
    def transformacion_lineal(base_dominio, base_codominio):
        """
        Calcula la matriz de una transformación lineal dado un conjunto de vectores
        de la base del dominio y sus correspondientes imágenes en el codominio.
        
        Args:
            base_dominio (list): Lista de vectores que forman una base del dominio
            base_codominio (list): Lista de vectores que son las imágenes de los vectores de la base
            
        Returns:
            list: Matriz de la transformación lineal
            
        Raises:
            ValueError: Si los vectores no tienen dimensiones adecuadas o no forman una base
            
        Example:
            matriz = AlgebraLineal.transformacion_lineal([[1, 0], [0, 1]], [[2, 1], [0, 3]])
            # Retorna: [[2, 0], [1, 3]]
        """
        
        # Validación de entrada
        if not base_dominio or not base_codominio:
            raise ValueError("Las bases del dominio y codominio no pueden estar vacías")
            
        n_vec = len(base_dominio)
        if len(base_codominio) != n_vec:
            raise ValueError("Las bases del dominio y codominio deben tener el mismo número de vectores")
        
        # Convertir a fracciones para cálculos exactos
        vec_V = [[Fraction(comp) for comp in vector] for vector in base_dominio]
        vec_W = [[Fraction(comp) for comp in vector] for vector in base_codominio]
        
        dim_v = len(vec_V[0])
        dim_w = len(vec_W[0])
        
        # Comprobar que todos los vectores tienen la dimensión correcta
        if any(len(v) != dim_v for v in vec_V):
            raise ValueError("Todos los vectores de la base del dominio deben tener la misma dimensión")
        if any(len(w) != dim_w for w in vec_W):
            raise ValueError("Todos los vectores de la base del codominio deben tener la misma dimensión")
        
        # Comprobar si n_vec == dim_v (condición para invertibilidad)
        if n_vec != dim_v:
            raise ValueError("La cantidad de vectores debe ser igual a su dimensión para formar una base invertible")
        
        # Crear matriz aumentada para inversión
        neq_cl = dim_v
        nvar_c = n_vec
        tot_cl = nvar_c + neq_cl
        
        mat_A = [[Fraction(0) for _ in range(tot_cl)] for _ in range(neq_cl)]
        
        # Llenar la matriz aumentada
        for i_mat in range(neq_cl):
            for j_mat in range(nvar_c):
                mat_A[i_mat][j_mat] = vec_V[j_mat][i_mat]
            mat_A[i_mat][nvar_c + i_mat] = Fraction(1)
        
        # Tolerancia para valores cercanos a cero
        tol_0 = Fraction(1, 10**9)
        
        # Eliminación gaussiana
        cur_pv = 0
        piv_cl = []
        for k_col in range(nvar_c):
            if cur_pv >= neq_cl:
                break
            
            max_vl = abs(mat_A[cur_pv][k_col])
            max_fl = cur_pv
            
            for i_fil in range(cur_pv + 1, neq_cl):
                if abs(mat_A[i_fil][k_col]) > max_vl:
                    max_vl = abs(mat_A[i_fil][k_col])
                    max_fl = i_fil
            
            if max_vl > tol_0:
                if max_fl != cur_pv:
                    mat_A[cur_pv], mat_A[max_fl] = mat_A[max_fl], mat_A[cur_pv]
                
                piv_cl.append(k_col)
                
                for i_row in range(cur_pv + 1, neq_cl):
                    if abs(mat_A[i_row][k_col]) > tol_0:
                        factr = mat_A[i_row][k_col] / mat_A[cur_pv][k_col]
                        for j_idx in range(k_col, tot_cl):
                            mat_A[i_row][j_idx] -= factr * mat_A[cur_pv][j_idx]
                cur_pv += 1
        
        # Verificar si la matriz es invertible
        if len(piv_cl) < nvar_c:
            raise ValueError("Los vectores de la base no son linealmente independientes. No puede encontrarse una única matriz de transformación.")
        
        # Finalizar eliminación gaussiana (sustitución hacia atrás)
        for i_pvt in range(len(piv_cl) - 1, -1, -1):
            piv_co = piv_cl[i_pvt]
            piv_vl = mat_A[i_pvt][piv_co]
            
            for j_idx in range(piv_co, tot_cl):
                mat_A[i_pvt][j_idx] /= piv_vl
            
            for r_abv in range(i_pvt):
                factr = mat_A[r_abv][piv_co]
                for j_idx in range(piv_co, tot_cl):
                    mat_A[r_abv][j_idx] -= factr * mat_A[i_pvt][j_idx]
        
        # Extraer matriz inversa
        inv_V = []
        for i_inv in range(neq_cl):
            row_i = []
            for j_inv in range(nvar_c, tot_cl):
                row_i.append(mat_A[i_inv][j_inv])
            inv_V.append(row_i)
        
        # Calcular la matriz de transformación
        fil_T = dim_w
        col_T = n_vec
        
        mat_T = [[Fraction(0) for _ in range(col_T)] for _ in range(fil_T)]
        
        for i_fil in range(fil_T):
            for j_col in range(col_T):
                suma_mul = Fraction(0)
                for k_idx in range(n_vec):
                    suma_mul += vec_W[k_idx][i_fil] * inv_V[k_idx][j_col]
                mat_T[i_fil][j_col] = suma_mul
        # Mantener las fracciones en el resultado para mayor precisión y legibilidad
        result_matrix = []
        for row in mat_T:
            result_row = []
            for elem in row:
                # Si es entero, devolver el entero, sino la fracción
                if elem.denominator == 1:
                    result_row.append(int(elem.numerator))
                else:
                    result_row.append(elem)  # Mantener como Fraction
            result_matrix.append(result_row)
        
        return result_matrix

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
        # Convertir el valor inicial a Fraction para mantener precisión exacta
        valor_inicial_frac = Fraction(valor_inicial) if not isinstance(valor_inicial, Fraction) else valor_inicial
        return [[valor_inicial_frac for _ in range(columnas)] for _ in range(filas)]
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
        identidad = [[Fraction(0) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            identidad[i][i] = Fraction(1)
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
        
        # Convertir a Fraction para mantener precisión exacta
        m1_frac = [[Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in fila] for fila in m1]
        m2_frac = [[Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in fila] for fila in m2]
        
        filas = len(m1_frac)
        columnas = len(m1_frac[0])
        resultado = [[Fraction(0) for _ in range(columnas)] for _ in range(filas)]
        
        for i in range(filas):
            for j in range(columnas):
                resultado[i][j] = m1_frac[i][j] + m2_frac[i][j]
                
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
        
        # Convertir a Fraction para mantener precisión exacta
        m1_frac = [[Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in fila] for fila in m1]
        m2_frac = [[Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in fila] for fila in m2]
        
        filas = len(m1_frac)
        columnas = len(m1_frac[0])
        resultado = [[Fraction(0) for _ in range(columnas)] for _ in range(filas)]
        
        for i in range(filas):
            for j in range(columnas):
                resultado[i][j] = m1_frac[i][j] - m2_frac[i][j]
                
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
        # Convertir a Fraction para mantener precisión exacta
        escalar_frac = Fraction(escalar) if not isinstance(escalar, Fraction) else escalar
        matriz_frac = [[Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in fila] for fila in matriz]
        
        filas = len(matriz_frac)
        columnas = len(matriz_frac[0])
        resultado = AlgebraLineal.crear_matriz(filas, columnas)
        
        for i in range(filas):
            for j in range(columnas):
                resultado[i][j] = escalar_frac * matriz_frac[i][j]
                
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
        # Convertir a Fraction para mantener precisión exacta
        m1_frac = [[Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in fila] for fila in m1]
        m2_frac = [[Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in fila] for fila in m2]
        
        filas_m1 = len(m1_frac)
        columnas_m1 = len(m1_frac[0])
        filas_m2 = len(m2_frac)
        
        if columnas_m1 != filas_m2:
            raise ValueError("Las dimensiones de las matrices no son compatibles para la multiplicación")
        
        columnas_m2 = len(m2_frac[0])
        resultado = [[Fraction(0) for _ in range(columnas_m2)] for _ in range(filas_m1)]
        
        for i in range(filas_m1):
            for j in range(columnas_m2):
                for k in range(columnas_m1):
                    resultado[i][j] += m1_frac[i][k] * m2_frac[k][j]
                    
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
        # Convertir a Fraction para mantener precisión exacta
        matriz_frac = [[Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in fila] for fila in matriz]
        
        filas = len(matriz_frac)
        columnas = len(matriz_frac[0])
        
        transpuesta = [[Fraction(0) for _ in range(filas)] for _ in range(columnas)]
        
        for i in range(filas):
            for j in range(columnas):
                transpuesta[j][i] = matriz_frac[i][j]
                
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
        # Mantener los valores originales, que pueden ser Fraction
        return [[matriz[i][j] for j in range(len(matriz[0])) if j != columna_excluida]
                for i in range(len(matriz)) if i != fila_excluida]@staticmethod
    def determinante(matriz):
        """
        Calcula el determinante de una matriz.
        Para matrices pequeñas, usa expansión por cofactores.
        Para matrices grandes (n > 3), usa eliminación gaussiana para mejor rendimiento.
        
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
        
        # Convertir a Fraction para cálculos exactos
        matriz_frac = [[Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in fila] for fila in matriz]
        
        # Caso base: matriz 1x1
        if filas == 1:
            return matriz_frac[0][0]
        
        # Caso base: matriz 2x2
        if filas == 2:
            return matriz_frac[0][0] * matriz_frac[1][1] - matriz_frac[0][1] * matriz_frac[1][0]
        
        # Caso base: matriz 3x3 (fórmula directa para optimización)
        if filas == 3:
            a, b, c = matriz_frac[0]
            d, e, f = matriz_frac[1]
            g, h, i = matriz_frac[2]
            return (a * e * i + b * f * g + c * d * h) - (c * e * g + a * f * h + b * d * i)
        
        # Para matrices grandes (n ≥ 4), usamos eliminación gaussiana para mejor rendimiento
        if filas >= 4:
            return AlgebraLineal._determinante_gauss(matriz)
        # Expansión por cofactores a lo largo de la primera fila para matrices medianas
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
        
        # Convertir a Fraction para cálculos exactos
        matriz_frac = [[Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in fila] for fila in matriz]
        
        # Calcular el determinante
        det = AlgebraLineal.determinante(matriz_frac)
        
        if abs(det) < AlgebraLineal.TOLERANCIA:  # Usar la tolerancia definida para evitar problemas de precisión
            raise ValueError("La matriz no es invertible (determinante = 0)")
        
        # Para una matriz 1x1, la inversa es trivial
        if filas == 1:
            return [[Fraction(1) / matriz_frac[0][0]]]
        
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
    def graficar_funcion(f, x_min, x_max, puntos=100, titulo="Gráfica de función", etiqueta_x="x", etiqueta_y="f(x)", mostrar_puntos_destacados=True):
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
            mostrar_puntos_destacados (bool, optional): Si es True, muestra puntos destacados de la función. Por defecto es True.
            
        Returns:
            None: Muestra la gráfica
            
        Example:
            AlgebraLineal.graficar_funcion(lambda x: x**2, -5, 5, titulo="Parábola")
        """
        x = [x_min + i * (x_max - x_min) / (puntos - 1) for i in range(puntos)]
        y = [f(valor_x) for valor_x in x]
        
        plt.figure(figsize=(10, 6))
        
        # Graficar la función principal
        plt.plot(x, y, label=f"f(x)")
        
        # Añadir algunos puntos destacados si se solicita
        if mostrar_puntos_destacados:
            # Seleccionamos algunos puntos destacados
            num_puntos_destacados = min(5, puntos)  # Máximo 5 puntos para no sobrecargar
            indices = [int(i * (puntos - 1) / (num_puntos_destacados - 1)) for i in range(num_puntos_destacados)]
            
            puntos_x = [x[i] for i in indices]
            puntos_y = [y[i] for i in indices]
            
            # Plotear los puntos destacados
            plt.scatter(puntos_x, puntos_y, color='red', zorder=5)
            
            # Añadir etiquetas para los puntos destacados
            for i in range(len(puntos_x)):
                plt.annotate(f"({puntos_x[i]:.2f}, {puntos_y[i]:.2f})", 
                           (puntos_x[i], puntos_y[i]),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center')
        
        plt.title(titulo)
        plt.xlabel(etiqueta_x)
        plt.ylabel(etiqueta_y)
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.legend()
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
    
    @staticmethod
    def visualizar_transformacion_lineal(matriz_transformacion, titulo="Visualización de Transformación Lineal", figsize=(12, 6), grid_lines=10, mostrar_etiquetas=True):
        """
        Visualiza el resultado de una transformación lineal calculada con el método transformacion_lineal.
        
        Args:
            matriz_transformacion (list): Matriz de transformación resultado del método transformacion_lineal
            titulo (str, optional): Título personalizado para la gráfica
            figsize (tuple, optional): Tamaño de la figura (ancho, alto)
            grid_lines (int, optional): Número de líneas de la cuadrícula
            mostrar_detalle (bool, optional): Mostrar detalles adicionales como la matriz en la gráfica
            
        Returns:
            None: Muestra la gráfica
            
        Example:
            matriz = AlgebraLineal.transformacion_lineal([[1, 0], [0, 1]], [[0, -1], [1, 0]])
            AlgebraLineal.visualizar_transformacion_lineal(matriz, "Rotación 90°")
        """
        matriz = np.array(matriz_transformacion)
        
        # Detectar si es una matriz 2x2 o 3x3
        filas, columnas = matriz.shape
        
        if filas == 2 and columnas == 2:
            # Caso 2D
            # Crear una cuadrícula de puntos
            x = np.linspace(-5, 5, grid_lines)
            y = np.linspace(-5, 5, grid_lines)
            X, Y = np.meshgrid(x, y)
            puntos = np.vstack([X.flatten(), Y.flatten()])
            
            # Aplicar la transformación
            transformados = matriz @ puntos
            
            # Reorganizar en cuadrículas para graficar
            X_transformado = transformados[0, :].reshape(grid_lines, grid_lines)
            Y_transformado = transformados[1, :].reshape(grid_lines, grid_lines)
            
            # Crear figura y subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Gráfica original
            ax1.set_title("Espacio Original")
            ax1.set_xlim(-6, 6)
            ax1.set_ylim(-6, 6)
            ax1.grid(True)
            
            # Dibujar cuadrícula original
            for i in range(grid_lines):
                ax1.plot(x, [y[i]] * len(x), 'b-', alpha=0.3)
                ax1.plot([x[i]] * len(y), y, 'b-', alpha=0.3)
            
            # Vectores base originales
            ax1.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='r', label="e₁")
            ax1.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='g', label="e₂")
            ax1.text(1.1, 0, "e₁=(1,0)", color='r')
            ax1.text(0, 1.1, "e₂=(0,1)", color='g')
            
            ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            ax1.axvline(x=0, color='k', linestyle='-', alpha=0.5)
            ax1.legend()
            
            # Gráfica transformada
            ax2.set_title("Espacio Transformado")
            
            # Calcular límites basados en los puntos transformados
            max_val = max(np.max(np.abs(X_transformado)), np.max(np.abs(Y_transformado)))
            ax2.set_xlim(-max_val*1.2, max_val*1.2)
            ax2.set_ylim(-max_val*1.2, max_val*1.2)
            ax2.grid(True)
            
            # Dibujar cuadrícula transformada
            for i in range(grid_lines):
                ax2.plot(X_transformado[i, :], Y_transformado[i, :], 'b-', alpha=0.3)
                ax2.plot(X_transformado[:, i], Y_transformado[:, i], 'b-', alpha=0.3)
            
            # Vectores base transformados
            vector_e1_transformado = matriz @ np.array([1, 0])
            vector_e2_transformado = matriz @ np.array([0, 1])
            
            ax2.quiver(0, 0, vector_e1_transformado[0], vector_e1_transformado[1], 
                     angles='xy', scale_units='xy', scale=1, color='r', label="T(e₁)")
            ax2.quiver(0, 0, vector_e2_transformado[0], vector_e2_transformado[1], 
                     angles='xy', scale_units='xy', scale=1, color='g', label="T(e₂)")
            
            # Etiquetas y detalles
            if mostrar_detalle:
                ax2.text(vector_e1_transformado[0]*1.1, vector_e1_transformado[1]*1.1, 
                        f"T(e₁)=({vector_e1_transformado[0]:.2f},{vector_e1_transformado[1]:.2f})", color='r')
                ax2.text(vector_e2_transformado[0]*1.1, vector_e2_transformado[1]*1.1, 
                        f"T(e₂)=({vector_e2_transformado[0]:.2f},{vector_e2_transformado[1]:.2f})", color='g')
                
                # Agregar matriz de transformación como texto
                matriz_str = f"T = [{matriz[0,0]:.2f} {matriz[0,1]:.2f}\n     {matriz[1,0]:.2f} {matriz[1,1]:.2f}]"
                ax2.text(0.05, 0.95, matriz_str, transform=ax2.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Agregar información de determinante
                det = np.linalg.det(matriz)
                det_info = f"det(T) = {det:.2f}"
                ax2.text(0.05, 0.85, det_info, transform=ax2.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            ax2.axvline(x=0, color='k', linestyle='-', alpha=0.5)
            ax2.legend()
            
        elif filas == 3 and columnas == 3:
            # Caso 3D
            # Crear figura y subplots
            fig = plt.figure(figsize=(15, 7))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
            
            # Crear un cubo unitario
            puntos = np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
            ]).T - 0.5  # Centrar en el origen
            
            # Conectar las esquinas (aristas del cubo)
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Cara inferior
                (4, 5), (5, 6), (6, 7), (7, 4),  # Cara superior
                (0, 4), (1, 5), (2, 6), (3, 7)   # Aristas verticales
            ]
            
            # Dibujar el cubo original
            ax1.set_title("Espacio Original")
            for i, j in edges:
                x = [puntos[0, i], puntos[0, j]]
                y = [puntos[1, i], puntos[1, j]]
                z = [puntos[2, i], puntos[2, j]]
                ax1.plot(x, y, z, 'b-', alpha=0.5)
            
            # Dibujar los vectores base
            ax1.quiver(0, 0, 0, 1, 0, 0, color='r', label="e₁")
            ax1.quiver(0, 0, 0, 0, 1, 0, color='g', label="e₂")
            ax1.quiver(0, 0, 0, 0, 0, 1, color='b', label="e₃")
            
            if mostrar_detalle:
                ax1.text(1.1, 0, 0, "e₁=(1,0,0)", color='r')
                ax1.text(0, 1.1, 0, "e₂=(0,1,0)", color='g')
                ax1.text(0, 0, 1.1, "e₃=(0,0,1)", color='b')
            
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_xlim(-1, 1)
            ax1.set_ylim(-1, 1)
            ax1.set_zlim(-1, 1)
            ax1.legend()
            
            # Aplicar la transformación
            puntos_transformados = matriz @ puntos
            
            # Dibujar el objeto transformado
            ax2.set_title("Espacio Transformado")
            for i, j in edges:
                x = [puntos_transformados[0, i], puntos_transformados[0, j]]
                y = [puntos_transformados[1, i], puntos_transformados[1, j]]
                z = [puntos_transformados[2, i], puntos_transformados[2, j]]
                ax2.plot(x, y, z, 'b-', alpha=0.5)
            
            # Dibujar los vectores base transformados
            vector_e1_transformado = matriz @ np.array([1, 0, 0])
            vector_e2_transformado = matriz @ np.array([0, 1, 0])
            vector_e3_transformado = matriz @ np.array([0, 0, 1])
            
            ax2.quiver(0, 0, 0, vector_e1_transformado[0], vector_e1_transformado[1], vector_e1_transformado[2], 
                     color='r', label="T(e₁)")
            ax2.quiver(0, 0, 0, vector_e2_transformado[0], vector_e2_transformado[1], vector_e2_transformado[2], 
                     color='g', label="T(e₂)")
            ax2.quiver(0, 0, 0, vector_e3_transformado[0], vector_e3_transformado[1], vector_e3_transformado[2], 
                     color='b', label="T(e₃)")
            
            # Etiquetas y detalles
            if mostrar_etiquetas:
                ax2.text(vector_e1_transformado[0]*1.1, vector_e1_transformado[1]*1.1, vector_e1_transformado[2]*1.1, 
                        f"T(e₁)=({vector_e1_transformado[0]:.1f},{vector_e1_transformado[1]:.1f},{vector_e1_transformado[2]:.1f})", color='r')
                ax2.text(vector_e2_transformado[0]*1.1, vector_e2_transformado[1]*1.1, vector_e2_transformado[2]*1.1, 
                        f"T(e₂)=({vector_e2_transformado[0]:.1f},{vector_e2_transformado[1]:.1f},{vector_e2_transformado[2]:.1f})", color='g')
                ax2.text(vector_e3_transformado[0]*1.1, vector_e3_transformado[1]*1.1, vector_e3_transformado[2]*1.1, 
                        f"T(e₃)=({vector_e3_transformado[0]:.1f},{vector_e3_transformado[1]:.1f},{vector_e3_transformado[2]:.1f})", color='b')
                
                # Mostrar matriz y determinante
                matriz_str = f"T = [{matriz[0,0]:.1f} {matriz[0,1]:.1f} {matriz[0,2]:.1f}\n     {matriz[1,0]:.1f} {matriz[1,1]:.1f} {matriz[1,2]:.1f}\n     {matriz[2,0]:.1f} {matriz[2,1]:.1f} {matriz[2,2]:.1f}]"
                det = np.linalg.det(matriz)
                info_text = matriz_str + f"\n\ndet(T) = {det:.2f}"
                
                # Colocar texto en un lugar visible
                ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=10,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
            ax2.set_xlim([-1.5, 1.5])
            ax2.set_ylim([-1.5, 1.5])
            ax2.set_zlim([-1.5, 1.5])
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.legend()
        else:
            raise ValueError(f"La matriz de transformación debe ser 2x2 o 3x3, pero tiene dimensión {filas}x{columnas}")
        
        plt.suptitle(titulo)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def graficar_transformacion_lineal_rotados(matriz, grid_lines=10, titulo="Transformación Lineal", mostrar_etiquetas=True, figsize=(12, 6)):
        """
        Visualiza una transformación lineal 2D.
        
        Args:
            matriz (list): Matriz de transformación 2x2
            grid_lines (int, optional): Número de líneas de cuadrícula. Por defecto es 10.
            titulo (str, optional): Título de la gráfica. Por defecto es "Transformación Lineal".
            mostrar_etiquetas (bool, optional): Si es True, muestra etiquetas con los valores de la matriz. Por defecto es True.
            figsize (tuple, optional): Tamaño de la figura. Por defecto es (12, 6).
            
        Returns:
            None: Muestra la gráfica
            
        Example:
            AlgebraLineal.graficar_transformacion_lineal_rotados([[0, -1], [1, 0]])  # Rotación de 90 grados
        """
        matriz = np.array(matriz)
        
        if matriz.shape != (2, 2):
            raise ValueError("La matriz debe ser 2x2 para visualizar la transformación")
        
        # Crear una cuadrícula de puntos
        x = np.linspace(-5, 5, grid_lines)
        y = np.linspace(-5, 5, grid_lines)
        X, Y = np.meshgrid(x, y)
        puntos = np.vstack([X.flatten(), Y.flatten()])
        
        # Aplicar la transformación
        transformados = np.dot(matriz, puntos)
        
        # Reorganizar en cuadrículas para graficar
        X_transformado = transformados[0, :].reshape(grid_lines, grid_lines)
        Y_transformado = transformados[1, :].reshape(grid_lines, grid_lines)
        
        # Crear la figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Graficar la cuadrícula original
        ax1.set_title("Cuadrícula Original")
        ax1.set_xlim(-6, 6)
        ax1.set_ylim(-6, 6)
        ax1.grid(True)
        
        # Dibujar líneas horizontales y verticales de la cuadrícula original
        for i in range(grid_lines):
            ax1.plot(x, [y[i]] * len(x), 'b-', alpha=0.3)  # Líneas horizontales
            ax1.plot([x[i]] * len(y), y, 'b-', alpha=0.3)  # Líneas verticales
        
        # Graficar los vectores base
        ax1.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='r', label="e₁")
        ax1.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='g', label="e₂")
        ax1.text(1.1, 0, "e₁=(1,0)", color='r')
        ax1.text(0, 1.1, "e₂=(0,1)", color='g')
        
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax1.axvline(x=0, color='k', linestyle='-', alpha=0.5)
        ax1.legend()
        
        # Graficar la cuadrícula transformada
        ax2.set_title("Cuadrícula Transformada")
        
        # Calcular los límites basados en los puntos transformados
        max_val = max(np.max(np.abs(X_transformado)), np.max(np.abs(Y_transformado)))
        ax2.set_xlim(-max_val*1.2, max_val*1.2)
        ax2.set_ylim(-max_val*1.2, max_val*1.2)
        ax2.grid(True)
        
        # Dibujar líneas horizontales y verticales de la cuadrícula transformada
        for i in range(grid_lines):
            ax2.plot(X_transformado[i, :], Y_transformado[i, :], 'b-', alpha=0.3)  # Horizontales transformadas
            ax2.plot(X_transformado[:, i], Y_transformado[:, i], 'b-', alpha=0.3)  # Verticales transformadas
        
        # Graficar los vectores base transformados
        vector_e1_transformado = matriz @ np.array([1, 0])
        vector_e2_transformado = matriz @ np.array([0, 1])
        
        ax2.quiver(0, 0, vector_e1_transformado[0], vector_e1_transformado[1], 
                  angles='xy', scale_units='xy', scale=1, color='r', label="T(e₁)")
        ax2.quiver(0, 0, vector_e2_transformado[0], vector_e2_transformado[1], 
                  angles='xy', scale_units='xy', scale=1, color='g', label="T(e₂)")
        
        # Agregar etiquetas con las coordenadas de los vectores transformados
        if mostrar_etiquetas:
            ax2.text(vector_e1_transformado[0]*1.1, vector_e1_transformado[1]*1.1, 
                    f"T(e₁)=({vector_e1_transformado[0]:.2f},{vector_e1_transformado[1]:.2f})", color='r')
            ax2.text(vector_e2_transformado[0]*1.1, vector_e2_transformado[1]*1.1, 
                    f"T(e₂)=({vector_e2_transformado[0]:.2f},{vector_e2_transformado[1]:.2f})", color='g')
            
            # Agregar matriz de transformación como texto en la gráfica
            matriz_str = f"T = [{matriz[0,0]:.2f} {matriz[0,1]:.2f}\n     {matriz[1,0]:.2f} {matriz[1,1]:.2f}]"
            ax2.text(0.05, 0.95, matriz_str, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.5)
        ax2.legend()
        
        plt.suptitle(titulo)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def graficar_transformacion_lineal_3d_rotados(matriz, grid_lines=5, titulo="Transformación Lineal 3D", mostrar_etiquetas=True, figsize=(15, 7)):
        """
        Visualiza una transformación lineal 3D.
        
        Args:
            matriz (list): Matriz de transformación 3x3
            grid_lines (int, optional): Número de líneas de cuadrícula. Por defecto es 5.
            titulo (str, optional): Título de la gráfica. Por defecto es "Transformación Lineal 3D".
            mostrar_etiquetas (bool, optional): Si es True, muestra etiquetas con los valores de la matriz. Por defecto es True.
            figsize (tuple, optional): Tamaño de la figura. Por defecto es (15, 7).
            
        Returns:
            None: Muestra la gráfica
            
        Example:
            # Rotación alrededor del eje z
            matriz_rotacion = [
                [math.cos(math.pi/4), -math.sin(math.pi/4), 0],
                [math.sin(math.pi/4), math.cos(math.pi/4), 0],
                [0, 0, 1]
            ]
            AlgebraLineal.graficar_transformacion_lineal_3d_rotados(matriz_rotacion)
        """
        matriz = np.array(matriz)
        
        if matriz.shape != (3, 3):
            raise ValueError("La matriz debe ser 3x3 para visualizar la transformación")
        
        # Crear la figura con dos subplots
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Crear un cubo unitario
        # Puntos de las esquinas del cubo
        puntos = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ]).T - 0.5  # Centrar en el origen
        
        # Conectar las esquinas (aristas del cubo)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Cara inferior
            (4, 5), (5, 6), (6, 7), (7, 4),  # Cara superior
            (0, 4), (1, 5), (2, 6), (3, 7)   # Aristas verticales
        ]
        
        # Dibujar el cubo original
        ax1.set_title("Figura Original")
        for i, j in edges:
            x = [puntos[0, i], puntos[0, j]]
            y = [puntos[1, i], puntos[1, j]]
            z = [puntos[2, i], puntos[2, j]]
            ax1.plot(x, y, z, 'b-', alpha=0.5)
        
        # Dibujar los vectores base
        ax1.quiver(0, 0, 0, 1, 0, 0, color='r', label="e₁")
        ax1.quiver(0, 0, 0, 0, 1, 0, color='g', label="e₂")
        ax1.quiver(0, 0, 0, 0, 0, 1, color='b', label="e₃")
        
        if mostrar_etiquetas:
            ax1.text(1.1, 0, 0, "e₁=(1,0,0)", color='r')
            ax1.text(0, 1.1, 0, "e₂=(0,1,0)", color='g')
            ax1.text(0, 0, 1.1, "e₃=(0,0,1)", color='b')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
        ax1.legend()
        
        # Aplicar la transformación a los puntos
        puntos_transformados = matriz @ puntos
        
        # Dibujar el objeto transformado
        ax2.set_title("Figura Transformada")
        for i, j in edges:
            x = [puntos_transformados[0, i], puntos_transformados[0, j]]
            y = [puntos_transformados[1, i], puntos_transformados[1, j]]
            z = [puntos_transformados[2, i], puntos_transformados[2, j]]
            ax2.plot(x, y, z, 'b-', alpha=0.5)
        
        # Dibujar los vectores base transformados
        vector_e1_transformado = matriz @ np.array([1, 0, 0])
        vector_e2_transformado = matriz @ np.array([0, 1, 0])
        vector_e3_transformado = matriz @ np.array([0, 0, 1])
        
        ax2.quiver(0, 0, 0, vector_e1_transformado[0], vector_e1_transformado[1], vector_e1_transformado[2], 
                  color='r', label="T(e₁)")
        ax2.quiver(0, 0, 0, vector_e2_transformado[0], vector_e2_transformado[1], vector_e2_transformado[2], 
                  color='g', label="T(e₂)")
        ax2.quiver(0, 0, 0, vector_e3_transformado[0], vector_e3_transformado[1], vector_e3_transformado[2], 
                  color='b', label="T(e₃)")
        
        # Etiquetas y detalles
        if mostrar_detalle:
            ax2.text(vector_e1_transformado[0]*1.1, vector_e1_transformado[1]*1.1, vector_e1_transformado[2]*1.1, 
                    f"T(e₁)=({vector_e1_transformado[0]:.1f},{vector_e1_transformado[1]:.1f},{vector_e1_transformado[2]:.1f})", color='r')
            ax2.text(vector_e2_transformado[0]*1.1, vector_e2_transformado[1]*1.1, vector_e2_transformado[2]*1.1, 
                    f"T(e₂)=({vector_e2_transformado[0]:.1f},{vector_e2_transformado[1]:.1f},{vector_e2_transformado[2]:.1f})", color='g')
            ax2.text(vector_e3_transformado[0]*1.1, vector_e3_transformado[1]*1.1, vector_e3_transformado[2]*1.1, 
                    f"T(e₃)=({vector_e3_transformado[0]:.1f},{vector_e3_transformado[1]:.1f},{vector_e3_transformado[2]:.1f})", color='b')
            
            # Mostrar matriz y determinante
            matriz_str = f"T = [{matriz[0,0]:.1f} {matriz[0,1]:.1f} {matriz[0,2]:.1f}\n     {matriz[1,0]:.1f} {matriz[1,1]:.1f} {matriz[1,2]:.1f}\n     {matriz[2,0]:.1f} {matriz[2,1]:.1f} {matriz[2,2]:.1f}]"
            det = np.linalg.det(matriz)
            info_text = matriz_str + f"\n\ndet(T) = {det:.2f}"
            
            # Colocar texto en un lugar visible
            ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax2.set_xlim([-1.5, 1.5])
        ax2.set_ylim([-1.5, 1.5])
        ax2.set_zlim([-1.5, 1.5])
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        
        plt.suptitle(titulo)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def graficar_transformacion_lineal_3d(matriz, grid_lines=5, titulo="Transformación Lineal 3D", mostrar_etiquetas=True, figsize=(15, 7)):
        """
        Visualiza una transformación lineal 3D.
        
        Args:
            matriz (list): Matriz de transformación 3x3
            grid_lines (int, optional): Número de líneas de cuadrícula. Por defecto es 5.
            titulo (str, optional): Título de la gráfica. Por defecto es "Transformación Lineal 3D".
            mostrar_etiquetas (bool, optional): Si es True, muestra etiquetas con los valores de la matriz. Por defecto es True.
            figsize (tuple, optional): Tamaño de la figura. Por defecto es (15, 7).
            
        Returns:
            None: Muestra la gráfica
            
        Example:
            # Rotación alrededor del eje z
            matriz_rotacion = [
                [math.cos(math.pi/4), -math.sin(math.pi/4), 0],
                [math.sin(math.pi/4), math.cos(math.pi/4), 0],
                [0, 0, 1]
            ]
            AlgebraLineal.graficar_transformacion_lineal_3d(matriz_rotacion)
        """
        matriz = np.array(matriz)
        
        if matriz.shape != (3, 3):
            raise ValueError("La matriz debe ser 3x3 para visualizar la transformación")
        
        # Crear la figura con dos subplots
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Crear un cubo unitario
        # Puntos de las esquinas del cubo
        puntos = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ]).T - 0.5  # Centrar en el origen
        
        # Conectar las esquinas (aristas del cubo)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Cara inferior
            (4, 5), (5, 6), (6, 7), (7, 4),  # Cara superior
            (0, 4), (1, 5), (2, 6), (3, 7)   # Aristas verticales
        ]
        
        # Dibujar el cubo original
        ax1.set_title("Figura Original")
        for i, j in edges:
            x = [puntos[0, i], puntos[0, j]]
            y = [puntos[1, i], puntos[1, j]]
            z = [puntos[2, i], puntos[2, j]]
            ax1.plot(x, y, z, 'b-', alpha=0.5)
        
        # Dibujar los vectores base
        ax1.quiver(0, 0, 0, 1, 0, 0, color='r', label="e₁")
        ax1.quiver(0, 0, 0, 0, 1, 0, color='g', label="e₂")
        ax1.quiver(0, 0, 0, 0, 0, 1, color='b', label="e₃")
        
        if mostrar_etiquetas:
            ax1.text(1.1, 0, 0, "e₁=(1,0,0)", color='r')
            ax1.text(0, 1.1, 0, "e₂=(0,1,0)", color='g')
            ax1.text(0, 0, 1.1, "e₃=(0,0,1)", color='b')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
        ax1.legend()
        
        # Aplicar la transformación a los puntos
        puntos_transformados = matriz @ puntos
        
        # Dibujar el objeto transformado
        ax2.set_title("Figura Transformada")
        for i, j in edges:
            x = [puntos_transformados[0, i], puntos_transformados[0, j]]
            y = [puntos_transformados[1, i], puntos_transformados[1, j]]
            z = [puntos_transformados[2, i], puntos_transformados[2, j]]
            ax2.plot(x, y, z, 'b-', alpha=0.5)
        
        # Dibujar los vectores base transformados
        vector_e1_transformado = matriz @ np.array([1, 0, 0])
        vector_e2_transformado = matriz @ np.array([0, 1, 0])
        vector_e3_transformado = matriz @ np.array([0, 0, 1])
        
        ax2.quiver(0, 0, 0, vector_e1_transformado[0], vector_e1_transformado[1], vector_e1_transformado[2], 
                  color='r', label="T(e₁)")
        ax2.quiver(0, 0, 0, vector_e2_transformado[0], vector_e2_transformado[1], vector_e2_transformado[2], 
                  color='g', label="T(e₂)")
        ax2.quiver(0, 0, 0, vector_e3_transformado[0], vector_e3_transformado[1], vector_e3_transformado[2], 
                  color='b', label="T(e₃)")
        
        # Etiquetas y detalles
        if mostrar_detalle:
            ax2.text(vector_e1_transformado[0]*1.1, vector_e1_transformado[1]*1.1, vector_e1_transformado[2]*1.1, 
                    f"T(e₁)=({vector_e1_transformado[0]:.1f},{vector_e1_transformado[1]:.1f},{vector_e1_transformado[2]:.1f})", color='r')
            ax2.text(vector_e2_transformado[0]*1.1, vector_e2_transformado[1]*1.1, vector_e2_transformado[2]*1.1, 
                    f"T(e₂)=({vector_e2_transformado[0]:.1f},{vector_e2_transformado[1]:.1f},{vector_e2_transformado[2]:.1f})", color='g')
            ax2.text(vector_e3_transformado[0]*1.1, vector_e3_transformado[1]*1.1, vector_e3_transformado[2]*1.1, 
                    f"T(e₃)=({vector_e3_transformado[0]:.1f},{vector_e3_transformado[1]:.1f},{vector_e3_transformado[2]:.1f})", color='b')
            
            # Mostrar matriz y determinante
            matriz_str = f"T = [{matriz[0,0]:.1f} {matriz[0,1]:.1f} {matriz[0,2]:.1f}\n     {matriz[1,0]:.1f} {matriz[1,1]:.1f} {matriz[1,2]:.1f}\n     {matriz[2,0]:.1f} {matriz[2,1]:.1f} {matriz[2,2]:.1f}]"
            det = np.linalg.det(matriz)
            info_text = matriz_str + f"\n\ndet(T) = {det:.2f}"
            
            # Colocar texto en un lugar visible
            ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax2.set_xlim([-1.5, 1.5])
        ax2.set_ylim([-1.5, 1.5])
        ax2.set_zlim([-1.5, 1.5])
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        
        plt.suptitle(titulo)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def graficar_transformacion_lineal(matriz, grid_lines=10, titulo="Transformación Lineal", mostrar_etiquetas=True, figsize=(12, 6)):
        """
        Visualiza una transformación lineal 2D.
        
        Args:
            matriz (list): Matriz de transformación 2x2
            grid_lines (int, optional): Número de líneas de cuadrícula. Por defecto es 10.
            titulo (str, optional): Título de la gráfica. Por defecto es "Transformación Lineal".
            mostrar_etiquetas (bool, optional): Si es True, muestra etiquetas con los valores de la matriz. Por defecto es True.
            figsize (tuple, optional): Tamaño de la figura. Por defecto es (12, 6).
            
        Returns:
            None: Muestra la gráfica
            
        Example:
            AlgebraLineal.graficar_transformacion_lineal([[0, -1], [1, 0]])  # Rotación de 90 grados
        """
        matriz = np.array(matriz)
        
        if matriz.shape != (2, 2):
            raise ValueError("La matriz debe ser 2x2 para visualizar la transformación")
        
        # Crear una cuadrícula de puntos
        x = np.linspace(-5, 5, grid_lines)
        y = np.linspace(-5, 5, grid_lines)
        X, Y = np.meshgrid(x, y)
        puntos = np.vstack([X.flatten(), Y.flatten()])
        
        # Aplicar la transformación
        transformados = np.dot(matriz, puntos)
        
        # Reorganizar en cuadrículas para graficar
        X_transformado = transformados[0, :].reshape(grid_lines, grid_lines)
        Y_transformado = transformados[1, :].reshape(grid_lines, grid_lines)
        
        # Crear la figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Graficar la cuadrícula original
        ax1.set_title("Cuadrícula Original")
        ax1.set_xlim(-6, 6)
        ax1.set_ylim(-6, 6)
        ax1.grid(True)
        
        # Dibujar líneas horizontales y verticales de la cuadrícula original
        for i in range(grid_lines):
            ax1.plot(x, [y[i]] * len(x), 'b-', alpha=0.3)  # Líneas horizontales
            ax1.plot([x[i]] * len(y), y, 'b-', alpha=0.3)  # Líneas verticales
        
        # Graficar los vectores base
        ax1.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='r', label="e₁")
        ax1.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='g', label="e₂")
        ax1.text(1.1, 0, "e₁=(1,0)", color='r')
        ax1.text(0, 1.1, "e₂=(0,1)", color='g')
        
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax1.axvline(x=0, color='k', linestyle='-', alpha=0.5)
        ax1.legend()
        
        # Graficar la cuadrícula transformada
        ax2.set_title("Cuadrícula Transformada")
        
        # Calcular los límites basados en los puntos transformados
        max_val = max(np.max(np.abs(X_transformado)), np.max(np.abs(Y_transformado)))
        ax2.set_xlim(-max_val*1.2, max_val*1.2)
        ax2.set_ylim(-max_val*1.2, max_val*1.2)
        ax2.grid(True)
        
        # Dibujar líneas horizontales y verticales de la cuadrícula transformada
        for i in range(grid_lines):
            ax2.plot(X_transformado[i, :], Y_transformado[i, :], 'b-', alpha=0.3)  # Horizontales transformadas
            ax2.plot(X_transformado[:, i], Y_transformado[:, i], 'b-', alpha=0.3)  # Verticales transformadas
        
        # Graficar los vectores base transformados
        vector_e1_transformado = matriz @ np.array([1, 0])
        vector_e2_transformado = matriz @ np.array([0, 1])
        
        ax2.quiver(0, 0, vector_e1_transformado[0], vector_e1_transformado[1], 
                  angles='xy', scale_units='xy', scale=1, color='r', label="T(e₁)")
        ax2.quiver(0, 0, vector_e2_transformado[0], vector_e2_transformado[1], 
                  angles='xy', scale_units='xy', scale=1, color='g', label="T(e₂)")
        
        # Agregar etiquetas con las coordenadas de los vectores transformados
        if mostrar_etiquetas:
            ax2.text(vector_e1_transformado[0]*1.1, vector_e1_transformado[1]*1.1, 
                    f"T(e₁)=({vector_e1_transformado[0]:.2f},{vector_e1_transformado[1]:.2f})", color='r')
            ax2.text(vector_e2_transformado[0]*1.1, vector_e2_transformado[1]*1.1, 
                    f"T(e₂)=({vector_e2_transformado[0]:.2f},{vector_e2_transformado[1]:.2f})", color='g')
            
            # Agregar matriz de transformación como texto en la gráfica
            matriz_str = f"T = [{matriz[0,0]:.2f} {matriz[0,1]:.2f}\n     {matriz[1,0]:.2f} {matriz[1,1]:.2f}]"
            ax2.text(0.05, 0.95, matriz_str, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.5)
        ax2.legend()
        
        plt.suptitle(titulo)
        plt.tight_layout()
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
                             o None (sistema sin solucion)
                  - tipo_solucion: String indicando el tipo de solución ("unica", "infinitas", "sin solucion")
            
        Example:
            resultado, tipo = AlgebraLineal.gauss_jordan([[1, 1, 1, 6], [2, -1, 3, 9], [3, 2, -4, 3]])
            # Para el sistema: x + y + z = 6, 2x - y + 3z = 9, 3x + 2y - 4z = 3
        """        # Hacer una copia profunda de la matriz para no modificar la original
        matriz = []
        for fila in matriz_aumentada:
            matriz.append([Fraction(elem) for elem in fila])
        
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
            if max_valor < AlgebraLineal.TOLERANCIA:  # Usar tolerancia para errores de punto flotante
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
          # Verificar si el sistema es sin solucion
        for i in range(fila_actual, filas):
            # Si hay una fila con todos ceros excepto el término independiente
            if all(abs(matriz[i][j]) < AlgebraLineal.TOLERANCIA for j in range(n_variables)) and abs(matriz[i][n_variables]) > AlgebraLineal.TOLERANCIA:
                return None, "sin solucion"
          # Verificar si el sistema tiene infinitas soluciones
        if fila_actual < n_variables:
            # Usar Fraction para mantener resultados como fracciones
            return [matriz[i][n_variables] for i in range(n_variables)], "infinitas"
        
        # El sistema tiene solución única
        # Devolver los resultados como fracciones para mayor claridad
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
                - tipo_solucion: String indicando el tipo de solución ("unica", "infinitas", "sin solucion")
        
        Example:
            solucion, tipo = AlgebraLineal.gauss([[1, 1, 1, 6], [2, -1, 3, 9], [3, 2, -4, 3]])
            # Para el sistema: x + y + z = 6, 2x - y + 3z = 9, 3x + 2y - 4z = 3
        """        # Hacer una copia profunda de la matriz para no modificar la original
        amat = []
        for fila in matriz_aumentada:
            amat.append([Fraction(elem) for elem in fila])
        n_eq = len(amat)  # Número de ecuaciones (filas)
        n_var = len(amat[0]) - 1  # Número de variables (columnas - 1)
        
        if verbose:
            print("\n--- Sistema ingresado ---")
            for i in range(n_eq):
                for j in range(n_var):
                    print(f"{amat[i][j]}x_{j+1}", end="")
                    if j < n_var - 1:
                        print(" + ", end="")
                print(f" = {amat[i][n_var]}")
        
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
            if abs(amat[k][k]) < AlgebraLineal.TOLERANCIA:
                is_sngl = True
                continue
            
            # Eliminar la variable actual de las ecuaciones siguientes
            for i in range(k + 1, n_eq):
                if abs(amat[i][k]) > AlgebraLineal.TOLERANCIA and abs(amat[k][k]) > AlgebraLineal.TOLERANCIA:
                    fact = amat[i][k] / amat[k][k]
                    for j in range(k, n_var + 1):
                        amat[i][j] -= fact * amat[k][j]
        if verbose:
            print("\n--- Matriz después de la Eliminación Gaussiana ---")
            for i in range(n_eq):
                for j in range(n_var + 1):
                    print(f"{amat[i][j]}\t", end="")
                print()
        
        # Verificar si el sistema es sin solucion
        is_incn = False
        for i in range(num_piv, n_eq):
            all_z_cof = True
            for j in range(n_var):
                if abs(amat[i][j]) > AlgebraLineal.TOLERANCIA:
                    all_z_cof = False
                    break
            
            if all_z_cof and abs(amat[i][n_var]) > AlgebraLineal.TOLERANCIA:
                is_incn = True
                break
        
        if is_incn:
            if verbose:
                print("\nEl sistema es sin solucion.")
            return None, "sin solucion"
        elif is_sngl or n_eq < n_var:
            if verbose:
                print("\nEl sistema tiene infinitas soluciones.")
            return None, "infinitas"
        else:            # Sustitución hacia atrás
            sol = [Fraction(0) for _ in range(n_var)]
            try:
                for i in range(n_var - 1, -1, -1):
                    curr_sum = Fraction(0)
                    for j in range(i + 1, n_var):
                        curr_sum += amat[i][j] * sol[j]
                    
                    if abs(amat[i][i]) < AlgebraLineal.TOLERANCIA:
                        if verbose:
                            print("\nEl sistema tiene infinitas soluciones o es sin solucion.")
                        return None, "infinitas"
                    
                    sol[i] = (amat[i][n_var] - curr_sum) / amat[i][i]
                if verbose:
                    print("\n--- Soluciones ---")
                    for i in range(n_var):
                        print(f"Solución x_{i+1} = {sol[i]}")
                
                return sol, "unica"
            
            except Exception:
                if verbose:
                    print("\nError en el cálculo: El sistema tiene infinitas soluciones o es sin solucion.")
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
        """        # Crear la matriz aumentada [A|b]
        matriz_aumentada = []
        for i in range(len(coeficientes)):
            fila = coeficientes[i].copy()
            fila.append(terminos_independientes[i])
            matriz_aumentada.append(fila)
            
        return AlgebraLineal.gauss_jordan(matriz_aumentada)
    # ======================== ANÁLISIS DE INDEPENDENCIA LINEAL ========================
    
    @staticmethod
    def calcular_rango(matriz):
        """
        Calcula el rango de una matriz utilizando eliminación gaussiana.
        
        Args:
            matriz (list): Matriz de entrada
            
        Returns:
            int: Rango de la matriz
        """
        # Hacer una copia profunda de la matriz para no modificar la original
        mat = [fila[:] for fila in matriz]
        filas = len(mat)
        if filas == 0:
            return 0
        columnas = len(mat[0])
        
        # Variable para rastrear la fila y columna actual durante la eliminación
        fila_actual = 0
        for columna_actual in range(columnas):
            # Si estamos al final de las filas, terminamos
            if fila_actual >= filas:
                break
                
            # Encontrar el pivote máximo en esta columna (por estabilidad numérica)
            max_fila = fila_actual
            max_valor = abs(mat[fila_actual][columna_actual])
            
            for i in range(fila_actual + 1, filas):
                if abs(mat[i][columna_actual]) > max_valor:
                    max_fila = i
                    max_valor = abs(mat[i][columna_actual])
            
            # Si el máximo es cero, esta columna ya está reducida
            if max_valor < AlgebraLineal.TOLERANCIA:
                continue
                
            # Intercambiar filas si es necesario
            if max_fila != fila_actual:
                mat[fila_actual], mat[max_fila] = mat[max_fila], mat[fila_actual]
                
            # Normalizar la fila del pivote
            pivote = mat[fila_actual][columna_actual]
            for j in range(columna_actual, columnas):
                mat[fila_actual][j] /= pivote
                
            # Eliminar esta variable de las otras filas
            for i in range(filas):
                if i != fila_actual:
                    factor = mat[i][columna_actual]
                    for j in range(columna_actual, columnas):
                        mat[i][j] -= factor * mat[fila_actual][j]
        
            fila_actual += 1
        
        # El rango es el número de filas no nulas
        return fila_actual
    
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
            return False, f"Todos los vectores deben tener la misma dimension"
        
        # Caso especial: si algún vector es el vector cero
        for i, v in enumerate(vectores):
            if all(abs(comp) < AlgebraLineal.TOLERANCIA for comp in v):
                return False, f"El vector {i+1} es el vector cero, por lo que el conjunto no es linealmente independiente."
        
        # Caso especial: si hay un solo vector no nulo
        if n_vectores == 1:
            return True, "Un conjunto con un solo vector no nulo es linealmente independiente."
        
        # Formar la matriz con los vectores como columnas para calcular su rango
        matriz = AlgebraLineal.transpuesta(vectores)
        
        # Calcular el rango de la matriz usando nuestro método optimizado
        rango = AlgebraLineal.calcular_rango(matriz)
        
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
        resultado = [Fraction(0) for _ in range(dimension)]
        
        # Convertir vectores y coeficientes a Fraction
        vectores_frac = [[Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in v] for v in vectores]
        coefs_frac = [Fraction(c) if not isinstance(c, Fraction) else c for c in coeficientes]
        
        # Sumar cada vector multiplicado por su coeficiente
        for j in range(len(vectores_frac)):
            vector = vectores_frac[j]
            coef = coefs_frac[j]
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
            tuple: (bool, coeficientes o mensaje)
            
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
        
        if tipo == "sin solucion":
            return False, "El vector no es combinación lineal del conjunto dado."
        
        # Verificar la solución reconstruyendo el vector
        reconstruccion = AlgebraLineal.combinacion_lineal(conjunto_vectores, solucion)
        # Comparar con cierta tolerancia debido a errores de punto flotante
        es_igual = all(abs(reconstruccion[i] - vector[i]) < AlgebraLineal.TOLERANCIA for i in range(dimension))
        
        if es_igual:
            return True, solucion
        else:
            return False, "El vector no es combinación lineal del conjunto dado."
    @staticmethod
    def _determinante_gauss(matriz):
        """
        Calcula el determinante usando eliminación gaussiana.
        Este método es más eficiente para matrices grandes.
        
        Args:
            matriz (list): Matriz cuadrada
            
        Returns:
            float: Determinante de la matriz
        """
        # Crear copia de la matriz para no modificar la original
        n = len(matriz)
        
        # Convertir a Fraction para cálculos exactos
        mat = [[Fraction(elem) if not isinstance(elem, Fraction) else elem for elem in fila] for fila in matriz]
        
        # Factor que rastrea el cambio en el determinante
        det = Fraction(1)
        
        # Eliminación gaussiana
        for i in range(n):
            # Si el elemento diagonal es cero, intercambiar con una fila posterior
            if abs(mat[i][i]) < AlgebraLineal.TOLERANCIA:
                # Buscar fila con elemento no cero en esta columna
                for k in range(i + 1, n):
                    if abs(mat[k][i]) > AlgebraLineal.TOLERANCIA:
                        mat[i], mat[k] = mat[k], mat[i]
                        # Cada intercambio cambia el signo del determinante
                        det = -det
                        break
                else:
                    # Si no se encontró fila con elemento no cero, el determinante es cero
                    return Fraction(0)
            
            # Multiplicar el determinante por el pivote diagonal
            det *= mat[i][i]
            
            # Eliminar elementos debajo del pivote
            for k in range(i + 1, n):
                factor = mat[k][i] / mat[i][i]
                for j in range(i, n):
                    mat[k][j] -= factor * mat[i][j]
        
        return det



