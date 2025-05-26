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
        """
        # Hacer una copia profunda de la matriz para no modificar la original
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
        else:
            # Sustitución hacia atrás
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
            
            except Exception as e:
                if verbose:
                    print(f"\nError en el cálculo: {e}")
                    print("El sistema tiene infinitas soluciones o es sin solucion.")
                return None, "indeterminado"
