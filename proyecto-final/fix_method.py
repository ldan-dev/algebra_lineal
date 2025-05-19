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
    import numpy as np
    import matplotlib.pyplot as plt
    
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
    
    # Agregar etiquetas con las coordenadas de los vectores transformados
    if mostrar_etiquetas:
        ax2.text(vector_e1_transformado[0]*1.1, vector_e1_transformado[1]*1.1, vector_e1_transformado[2]*1.1, 
                f"T(e₁)=({vector_e1_transformado[0]:.1f},{vector_e1_transformado[1]:.1f},{vector_e1_transformado[2]:.1f})", color='r')
        ax2.text(vector_e2_transformado[0]*1.1, vector_e2_transformado[1]*1.1, vector_e2_transformado[2]*1.1, 
                f"T(e₂)=({vector_e2_transformado[0]:.1f},{vector_e2_transformado[1]:.1f},{vector_e2_transformado[2]:.1f})", color='g')
        ax2.text(vector_e3_transformado[0]*1.1, vector_e3_transformado[1]*1.1, vector_e3_transformado[2]*1.1, 
                f"T(e₃)=({vector_e3_transformado[0]:.1f},{vector_e3_transformado[1]:.1f},{vector_e3_transformado[2]:.1f})", color='b')
        
        # Agregar matriz de transformación como texto en la gráfica
        matriz_str = f"T = [{matriz[0,0]:.1f} {matriz[0,1]:.1f} {matriz[0,2]:.1f}\n     {matriz[1,0]:.1f} {matriz[1,1]:.1f} {matriz[1,2]:.1f}\n     {matriz[2,0]:.1f} {matriz[2,1]:.1f} {matriz[2,2]:.1f}]"
        # Use a fixed position for the text (at the corner of the axes)
        ax2.text(max_val*0.8, -max_val*0.8, -max_val*0.8, matriz_str, fontsize=10)
    
    # Calcular los límites para mantener la proporción en la vista transformada
    max_val = max(np.max(np.abs(puntos_transformados)))
    ax2.set_xlim(-max_val*1.2, max_val*1.2)
    ax2.set_ylim(-max_val*1.2, max_val*1.2)
    ax2.set_zlim(-max_val*1.2, max_val*1.2)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    plt.suptitle(titulo)
    plt.tight_layout()
    plt.show()
