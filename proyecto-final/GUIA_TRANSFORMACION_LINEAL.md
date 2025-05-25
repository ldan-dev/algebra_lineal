# Guía para usar los métodos de Transformación Lineal en la Interfaz

## 1. Método `transformacion_lineal`

Este método calcula la matriz de una transformación lineal dado un conjunto de vectores base del dominio y sus correspondientes imágenes en el codominio.

### Parámetros en la interfaz:

- **Base del Dominio**: Matriz donde cada fila representa un vector de la base del dominio.
- **Base del Codominio**: Matriz donde cada fila representa el vector imagen correspondiente.

### Ejemplo:

Para representar la rotación de 90 grados antihorario en R²:

**Base del Dominio**:
```
1 0
0 1
```

**Base del Codominio**:
```
0 -1
1 0
```

Esto nos dará como resultado la matriz de transformación: `[[0, -1], [1, 0]]`.

## 2. Método `visualizar_transformacion_lineal`

Este método visualiza el efecto de una matriz de transformación, mostrando el espacio original y el transformado.

### Parámetros en la interfaz:

- **Matriz de Transformación**: La matriz que representa la transformación lineal.
- **Título**: Título personalizado para la gráfica.
- **Líneas de cuadrícula**: Número de líneas de la cuadrícula (valor recomendado: 10).
- **Mostrar detalles**: Si se activa, muestra detalles como las coordenadas y la matriz en la gráfica.
- **Tamaño de figura**: El tamaño de la figura en pulgadas (ancho, alto).

### Comportamiento:

- Para matrices 2x2: Muestra una transformación en el plano (R²).
- Para matrices 3x3: Muestra una transformación en el espacio (R³).

### Ejemplos:

**Rotación 45 grados (2D)**:
```
0.7071 -0.7071
0.7071 0.7071
```

**Escalamiento (2D)**:
```
2 0
0 3
```

**Rotación alrededor del eje Z (3D)**:
```
0.7071 -0.7071 0
0.7071 0.7071 0
0 0 1
```

## Consejos de uso:

1. Al introducir matrices, asegúrese de que las dimensiones sean correctas (2x2 o 3x3).
2. Para transformaciones en 2D, use matrices 2x2.
3. Para transformaciones en 3D, use matrices 3x3.
4. Si la visualización no aparece correctamente, intente ajustar el número de líneas de cuadrícula.
5. Puede combinar ambos métodos: primero calcular la matriz con `transformacion_lineal` y luego visualizarla con `visualizar_transformacion_lineal`.
