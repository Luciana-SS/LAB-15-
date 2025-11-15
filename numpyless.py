"""
╔════════════════════════════════════════════════════════════════════════╗
║                             NumpyLess                                ║
║                  Pure Python Linear Algebra Library                    ║
║                      (NumPy-less, stress-more!)                        ║
╚════════════════════════════════════════════════════════════════════════╝

Una biblioteca minimalista de álgebra lineal que implementa operaciones
tipo NumPy usando solo Python puro. ¡Perfecta para entender qué pasa
"bajo el cap

Uso Recomendado:
    import numpyless as npl

    # O para máxima ironía:
    import numpyless as np  # ¡Cuidado con esto!

Tipos de Datos:
- Vector: list[float] - Un array 1D de flotantes
- Matriz: list[list[float]] - Un array 2D de flotantes (filas x columnas)
"""

# --- Alias de Tipos Nativos ---
Vector = list[float]
Matriz = list[list[float]]

# -------------------------------------------------------------------
# Sección 1: Creación de Arrays (⭐ Básico)
# -------------------------------------------------------------------


def zeros(shape: tuple[int, int]) -> Matriz:
    filas = shape[0]
    columnas = shape [1]
    matriz_zeros = []
    for i in range(filas):
        fila = []
        for j in range(columnas):
            fila.append(0)
            matriz_zeros.append(fila)
    return matriz_zeros
    

def ones(shape: tuple[int, int]) -> Matriz:
    matriz_unos = []
    columnas = shape[1]
    filas = shape[0]
    columna = []
    for i in range(filas):
        columna.append(1.0)
    for i in range(columnas):
        matriz_unos.append(columna)
    return matriz_unos




def identity(n: int) -> Matriz:
    """Crea una matriz identidad cuadrada.

    Equivalente en NumPy: np.identity(n)

    Args:
        n: El tamaño (número de filas y columnas) de la matriz.

    Returns:
        Matriz: Una matriz identidad de n x n.

    Ejemplo:
        >>> identity(3)
        [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]]

    Pista: La diagonal tiene 1.0 cuando fila == columna
    """
    columnas = n
    filas = n

    matriz_identidad = []

    for i in range (filas):
        fila = []
        for j in range (columnas):
            if j == i:
                fila.append(1)
            else:
                fila.append(0)
        
        matriz_identidad.append(fila)
    return matriz_identidad


# -------------------------------------------------------------------
# Sección 2: Información de Arrays (⭐ Básico)
# -------------------------------------------------------------------


def shape(A: Matriz) -> tuple[int, int]:
    """Devuelve las dimensiones de una matriz como (filas, columnas).

    Equivalente en NumPy: A.shape

    Args:
        A: La matriz de entrada.

    Returns:
        tuple[int, int]: Una tupla (filas, columnas).

    Ejemplo:
        >>> shape([[1, 2, 3], [4, 5, 6]])
        (2, 3)

    Pista: len(A) da filas, len(A[0]) da columnas
    """
    raise NotImplementedError("Función no implementada.")


def transpose(A: Matriz) -> Matriz:
    filas = len(A)        
    columnas = len(A[0]) 
    transpuesta = []
    for j in range(columnas):
        nueva_fila = []  
        for i in range(filas):
            nueva_fila.append(A[i][j]) 
        transpuesta.append(nueva_fila)

    return transpuesta



# -------------------------------------------------------------------
# Sección 3: Operaciones con Vectores (⭐⭐ Intermedio)
# -------------------------------------------------------------------


def dot(v: Vector, w: Vector) -> float:
    if len(v) == len(w):
        producto_punto = 0.0
        for n, m in zip(v, w):
            producto = n * m
            producto_punto += producto
        return producto_punto
    else:
        raise ValueError("Los vectores tienen dimensiones diferentes")
            
            


def add(v: Vector, w: Vector) -> Vector:
    """Suma dos vectores elemento a elemento.

    Equivalente en NumPy: v + w

    Args:
        v: El primer vector.
        w: El segundo vector.

    Returns:
        Vector: El vector resultante de la suma.

    Raises:
        ValueError: Si los vectores no tienen la misma dimensión.

    Ejemplo:
        >>> add([1, 2], [3, 4])
        [4.0, 6.0]

    Pista: Usa listas por comprensión con zip()
    """
    
    dimension_v = len(v)
    dimension_w = len(w)

    if dimension_v != dimension_w:
        raise ValueError ("los vectores no tienen la misma dimensión")

    vector = []

    for vector1, vector2 in zip(v, w):
        suma = vector1 + vector2
        vector.append(suma)

    return vector
    

def multiply(c: float, v: Vector) -> Vector:
    """Multiplica cada elemento de un vector por un escalar.

    Equivalente en NumPy: c * v

    Args:
        c: El escalar.
        v: El vector.

    Returns:
        Vector: El vector resultante escalado.

    Ejemplo:
        >>> multiply(2.5, [1, 2, 3])
        [2.5, 5.0, 7.5]

    Pista: Multiplica c por cada elemento
    """
    raise NotImplementedError("Función no implementada.")


def norm(v: Vector) -> float:
    """Calcula la magnitud (norma L2) de un vector.

    Fórmula: ||v|| = sqrt(v[0]² + v[1]² + ... + v[n]²)

    Equivalente en NumPy: np.linalg.norm(v)

    Args:
        v: El vector.

    Returns:
        float: La magnitud del vector.

    Ejemplo:
        >>> norm([3, 4])
        5.0  # = sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25)

    Pista: Usa dot(v, v) y luego sqrt() del módulo math
    """

    import math 
    suma = 0 
    for x in v:
        suma += x**2
    norma = sqrt(suma)
    return norma 
    
    

# -------------------------------------------------------------------
# Sección 4: Operaciones con Matrices (⭐⭐ Intermedio)
# -------------------------------------------------------------------


def add_matrices(A: Matriz, B: Matriz) -> Matriz:
    matriz_suma = []
    contador = 0
    if len(A) == len(B) and len(A[0]) == len(B[0]):
        for fila_A, fila_B in zip(A, B):
            matriz_suma.append([])
            for elemento_A, elemento_B in zip(fila_A, fila_B):
                suma = elemento_A + elemento_B
                matriz_suma[contador].append(suma)
            contador += 1
        return matriz_suma
    else:
        raise ValueError("Las matrices no se pueden sumar")
        
    """Suma dos matrices elemento a elemento.

    Equivalente en NumPy: A + B

    Args:
        A: La primera matriz.
        B: La segunda matriz.

    Returns:
        Matriz: La matriz resultante de la suma.

    Raises:
        ValueError: Si las matrices no tienen la misma forma.

    Ejemplo:
        >>> add_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        [[6.0, 8.0], [10.0, 12.0]]

    Pista: Suma elemento a elemento, fila por fila
    """
    raise NotImplementedError("Función no implementada.")


def multiply_matrix(c: float, A: Matriz) -> Matriz:
    """Multiplica cada elemento de una matriz por un escalar.

    Equivalente en NumPy: c * A

    Args:
        c: El escalar.
        A: La matriz.

    Returns:
        Matriz: La matriz resultante escalada.

    Ejemplo:
        >>> multiply_matrix(2, [[1, 2], [3, 4]])
        [[2.0, 4.0], [6.0, 8.0]]

    Pista: Similar a multiply() pero para cada fila
    """

    filas = len (A)
    columnas = len(A[0])

    Matriz = []

    for i in range (filas):
        nueva_fila = []
        for j in range (columnas):
            dato = A[i][j]
            multiplicacion = c * dato
            nueva_fila.append(multiplicacion)

        Matriz.append(nueva_fila)

    return Matriz



def matmul(A: Matriz, B: Matriz | Vector) -> Matriz | Vector:
    """Multiplica una matriz A por una matriz B o vector v.

    Regla: El número de columnas de A debe ser igual al número de
           filas de B (o longitud de v).

    Equivalente en NumPy: A @ B

    Args:
        A: La matriz izquierda (m × n).
        B: La matriz derecha (n × p) o vector (n).

    Returns:
        Matriz (m × p) o Vector (m): El resultado de la multiplicación.

    Raises:
        ValueError: Si las dimensiones no son compatibles.

    Ejemplos:
        >>> matmul([[1, 2]], [3, 4])
        [11.0]  # = [1*3 + 2*4]

        >>> matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        [[19.0, 22.0], [43.0, 50.0]]

    Pista: Para matrices, cada elemento resultado[i][j] es el
           producto punto de la fila i de A con la columna j de B
    """
    raise NotImplementedError("Función no implementada.")


# -------------------------------------------------------------------
# Sección 5: Álgebra Lineal (⭐⭐⭐ Avanzado - Opcional/Extra)
# -------------------------------------------------------------------


def det(A: Matriz) -> float:
    """Calcula el determinante de una matriz cuadrada.

    NOTA: Esta es la función más difícil. Es opcional pero da puntos extra.

    Para matriz 2×2:
        det([[a, b], [c, d]]) = a*d - b*c

    Para matriz 3×3 y mayores:
        Usa expansión de cofactores (recursivo) o eliminación gaussiana.

    Equivalente en NumPy: np.linalg.det(A)

    Args:
        A: La matriz cuadrada.

    Returns:
        float: El valor del determinante.

    Raises:
        ValueError: Si la matriz no es cuadrada.

    Ejemplos:
        >>> det([[4, 3], [2, 1]])
        -2.0  # = 4*1 - 3*2

        >>> det([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        1.0  # determinante de identidad = 1

    Pistas:
    - Caso base: matriz 1×1 devuelve el único elemento
    - Caso 2×2: usa la fórmula directa
    - Caso 3×3+: expansión por primera fila (recursivo)
    """
    raise NotImplementedError("Función no implementada.")






