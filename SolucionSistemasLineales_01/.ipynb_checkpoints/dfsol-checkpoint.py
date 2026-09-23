import numpy as np

def sol_analitica(x, TA, TB, q, L, k):
    """
    Calcula la temperatura usando la fórmula (2).
    """
    return ((TB - TA)/L + q /(2*k) * (L - x) ) * x + TA

def sistema_lineal(TA, TB, q, L, k, N):
    # Calculo de la separación entre puntos
    h = L / (N+1)

    # Definición inicial del vector del lado derecho (RHS)
    Q = np.full(N, q * h**2 / k)
    
    # Definición inicial de la matriz
    A = np.identity(N)

    # Primer renglón de la matriz
    A[0,0] = 2
    A[0,1] = -1

    # Renglones internos de la matriz
    for i in range(1,N-1):
        A[i,i-1] = -1
        A[i,i]   = 2
        A[i,i+1] = -1

    # Último renglón de la matriz
    A[-1,-2] = -1
    A[-1,-1] = 2
    
    # Aplicación de las condiciones de frontera
    Q[0]  += TA
    Q[-1] += TB
    
    # Regresa la matriz y el RHS
    return A, Q