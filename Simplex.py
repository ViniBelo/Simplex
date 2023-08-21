# fx = Função objetivo
# x_B = Básica, ou seja, restrições
# x_N = Não básica
# b Vetor dos resultados das restrições

import numpy as np

obj = 'min'
fx = np.array([1, 0])

x_B = np.array([[1, 0],
                [0, 1],])

x_N = np.array([[1, 1, 1],
                [2, -1, 3]])

b = np.array([3, 4])

sinal = ['=', '<=']

m = x_N.shape[0]
n = x_B.shape[0] * x_B.shape[1]

# Precisamos entender as variáveis b e n.
# def troca(mat, x_B, b, x_N, n, i, j):
#     aux = b[i]
#     b[i] = n[j]
#     n[j] = aux
#     x_B = attSubMatriz(mat, b)
#     x_N = attSubMatriz(mat, n)
#     return x_B, b, x_N, n

# def Multiplicacao_vetores(vetorA: list, vetorB:list) -> float:
#     res = np.dot(vetorA, vetorB)
#     return res

def multiplicacaoDeMatrizes(matrizA, matrizB):
    return np.matmul(matrizA, matrizB)

# Verificação da necessidade da Fase I
def verificacaoFaseI (obj):
    # Primeira verificação
    if obj == 'max':
        for i in range(len(fx)):
            fx[i] *= -1
        obj = 'min'

    # Segunda verificação
    for i in range(m):
        if b[i] < 0:
            for j in range(m):
                x_B[i][j] *= -1
                # Inverter sinal
    
    # Terceira verificação
    if sinal:
        return faseI()


# Formulação do problema artificial
def formulaProblemaArtificial ():
    # Desenvolver na sequência!
    return 0

# 2.2) {custos relativos}
def calcular_custos_relativos(lambdaT, aNj):
    custos_relativos = []

    for j in range(len(lambdaT)):
        custo_relativo = fx[j] - np.dot(lambdaT, aNj[:, j])
        custos_relativos.append(custo_relativo)
    
    return custos_relativos

# Início da iteração simplex - Fase I
def faseI ():
    # Passo 1: {cálculo da solução básica}
    x_chapeu_B = np.zeros(n)
    inversaB = np.linalg.inv(x_B)
    x_chapeu_B = multiplicacaoDeMatrizes(b, inversaB)
    x_chapeu_N = np.zeros(x_N.shape[1])
    print(x_chapeu_B)

    # Passo 2: {cálculo dos custos relativos}
    #     2.1) {vetor multiplicador simplex}
    cBT = fx
    lambdaT = multiplicacaoDeMatrizes(inversaB, cBT)
    print(lambdaT)

    #     2.2) {custos relativos}
    custos_relativos = calcular_custos_relativos(lambdaT, x_N)
    print(f'custos_relativos: {custos_relativos}')

    #     2.3) {determinação da variável a entrar na base}
    k = np.argmin(custos_relativos)  # Índice da variável com menor custo relativo
    print(f'Variável a entrar na base: x_N{k+1}')




if __name__ == '__main__':
    print(verificacaoFaseI(obj))
