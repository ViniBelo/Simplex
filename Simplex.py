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
    else:
        return 'Não é necessário entrar na Fase I'


# Formulação do problema artificial
def formulaProblemaArtificial ():
    # Desenvolver na sequência!
    return 0

# 2.2) {custos relativos}
def calcular_custos_relativos(lambdaT, x_N, x_chapeu_N):
    custos_relativos = []

    for j in range(3):
        coluna_index = j
        coluna = x_N[:, coluna_index]
        print(f'coluna: {coluna}')
        custo_relativo = x_chapeu_N[j] - np.dot(lambdaT, coluna)
        custos_relativos.append(custo_relativo)
    
    return custos_relativos


# Início da iteração simplex - Fase I
def faseI ():
    # Passo 1: {cálculo da solução básica}
    x_chapeu_B = np.zeros(n)
    inversaB = np.linalg.inv(x_B)
    x_chapeu_B = multiplicacaoDeMatrizes(b, inversaB)
    x_chapeu_N = np.zeros(x_N.shape[1])
    print(f'x_chapeu_B: {x_chapeu_B}')
    print(f'x_chapeu_N: {x_chapeu_N}')

    # Passo 2: {cálculo dos custos relativos}
    #     2.1) {vetor multiplicador simplex}
    cBT = fx
    lambdaT = multiplicacaoDeMatrizes(inversaB, cBT)
    print(f'lambdaT: {lambdaT}')

    #     2.2) {custos relativos}
    print(f'x_N:\n{x_N}')
    custos_relativos = calcular_custos_relativos(lambdaT, x_N, x_chapeu_N)
    print(f'custos_relativos: {custos_relativos}')

    #     2.3) {determinação da variável a entrar na base}
    k = np.argmin(custos_relativos)  # Índice da variável com menor custo relativo
    print(f'Variável a entrar na base: x_N{k+1}')

    # Passo 3: {teste de otimalidade}
    if all(custo >= 0 for custo in custos_relativos):
        print('Solução ótima encontrada.')
        #return 0
    else:
        'Solução não é ótima.'
        #return faseII()
    
    # Passo 4: {cálculo da direção simplex}
    y = np.dot(inversaB, x_N[:, k])
    print(f'y = {y}')

    # Passo 5: {determina¸c˜ao do passo e vari´avel a sair da base}
    if all(y <= 0 for y in y):
        print("Problema não tem solução ótima finita. Problema Original Infactível.")
        return
    else:
        min_ratio = float('inf')
        saindo_da_base = -1

        for i in range(m):
            if y[i] > 0:
                print(f'ratio = {x_chapeu_B[i]} / {y[i]}')
                ratio = x_chapeu_B[i] / y[i]
                print(f'Resultado ratio: {ratio}')
                if ratio < min_ratio:
                    min_ratio = ratio
                    saindo_da_base = i

        if saindo_da_base == -1:
            print("Não foi possível determinar a variável a sair da base.")
            return
        else:
            print(f"Variável a sair da base: x_B{saindo_da_base + 1}") #x_b2 = x4
            passo = min_ratio
            print(f"Passo: {passo}")

# def faseII():

if __name__ == '__main__':
    print(verificacaoFaseI(obj))
