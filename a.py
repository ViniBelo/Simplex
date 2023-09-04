from copy import deepcopy
import numpy as np


#Funções Auxiliares:
#Função de mostrar matriz
def print_matriz(matriz):
    for linha in matriz:
        print(linha)


#Função de imprimir resultado final
def imprimir_resultado(funciona, otima, basicas, fx):
    if funciona:
        print("A solução factível ótima é:")
        resultado_final = calculo_funcaoZ(fx, otima, basicas)
        print(', '.join(
            [f'x{basicas[i]} = {otima[i]}'
             for i in range(len(otima))]) + f', z = {resultado_final}')
        print(
            "\n------------------------------ Finalizou! ------------------------------"
        )
    else:
        print(
            "Em algum momento não é possível fazer a inversa ou a direção simplex é <= 0."
        )
        print(
            "------------------------------ Finalizou! -------------------------------"
        )


#Função para calcular as submatrizes básicas e não básicas
def calcula_submatriz(matrizA: list, vetorX: list) -> float:
    subM = []
    for j in range(len(matrizA)):
        linha = []
        for i in range(len(vetorX)):
            linha.append(matrizA[j][vetorX[i]])
        subM.append(linha)
    return subM


#Função de troca de linha
def troca_linha(matriz, i, j):
    matriz[i], matriz[j] = matriz[j], matriz[i]


#Função para calcular a inversa utilizando eliminação de gauss
def calcula_inversa(matrizA: list) -> float:
    det = np.linalg.det(matrizA)
    if (det != 0):
        n = len(matrizA)
        inv = [[0] * n for _ in range(n)]
        for i in range(n):
            inv[i][i] = 1
        for i in range(n):
            if matrizA[i][i] == 0:
                c = 1
                while i + c < n and matrizA[i + c][i] == 0:
                    c += 1
                if i + c == n:
                    return False  # A matriz não é inversível
                troca_linha(matrizA, i, i + c)
                troca_linha(inv, i, i + c)
            pivo = matrizA[i][i]
            for j in range(n):
                matrizA[i][j] /= pivo
                inv[i][j] /= pivo
            for j in range(n):
                if i != j:
                    fator = matrizA[j][i]
                    for k in range(n):
                        matrizA[j][k] -= fator * matrizA[i][k]
                        inv[j][k] -= fator * inv[i][k]
        for i in range(n):
            if matrizA[i][i] != 1:
                p = 1 / matrizA[i][i]
                for j in range(n):
                    matrizA[i][j] *= p
                    inv[i][j] *= p
        return inv
    return False


# Simplec
# Fase I:
def separa_matriz(funcaoObjetiva: list, restricoes: list) -> list:
    # Definicao das variaveis de folga
    ineq = []
    tamanho = len(restricoes)
    for i in range(tamanho):
        cond = restricoes[i][-2]
        if cond != '=':
            funcaoObjetiva.append(0.0)
            if (cond == '<='):
                ineq.append(1.0)
            else:
                ineq.append(-1.0)
        else:
            ineq.append(0.0)
    # Fim da definicao das variaveis de folga
    # Inicio do aumento da matriz A
    matrizA = []
    for i in range(tamanho):
        linha = restricoes[i][:-2]
        tamanho2 = len(ineq)
        for j in range(tamanho2):
            if (i == j):
                linha.append(ineq[j])
            else:
                linha.append(0.0)
        matrizA.append(linha)
    # Fim do aumento da matriz A
    # Inicio da definicao basicas e nao basicas
    basicas = []
    naoBasicas = []
    tam = len(funcaoObjetiva)
    for i in range(tam):
        if (i < tam - tamanho):
            naoBasicas.append(i)
        else:
            basicas.append(i)
    basicas.sort()
    naoBasicas.sort()
    # Fim da definicao basicas e nao basicas
    # Inicio da criacao da matriz de termos independentes(b)
    b = []
    for i in range(tamanho):
        b.append(restricoes[i][-1])
    # Fim da criacao da matriz de termos independentes(b)
    # Restrição B tem elemento menor que 0
    for i in range(len(b)):
        if b[i] < 0:
            for j in range(len(matrizA[i])):
                matrizA[i][j] = -matrizA[i][j]
            b[i] = -b[i]

    return matrizA, basicas, naoBasicas, b


# Fase II: {início da iteração}

# Passo 1: {cálculo da solução básica}


def calculo_relativo(BInv: list, b: list) -> float:
    return np.matmul(BInv, np.transpose(np.matrix(b)))


#   Passo 2: {cálculo dos custos relativos}


#       2.1) {vetor multiplicador simplex}
def calcula_custo(funcaoObjetiva: list, variaveis: list) -> float:
    custoBasico = [0] * len(variaveis)
    for i in range(len(custoBasico)):
        custoBasico[i] = funcaoObjetiva[variaveis[i]]
    return custoBasico


def calcula_lambda(custoBasico: list, basicaInversa: list) -> float:
    return np.matmul(custoBasico, basicaInversa)


#       2.2) {custos relativos}
def custos_Relativos(lambdaSimplex: list, custoNaoBasico: list,
                     matrizNaoBasica: list) -> float:
    naoBasicaTransposta = np.transpose(matrizNaoBasica)
    for i in range(len(custoNaoBasico)):
        custoNaoBasico[i] -= (np.dot(lambdaSimplex, naoBasicaTransposta[i]))
    return custoNaoBasico


#       2.3) {determinação da variável a entrar na base}
def calcula_k(custoRelativoNaoBasico: list) -> int:
    return custoRelativoNaoBasico.index(min(custoRelativoNaoBasico))


#   Passo 3: {teste de otimalidade}
def verifica_otimo(custoRelativoNaoBasico: list, k: int) -> bool:
    return custoRelativoNaoBasico[k] >= 0


#   Passo 4: {cálculo da direção simplex}
def direcao_s(BasicaInversa: list, matrizA: list, k: int,
              naoBasicas: list) -> float:
    colunaK = [matrizA[i][naoBasicas[k]] for i in range(len(matrizA))]
    colunaK = np.transpose(colunaK)
    y = np.matmul(BasicaInversa, colunaK)
    return y


#   Passo 5: {determinação do passo e variável a sair da base}
def calcula_l(y: list, xRelativoBasico: list) -> int:
    seguro = any(y[i] > 0 for i in range(len(y)))
    if not seguro:
        return False
    razoes = [
        xRelativoBasico[i] / y[i] if y[i] > 0 else float('inf')
        for i in range(len(xRelativoBasico))
    ]
    passo = min(razoes)
    l = razoes.index(passo)
    return l


#   Passo 6: {atualização: nova partição básica, troque a l-ésima coluna de B pela k-ésima
#   coluna de N}


def troca_linhas_k_l(basicas: list, naoBasicas: list, k: int, l: int) -> list:
    basicas[l], naoBasicas[k] = naoBasicas[k], basicas[l]
    return basicas, naoBasicas


# 3. Calcula função final
def calculo_funcaoZ(funcaoObjetiva: list, xRelativoBasico: list,
                    basicas: list) -> float:
    resultado = sum(funcaoObjetiva[basicas[i]] * xRelativoBasico[i]
                    for i in range(len(xRelativoBasico)))
    return resultado


# Simplex
def calculo_simplex(tipoProblema, funcaoObjetiva, restricoes):
    it = 0
    maxit = 20
    otima = []
    funciona = True
    matrizA, basicas, naoBasicas, b = separa_matriz(funcaoObjetiva, restricoes)
    fx = deepcopy(funcaoObjetiva)
    tamanho = len(funcaoObjetiva)
    if tipoProblema == 'max':
        for i in range(tamanho):
            funcaoObjetiva[i] *= -1

    while it < maxit:
        print(f'\nit: {it+1}')
        matrizBasica = calcula_submatriz(matrizA, basicas)
        matrizNaoBasica = calcula_submatriz(matrizA, naoBasicas)

        print('MatrizA: ')
        print_matriz(matrizA)

        print('Matriz Básica: ')
        print_matriz(matrizBasica)

        print('Matriz Não Básica: ')
        print_matriz(matrizNaoBasica)

        matrizBasicaInversa = calcula_inversa(matrizBasica)

        print('Matriz Básica Inversa: ')
        print_matriz(matrizBasicaInversa)

        if matrizBasicaInversa is False:
            funciona = False
            break

        xRelativo = calculo_relativo(matrizBasicaInversa, b)

        custoBasico = calcula_custo(funcaoObjetiva, basicas)
        print('Custo Básica: ', custoBasico)

        lambdaTransposto = calcula_lambda(custoBasico, matrizBasicaInversa)

        custoNaoBasico = calcula_custo(funcaoObjetiva, naoBasicas)
        print('Custo Não Básica: ', custoNaoBasico)

        custoRelativoNaoBasico = custos_Relativos(lambdaTransposto,
                                                  custoNaoBasico,
                                                  matrizNaoBasica)
        print('Custo Relativo Não Basico: ', custoRelativoNaoBasico)
        k = calcula_k(custoRelativoNaoBasico)
        if verifica_otimo(custoRelativoNaoBasico, k):
            print(
                "\n-------------------------------- Ótimo! --------------------------------\n"
            )
            otima = xRelativo
            funciona = True
            break
        print(
            "\n------------------------------ Não ótimo! ------------------------------\n"
        )
        y = direcao_s(matrizBasicaInversa, matrizA, k, naoBasicas)
        l = calcula_l(y, xRelativo)
        if isinstance(l, bool) and l is False:
            funciona = False
            break
        basicas, naoBasicas = troca_linhas_k_l(basicas, naoBasicas, k, l)
        it += 1
    # Fim do laco de repeticao simplex
    imprimir_resultado(funciona, otima, basicas, fx)


# Chama a função main
if __name__ == "__main__":
    # Exercício 2.e) Cap 5
    tipoProblema = 'min'
    funcaoObjetiva = [-5, -2]
    restricoes = [
        [7, -5, '<=', 13],
        [3, 2, '<=', 17],
        [0, 1, '<=', 2],
        [1, 0, '>=', 4],
        # [1, 0,'<=',3],
    ]
    print("\nCalculo 1")
    calculo_simplex(tipoProblema, funcaoObjetiva, restricoes)
    # # Exercício 1.k) Cap 5
    # tipoProblema = 'min'
    # funcaoObjetiva = [-1, -1, 0]
    # restricoes = [
    #     [1, 0, 3, '>=', 1],
    #     [1, -3, -1, '>=', 1],
    #     [1, -1, 5, '>=', 5],
    #     [1, 1, 1, '<=', 5]
    # ]
    # print("\nCalculo 2")
    # calculo_simplex(tipoProblema, funcaoObjetiva, restricoes)
    # # Exercício 1.d) Cap 5
    # tipoProblema = 'max'
    # funcaoObjetiva = [3, 1]
    # restricoes = [
    #               [2, 1, '>=', 30],
    #               [1, 4, '<=', 40]
    #              ]
    # print("\nCalculo 3")
    # calculo_simplex(tipoProblema, funcaoObjetiva, restricoes)