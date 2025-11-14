import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

horas = np.array([2, 4, 6, 8])
resultado = np.array([0, 0, 1, 1])
X = sm.add_constant(horas) # Adiciona a coluna de 1 p/ o intercepto (beta0)

def sigmoid(z):
    """
    Função de ativação sigmoide.
    """
    return 1 / (1 + np.exp(-z))

def gradienteDescendente(X, y, eta=0.5, T=20000):
    """
    Implementa o algoritmo de Gradiente Descendente para Regressão Logística.
    
    Args:
        X (np.ndarray): Conjunto de treinamento com features.
        y (np.ndarray): Rótulos de treinamento.
        eta (float): Taxa de aprendizado.
        T (int): Número máximo de iterações.
    Returns:
        np.ndarray: Vetor de parâmetros estimados (beta).
    """
    beta = np.random.uniform(0, 1, size=X.shape[1])
    n_samples = len(y)
    
    for t in range(T):
        # Calcula predições: p_hat ← σ(X @ beta)
        p = sigmoid(X @ beta)

        # Calcula gradiente médio: g ← (1/n) * X.T @ (p_hat - y)
        g = (X.T @ (p - y)) / n_samples

        # Atualiz parâmetros: beta ← beta - eta * g
        beta -= eta * g

        # Critério de parada: Se a norma do vetor de atualização for pequena
        if np.linalg.norm(eta * g) < 1e-8:
            break
            
    return beta

betas_estimados = gradienteDescendente(X, resultado)
beta0, beta1 = betas_estimados[0], betas_estimados[1]

print(f"Valores estimados:")
print(f"beta0 = {beta0:.2f}")
print(f"beta1 = {beta1:.2f}")

x_values = np.linspace(min(horas) - 2, max(horas) + 2, 100)
sigmoid_curve = sigmoid(beta0 + beta1 * x_values)

plt.figure(figsize=(8, 6))
plt.plot(x_values, sigmoid_curve, color='blue', label='Curva Sigmoide Ajustada')
plt.scatter(horas, resultado, color='red', zorder=5, label='Dados de Treinamento')
plt.title('Regressão Logística com Gradiente Descendente')
plt.xlabel('Horas de Estudo')
plt.ylabel('Probabilidade de Aprovação (0 ou 1)')
plt.grid(True)
plt.legend()
plt.show()