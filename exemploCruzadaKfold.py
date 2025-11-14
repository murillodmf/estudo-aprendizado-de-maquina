import numpy as np

# -------------------------------
# Algoritmo: Validação Cruzada k-Fold
# -------------------------------

def ValidacaoCruzada(dados, k, modelo, metrica, seed=None):
    # 2: Dividir dados em k subconjuntos de tamanhos aproximadamente iguais
    n = len(dados[0])
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    metricas = []

    # 3: para i = 1,...,k faça
    for i in range(k):
        # 4: dados_teste ← subconjunto i
        dados_teste = folds[i]

        # 5: dados_treinamento ← união dos outros k − 1 subconjuntos
        dados_treinamento = np.hstack([folds[j] for j in range(k) if j != i])

        X, y = dados
        X_train, y_train = X[dados_treinamento], y[dados_treinamento]
        X_test, y_test = X[dados_teste], y[dados_teste]

        # 6: Treinar modelo usando dados_treinamento
        modelo_treinado = modelo(X_train, y_train)

        # 7: Avaliar modelo usando dados_teste
        y_pred = modelo_treinado["predict"](X_test)

        # 8: Armazenar métrica de desempenho
        valor_metrica = metrica(y_test, y_pred)
        metricas.append(valor_metrica)

        print(f"Fold {i+1}: Métrica = {valor_metrica:.4f}")

    # 10: retorne Média das métricas de desempenho
    return np.mean(metricas)


# -------------------------------
# Exemplo de uso com Regressão Linear
# -------------------------------

def modelo_linear(X_train, y_train):
    """Treina regressão linear via pseudo-inversa."""
    X_design = np.hstack([np.ones((X_train.shape[0],1)), X_train])
    theta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y_train

    def predict(X):
        X_design = np.hstack([np.ones((X.shape[0],1)), X])
        return X_design @ theta

    return {"theta": theta, "predict": predict}

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


# -------------------------------
# Executando exemplo
# -------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    n = 20
    X = np.linspace(0, 10, n).reshape(-1,1)
    y = 4.0 + 3.0 * X[:,0] + np.random.normal(scale=2.0, size=n)
    y = y.reshape(-1,1)

    media_mse = ValidacaoCruzada((X, y), k=5, modelo=modelo_linear, metrica=mse, seed=42)
    print(f"\nMédia das métricas de desempenho (MSE): {media_mse:.4f}")
