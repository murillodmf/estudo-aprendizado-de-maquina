import pandas as pd
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.cond_probs = {}
        self.classes = []
        self.features = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        X: DataFrame com os atributos
        y: Series com a classe
        """
        N = len(y)
        self.classes = y.unique()
        self.features = X.columns

        self.class_priors = (y.value_counts() / N).to_dict()

        self.cond_probs = {}
        for c in self.classes:
            X_c = X[y == c]
            self.cond_probs[c] = {}
            for feat in self.features:
                counts = X_c[feat].value_counts(normalize=True)
                self.cond_probs[c][feat] = counts.to_dict()

    def predict(self, x: dict):
        """
        x: dicionário com atributos do exemplo
        """
        probs = {}
        for c in self.classes:
            prob = self.class_priors[c]
            for feat in self.features:
                prob *= self.cond_probs[c][feat].get(x[feat], 0)
            probs[c] = prob

        return max(probs, key=probs.get), probs



# Dataset da Parte 1
X = pd.DataFrame([
    {"Clima": "Ensolarado", "Humor": "Feliz"},
    {"Clima": "Ensolarado", "Humor": "Triste"},
    {"Clima": "Chuvoso", "Humor": "Feliz"},
    {"Clima": "Chuvoso", "Humor": "Triste"},
    {"Clima": "Ensolarado", "Humor": "Feliz"},
])

y = pd.Series(["Sim", "Não", "Não", "Não", "Sim"])

# Treinamento
nb = NaiveBayes()
nb.fit(X, y)

# Teste
exemplo = {"Clima": "Ensolarado", "Humor": "Feliz"}
pred, probs = nb.predict(exemplo)

print("Exemplo:", exemplo)
print("Probabilidades:", probs)
print("Classe prevista:", pred)
