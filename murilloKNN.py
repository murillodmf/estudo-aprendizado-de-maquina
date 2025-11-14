import numpy as np
from collections import Counter

class KNN:
    """
    Implementação do classificador k-Nearest Neighbors (k-NN) do zero.

    Parâmetros
    ----------
    k : int, opcional (default=3)
        Número de vizinhos a serem considerados para a votação.
    metric : str, opcional (default='euclidean')
        Métrica de distância a ser usada. Opções: 'euclidean', 
        'manhattan', 'chebyshev', 'minkowski'.
    p : int, opcional (default=None)
        Parâmetro para a distância de Minkowski. Se a métrica for 
        'minkowski', 'p' deve ser especificado.
    """
    def __init__(self, k=3, metric='euclidean', p=None):
        if metric == 'minkowski' and p is None:
            raise ValueError("O parâmetro 'p' deve ser especificado para a distância de Minkowski.")
        
        self.k = k
        self.metric = metric
        self.p = p

    def _calculate_distance(self, p1, p2):
        """Calcula a distância entre dois pontos p1 e p2."""
        p1 = np.array(p1)
        p2 = np.array(p2)
        
        if self.metric == 'euclidean':
            # Distância Euclidiana (L2)
            return np.sqrt(np.sum((p1 - p2)**2))
        
        elif self.metric == 'manhattan':
            # Distância de Manhattan (L1)
            return np.sum(np.abs(p1 - p2))
            
        elif self.metric == 'chebyshev':
            # Distância de Chebyshev (L-inf)
            return np.max(np.abs(p1 - p2))

        elif self.metric == 'minkowski':
            # Distância de Minkowski (Lp)
            return np.power(np.sum(np.power(np.abs(p1 - p2), self.p)), 1/self.p)



    def fit(self, X_train, y_train):

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _predict_single_point(self, x_test_point):
        """Prevê a classe de um único ponto de teste."""
        
        distances = [self._calculate_distance(x_test_point, x_train_point) for x_train_point in self.X_train]
        
        
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_distances = [distances[i] for i in k_indices]


        vote_counts = Counter(k_nearest_labels)
        
        max_count = max(vote_counts.values())
        
        tied_classes = [cls for cls, count in vote_counts.items() if count == max_count]

        if len(tied_classes) == 1:
            return tied_classes[0]
        
        else:
            weighted_votes = {cls: 0 for cls in tied_classes}
            
            for label, dist in zip(k_nearest_labels, k_nearest_distances):
                if label in tied_classes:
                    weight = 1 / (dist + 1e-9)
                    weighted_votes[label] += weight
            
            return max(weighted_votes, key=weighted_votes.get)

    def predict(self, X_test):
        """
        Prevê as classes para um conjunto de dados de teste.

        Parâmetros
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Dados de teste.

        Retorna
        -------
        predictions : array, shape (n_samples,)
            Array com as classes preditas para cada amostra em X_test.
        """
        predictions = [self._predict_single_point(x_test_point) for x_test_point in X_test]
        return np.array(predictions)