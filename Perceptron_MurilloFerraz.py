import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, eta=1.0, epochs=20, seed=42):
        self.eta = eta
        self.epochs = epochs
        self.seed = seed
        self.w = None
        self.b = None
        self.errors_history = []
    
    def fit(self, X, y):
        np.random.seed(self.seed)
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        
        # Treinamento
        for epoch in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                z = np.dot(self.w, xi) + self.b
                
                if yi * z <= 0:
                    self.w += self.eta * yi * xi
                    self.b += self.eta * yi
                    errors += 1
            
            self.errors_history.append(errors)
            
            if errors == 0:
                print(f"Convergiu na época {epoch + 1}")
                break
        
        return self
    
    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return np.where(z > 0, 1, -1)
    
    def plot_decision_boundary(self, X, y, title="Fronteira de Decisão"):
        if X.shape[1] != 2:
            print("Erro: plot_decision_boundary funciona apenas para dados 2D")
            return
        
        plt.figure(figsize=(10, 8))
        
        for label in np.unique(y):
            mask = y == label
            color = 'green' if label == 1 else 'red'
            marker = 'o'
            plt.scatter(X[mask, 0], X[mask, 1], 
                       c=color, marker=marker, s=150, 
                       label=f'y = {label:+d}', edgecolors='black', linewidth=2)
        
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        
        if abs(self.w[1]) > 1e-10:  # nao deixar dividir por 0
            x1_line = np.linspace(x_min, x_max, 100)
            x2_line = -(self.w[0] * x1_line + self.b) / self.w[1]
            plt.plot(x1_line, x2_line, 'orange', linewidth=3, 
                    label=f'Fronteira: {self.w[0]:.2f}x₁ + {self.w[1]:.2f}x₂ + {self.b:.2f} = 0')
        else:
            x1_boundary = -self.b / self.w[0]
            plt.axvline(x=x1_boundary, color='orange', linewidth=3,
                       label=f'Fronteira: x₁ = {x1_boundary:.2f}')
        
        xx1, xx2 = np.meshgrid(np.linspace(x_min, x_max, 100),
                               np.linspace(y_min, y_max, 100))
        Z = self.predict(np.c_[xx1.ravel(), xx2.ravel()])
        Z = Z.reshape(xx1.shape)
        
        plt.contourf(xx1, xx2, Z, alpha=0.2, levels=[-1.5, 0, 1.5], 
                    colors=['red', 'green'])
        
        plt.xlabel('x₁', fontsize=14, fontweight='bold')
        plt.ylabel('x₂', fontsize=14, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.tight_layout()
        plt.show()


#teste do and
print("\nPROBLEMA AND")

X_and = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

y_and = np.array([-1, -1, -1, 1])

print("\nDados de entrada (AND):")
print("X:", X_and)
print("y:", y_and)

perceptron_and = Perceptron(eta=1.0, epochs=20, seed=42)
perceptron_and.fit(X_and, y_and)

print(f"\nPesos finais: w = {perceptron_and.w}")
print(f"Viés final: b = {perceptron_and.b}")

y_pred_and = perceptron_and.predict(X_and)
print("\nPredições:")
for i, (xi, yi_true, yi_pred) in enumerate(zip(X_and, y_and, y_pred_and)):
    status = "✓" if yi_true == yi_pred else "✗"
    print(f"  x = {xi}, y_true = {yi_true:+d}, y_pred = {yi_pred:+d} {status}")

perceptron_and.plot_decision_boundary(X_and, y_and, title="Problema AND")


# teste do or
print("\nPROBLEMA OR")

X_or = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])

y_or = np.array([-1, 1, 1, 1])

print("\nDados de entrada (OR):")
print("X:", X_or)
print("y:", y_or)

perceptron_or = Perceptron(eta=1.0, epochs=20, seed=42)
perceptron_or.fit(X_or, y_or)

print(f"\nPesos finais: w = {perceptron_or.w}")
print(f"Viés final: b = {perceptron_or.b}")

y_pred_or = perceptron_or.predict(X_or)
print("\nPredições:")
for i, (xi, yi_true, yi_pred) in enumerate(zip(X_or, y_or, y_pred_or)):
    status = "✓" if yi_true == yi_pred else "✗"
    print(f"  x = {xi}, y_true = {yi_true:+d}, y_pred = {yi_pred:+d} {status}")

perceptron_or.plot_decision_boundary(X_or, y_or, title="Problema OR")
