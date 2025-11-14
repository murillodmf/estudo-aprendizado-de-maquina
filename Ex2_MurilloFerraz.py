import numpy as np

y = np.array([46800, 49200, 47900, 51300, 50100])
x1 = np.array([24, 26, 25, 28, 27])
x2 = np.array([540, 580, 560, 600, 590])
x3 = np.array([410, 415, 405, 420, 418])

X = np.column_stack((np.ones(len(y)), x1, x2, x3))

b = np.linalg.inv(X.T @ X) @ X.T @ y

print("Coeficientes do modelo:")
print(f"b0 (intercepto): {b[0]:.4f}")
print(f"b1 (temp):       {b[1]:.4f}")
print(f"b2 (prod):       {b[2]:.4f}")
print(f"b3 (horas):      {b[3]:.4f}")

y_pred = X @ b

print("\nValores previstos vs reais:")
for i in range(len(y)):
    print(f"Mês {i+1}: Real = {y[i]:.1f}  |  Previsto = {y_pred[i]:.1f}")

rmse = np.sqrt(np.mean((y - y_pred)**2))
print(f"\nRMSE: {rmse:.2f}")
