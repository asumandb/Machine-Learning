import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

diabetes = load_diabetes()
X = diabetes.data[:, 2].reshape(-1, 1)
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, color = "blue", label = "Gerçek İlerleme")
plt.plot(X_test, y_pred, color = "red", label = "Tahmin Edilen İlerleme")
plt.title("Diyabet Veri Seti İle Hastalık İlerlemesi Tahmini")
plt.xlabel("Vücut Kitle İndeksi(BMI)")
plt.ylabel("Hastalığın İlerlemesi")
plt.legend()
plt.show()
