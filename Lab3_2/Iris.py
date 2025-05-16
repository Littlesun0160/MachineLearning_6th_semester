import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print("Имена сортов:", iris.target_names)
X = df[['petal length (cm)', 'petal width (cm)']]
y = df['target']
iris_colors = ['#cb416b', 'pink', '#a2cffe']

#1) Разбиваем данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#3 и 4) Многоклассовая логистическая регрессия
model = LogisticRegression(random_state=0, solver='lbfgs')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.4f}")
print(f"Коэффициенты наклона: {model.coef_}")
print(f"Коэффициент пересечения: {model.intercept_}")

x_min, x_max = X['petal length (cm)'].min() - 1, X['petal length (cm)'].max() + 1
y_min, y_max = X['petal width (cm)'].min() - 1, X['petal width (cm)'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

grid_data = np.c_[xx.ravel(), yy.ravel()]
grid_df = pd.DataFrame(grid_data, columns=['petal length (cm)', 'petal width (cm)'])
Z = model.predict(grid_df)
Z = Z.reshape(xx.shape)

#Изобразим полученное решение
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)
for i in range(3):
    subset = df[df['target'] == i]
    plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'], color=iris_colors[i], label=iris.target_names[i])
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Многоклассовая логистическая регрессия')
plt.legend()
plt.show()