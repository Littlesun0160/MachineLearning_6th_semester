import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Загрузка набора данных diabetes и исследование
diabetes = datasets.load_diabetes()
print("Описание набора данных:\n", diabetes.DESCR)
print("Имена признаков:\n", diabetes.feature_names)
# В данной задаче используем столбец 'bmi' (индекс в массиве 2)
index = 2
X = diabetes.data[:, np.newaxis, index]
Y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Реализация Scikit-Learn
sklearn = LinearRegression()
sklearn.fit(X_train, y_train)
sklearn_predictions = sklearn.predict(X_test)

print("\nЛинейная регрессия при помощи Scikit-Learn:")
a = sklearn.coef_[0]
b = sklearn.intercept_
print(f"Коэффициент наклона: {a}")
print(f"Коэффициент пересечения: {b}")
print(f"Таким образом, имеем уравнение регрессии: y = {a} * x + {b}")

# Используем метод наименьших квадратов
class Least_Squares_Method:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    def fit(self, X, y):
        X_b = np.concatenate((np.ones((len(X), 1)), X), axis=1)
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
    def predict(self, X):
        return self.intercept_ + X @ self.coef_

my_algorithm = Least_Squares_Method()
my_algorithm.fit(X_train, y_train)
my_predictions = my_algorithm.predict(X_test)

print("\nСобственный алгоритм регрессии (метод наименьших квадратов):")
a = my_algorithm.coef_[0]
b = my_algorithm.intercept_
print(f"Коэффициент наклона: {a}")
print(f"Коэффициент пересечения: {b}")
print(f"Таким образом, имеем уравнение регрессии: y = {a} * x + {b}")

#Вывод графиков
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='purple', label='Данные точки')
plt.plot(X_test, sklearn_predictions, color='red', linewidth=3, label='Scikit-Learn метод')
plt.plot(X_test, my_predictions, color='pink', linewidth=3, label='Мой метод')
plt.xlabel(diabetes.feature_names[index])
plt.ylabel('Target')
plt.title('Линейная регрессия разными методами')
plt.legend()
plt.show()

# Таблица с предсказаниями
results = pd.DataFrame({'Исходные данные': y_test,'Scikit-Learn предсказание': sklearn_predictions,
        'Мое предсказание': my_predictions})
print("\nТаблица предсказаний:")
print(results)