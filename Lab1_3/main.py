import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import check_array

# Загрузка набора данных diabetes
diabetes = datasets.load_diabetes()

# Используем столбец 'bmi' (индекс в массиве 2)
index = 2
X = diabetes.data[:, np.newaxis, index]
Y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Реализация Scikit-Learn
sklearn = LinearRegression()
sklearn.fit(X_train, y_train)
sklearn_predictions = sklearn.predict(X_test)

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

# Реализация MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Метрики для Scikit-Learn модели
print("\nМетрики для Scikit-Learn модели:")
print(f"MAE: {mean_absolute_error(y_test, sklearn_predictions):.2f}")
print(f"R2: {r2_score(y_test, sklearn_predictions):.2f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, sklearn_predictions):.2f}%")

# Метрики для собственной модели
print("\nМетрики для моей модели:")
print(f"MAE: {mean_absolute_error(y_test, my_predictions):.2f}")
print(f"R2: {r2_score(y_test, my_predictions):.2f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, my_predictions):.2f}%")

# Вывод о качестве модели
print("\nВывод о моделях:")
print("Обе модели (Scikit-Learn и собственная реализация метода наименьших квадратов) показывают ")
print("похожие результаты, что показано одинаковыми значениями метрик MAE, R2 и MAPE.  Это очевидно, ")
print("потому как обе модели реализуют один и тот же алгоритм линейной регрессии")