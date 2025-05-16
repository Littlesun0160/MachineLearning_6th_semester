import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print("Имена сортов:", iris.target_names)
iris_colors = ['#cb416b','pink','#a2cffe']

# Визуализация сортов зависимости
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i in range(3):
    subset = df[df['target'] == i]
    plt.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], color = iris_colors[i], label=iris.target_names[i])
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Sepal Length -- Sepal Width')
plt.legend()

plt.subplot(1, 2, 2)
for i in range(3):
    subset = df[df['target'] == i]
    plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'], color = iris_colors[i], label=iris.target_names[i])
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Petal Length -- Petal Width')
plt.legend()
plt.tight_layout()

#Вывод всего датасета
sns.pairplot(df, hue='target', palette='husl')

#Берем 2 датасета: setosa и versicolor \ versicolor и virginica
df1 = df[df['target'].isin([0, 1])].copy()
df2 = df[df['target'].isin([1, 2])].copy()

#Обучение и оценка модели
def train_and_test(df, test_size=0.3, random_state=42):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели: {accuracy:.4f}")
    print(f"Коэффициенты наклона: {clf.coef_}")
    print(f"Коэффициент пересечения: {clf.intercept_}")
    return clf, X_test, y_test

print("\nОбучение модели для датасета №1 (setosa и versicolor):")
model1, X_test1, y_test1 = train_and_test(df1)
print("\nОбучение модели для датасета №2 (versicolor и virginica):")
model2, X_test2, y_test2 = train_and_test(df2)

#генерация случайного датасета и бинарная классификация
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='berlin')
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.title("Случайно сгенерированный датасет")
plt.colorbar()

df_random = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
df_random['target'] = y
print("\nОбучение модели для случайного датасета:")
model_random, X_test_random, y_test_random = train_and_test(df_random)

plt.show()
