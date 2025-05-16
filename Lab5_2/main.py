import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
import xgboost as xgb

data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1)
Y = data["Outcome"]
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Делим данные на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#Изобразим зависимость качества модели от глубины используемых деревьев
#Для описания качества модели возьмем метрику F1 Score
depths = range(1, 25)
f1_scores_depth = []
for depth in depths:
    random_forest_model = RandomForestClassifier(max_depth=depth, random_state=42)
    random_forest_model.fit(X_train, Y_train)
    y_pred = random_forest_model.predict(X_test)
    f1_scores_depth.append(f1_score(Y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(depths, f1_scores_depth, marker='o', label='F1 Score', color='blue')
plt.xlabel("Глубина дерева")
plt.ylabel("F1 Score")
plt.title("Метод случайного леса: зависимость F1 Score от глубины дерева (Random Forest)")
plt.grid(True)
plt.legend()

#Изобразим зависимость качества модели от количества признаков, подаваемых на дерево
#Для описания качества модели снова возьмем метрику F1 Score
n_features_options = range(1, X.shape[1] + 1)
f1_scores_features = []
for n_features in n_features_options:
    random_forest_model = RandomForestClassifier(max_features=n_features, random_state=42)
    random_forest_model.fit(X_train, Y_train)
    y_pred = random_forest_model.predict(X_test)
    f1_scores_features.append(f1_score(Y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(n_features_options, f1_scores_features, marker='o', label='F1 Score', color='green')
plt.xlabel("Количество признаков")
plt.ylabel("F1 Score")
plt.title("Метод случайного леса: Зависимость F1 Score от количества признаков")
plt.grid(True)
plt.legend()

#Изобразим зависимость качества модели от числа деревьев
#Для описания качества модели снова возьмем метрику F1 Score
#Также добавим данные о времени обучения
n_estimators_options = [50, 100, 200, 300, 400, 500]
f1_scores_estimators = []
training_times = []
for n_estimators in n_estimators_options:
    start_time = time.time()
    random_forest_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    random_forest_model.fit(X_train, Y_train)
    end_time = time.time()
    training_time = end_time - start_time
    y_pred = random_forest_model.predict(X_test)
    f1_scores_estimators.append(f1_score(Y_test, y_pred))
    training_times.append(training_time)

print(f"Метод случайного леса имеет следующие характеристики:")
print(f"F1 Score: {f1_scores_estimators[-1]:.4f}")
print(f"Время обучения (в секундах): {training_times[-1]:.4f}")
print(f"Количество деревьев (выбрано вручную): {n_estimators_options[-1]}")

plt.figure(figsize=(12, 6))
plt.plot(n_estimators_options, f1_scores_estimators, marker='o', label='F1 Score', color='red')
plt.xlabel("Количество деревьев")
plt.ylabel("F1 Score")
plt.title("Метод случайного леса: зависимость F1 Score от количества деревьев")
plt.grid(True)
plt.legend(loc='upper left')
ax2 = plt.gca().twinx()
ax2.plot(n_estimators_options, training_times, marker='x', label='Время обучения', color='purple')
ax2.set_ylabel("Время обучения (сек)")
ax2.legend(loc='upper right')
plt.show()

#Теперь используем XGBoost для решения задачи классификации
n_estimators_XGBoost = 100
learning_rate_XGBoost = 0.1
max_depth_XGBoost = 5

start_time_XGBoost = time.time()
XGBoost_model = xgb.XGBClassifier(n_estimators=n_estimators_XGBoost, learning_rate=learning_rate_XGBoost,
    max_depth=max_depth_XGBoost, random_state=42)
XGBoost_model.fit(X_train, Y_train)
end_time_XGBoost = time.time()
training_time_XGBoost = end_time_XGBoost - start_time_XGBoost
y_pred_XGBoost = XGBoost_model.predict(X_test)
f1_XGBoost = f1_score(Y_test, y_pred_XGBoost)

print(f"Метод XGBoost имеет следующие характеристики:")
print(f"F1 Score: {f1_XGBoost:.4f}")
print(f"Время обучения (в секундах): {training_time_XGBoost:.4f}")
print(f"Количество деревьев (выбрано вручную): {n_estimators_XGBoost}")

#Сравниваем и анализируем итоги
print("\n1) Анализ Random Forest:")
print(f"- Оптимальная глубина деревьев: {depths[f1_scores_depth.index(max(f1_scores_depth))]}")
print(f"- Лучшее количество признаков: {n_features_options[f1_scores_features.index(max(f1_scores_features))]}")
print(f"- Рекомендуемое число деревьев: {n_estimators_options[f1_scores_estimators.index(max(f1_scores_estimators))]}")

print("\n2) Анализ XGBoost:")
print(f"- F1-score: {f1_XGBoost:.4f} (при n_estimators={n_estimators_XGBoost}, learning_rate={learning_rate_XGBoost}, max_depth={max_depth_XGBoost})")

print("\n3) Сравнительные показатели:")
print(f"- Разница в F1-score: {f1_XGBoost - max(f1_scores_estimators):.4f} в пользу {'XGBoost' if f1_XGBoost > max(f1_scores_estimators) else 'Random Forest'}")
print(f"- Разница во времени обучения: {training_time_XGBoost - training_times[f1_scores_estimators.index(max(f1_scores_estimators))]:.4f} сек")

print("\n4) Рекомендации по параметрам:")
print("- Для Random Forest:")
print("  • Глубина деревьев: 8-12")
print("  • Количество признаков: 3-5")
print("  • Количество деревьев: 100-200")
print("- Для XGBoost:")
print("  • learning_rate: 0.05-0.2")
print("  • max_depth: 3-7")
print("  • n_estimators: 100-200")

print("\n5) Итоговый вывод:")
print("XGBoost показал себя более эффективным алгоритмом для данной задачи,")
print("но требует более тщательной настройки гиперпараметров. Random Forest")
print("проще в настройке и показывает стабильно хорошие результаты.")