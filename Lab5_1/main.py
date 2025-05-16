import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import graphviz
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1)
Y = data["Outcome"]
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Делим данные на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#Метод логической регрессии
logistic_regression = LogisticRegression(random_state=42)
logistic_regression.fit(X_train, Y_train)
logistic_predictions = logistic_regression.predict(X_test)
print("\nМодель логической регрессии имеет следующие метрики:")
print(f"Общая точность (Accuracy): {accuracy_score(Y_test, logistic_predictions):.4f}")
print(f"Точность положительных предсказаний (Precision): {precision_score(Y_test, logistic_predictions):.4f}")
print(f"Полнота (Recall): {recall_score(Y_test, logistic_predictions):.4f}")
f1_logistic_regression = f1_score(Y_test, logistic_predictions)
print(f"F1 Score: {f1_logistic_regression:.4f}")

#метод решающих деревьев
decision_trees = DecisionTreeClassifier(random_state=42)
decision_trees.fit(X_train, Y_train)
decision_trees_predictions = decision_trees.predict(X_test)
print("\nМодель решающих деревьев имеет следующие метрики:")
print(f"Общая точность (Accuracy): {accuracy_score(Y_test, decision_trees_predictions):.4f}")
print(f"Точность положительных предсказаний (Precision): {precision_score(Y_test, decision_trees_predictions):.4f}")
print(f"Полнота (Recall): {recall_score(Y_test, decision_trees_predictions):.4f}")
f1_decision_trees = f1_score(Y_test, decision_trees_predictions)
print(f"F1 Score: {f1_decision_trees:.4f}")

#Выбираем лучшую модель
print("\nСравним обе модели между собой:")
if f1_logistic_regression > f1_decision_trees:
    print("Логическая регрессия лучше согласно метрике F1 Score")
    print(f"{f1_logistic_regression:.4f} > {f1_decision_trees:.4f}")
elif f1_logistic_regression < f1_decision_trees:
    print("Модель решающих деревьев лучше согласно метрике F1 Score")
    print(f"{f1_logistic_regression:.4f} < {f1_decision_trees:.4f}")
else:
    print("Метрика F1 Score не даёт информации, какая из моделей лучше")
    print(f"{f1_logistic_regression:.4f} = {f1_decision_trees:.4f}")

#Исследуем зависимость метрики F1 Score от глубины решающего дерева
#Почему именно ее? Она является одной из самых показательных метрик качества модели.
#Поэтому нам важно знать, насколько качество и эффективность модели (а точнее, баланс между
#Точностью (Precision) и Полнотой (Recall)) зависят от глубины
depths = range(1, 15)
f1_scores = []
for depth in depths:
    decision_trees = DecisionTreeClassifier(max_depth=depth, random_state=42)
    decision_trees.fit(X_train, Y_train)
    decision_trees_predictions = decision_trees.predict(X_test)
    f1_scores.append(f1_score(Y_test, decision_trees_predictions))

plt.figure(figsize=(10, 6))
plt.plot(depths, f1_scores, marker='o', c='#39ad48')
plt.xlabel("Глубина дерева")
plt.ylabel("F1 Score")
plt.title("Зависимость метрики F1 Score от глубины решающего дерева")
plt.grid(True)

#Определим лучший уровень глубины дерева для оптимизации обучения
optimal_depth = depths[np.argmax(f1_scores)]
print(f"\nСамый оптимальный уровень глубины дерева: {optimal_depth}")

#Модель с оптимальной глубиной
decision_trees_optimal = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
decision_trees_optimal.fit(X_train, Y_train)
decision_trees_predictions_optimal = decision_trees_optimal.predict(X_test)
decision_trees_probabilities_optimal = decision_trees_optimal.predict_proba(X_test)[:, 1]

dot_data = export_graphviz(decision_trees_optimal, out_file=None, feature_names=data.drop("Outcome", axis=1).columns,
    class_names=["No Diabetes", "Diabetes"], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("tree_diabetes")

#Выведем график значимости признаков в решающем дереве в виде дистограммы
feature_importances = decision_trees_optimal.feature_importances_
features = data.drop("Outcome", axis=1).columns
plt.figure(figsize=(10, 6))
plt.bar(features, feature_importances, color='#980002')
plt.xlabel("Признак")
plt.ylabel("Важность")
plt.title("Важность признаков в дереве")
plt.xticks(rotation=45, ha='right', c='#c0737a')
plt.tight_layout()

#Выведем также PR кривую
precision, recall, thresholds = precision_recall_curve(Y_test, decision_trees_probabilities_optimal)
pr_auc = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}', color='#5729ce')
plt.title('Кривая PR для модели с оптимальной глубиной')
plt.xlabel('Полнота')
plt.ylabel('Точность положит. предсказаний')
plt.legend()
plt.grid(True)

#и ROC кривую
fpr, tpr, thresholds = roc_curve(Y_test, decision_trees_probabilities_optimal)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='#b5485d')
plt.plot([0, 1], [0, 1], '--', color='#ffbacd')
plt.title('Кривая ROC для модели с оптимальной глубиной')
plt.xlabel('Чувствительность')
plt.ylabel('1-Специфичность')
plt.legend()
plt.grid(True)

#Возьмем параметр max_features решающего дерева и рассмотрим зависимость метрики F1 Score от него
max_features_options = [1,2,3,4,5,6,7,8,9,10,12,14,16]
f1_scores_max_features = []
for max_feature in max_features_options:
    decision_trees = DecisionTreeClassifier(max_depth=optimal_depth, max_features=max_feature, random_state=42)
    decision_trees.fit(X_train, Y_train)
    decision_trees_predictions = decision_trees.predict(X_test)
    f1_scores_max_features.append(f1_score(Y_test, decision_trees_predictions))

plt.figure(figsize=(12, 6))
plt.plot(range(len(max_features_options)), f1_scores_max_features, marker='o', color='#448ee4')
plt.xlabel("количество признаков")
plt.ylabel("F1-score")
plt.title("Зависимость F1-score от параметра max_features")
plt.xticks(range(len(max_features_options)), max_features_options, rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()