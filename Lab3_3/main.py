import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 1 часть. метрики Титаника
df = pd.read_csv('Titanic.csv')

#Делаем предобработку как в 3.2
df.dropna(inplace=True)
columns_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
df.drop(columns=columns_to_drop, inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked']) + 1

#Делим данные на обучающую и тестовую выборки и обучаем модель
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_logical_regression = model_lr.predict(X_test)

#Дополняем метриками
accuracy_logical_regression = accuracy_score(y_test, y_pred_logical_regression)
precision_logical_regression = precision_score(y_test, y_pred_logical_regression)
recall_logical_regression = recall_score(y_test, y_pred_logical_regression)
f1_logical_regression = f1_score(y_test, y_pred_logical_regression)

print("Модель логической регрессии имеет следующие метрики:")
print(f"Общая точность (Accuracy): {accuracy_logical_regression:.4f}")
print(f"Точность положительных предсказаний (Precision): {precision_logical_regression:.4f}")
print(f"Полнота (Recall): {recall_logical_regression:.4f}")
print(f"F1 Score: {f1_logical_regression:.4f}")

#Выведем конкретно графики для логической регрессии: матрицу ошибок,кривую PR и кривую ROC
#Матрица ошибок
cm_logical_regression = confusion_matrix(y_test, y_pred_logical_regression)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_logical_regression, annot=True, fmt='d', cmap='Wistia')
plt.title('Логическая регрессия: матрица ошибок')
plt.xlabel('Предсказание')
plt.ylabel('Исходные данные')

#кривая PR
precision_curve_logical_regression, recall_curve_logical_regression, _ = precision_recall_curve(y_test, model_lr.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(recall_curve_logical_regression, precision_curve_logical_regression, color='#fec615')
plt.xlabel('Полнота')
plt.ylabel('Точность положит. предсказаний')
plt.title('Логическая регрессия: кривая PR')

#кривая ROC
fpr_logical_regression, tpr_logical_regression, thresholds_logical_regression = roc_curve(y_test, model_lr.predict_proba(X_test)[:, 1])
roc_auc_logical_regression = auc(fpr_logical_regression, tpr_logical_regression)
plt.figure(figsize=(8, 6))
plt.plot(fpr_logical_regression, tpr_logical_regression, color='#fec615', lw=2, label=f'ROC AUC = {roc_auc_logical_regression:.2f}')
plt.plot([0, 1], [0, 1], color='#c44240', lw=2, linestyle='--')
plt.xlabel('Чувствительность')
plt.ylabel('1-Специфичность')
plt.title('ROC-кривая (Логистическая регрессия)')
plt.legend(loc="lower right")
plt.show()

#Сделаем выаод о качестве модели
print("\nСудя по полученным значениям и графикам мы можем сделать выводы:")
print("- Модель показывает хорошую точность, она правильно предсказывает в большинстве случаев")
print("- Значение F1-меры говорит о том, что она хорошо находит то, что нужно, и не выдает ")
print("слишком много ложных срабатываний (значения полноты и точности сбалансированы)")
print("- Анализ кривой PR подтверждает способность модели сохранять высокую точность даже при увеличении полноты")
print("- AUC ROC > 0.8. Это означает, что модель обладает высоким уровнем разделения классов (но не идеальным)")

#2 часть. модели опорных векторов и ближайшиз соседей

#модель Support Vector Machine(SVM) - опорных векторов
model_SVM = SVC(probability=True)
model_SVM.fit(X_train, y_train)
y_pred_SVM = model_SVM.predict(X_test)

#Аналогично пишем метрики
accuracy_SVM = accuracy_score(y_test, y_pred_SVM)
precision_SVM = precision_score(y_test, y_pred_SVM)
recall_SVM = recall_score(y_test, y_pred_SVM)
f1_SVM = f1_score(y_test, y_pred_SVM)
print("\nМодель SVM имеет следующие метрики:")
print(f"Общая точность (Accuracy): {accuracy_SVM:.4f}")
print(f"Точность положительных предсказаний (Precision): {precision_SVM:.4f}")
print(f"Полнота (Recall): {recall_SVM:.4f}")
print(f"F1 Score: {f1_SVM:.4f}")

#Выводим матрицу ошибок. Кривая PR и кривая ROC позже будут выведены на общем графике для сравнения
cm_SVM = confusion_matrix(y_test, y_pred_SVM)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_SVM, annot=True, fmt='d', cmap='Greens')
plt.title('Модель SVM: матрица ошибок')
plt.xlabel('Предсказание')
plt.ylabel('Исходные данные')

#модель K-Nearest Neighbors(KNN) - ближайших соседей
model_KNN = KNeighborsClassifier(n_neighbors=5)
model_KNN.fit(X_train, y_train)
y_pred_KNN = model_KNN.predict(X_test)

#Аналогично пишем метрики
accuracy_KNN = accuracy_score(y_test, y_pred_KNN)
precision_KNN = precision_score(y_test, y_pred_KNN)
recall_KNN = recall_score(y_test, y_pred_KNN)
f1_KNN = f1_score(y_test, y_pred_KNN)
print("\nМодель KNN имеет следующие метрики:")
print(f"Общая точность (Accuracy): {accuracy_KNN:.4f}")
print(f"Точность положительных предсказаний (Precision): {precision_KNN:.4f}")
print(f"Полнота (Recall): {recall_KNN:.4f}")
print(f"F1 Score: {f1_KNN:.4f}")

#Выводим матрицу ошибок. Кривая PR и кривая ROC позже будут выведены на общем графике для сравнения
cm_KNN = confusion_matrix(y_test, y_pred_KNN)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_KNN, annot=True, fmt='d', cmap='Purples')
plt.title('Модель KNN: матрица ошибок')
plt.xlabel('Предсказание')
plt.ylabel('Исходные данные')

#Изобразим PR кривую для всех моделей. Так будет проще оценивать и сравнивать качество моделей
precision_curve_logical_regression, recall_curve_logical_regression, _ = precision_recall_curve(y_test, model_lr.predict_proba(X_test)[:, 1])
precision_curve_SVM, recall_curve_SVM, _ = precision_recall_curve(y_test, model_SVM.predict_proba(X_test)[:, 1])
precision_curve_KNN, recall_curve_KNN, _ = precision_recall_curve(y_test, model_KNN.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(recall_curve_logical_regression, precision_curve_logical_regression, color='#fec615', label='Логистическая регрессия')
plt.plot(recall_curve_SVM, precision_curve_SVM, color='#25a36f', label='SVM')
plt.plot(recall_curve_KNN, precision_curve_KNN, color='#6d5acf', label='KNN')
plt.xlabel('Полнота')
plt.ylabel('Точность положит. предсказаний')
plt.title('Кривая PR для всех моделей')
plt.legend()

#Изобразим ROC кривую для всех моделей. Так будет проще оценивать и сравнивать эффективность моделей
fpr_logical_regression, tpr_logical_regression, thresholds_logical_regression = roc_curve(y_test, model_lr.predict_proba(X_test)[:, 1])
roc_auc_logical_regression = auc(fpr_logical_regression, tpr_logical_regression)
fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(y_test, model_SVM.predict_proba(X_test)[:, 1])
roc_auc_SVM = auc(fpr_SVM, tpr_SVM)
fpr_KNN, tpr_KNN, thresholds_KNN = roc_curve(y_test, model_KNN.predict_proba(X_test)[:, 1])
roc_auc_KNN = auc(fpr_KNN, tpr_KNN)

plt.figure(figsize=(8, 6))
plt.plot(fpr_logical_regression, tpr_logical_regression, color='#fec615', lw=2, label=f'Логистическая регрессия (AUC = {roc_auc_logical_regression:.2f})')
plt.plot(fpr_SVM, tpr_SVM, color='#25a36f', lw=2, label=f'SVM (AUC = {roc_auc_SVM:.2f})')
plt.plot(fpr_KNN, tpr_KNN, color='#6d5acf', lw=2, label=f'KNN (AUC = {roc_auc_KNN:.2f})')
plt.plot([0, 1], [0, 1], color='#c44240', lw=2, linestyle='--')
plt.title('Кривая ROC для всех моделей')
plt.xlabel('Чувствительность')
plt.ylabel('1-Специфичность')
plt.legend()

#Будем сравнивать модели по значениям ROC AUC и F1 Score
print("\nСравним полученные данные о моделях и выберем лучшую")
roc_auc_models = [roc_auc_logical_regression, roc_auc_SVM, roc_auc_KNN]
max_roc_auc = max(roc_auc_models)
max_index = roc_auc_models.index(max_roc_auc)
model_names = ["Логистическая регрессия", "SVM", "KNN"]
print(f"По показателю ROC AUC лучшей моделью является: {model_names[max_index]}")
print(f"Значение показателя ROC AUC: {roc_auc_models[max_index]}")

f1_models = [f1_logical_regression, f1_SVM, f1_KNN]
max_f1 = max(f1_models)
max_index = f1_models.index(max_f1)
model_names = ["Логистическая регрессия", "SVM", "KNN"]
print(f"\nПо показателю f1 лучшей моделью является: {model_names[max_index]}")
print(f"Значение показателя f1: {f1_models[max_index]}")
plt.show()