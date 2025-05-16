import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('Titanic.csv')
#1) Удаляем строки с пропусками
initial_data_len = len(df)
df.dropna(inplace=True)
rows_after_dropna = len(df)
print("Успешно удалены строки с пропусками")
print(f"Изначальная длина: {initial_data_len}")
print(f"Измененная длина: {rows_after_dropna}")

#2)Удаляем столбцы с нечисловыми значениями
columns_to_drop = ['Name', 'Ticket', 'Cabin']
df.drop(columns=columns_to_drop, inplace=True)

#3) Перекодируем Sex и Embarked
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked']) + 1

#4) Удалим PassengerId
df.drop(columns=['PassengerId'], inplace=True)

#5) Вычислим процент потерянных данных
rows_after_all_cleaning = len(df)
lost_data_percent = ((initial_data_len - rows_after_all_cleaning) / initial_data_len) * 100
print(f"Процент потерянных данных: {lost_data_percent:.2f}%")

#Изобразим исходные и измененные данные для наглядности
def plot_feature(feature_name):
    plt.figure(figsize=(12, 6))

    #График до предобработки
    plt.subplot(1, 2, 1)
    df_1 = pd.read_csv('Titanic.csv')
    if df_1[feature_name].dtype == 'object' or len(df_1[feature_name].unique()) < 10:
        sns.countplot(x=feature_name, data=df_1, color='#dfc5fe')
    else:
        sns.histplot(df_1[feature_name].dropna(), color='#dfc5fe', kde=True)
    plt.title(f'Частота признака {feature_name} до предобработки')
    plt.xlabel(feature_name)
    plt.ylabel('Частота')

    # График после предобработки
    plt.subplot(1, 2, 2)
    if df[feature_name].dtype == 'object' or len(df[feature_name].unique()) < 10:
        sns.countplot(x=feature_name, data=df, color='#d1ffbd')
    else:
        sns.histplot(df[feature_name], color='#d1ffbd', kde=True)
    plt.title(f'Частота признака {feature_name} послед предобработки')
    plt.xlabel(feature_name)
    plt.ylabel('Частота')
    plt.tight_layout()

features_to_plot = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass', 'Sex', 'Embarked']
for feature in features_to_plot:
    plot_feature(feature)

#1) Разбиваем данные на обучающую и тестовую выборки
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

#2 и 3) Метод логистической регрессии
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nТочность модели: {accuracy:.4f}")

#4) Влияние признака Embarked на точность модели. Для этого создаем модель без Embarked и изучаем точность
X_no_embarked = df.drop(columns=['Survived', 'Embarked'])
y = df['Survived']
X_train_no_embarked, X_test_no_embarked, y_train_no_embarked, y_test_no_embarked = train_test_split(X_no_embarked, y, test_size=0.3, random_state=42)
model_no_embarked = LogisticRegression(max_iter=1000)
model_no_embarked.fit(X_train_no_embarked, y_train_no_embarked)
y_pred_no_embarked = model_no_embarked.predict(X_test_no_embarked)
accuracy_no_embarked = accuracy_score(y_test_no_embarked, y_pred_no_embarked)
print(f"Точность модели без Embarked: {accuracy_no_embarked:.4f}")

if accuracy > accuracy_no_embarked:
    print("В итоге с Embarked точность модели выше")
elif accuracy < accuracy_no_embarked:
    print("В итоге без Embarked точность модели выше")
else:
    print("В итоге Embarked никак не влияет на точность модели")
plt.show()
