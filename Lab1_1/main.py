import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_data_from_file(filepath):
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        print(f"Ошибка! Файл '{filepath}' не найден.")
        return None

#Функция для вывода статистической информации о столбце
def column_info(data, column_name):
    column = data[column_name].values
    print(f"\nИнформация о столбце '{column_name}'")
    print(f"Количество данных: {len(column)}")
    print(f"Минимальное значение: {np.min(column):.2f}")
    print(f"Максимальное значение: {np.max(column):.2f}")
    print(f"Среднее значение: {np.mean(column):.2f}")

#Функция для отрисовки графиков исходных данных
def plot_data(data, column_X, column_Y):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[column_X], data[column_Y], color='purple')
    plt.xlabel(column_X)
    plt.ylabel(column_Y)
    plt.title('Изображение исходных точек')
    plt.grid(True)
    return plt.gcf()

#Функция для отрисовки графиков регрессии
def plot_regression_line(data, column_X, column_Y, a, b):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[column_X], data[column_Y],color='purple', label='Исходные точки')
    values_X = np.linspace(data[column_X].min(), data[column_X].max(), 100)
    values_Y = a * values_X + b
    plt.plot(values_X, values_Y, color='red', label='Регрессионная прямая')

    plt.xlabel(column_X)
    plt.ylabel(column_Y)
    plt.title('Изображение регрессионной прямой')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

#Функция для отрисовка графиков ошибок
def plot_errors(data, column_X, column_Y, a, b):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[column_X], data[column_Y], color='purple', label='Исходные точки')
    values_X = np.linspace(data[column_X].min(), data[column_X].max(), 100)
    values_Y = a * values_X + b
    plt.plot(values_X, values_Y, color='red', label='Регрессионная прямая')

    for i in range(len(data)):
        x = data[column_X].iloc[i]
        y = data[column_Y].iloc[i] #Исходное значение y
        predicted_Y = a * x + b #Значение у согласно регрессионной прямой, прогнозируемое значение
        error = y - predicted_Y

        plt.gca().add_patch(plt.Rectangle((x, min(y, predicted_Y)),
                                          0.3, abs(error),
                                          fc='pink', alpha=0.3,
                                          label='Квадрат ошибки' if i == 0 else ""))
    plt.xlabel(column_X)
    plt.ylabel(column_Y)
    plt.title('Изображение квадратов ошибок')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def main():
    filepath = input("Введите путь к файлу с данными:")
    data = read_data_from_file(filepath)
    if data is None:
        return None

    print("\nВ исходных данных имеются столбцы ", data.columns.tolist())
    columnX = input("Введите имя столбца для оси X: ")
    columnY = input("Введите имя столбца для оси Y: ")
    if columnX not in data.columns or columnY not in data.columns:
        print("Ошибка! Таких столбцов не существует в данной сводке")
        return None

    # Выводим информацию о каждом столбце
    column_info(data, columnX)
    column_info(data, columnY)


    #Находим коэффициенты линейной регрессии
    x = data[columnX]
    y = data[columnY]
    a = (len(x) * np.sum(x * y) - np.sum(x) * np.sum(y)) / (len(x) * np.sum(x ** 2) - np.sum(x) ** 2)
    b = (np.sum(y) - a * np.sum(x)) / len(x)
    print(f"\nУравнение линейной регрессии имеет вид: y = {a:.2f} * x + {b:.2f}")
    print(f"где y = {columnY}, x = {columnX}")

    # Выводим графики исходных данных, линейной регрессии и квадратов ошибок
    source_data_plot = plot_data(data, columnX, columnY)
    regression_line_plot = plot_regression_line(data, columnX, columnY, a, b)
    error_squares_plot = plot_errors(data, columnX, columnY, a, b)
    plt.show()
    return None

if __name__ == "__main__":
    main()
