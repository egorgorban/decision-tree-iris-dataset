import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()


labels = np.array([[iris['target_names'][iris['target'][i]]]
                   for i in range(len(iris['target']))], dtype=object)

# строчки будут вида [4.9 3.1 1.5 0.2 'setosa'], [5.5 2.4 3.8 1.1 'versicolor'] и т.д.
data = np.append(np.array(iris['data'], dtype=object), labels, axis=1)

# print(data)

# разделяем данные на обучающие и тренировочные


# перемешиваем строчки (при перезапуске перемешивается заново)
np.random.shuffle(data)

training_data, testing_data = np.split(data, [int(0.9*len(data))])

# print(training_data)
# print(testing_data)


def class_counts(arr):
    """считает количество цветков каждого типа"""
    counts = {}
    for row in arr:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


header = ["sepal length", "sepal width", "petal length", "petal width"]


class Question:
    """Вопросы, которые используются чтобы разделить датасет на два подмножества."""

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        """ Удовлетворяет ли элемент условию вопроса"""
        return example[self.column] >= self.value

    def __repr__(self):
        return f"Is {header[self.column]} >= {str(self.value)}"


def partition(arr, question):
    """В соответствии с ответами на вопрос делит датасет на два помножества."""
    true_arr, false_arr = [], []
    for row in arr:
        if question.match(row):
            true_arr.append(row)
        else:
            false_arr.append(row)
    return np.array(true_arr), np.array(false_arr)


def gini(arr):
    """Считает загрязнённость датасета. Для каждой метки находится
    вероятность того, что случайно выбранный элемент датасета имеет эту метку. """
    counts = class_counts(arr)
    impurity = 1
    for label in counts:
        probability_of_label = counts[label] / float(len(arr))
        impurity -= probability_of_label**2
    return impurity


def info_gain(left, right, current_uncertainty):
    """считает насколько хорошо данный вопрос делит датасет.
    от изначальной загрязненности отнимается взвешенная загрязненность подмножеств"""
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(arr):
    """Находится лучший вопрос (на котором функция info_gain принимает наибольшее значение)
    Возвращается лучший вопрос и значение info_gain на нём"""
    best_gain = 0
    best_question = None
    current_uncertainty = gini(arr)
    n_features = len(arr[0]) - 1

    for col in range(n_features):

        values = set([row[col] for row in arr])

        for val in values:

            question = Question(col, val)

            # пытаемся разделить
            true_arr, false_arr = partition(arr, question)

            # Если хоть одно подмножество длины ноль, ищем другой вопрос
            if not len(true_arr)*len(false_arr):
                continue

            # Иначе считаем, насколько изменилась загрязнённость
            gain = info_gain(true_arr, false_arr, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    """Лист дерева. Хранит словарь {label: count_of_label}"""

    def __init__(self, arr):
        self.predictions = class_counts(arr)


class Decision_Node:
    """Внутренний узел дерева. 
    Хранит вопрос, который делит этот узел и ветки, на которые поделится датасет"""

    def __init__(self, question, true_branch, false_branch, arr):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.arr = arr


def build_tree(arr):
    """Решающее дерево"""

    gain, question = find_best_split(arr)

    if gain == 0:
        return Leaf(arr)

    true_arr, false_arr = partition(arr, question)

    true_branch = build_tree(true_arr)
    false_branch = build_tree(false_arr)

    return Decision_Node(question, true_branch, false_branch, arr)


def print_tree(node, spacing=""):
    """Нарисовать схему вопросов, характеризующую это дерево"""

    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    print(spacing + str(node.question))

    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

# функции для отображения каждого листа


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    total = float(sum(counts.values()))
    probs = {}
    for label in counts.keys():
        probs[label] = str(int(counts[label] / total * 100)) + "%"
    return probs


my_tree = build_tree(training_data)
print_tree(my_tree)

summa = 0
for row in testing_data:
    predictions = classify(row, my_tree)
    print(f"Метка: {row[-1]}. Предполагаемая метка: {print_leaf(predictions)}")
    cur_sum = 0
    for pred in predictions.items():
        if row[-1] == pred[0]:
            cur_sum += pred[1]
    summa += cur_sum/float(sum(predictions.values()))
print('Точность этого дерева:', summa/len(testing_data))


# Рисуем как каждый вопрос разделяет данные
# Вертикальная линия символизирует разделение на два подмножества
# Точки - элементы множества со значением параметра, откладываемым по оси абсцисс
# значение по ординате символизирует вид цветка. 1 - setosa, 3 - virginica, 5 - versicolor


def plot_tree(tree):
    if isinstance(tree, Leaf):
        return
    X = [tree.arr[:, tree.question.column]]
    Y = [1 if tree.arr[i][-1] == 'setosa' else 3 if tree.arr[i]
         [-1] == 'virginica' else 5 for i in range(len(tree.arr))]
    plt.title(tree.question.__repr__())
    plt.xlabel(header[tree.question.column])
    plt.ylabel('Вид цветка')

    plt.scatter(X, Y)
    plt.vlines(tree.question.value, 0.5, 5.5)
    plt.show()
    plot_tree(tree.false_branch)
    plot_tree(tree.true_branch)


plot_tree(my_tree)
