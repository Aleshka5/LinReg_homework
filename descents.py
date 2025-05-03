from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    """
    Класс для вычисления длины шага.

    Parameters
    ----------
    lambda_ : float, optional
        Начальная скорость обучения. По умолчанию 1e-3.
    s0 : float, optional
        Параметр для вычисления скорости обучения. По умолчанию 1.
    p : float, optional
        Степенной параметр для вычисления скорости обучения. По умолчанию 0.5.
    iteration : int, optional
        Текущая итерация. По умолчанию 0.

    Methods
    -------
    __call__()
        Вычисляет скорость обучения на текущей итерации.
    """

    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __init__(self, lambda_: float = 1e-3, s0: float = 1, p: float = 0.5):
        self.lambda_ = lambda_
        self.s0 = s0
        self.p = p
        self.iteration = 0

    def __call__(self):
        """
        Вычисляет скорость обучения по формуле lambda * (s0 / (s0 + t))^p.

        Returns
        -------
        float
            Скорость обучения на текущем шаге.
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    """
    Перечисление для выбора функции потерь.

    Attributes
    ----------
    MSE : auto
        Среднеквадратическая ошибка.
    MAE : auto
        Средняя абсолютная ошибка.
    LogCosh : auto
        Логарифм гиперболического косинуса от ошибки.
    Huber : auto
        Функция потерь Хьюбера.
    """

    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    Базовый класс для всех методов градиентного спуска.

    Parameters
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float, optional
        Параметр скорости обучения. По умолчанию 1e-3.
    loss_function : LossFunction, optional
        Функция потерь, которая будет оптимизироваться. По умолчанию MSE.

    Attributes
    ----------
    w : np.ndarray
        Вектор весов модели.
    lr : LearningRate
        Скорость обучения.
    loss_function : LossFunction
        Функция потерь.

    Methods
    -------
    step(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Шаг градиентного спуска.
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов на основе градиента. Метод шаблон.
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь по весам. Метод шаблон.
    calc_loss(x: np.ndarray, y: np.ndarray) -> float
        Вычисление значения функции потерь.
    predict(x: np.ndarray) -> np.ndarray
        Вычисление прогнозов на основе признаков x.
    """

    def __init__(
        self,
        dimension: int,
        lambda_: float = 1e-3,
        loss_function: LossFunction = LossFunction.MSE,
    ):
        """
        Инициализация базового класса для градиентного спуска.

        Parameters
        ----------
        dimension : int
            Размерность пространства признаков.
        lambda_ : float
            Параметр скорости обучения.
        loss_function : LossFunction
            Функция потерь, которая будет оптимизирована.

        Attributes
        ----------
        w : np.ndarray
            Начальный вектор весов, инициализированный случайным образом.
        lr : LearningRate
            Экземпляр класса для вычисления скорости обучения.
        loss_function : LossFunction
            Выбранная функция потерь.
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Выполнение одного шага градиентного спуска.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Разность между текущими и обновленными весами.
        """

        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Шаблон функции для обновления весов. Должен быть переопределен в подклассах.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь по весам.

        Returns
        -------
        np.ndarray
            Разность между текущими и обновленными весами. Этот метод должен быть реализован в подклассах.
        """
        raise NotImplementedError("BaseDescent update_weights function not implemented")

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Шаблон функции для вычисления градиента функции потерь по весам. Должен быть переопределен в подклассах.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь по весам. Этот метод должен быть реализован в подклассах.
        """
        raise NotImplementedError("BaseDescent calc_gradient function not implemented")

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Вычисление значения функции потерь с использованием текущих весов.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        float
            Значение функции потерь.
        """
        y_pred = self.predict(x)
        error = y_pred - y
        if self.loss_function == LossFunction.MSE:
            return np.mean(error**2)

        elif self.loss_function == LossFunction.MAE:
            return np.mean(np.abs(error))

        elif self.loss_function == LossFunction.LogCosh:
            return np.mean(np.log(np.cosh(error)))

        elif self.loss_function == LossFunction.Huber:
            delta = 1.0
            is_small_error = np.abs(error) <= delta
            squared_loss = 0.5 * error**2
            linear_loss = delta * (np.abs(error) - 0.5 * delta)
            return np.mean(np.where(is_small_error, squared_loss, linear_loss))

        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Расчет прогнозов на основе признаков x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        np.ndarray
            Прогнозируемые значения.
        """
        return x @ self.w


class VanillaGradientDescent(BaseDescent):
    """
    Класс полного градиентного спуска.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с учетом градиента.
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь по весам.
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов на основе градиента.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь по весам.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        learning_rate = self.lr()
        self.w -= learning_rate * gradient
        return (-1) * learning_rate * gradient

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь по весам.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков (n_samples, n_features).
        y : np.ndarray
            Массив целевых переменных (n_samples,).

        Returns
        -------
        np.ndarray
            Градиент функции потерь по весам.
        """
        y_pred = self.predict(x)
        error = y_pred - y

        if self.loss_function == LossFunction.MSE:
            return 2 * x.T @ error / len(y)

        elif self.loss_function == LossFunction.MAE:
            return x.T @ np.sign(error) / len(y)

        elif self.loss_function == LossFunction.LogCosh:
            return x.T @ np.tanh(error) / len(y)

        elif self.loss_function == LossFunction.Huber:
            delta = 1.0
            is_small_error = np.abs(error) <= delta
            grad = np.where(is_small_error, error, delta * np.sign(error))
            return x.T @ grad / len(y)

        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")


class StochasticDescent(VanillaGradientDescent):
    """
    Класс стохастического градиентного спуска.

    Parameters
    ----------
    batch_size : int, optional
        Размер мини-пакета. По умолчанию 50.

    Attributes
    ----------
    batch_size : int
        Размер мини-пакета.

    Методы
    -------
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь по мини-пакетам.
    """

    def __init__(
        self,
        dimension: int,
        lambda_: float = 1e-3,
        batch_size: int = 50,
        loss_function: LossFunction = LossFunction.MSE,
    ):

        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь по мини-пакетам.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь по весам, вычисленный по мини-пакету.
        """
        # Случайная выборка мини-пакета
        indices = np.random.choice(len(x), self.batch_size, replace=False)
        x_batch = x[indices]
        y_batch = y[indices]

        # Вычисление предсказаний и ошибки
        y_pred = self.predict(x_batch)
        error = y_pred - y_batch

        # Вычисление градиента по выбранной функции потерь
        if self.loss_function == LossFunction.MSE:
            return 2 * x_batch.T @ error / self.batch_size

        elif self.loss_function == LossFunction.MAE:
            return x_batch.T @ np.sign(error) / self.batch_size

        elif self.loss_function == LossFunction.LogCosh:
            return x_batch.T @ np.tanh(error) / self.batch_size

        elif self.loss_function == LossFunction.Huber:
            delta = 1.0
            is_small_error = np.abs(error) <= delta
            grad = np.where(is_small_error, error, delta * np.sign(error))
            return x_batch.T @ grad / self.batch_size

        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")


class MomentumDescent(VanillaGradientDescent):
    """
    Класс градиентного спуска с моментом.

    Параметры
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float
        Параметр скорости обучения.
    loss_function : LossFunction
        Оптимизируемая функция потерь.

    Атрибуты
    ----------
    alpha : float
        Коэффициент момента.
    h : np.ndarray
        Вектор момента для весов.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с использованием момента.
    """

    def __init__(
        self,
        dimension: int,
        lambda_: float = 1e-3,
        loss_function: LossFunction = LossFunction.MSE,
    ):
        """
        Инициализация класса градиентного спуска с моментом.

        Parameters
        ----------
        dimension : int
            Размерность пространства признаков.
        lambda_ : float
            Параметр скорости обучения.
        loss_function : LossFunction
            Оптимизируемая функция потерь.
        """
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с использованием момента.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        learning_rate = self.lr()

        # Обновление момента
        self.h = self.alpha * self.h + learning_rate * gradient

        # Обновление весов с учетом момента
        self.w -= self.h
        return (-1) * self.h


class Adam(VanillaGradientDescent):
    """
    Класс градиентного спуска с адаптивной оценкой моментов (Adam).

    Параметры
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float
        Параметр скорости обучения.
    loss_function : LossFunction
        Оптимизируемая функция потерь.

    Атрибуты
    ----------
    eps : float
        Малая добавка для предотвращения деления на ноль.
    m : np.ndarray
        Векторы первого момента.
    v : np.ndarray
        Векторы второго момента.
    beta_1 : float
        Коэффициент распада для первого момента.
    beta_2 : float
        Коэффициент распада для второго момента.
    iteration : int
        Счетчик итераций.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с использованием адаптивной оценки моментов.
    """

    def __init__(
        self,
        dimension: int,
        lambda_: float = 1e-3,
        loss_function: LossFunction = LossFunction.MSE,
    ):
        """
        Инициализация класса Adam.

        Parameters
        ----------
        dimension : int
            Размерность пространства признаков.
        lambda_ : float
            Параметр скорости обучения.
        loss_function : LossFunction
            Оптимизируемая функция потерь.
        """
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с использованием адаптивной оценки моментов.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        self.iteration += 1
        learning_rate = self.lr()

        # Обновляем первый и второй моменты
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient**2)

        # Смещения для моментов
        m_hat = self.m / (1 - self.beta_1**self.iteration)
        v_hat = self.v / (1 - self.beta_2**self.iteration)

        # Обновляем веса
        self.w -= learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
        return (-1) * learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


class BaseDescentReg(BaseDescent):
    """
    Базовый класс для градиентного спуска с регуляризацией.

    Параметры
    ----------
    *args : tuple
        Аргументы, передаваемые в базовый класс.
    mu : float, optional
        Коэффициент регуляризации. По умолчанию равен 0.
    **kwargs : dict
        Ключевые аргументы, передаваемые в базовый класс.

    Атрибуты
    ----------
    mu : float
        Коэффициент регуляризации.

    Методы
    -------
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь с учетом L2 регуляризации по весам.
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        Инициализация базового класса для градиентного спуска с регуляризацией.
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь и L2 регуляризации по весам.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь с учетом L2 регуляризации по весам.
        """
        l2_gradient: np.ndarray = np.zeros_like(x.shape[1])

        return super().calc_gradient(x, y) + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Класс полного градиентного спуска с регуляризацией.
    """

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь и L2 регуляризации по весам.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь с учетом L2 регуляризации.
        """
        # Предсказания модели
        y_pred = self.predict(x)

        # Вычисление ошибки
        error = y_pred - y

        # Градиент MSE: (2/n) * X^T (Xw - y)
        grad = 2 * x.T @ error / x.shape[0]

        # Добавляем градиент от L2 регуляризации: 2 * mu * w
        grad += 2 * self.mu * self.w

        return grad

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов на основе градиента.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь с регуляризацией.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        learning_rate = self.lr()
        w_old = self.w.copy()
        self.w -= learning_rate * gradient
        return self.w - w_old

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Расчет прогнозов на основе признаков x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        np.ndarray
            Прогнозируемые значения.
        """
        return x @ self.w

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Вычисление значения функции потерь с учетом регуляризации.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        float
            Значение функции потерь.
        """
        y_pred = self.predict(x)
        error = y_pred - y

        if self.loss_function == LossFunction.MSE:
            loss = np.mean(error**2)

        elif self.loss_function == LossFunction.MAE:
            loss = np.mean(np.abs(error))

        elif self.loss_function == LossFunction.LogCosh:
            loss = np.mean(np.log(np.cosh(error)))

        elif self.loss_function == LossFunction.Huber:
            delta = 1.0
            is_small_error = np.abs(error) <= delta
            squared_loss = 0.5 * (error**2)
            linear_loss = delta * (np.abs(error) - 0.5 * delta)
            loss = np.mean(np.where(is_small_error, squared_loss, linear_loss))

        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")

        # Добавляем регуляризацию
        loss += self.mu * np.sum(self.w**2)

        return loss


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Класс стохастического градиентного спуска с регуляризацией.
    """

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь с учетом L2 регуляризации по весам на мини-батче.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь с учетом L2 регуляризации.
        """
        # Случайный выбор индексов для мини-батча
        indices = np.random.choice(x.shape[0], self.batch_size, replace=False)
        x_batch = x[indices]
        y_batch = y[indices]

        # Предсказания модели
        y_pred = x_batch @ self.w
        error = y_pred - y_batch

        # Градиент MSE по мини-батчу
        grad = 2 * x_batch.T @ error / self.batch_size

        # Добавляем L2 регуляризацию: 2 * mu * w
        grad += 2 * self.mu * self.w

        return grad

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов на основе градиента.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        learning_rate = self.lr()
        w_old = self.w.copy()
        self.w -= learning_rate * gradient
        return self.w - w_old


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Класс градиентного спуска с моментом и регуляризацией.
    """

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь с учетом L2 регуляризации.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь с учетом L2 регуляризации.
        """
        # Предсказания
        y_pred = x @ self.w
        error = y_pred - y

        # Градиент MSE
        grad = 2 * x.T @ error / x.shape[0]

        # Добавление L2 регуляризации: 2 * mu * w
        grad += 2 * self.mu * self.w

        return grad

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с использованием момента.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        learning_rate = self.lr()
        w_old = self.w.copy()

        self.h = self.alpha * self.h + learning_rate * gradient
        self.w -= self.h

        return self.w - w_old


class AdamReg(BaseDescentReg, Adam):
    """
    Класс адаптивного градиентного алгоритма с регуляризацией (AdamReg).
    """

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь с учетом L2 регуляризации.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь с учетом L2 регуляризации.
        """
        # Предсказания
        y_pred = x @ self.w
        error = y_pred - y

        # Градиент MSE
        grad = 2 * x.T @ error / x.shape[0]

        # Добавляем L2 регуляризацию
        grad += 2 * self.mu * self.w

        return grad

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с использованием адаптивной оценки моментов (Adam).

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        self.iteration += 1
        learning_rate = self.lr()

        # Обновляем моменты
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradient**2)

        # Коррекция смещений
        m_hat = self.m / (1 - self.beta_1**self.iteration)
        v_hat = self.v / (1 - self.beta_2**self.iteration)

        # Сохраняем старые веса
        w_old = self.w.copy()

        # Обновляем веса
        self.w -= learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

        return self.w - w_old


def get_descent(descent_config: dict) -> BaseDescent:
    """
    Создает экземпляр класса градиентного спуска на основе предоставленной конфигурации.

    Параметры
    ----------
    descent_config : dict
        Словарь конфигурации для выбора и настройки класса градиентного спуска. Должен содержать ключи:
        - 'descent_name': строка, название метода спуска ('full', 'stochastic', 'momentum', 'adam').
        - 'regularized': булево значение, указывает на необходимость использования регуляризации.
        - 'kwargs': словарь дополнительных аргументов, передаваемых в конструктор класса спуска.

    Возвращает
    -------
    BaseDescent
        Экземпляр класса, реализующего выбранный метод градиентного спуска.

    Исключения
    ----------
    ValueError
        Вызывается, если указано неправильное имя метода спуска.

    Примеры
    --------
    >>> descent_config = {
    ...     'descent_name': 'full',
    ...     'regularized': True,
    ...     'kwargs': {'dimension': 10, 'lambda_': 0.01, 'mu': 0.1}
    ... }
    >>> descent = get_descent(descent_config)
    >>> isinstance(descent, BaseDescent)
    True
    """
    descent_name = descent_config.get("descent_name", "full")
    regularized = descent_config.get("regularized", False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        "full": (
            VanillaGradientDescent if not regularized else VanillaGradientDescentReg
        ),
        "stochastic": StochasticDescent if not regularized else StochasticDescentReg,
        "momentum": MomentumDescent if not regularized else MomentumDescentReg,
        "adam": Adam if not regularized else AdamReg,
    }

    if descent_name not in descent_mapping:
        raise ValueError(
            f"Incorrect descent name, use one of these: {descent_mapping.keys()}"
        )

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get("kwargs", {}))
