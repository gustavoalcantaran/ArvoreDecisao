"""
dados.py
--------
Carregamento e divisão da base de dados Diabetes.
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def carregar_dados():
    """Retorna X e y da base Diabetes."""
    data = load_diabetes()
    return data.data, data.target


def dividir(X, y, test_size=0.2, random_state=0):
    """Divide X, y em treino/teste."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def gerar_splits(X, y, n=10, test_size=0.2):
    """Gera n divisões treino/teste com seeds diferentes."""
    for i in range(n):
        yield dividir(X, y, test_size=test_size, random_state=i)