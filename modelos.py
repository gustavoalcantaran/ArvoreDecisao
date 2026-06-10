"""
modelos.py
----------
Funções para treinar e avaliar Árvore de Regressão e Random Forest.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from dados import carregar_dados, gerar_splits


def avaliar_arvore(X, y, n_splits=10, max_depth=None):
    """
    Treina uma DecisionTreeRegressor em n_splits divisões.
    Retorna listas de MAE treino/teste e o último modelo treinado.
    """
    maes_treino, maes_teste = [], []
    modelo = None

    for X_train, X_test, y_train, y_test in gerar_splits(X, y, n=n_splits):
        modelo = DecisionTreeRegressor(max_depth=max_depth,
                                       random_state=len(maes_treino))
        modelo.fit(X_train, y_train)
        maes_treino.append(mean_absolute_error(y_train, modelo.predict(X_train)))
        maes_teste.append(mean_absolute_error(y_test,  modelo.predict(X_test)))

    return maes_treino, maes_teste, modelo


def avaliar_random_forest(X, y, n_splits=10, max_depth=None, n_estimators=100):
    """
    Treina uma RandomForestRegressor em n_splits divisões.
    Retorna listas de MAE treino/teste.
    """
    maes_treino, maes_teste = [], []
    ultimo_modelo, ultimo_X_test, ultimo_y_test = None, None, None

    for X_train, X_test, y_train, y_test in gerar_splits(X, y, n=n_splits):
        seed = len(maes_treino)
        modelo = RandomForestRegressor(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       random_state=seed)
        modelo.fit(X_train, y_train)
        maes_treino.append(mean_absolute_error(y_train, modelo.predict(X_train)))
        maes_teste.append(mean_absolute_error(y_test,  modelo.predict(X_test)))
        ultimo_modelo, ultimo_X_test, ultimo_y_test = modelo, X_test, y_test

    return maes_treino, maes_teste, ultimo_modelo, ultimo_X_test, ultimo_y_test


def resumo(maes_treino, maes_teste, label=""):
    """Imprime média ± DP de treino e teste."""
    if label:
        print(f"\n  [{label}]")
    print(f"  Média MAE Teste : {np.mean(maes_teste):.2f}  ±  {np.std(maes_teste):.2f}")
    print(f"  Média MAE Treino: {np.mean(maes_treino):.2f}  ±  {np.std(maes_treino):.2f}")