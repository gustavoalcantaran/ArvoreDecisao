"""
experimentos.py
---------------
Implementação das 4 partes do laboratório.
Cada função executa a parte, imprime os resultados e retorna os dados
necessários para os gráficos.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error

from dados import gerar_splits
from modelos import avaliar_arvore, avaliar_random_forest, resumo


# ──────────────────────────────────────────────────────────────────────────────
# Parte 1 — Árvore de Regressão (10 divisões)
# ──────────────────────────────────────────────────────────────────────────────
def parte1(X, y):
    print("\n" + "─" * 60)
    print("PARTE 1 — Árvore de Regressão (10 divisões treino/teste)")
    print("─" * 60)

    maes_treino, maes_teste, ultimo_modelo = avaliar_arvore(X, y, n_splits=10)

    for i, (tr, te) in enumerate(zip(maes_treino, maes_teste), 1):
        print(f"  Divisão {i:02d} | MAE Treino: {tr:.2f}  |  MAE Teste: {te:.2f}")

    resumo(maes_treino, maes_teste)

    # 5 exemplos do último split (seed=9)
    splits = list(gerar_splits(X, y, n=10))
    _, X_test, _, y_test = splits[-1]
    y_pred = ultimo_modelo.predict(X_test)

    print("\n  5 exemplos — último modelo (divisão 10):")
    print(f"  {'Real':>8}  {'Previsto':>8}")
    for real, prev in zip(y_test[:5], y_pred[:5]):
        print(f"  {real:>8.1f}  {prev:>8.1f}")

    return maes_treino, maes_teste


# ──────────────────────────────────────────────────────────────────────────────
# Parte 2 — Random Forest (10 divisões)
# ──────────────────────────────────────────────────────────────────────────────
def parte2(X, y):
    print("\n" + "─" * 60)
    print("PARTE 2 — Random Forest (10 divisões treino/teste)")
    print("─" * 60)

    maes_treino, maes_teste, ultimo_modelo, X_test, y_test = \
        avaliar_random_forest(X, y, n_splits=10, n_estimators=100)

    for i, (tr, te) in enumerate(zip(maes_treino, maes_teste), 1):
        print(f"  Divisão {i:02d} | MAE Treino: {tr:.2f}  |  MAE Teste: {te:.2f}")

    resumo(maes_treino, maes_teste)

    print("\n  5 exemplos — último modelo (divisão 10):")
    print(f"  {'Real':>8}  {'Previsto':>8}")
    for real, prev in zip(y_test[:5], ultimo_modelo.predict(X_test)[:5]):
        print(f"  {real:>8.1f}  {prev:>8.1f}")

    return maes_treino, maes_teste


# ──────────────────────────────────────────────────────────────────────────────
# Parte 3 — Efeito da profundidade (depth 1–10)
# ──────────────────────────────────────────────────────────────────────────────
def parte3(X, y):
    print("\n" + "─" * 60)
    print("PARTE 3 — Efeito da profundidade da árvore (1 a 10)")
    print("─" * 60)

    depths = list(range(1, 11))
    media_teste,  std_teste  = [], []
    media_treino, std_treino = [], []

    for depth in depths:
        tr_list, te_list, _ = avaliar_arvore(X, y, n_splits=10, max_depth=depth)
        media_treino.append(np.mean(tr_list));  std_treino.append(np.std(tr_list))
        media_teste.append(np.mean(te_list));   std_teste.append(np.std(te_list))
        print(f"  Depth {depth:2d} | Teste: {np.mean(te_list):.2f} ± {np.std(te_list):.2f}"
              f"  | Treino: {np.mean(tr_list):.2f} ± {np.std(tr_list):.2f}")

    best_depth = depths[np.argmin(media_teste)]
    print(f"\n  Melhor profundidade (menor MAE teste): {best_depth}")

    return depths, media_treino, std_treino, media_teste, std_teste, best_depth


# ──────────────────────────────────────────────────────────────────────────────
# Parte 4 — Efeito do número de árvores
# ──────────────────────────────────────────────────────────────────────────────
def parte4(X, y, best_depth):
    print("\n" + "─" * 60)
    print(f"PARTE 4 — Número de árvores (depth={best_depth} fixo)")
    print("─" * 60)

    n_trees_list = [5, 10, 20, 40, 80]
    media_teste,  std_teste  = [], []
    media_treino, std_treino = [], []

    for n_trees in n_trees_list:
        tr_list, te_list, _, _, _ = avaliar_random_forest(
            X, y, n_splits=10, max_depth=best_depth, n_estimators=n_trees
        )
        media_treino.append(np.mean(tr_list));  std_treino.append(np.std(tr_list))
        media_teste.append(np.mean(te_list));   std_teste.append(np.std(te_list))
        print(f"  n_trees={n_trees:3d} | Teste: {np.mean(te_list):.2f} ± {np.std(te_list):.2f}"
              f"  | Treino: {np.mean(tr_list):.2f} ± {np.std(tr_list):.2f}")

    return n_trees_list, media_treino, std_treino, media_teste, std_teste