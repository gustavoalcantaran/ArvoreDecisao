"""
main.py
-------
Ponto de entrada do laboratório.
Execute:  python main.py
"""

import sys
import os

# Garante que a pasta do script está no path, independente de onde for executado
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dados import carregar_dados
from experimentos import parte1, parte2, parte3, parte4
from graficos import gerar_figura_completa


def main():
    print("=" * 60)
    print("  PRÁTICA DE LABORATÓRIO — ÁRVORE DE DECISÃO / RANDOM FOREST")
    print("  Base de dados: Diabetes (scikit-learn)")

    X, y = carregar_dados()
    print(f"  Amostras: {X.shape[0]}  |  Atributos: {X.shape[1]}")
    print("=" * 60)

    mae_treino_p1, mae_teste_p1 = parte1(X, y)
    mae_treino_p2, mae_teste_p2 = parte2(X, y)

    depths, media_treino_depth, std_treino_depth, \
        media_teste_depth, std_teste_depth, best_depth = parte3(X, y)

    n_trees_list, media_treino_nt, std_treino_nt, \
        media_teste_nt, std_teste_nt = parte4(X, y, best_depth)

    resultados = {
        "mae_treino_p1": mae_treino_p1,  "mae_teste_p1": mae_teste_p1,
        "mae_treino_p2": mae_treino_p2,  "mae_teste_p2": mae_teste_p2,
        "depths":              depths,
        "media_treino_depth":  media_treino_depth,
        "std_treino_depth":    std_treino_depth,
        "media_teste_depth":   media_teste_depth,
        "std_teste_depth":     std_teste_depth,
        "best_depth":          best_depth,
        "n_trees_list":        n_trees_list,
        "media_treino_nt":     media_treino_nt,
        "std_treino_nt":       std_treino_nt,
        "media_teste_nt":      media_teste_nt,
        "std_teste_nt":        std_teste_nt,
    }

    gerar_figura_completa(resultados)

    print("\n" + "=" * 60)
    print("  FIM DO LABORATÓRIO")
    print("=" * 60)


if __name__ == "__main__":
    main()