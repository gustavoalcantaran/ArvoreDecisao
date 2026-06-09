"""
graficos.py
-----------
Funções de visualização para os resultados do laboratório.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Paleta de cores ────────────────────────────────────────────────────────────
COLORS = {
    "treino":  "#4fc3f7",
    "teste":   "#ef5350",
    "treino2": "#80cbc4",
    "teste2":  "#ffb74d",
    "text":    "#e0e0e0",
    "grid":    "#2a2a3a",
    "bg":      "#1a1a2e",
    "fig_bg":  "#0f1117",
}


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(COLORS["bg"])
    ax.set_title(title, color=COLORS["text"], fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=COLORS["text"], fontsize=10)
    ax.set_ylabel(ylabel, color=COLORS["text"], fontsize=10)
    ax.tick_params(colors=COLORS["text"])
    ax.spines[:].set_color(COLORS["grid"])
    ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(facecolor="#1e1e2e", edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9)


def plot_mae_por_divisao(ax, mae_teste_arv, mae_treino_arv,
                             mae_teste_rf,  mae_treino_rf):
    """Gráfico de linhas: MAE por divisão treino/teste (Partes 1 & 2)."""
    x = list(range(1, len(mae_teste_arv) + 1))
    ax.plot(x, mae_teste_arv,  marker="o", color=COLORS["teste"],   linewidth=2, label="Teste (Árvore)")
    ax.plot(x, mae_treino_arv, marker="s", color=COLORS["treino"],  linewidth=2, linestyle="--", label="Treino (Árvore)")
    ax.plot(x, mae_teste_rf,   marker="^", color=COLORS["teste2"],  linewidth=2, label="Teste (RF)")
    ax.plot(x, mae_treino_rf,  marker="D", color=COLORS["treino2"], linewidth=2, linestyle="--", label="Treino (RF)")
    ax.set_xticks(x)
    _style_ax(ax, "Partes 1 & 2 — MAE por Divisão", "Divisão", "MAE")


def plot_comparacao_media(ax, mae_treino_arv, mae_teste_arv,
                              mae_treino_rf,  mae_teste_rf):
    """Gráfico de barras: comparação média ± DP entre modelos (Partes 1 & 2)."""
    labels  = ["Árvore\nTreino", "Árvore\nTeste", "RF\nTreino", "RF\nTeste"]
    medias  = [np.mean(mae_treino_arv), np.mean(mae_teste_arv),
               np.mean(mae_treino_rf),  np.mean(mae_teste_rf)]
    stds    = [np.std(mae_treino_arv),  np.std(mae_teste_arv),
               np.std(mae_treino_rf),   np.std(mae_teste_rf)]
    cores   = [COLORS["treino"], COLORS["teste"], COLORS["treino2"], COLORS["teste2"]]

    bars = ax.bar(labels, medias, yerr=stds, color=cores, capsize=6,
                  error_kw={"ecolor": COLORS["text"], "linewidth": 1.5}, width=0.5)
    for bar, val in zip(bars, medias):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom",
                color=COLORS["text"], fontsize=9, fontweight="bold")

    ax.set_facecolor(COLORS["bg"])
    ax.set_title("Partes 1 & 2 — Comparação Média ± DP",
                 color=COLORS["text"], fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("MAE Médio", color=COLORS["text"], fontsize=10)
    ax.tick_params(colors=COLORS["text"])
    ax.spines[:].set_color(COLORS["grid"])
    ax.grid(True, axis="y", color=COLORS["grid"], linestyle="--", linewidth=0.6, alpha=0.7)


def plot_profundidade(ax, depths, media_teste, std_teste,
                           media_treino, std_treino, best_depth):
    """Gráfico errorbar: MAE vs profundidade da árvore (Parte 3)."""
    ax.errorbar(depths, media_teste,  yerr=std_teste,
                marker="o", color=COLORS["teste"],  capsize=5, linewidth=2, label="Teste")
    ax.errorbar(depths, media_treino, yerr=std_treino,
                marker="s", color=COLORS["treino"], capsize=5, linewidth=2,
                linestyle="--", label="Treino")
    ax.axvline(best_depth, color="#ffd54f", linestyle=":", linewidth=1.5,
               label=f"Melhor depth={best_depth}")
    ax.set_xticks(depths)
    _style_ax(ax, "Parte 3 — MAE vs Profundidade da Árvore",
              "Profundidade", "MAE Médio ± DP")


def plot_n_arvores(ax, n_trees_list, media_teste, std_teste,
                        media_treino, std_treino, best_depth):
    """Gráfico errorbar: MAE vs número de árvores (Parte 4)."""
    ax.errorbar(n_trees_list, media_teste,  yerr=std_teste,
                marker="o", color=COLORS["teste"],  capsize=5, linewidth=2, label="Teste")
    ax.errorbar(n_trees_list, media_treino, yerr=std_treino,
                marker="s", color=COLORS["treino"], capsize=5, linewidth=2,
                linestyle="--", label="Treino")
    ax.set_xticks(n_trees_list)
    _style_ax(ax, f"Parte 4 — MAE vs Nº de Árvores (depth={best_depth})",
              "Nº de Árvores", "MAE Médio ± DP")


def _salvar(fig, caminho):
    plt.savefig(caminho, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Gráfico salvo em: {caminho}")


def _nova_figura():
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(COLORS["fig_bg"])
    return fig, ax


def gerar_figura_completa(resultados):
    """
    Salva 4 gráficos separados numa pasta 'graficos/' ao lado deste arquivo.
    """
    import os
    pasta_saida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graficos")
    os.makedirs(pasta_saida, exist_ok=True)

    # ── Gráfico 1: MAE por divisão ────────────────────────────────────────────
    fig, ax = _nova_figura()
    plot_mae_por_divisao(ax,
        resultados["mae_teste_p1"], resultados["mae_treino_p1"],
        resultados["mae_teste_p2"], resultados["mae_treino_p2"])
    _salvar(fig, os.path.join(pasta_saida, "grafico1_mae_por_divisao.png"))

    # ── Gráfico 2: Comparação média ───────────────────────────────────────────
    fig, ax = _nova_figura()
    plot_comparacao_media(ax,
        resultados["mae_treino_p1"], resultados["mae_teste_p1"],
        resultados["mae_treino_p2"], resultados["mae_teste_p2"])
    _salvar(fig, os.path.join(pasta_saida, "grafico2_comparacao_media.png"))

    # ── Gráfico 3: Profundidade ───────────────────────────────────────────────
    fig, ax = _nova_figura()
    plot_profundidade(ax,
        resultados["depths"],
        resultados["media_teste_depth"], resultados["std_teste_depth"],
        resultados["media_treino_depth"], resultados["std_treino_depth"],
        resultados["best_depth"])
    _salvar(fig, os.path.join(pasta_saida, "grafico3_profundidade.png"))

    # ── Gráfico 4: Número de árvores ──────────────────────────────────────────
    fig, ax = _nova_figura()
    plot_n_arvores(ax,
        resultados["n_trees_list"],
        resultados["media_teste_nt"], resultados["std_teste_nt"],
        resultados["media_treino_nt"], resultados["std_treino_nt"],
        resultados["best_depth"])
    _salvar(fig, os.path.join(pasta_saida, "grafico4_n_arvores.png"))