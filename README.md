# ArvoreDecisao

Projeto para experimentar e visualizar modelos de árvore de decisão aplicados a conjuntos de dados.

**Descrição**
- **Objetivo:** implementar, testar e gerar gráficos que mostrem o desempenho de árvores de decisão.

**Como executar**
Instale as dependências (se necessário):

```bash
pip install -r requirements.txt
```

Executar os experimentos e gerar os gráficos:

```bash
python main.py
python graficos.py
```

**Visualizações (gráficos)**
As imagens abaixo foram geradas pelos scripts do projeto e estão na pasta `graficos/`.

- **MAE por divisão:**
	![MAE por divisão](graficos/grafico1_mae_por_divisao.png)

- **Comparação de médias:**
	![Comparação de médias](graficos/grafico2_comparacao_media.png)

- **Profundidade do modelo:**
	![Profundidade do modelo](graficos/grafico3_profundidade.png)

- **Número de árvores vs. desempenho:**
	![Número de árvores](graficos/grafico4_n_arvores.png)