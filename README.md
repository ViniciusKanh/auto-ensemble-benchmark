# 📦 Auto-Ensemble-Benchmark

### 🔬 Benchmark automático de modelos *ensemble* para classificação — métricas padronizadas, ranking e validação cruzada

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.1-4ade80?style=for-the-badge" alt="version" />
  <img src="https://img.shields.io/badge/python-3.9%2B-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="python" />
  <img src="https://img.shields.io/badge/license-MIT-22c55e?style=for-the-badge" alt="license" />
</p>

---

## Sumário

1. [Visão geral](#vis%C3%A3o-geral)
2. [Instalação](#instala%C3%A7%C3%A3o)
3. [Exemplos de uso](#exemplos-de-uso)

   * [Exemplo rápido (hold-out)](#exemplo-r%C3%A1pido-hold-out)
   * [Validação cruzada (CV)](#valida%C3%A7%C3%A3o-cruzada-cv)
4. [Explicação dos resultados e formato dos arquivos](#explica%C3%A7%C3%A3o-dos-resultados-e-formato-dos-arquivos)
5. [Interpretação metodológica e recomendações](#interpreta%C3%A7%C3%A3o-metodol%C3%B3gica-e-recomenda%C3%A7%C3%B5es)
6. [API — referência sucinta](#api---refer%C3%AAncia-sucinta)
7. [Roadmap](#roadmap)
8. [Contribuição](#contribui%C3%A7%C3%A3o)
9. [Licença](#licen%C3%A7a)
10. [Sobre o autor](#sobre-o-autor)
11. [Changelog curto](#changelog-curto)

---

## Visão geral

`auto-ensemble-benchmark` é uma biblioteca Python projetada para automatizar a comparação de classificadores *ensemble* por meio de métricas padronizadas, rankings por métrica, ranking agregado e validação cruzada. O objetivo é proporcionar um fluxo reprodutível e científico para:

* Treinar e avaliar múltiplos ensembles baseline (RandomForest, ExtraTrees, GradientBoosting, AdaBoost, Bagging com RF).
* Calcular métricas padrão (accuracy, f1, recall, precision) com suporte a médias para problemas multiclasse/binário.
* Gerar colunas de ranking por métrica e um `overall_rank` agregador.
* Executar validação cruzada estratificada retornando média e desvio-padrão.
* Persistir resultados em CSV para documentação experimental e relatórios científicos.

Aplicações típicas: pesquisa acadêmica (benchmarks reprodutíveis), avaliação de baselines por cientistas de dados, provas de conceito rápidas em conjuntos de dados variados.

---

## Instalação

### Via PyPI (recomendado)

```bash
pip install auto-ensemble-benchmark
```

### Modo desenvolvimento (instalação local)

```bash
git clone https://github.com/ViniciusKanh/auto-ensemble-benchmark.git
cd auto-ensemble-benchmark

# criar ambiente virtual (exemplo)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -e .
```

---

## Exemplos de uso

> Todos os exemplos assumem importações padrão do `scikit-learn`. Comentários dos trechos de código estão em Português.

### Exemplo rápido — hold-out

```python
# Exemplo mínimo: hold-out com dataset wine
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from auto_ensemble_benchmark import AutoEnsembleClassifier

# Carrega dados
data = load_wine()
X, y = data.data, data.target

# Particiona treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Instancia o avaliador automático (padrões)
auto = AutoEnsembleClassifier(
    primary_metric="accuracy",   # métrica principal para ranking
    add_rank_columns=True,       # inclui colunas de ranking
    random_state=42,
    n_jobs=-1
)

# Treina e avalia (treina nos dados de treino e avalia no hold-out)
df_holdout = auto.fit_evaluate(X_train, y_train, X_test, y_test, media="macro")

# Resultado: DataFrame com métricas e rankings
print(df_holdout)

# Sumário interpretável
print(auto.summarize_results(top_k=3))
```

### Validação cruzada (CV) — estimativa robusta

```python
# Avaliação com validação cruzada estratificada
df_cv = auto.fit_evaluate_cv(
    X, y,
    cv=5,
    media="macro",
    random_state_cv=123,
    shuffle=True
)

# df_cv contém <metrica>_mean e <metrica>_std para cada modelo
print(df_cv)
```
 
### Uso avançado — sobrescrita de hiperparâmetros e salvamento automático

```python
auto = AutoEnsembleClassifier(
    metricas=["accuracy", "f1", "recall", "precision"],
    primary_metric="f1",
    model_overrides={
        "RandomForest": {"n_estimators": 500, "max_depth": None},
        "ExtraTrees": {"n_estimators": 800},
    },
    output_dir="resultados_bench",
    experiment_name="exp_v1",
    save_on_evaluate=True,
    save_on_cv=True
)

df_holdout = auto.fit_evaluate(X_train, y_train, X_test, y_test, media="macro")
# Arquivos gerados:
# resultados_bench/exp_v1_holdout_results.csv
# resultados_bench/exp_v1_cv_results.csv
```

---

## Explicação dos resultados e formato dos arquivos

A biblioteca retorna `pandas.DataFrame` com as colunas descritas abaixo. Caso `save_on_*` esteja ativado, gera CSVs contendo as mesmas informações, mais metadados.

### Colunas de métricas (hold-out)

* `accuracy` — fração de previsões corretas.
* `f1` — pontuação F1 (dependente de `media` para problemas multiclasse).
* `recall` — sensibilidade (TP / (TP + FN)).
* `precision` — precisão (TP / (TP + FP)).

> Todas as métricas seguem a API do `scikit-learn`. Use o parâmetro `media` para ajustar o cálculo (`"binary"`, `"macro"`, `"micro"`, `"weighted"`).

### Colunas de ranking

* `rank_accuracy`, `rank_f1`, `rank_recall`, `rank_precision` — posição ordinal por métrica (1 = melhor).
* `overall_rank` — soma das posições das métricas consideradas; menor valor indica melhor desempenho agregado.

**Nota interpretativa:** `overall_rank` é um agregador de ordens e serve como critério sumarizador. Ele não substitui testes estatísticos de significância entre modelos.

### Validação cruzada (CV)

Saída de `fit_evaluate_cv` possui colunas:

* `<metrica>_mean` — média da métrica nas folds.
* `<metrica>_std` — desvio-padrão entre folds (medida de estabilidade).

Use `*_std` para examinar robustez: elevado desvio indica sensibilidade a particionamentos.

### Arquivos gerados

* `{output_dir}/{experiment_name}_holdout_results.csv` — resultados do hold-out.
* `{output_dir}/{experiment_name}_cv_results.csv` — resultados da validação cruzada.

Os CSVs incluem colunas: `modelo`, métricas, colunas de ranking, parâmetros aplicados (se sobrescritos), timestamp de execução.

---

## Interpretação metodológica e recomendações

1. **Seleção da métrica principal (`primary_metric`)**

   * Para dados desbalanceados, priorize `f1` (ou `recall`/`precision` conforme custo de falsos negativos/positivos).
   * Para problemas multiclasse, utilize `media="macro"` e prefira `f1` como métrica agregada.

2. **Hold-out vs CV**

   * Hold-out é adequado para inspeção e diagnóstico rápidos.
   * Validação cruzada fornece estimativas mais estáveis e deve ser usada para relatórios científicos ou quando o conjunto de dados é pequeno.

3. **Estabilidade vs desempenho pontual**

   * Compare `*_mean` com `*_std` na CV. Um modelo com média ligeiramente inferior, mas menor desvio, pode ser preferível pela maior robustez.

4. **Comparações estatísticas**

   * Ao comparar top-k modelos, realize testes pareados (ex.: Wilcoxon, t-test pareado dependendo da normalidade) sobre as métricas nas folds. Use correção para múltiplos testes quando necessário.

5. **Reprodutibilidade**

   * Defina `random_state` e `random_state_cv` quando for reportar resultados; versione scripts e CSVs de saída para rastreabilidade.

---

## API — referência sucinta

```python
AutoEnsembleClassifier(
    modelos=None,
    metricas=None,
    primary_metric="accuracy",
    add_rank_columns=True,
    random_state=42,
    n_jobs=-1,
    model_overrides=None,
    output_dir=None,
    experiment_name=None,
    save_on_evaluate=False,
    save_on_cv=False,
)
```

### Métodos principais

* `fit(X_train, y_train)` — treina todos os modelos.
* `evaluate(X_test, y_test, media="auto")` — avalia modelos no conjunto de teste.
* `fit_evaluate(X_train, y_train, X_test, y_test, media="auto")` — atalho: treina e avalia.
* `fit_evaluate_cv(X, y, cv=5, media="auto", random_state_cv=42, shuffle=True)` — executa CV estratificada.
* `get_results()` — retorna `DataFrame` dos últimos resultados hold-out.
* `get_results_cv()` — retorna `DataFrame` dos últimos resultados de CV.
* `summarize_results(top_k=3)` — resumo textual dos top-k modelos (métrica principal).

> Implementações de modelos padrão: `RandomForestClassifier`, `ExtraTreesClassifier`, `GradientBoostingClassifier`, `AdaBoostClassifier`, `BaggingClassifier(base_estimator=RandomForest)`.

---

## Licença

Distribuído sob licença **MIT** — consulte o arquivo `LICENSE` para termos completos.

---

## Sobre o autor

**Vinicius de Souza Santos**
Pesquisador em Ciência da Computação (UNESP) — ênfase em Machine Learning, Feature Selection e experimentação empírica.

* **GitHub:** [https://github.com/ViniciusKanh](https://github.com/ViniciusKanh)
* **LinkedIn:** [https://www.linkedin.com/in/vinicius-souza-santoss/](https://www.linkedin.com/in/vinicius-souza-santoss/)
* **E-mail profissional:** [vinicius-souza.santos@unesp.br](mailto:vinicius-souza.santos@unesp.br)

**Resumo das competências:** concepção e execução de benchmarks reprodutíveis, validação cruzada estratificada, experimentação empírica com scikit-learn, engenharia de pipelines de avaliação.

---

### Notas finais

Este projeto foi desenvolvido com a proposta de oferecer uma biblioteca simples, transparente e científica para benchmark automatizado de modelos ensemble, focando em reprodutibilidade, rigor estatístico e facilidade de uso para pesquisadores, estudantes e profissionais da área.

