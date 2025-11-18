# 📦 Auto-Ensemble-Benchmark  
### 🔥 Benchmark automático de modelos *ensemble* com métricas completas, ranking e validação cruzada

<p align="center">
  <img src="https://img.shields.io/badge/status-0.1.0-4ade80?style=for-the-badge" />
  <img src="https://img.shields.io/badge/python-3.9+-3776ab?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/license-MIT-22c55e?style=for-the-badge" />
</p>

---

## 🌟 Visão Geral

`auto-ensemble-benchmark` é uma biblioteca em Python para **comparar automaticamente múltiplos modelos ensemble de classificação**, calculando métricas padronizadas e organizando os resultados em **tabelas claras, ordenadas e interpretáveis**.

Ela foi pensada para:

- 💡 **Pesquisadores** que precisam comparar rapidamente ensembles em diferentes bases.  
- 🧪 **Data Scientists** que querem um benchmark rápido de modelos baselines.  
- 📊 **Estudos acadêmicos** que requerem métricas reprodutíveis e relatórios consistentes.  

Com poucas linhas de código, você consegue:

- Treinar vários ensembles de uma vez (RandomForest, ExtraTrees, GradientBoosting, AdaBoost, Bagging com RF).  
- Obter uma tabela com:

  - `accuracy`  
  - `f1`  
  - `recall`  
  - `precision`  
  - Ranking por métrica (`rank_<métrica>`)  
  - Ranking global (`overall_rank`)

- Rodar **validação cruzada (k-fold)** e obter média/desvio-padrão das métricas.  
- Salvar tudo em **CSV**, organizado por experimento.

---

## 🧩 Principais Recursos

- 🔁 **Ensembles prontos para uso**
  - `RandomForestClassifier`
  - `ExtraTreesClassifier`
  - `GradientBoostingClassifier`
  - `AdaBoostClassifier`
  - `BaggingClassifier` com base em RandomForest

- 📊 **Métricas já calculadas**
  - `accuracy`
  - `f1`
  - `recall`
  - `precision`

- 🏆 **Ranking automático**
  - Ranking por métrica: `rank_accuracy`, `rank_f1`, etc.
  - Ranking global: `overall_rank` (soma das posições, menor = melhor).

- 🎯 **Métrica principal configurável**
  - `primary_metric="accuracy"` (padrão) ou `"f1"`, `"recall"`, `"precision"`.

- 🔧 **Customização de hiperparâmetros**
  - Via `model_overrides={"RandomForest": {"n_estimators": 500}, ...}` sem precisar recriar tudo na mão.

- 📂 **Persistência de resultados**
  - Define pasta (`output_dir`) e nome de experimento (`experiment_name`).
  - Salva automaticamente:
    - Resultados de hold-out
    - Resultados de validação cruzada
  - Formato CSV, pronto para análise posterior / relatórios.

- 🧪 **Validação cruzada integrada**
  - `fit_evaluate_cv(X, y, cv=5, ...)`  
  - Retorna métricas `_mean` e `_std` para cada modelo.

---

## 📦 Instalação

### 1️⃣ Clonar o repositório

```bash
git clone https://github.com/ViniciusKanh/auto-ensemble-benchmark.git
cd auto-ensemble-benchmark
````

### 2️⃣ Criar e ativar ambiente virtual (recomendado)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3️⃣ Instalar dependências em modo desenvolvimento

```bash
pip install -e .
```

Isso irá:

* Instalar o pacote `auto-ensemble-benchmark` em modo editável.
* Permitir que você edite o código e teste sem reinstalar.

> 💡 *Quando (e se) houver publicação no PyPI, a instalação ficará tão simples quanto:*
> `pip install auto-ensemble-benchmark`

---

## 🚀 Exemplo Rápido (Wine Dataset)

Uso mínimo da biblioteca com o dataset clássico `wine` do scikit-learn:

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from auto_ensemble_benchmark import AutoEnsembleClassifier

# 1. Dados
data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

# 2. Instanciar o avaliador automático (modo default)
auto = AutoEnsembleClassifier()

# 3. Treinar + avaliar (hold-out)
resultados_holdout = auto.fit_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    media="macro",  # média macro para problema multiclasse
)

print("Resultados (hold-out):")
print(resultados_holdout)

# 4. Resumo textual
print("\nResumo interpretado:")
print(auto.summarize_results(top_k=3))
```

Saída típica (resumida):

```text
Resultados (hold-out):
                          accuracy        f1    recall  precision  rank_accuracy  rank_f1  ...
modelo
RandomForest              1.000000  1.000000  1.000000   1.000000              1        1
ExtraTrees                1.000000  1.000000  1.000000   1.000000              1        1
Bagging_RandomForestBase  1.000000  1.000000  1.000000   1.000000              1        1
AdaBoost                  0.98...   ...
GradientBoosting          0.96...   ...

Resumo interpretado:
Melhor modelo (segundo accuracy): RandomForest com accuracy = 1.0000.
Top modelos (hold-out):
  1. RandomForest -> ...
  2. ExtraTrees -> ...
  3. Bagging_RandomForestBase -> ...
```

---

## ⚙️ Uso com Parâmetros Avançados

Exemplo configurando:

* métrica principal (`primary_metric="f1"`)
* sobrescrita de hiperparâmetros (`model_overrides`)
* pasta de saída e nome de experimento
* salvamento automático em CSV

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from auto_ensemble_benchmark import AutoEnsembleClassifier

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=123,
    stratify=y,
)

auto = AutoEnsembleClassifier(
    metricas=["accuracy", "f1", "recall", "precision"],
    primary_metric="f1",           # ranking guiado por F1
    add_rank_columns=True,         # inclui rank_<métrica> e overall_rank
    model_overrides={
        "RandomForest": {"n_estimators": 500, "max_depth": None},
        "ExtraTrees": {"n_estimators": 800},
    },
    output_dir="resultados_bench",     # pasta para salvar CSVs
    experiment_name="wine_ensembles",  # prefixo dos arquivos
    save_on_evaluate=True,            # salva hold-out
    save_on_cv=True,                  # salva cross-validation
)

resultados_holdout = auto.fit_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    media="macro",
)

print(resultados_holdout)

resultados_cv = auto.fit_evaluate_cv(
    X,
    y,
    cv=5,
    media="macro",
    random_state_cv=123,
    shuffle=True,
)

print(resultados_cv)
```

Arquivos gerados (exemplo):

* `resultados_bench/wine_ensembles_holdout_results.csv`
* `resultados_bench/wine_ensembles_cv_results.csv`

---

## 🧪 Exemplo em Dataset Mais Difícil

Para testar a biblioteca em um cenário mais desafiador, podemos usar `make_classification` com:

* classes desbalanceadas
* ruído nos rótulos (`flip_y`)
* separação moderada entre classes

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from auto_ensemble_benchmark import AutoEnsembleClassifier

# Dataset sintético difícil (binário)
X, y = make_classification(
    n_samples=2000,
    n_features=30,
    n_informative=10,
    n_redundant=5,
    n_clusters_per_class=2,
    weights=[0.7, 0.3],   # desbalanceado
    flip_y=0.05,          # 5% de ruído de rótulo
    class_sep=0.8,        # separação moderada
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

auto = AutoEnsembleClassifier(
    metricas=["accuracy", "f1", "recall", "precision"],
    primary_metric="f1",  # aqui o F1 faz mais sentido como métrica principal
    add_rank_columns=True,
)

resultados_holdout = auto.fit_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    media="binary",  # problema binário
)

print("Resultados (hold-out) - dataset difícil:")
print(resultados_holdout)

print("\nResumo interpretado:")
print(auto.summarize_results(top_k=3))

# Inspecionar o melhor modelo em detalhes
best_model = auto.best_model_

if best_model is not None:
    y_pred = best_model.predict(X_test)

    print("\nMatriz de confusão (melhor modelo):")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report (melhor modelo):")
    print(classification_report(y_test, y_pred, digits=4))
```

Aqui você verá métricas mais realistas (por exemplo, accuracy ~0.85, f1 ~0.74) e diferenças claras entre os ensembles.

---

## 📚 API – Referência Rápida

### `AutoEnsembleClassifier`

```python
from auto_ensemble_benchmark import AutoEnsembleClassifier
```

#### Construtor

```python
auto = AutoEnsembleClassifier(
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

* `modelos` (`dict[str, estimator]`, opcional)
  Se `None`, usa os modelos padrão do `model_zoo` (RandomForest, ExtraTrees, etc).

* `metricas` (`list[str]`, opcional)
  Lista de métricas a calcular. Suporta:
  `["accuracy", "f1", "recall", "precision"]`
  Se `None`, usa todas.

* `primary_metric` (`str`)
  Métrica principal usada para:

  * ordenar a tabela de resultados
  * definir o “melhor modelo” (`best_model_` / `best_model_name_`)

* `add_rank_columns` (`bool`)
  Se `True`, adiciona colunas `rank_<métrica>` e `overall_rank`.

* `random_state` (`int`)
  Semente padrão usada na criação dos modelos default.

* `n_jobs` (`int`)
  Número de núcleos para paralelização (quando suportado pela implementação do modelo).

* `model_overrides` (`dict[str, dict]`, opcional)
  Sobrescreve hiperparâmetros dos modelos padrões. Exemplo:

  ```python
  model_overrides = {
      "RandomForest": {"n_estimators": 500, "max_depth": 10},
      "ExtraTrees": {"n_estimators": 1000},
  }
  ```

* `output_dir` (`str`, opcional)
  Caminho para pasta onde salvar CSVs com resultados. Se `None`, não salva.

* `experiment_name` (`str`, opcional)
  Prefixo dos arquivos CSV. Default: `"auto_ensemble_experiment"`.

* `save_on_evaluate` (`bool`)
  Se `True`, salva resultados de hold-out ao final de `evaluate`/`fit_evaluate`.

* `save_on_cv` (`bool`)
  Se `True`, salva resultados de `fit_evaluate_cv`.

---

### Métodos principais

#### `fit(X_train, y_train)`

Treina todos os modelos com os dados de treino.

```python
auto.fit(X_train, y_train)
```

#### `evaluate(X_test, y_test, media="auto")`

Avalia todos os modelos com dados de teste.

```python
df_resultados = auto.evaluate(X_test, y_test, media="macro")
```

* `media`:

  * `"auto"` → detecta binário vs. multiclasse e escolhe `'binary'` ou `'macro'`.
  * ou qualquer valor aceito pelas métricas do scikit-learn (`"binary"`, `"macro"`, `"micro"`, `"weighted"` etc.).

Retorna `DataFrame` com:

* colunas de métricas (`accuracy`, `f1`, `recall`, `precision`).
* colunas de ranking (`rank_<métrica>`) se `add_rank_columns=True`.
* `overall_rank` (soma dos ranks).

#### `fit_evaluate(X_train, y_train, X_test, y_test, media="auto")`

Atalho: treina e avalia em uma única chamada.

```python
df_resultados = auto.fit_evaluate(
    X_train, y_train, X_test, y_test, media="macro"
)
```

#### `fit_evaluate_cv(X, y, cv=5, media="auto", random_state_cv=42, shuffle=True)`

Avaliação com validação cruzada estratificada.

```python
df_cv = auto.fit_evaluate_cv(
    X, y,
    cv=5,
    media="macro",
    random_state_cv=42,
    shuffle=True,
)
```

Retorna `DataFrame` com:

* `<métrica>_mean`
* `<métrica>_std`

para cada modelo.

#### `get_results()`

Retorna o último `DataFrame` de resultados de hold-out.

```python
df_holdout = auto.get_results()
```

#### `get_results_cv()`

Retorna o último `DataFrame` de resultados de cross-validation.

```python
df_cv = auto.get_results_cv()
```

#### `summarize_results(top_k=3)`

Gera um resumo textual do desempenho dos modelos no hold-out.

```python
print(auto.summarize_results(top_k=3))
```

Exemplo de saída:

```text
Melhor modelo (segundo f1): GradientBoosting com f1 = 0.7692.

Top modelos (hold-out):
  1. GradientBoosting -> accuracy=0.8650, f1=0.7692, recall=0.7181, precision=0.8282, overall_rank=8.0
  2. RandomForest -> ...
  3. Bagging_RandomForestBase -> ...

Diferença entre o 1º e o 2º em f1: 0.0065.
```

---

## 🛣️ Roadmap (Ideias Futuras)

* ✅ Versão inicial para classificação com ensembles.
* ⏳ `AutoEnsembleRegressor` (versão para regressão, com RMSE, MAE, R² etc.).
* ⏳ Integração com XGBoost, LightGBM e CatBoost.
* ⏳ Geração automática de relatório em Markdown / LaTeX para artigos.
* ⏳ Plots automáticos (boxplots das métricas, barras com intervalos de confiança).
* ⏳ Publicação no PyPI.

---

## 🤝 Contribuindo

Contribuições são bem-vindas:

1. Faça um fork do repositório.
2. Crie uma branch para sua feature/fix:

   ```bash
   git checkout -b minha-feature
   ```
3. Faça commits claros.
4. Abra um Pull Request explicando:

   * o problema
   * a solução adotada
   * se possível, inclua exemplos/prints das métricas.

Sugestões de contribuições:

* Novos modelos no `model_zoo` (ensembles adicionais).
* Testes unitários com `pytest`.
* Melhorias no README / documentação.
* Integração com mais tipos de datasets e exemplos.

---

## 📄 Licença

Este projeto é distribuído sob a licença **MIT**.
Você é livre para usar, modificar e distribuir, desde que mantenha os créditos originais.

---

Claro — segue um tópico profissional, elegante, acadêmico e bem-estruturado com **créditos completos**, **cartão profissional**, **foto**, **links**, mantendo o estilo formal e técnico do restante do README.

Você só precisa **copiar e colar no final do seu README**:

---

## 👤 Autor – Sobre o Pesquisador

### **Vinicius de Souza Santos**  
**Pesquisador em Ciência da Computação – UNESP**  
**Ênfase em Data Science, Machine Learning e Engenharia de Modelos**

<p align="left">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQPGe7oWAt4D6YLkx8btAGVGFCecJa1oXFzAA&s" width="180" style="border-radius:12px;"/>
</p>

---

### 🧑‍🎓 Formação & Atuação
- Mestrando em **Ciência da Computação** pela **UNESP**  
- Pesquisador com foco em:
  - Feature Selection  
  - Benchmarking de modelos  
  - Métodos Ensemble  
  - Engenharia de Dados aplicada a Machine Learning  
- Engenheiro de Computação em formação  
- Experiência com Python, Scikit-Learn, Data Mining, Experimentação e validação de modelos

---

### 🌐 Redes e Contato

- **LinkedIn:**  
  👉 https://www.linkedin.com/in/vinicius-souza-santoss/

- **GitHub:**  
  👉 https://github.com/ViniciusKanh

- **E-mail acadêmico:**  
  👉 vinicius-souza.santos@unesp.br

---

### 💬 Nota do Autor

Este projeto foi desenvolvido com a proposta de oferecer uma biblioteca simples, transparente e científica para benchmark automatizado de modelos ensemble, focando em reprodutibilidade, rigor estatístico e facilidade de uso para pesquisadores, estudantes e profissionais da área.

---

