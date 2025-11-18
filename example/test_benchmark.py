# test_benchmark.py
# Testes da biblioteca auto_ensemble_benchmark:
# 1) Uso simples com Wine
# 2) Uso parametrizado com Wine
# 3) Uso em dataset mais difícil (make_classification)

from sklearn.datasets import load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from auto_ensemble_benchmark import AutoEnsembleClassifier


def teste_simples_wine():
    """
    Teste básico com o dataset Wine:
    - Usa os modelos ensemble padrão
    - Usa métricas padrão (accuracy, f1, recall, precision)
    - Não salva nada em disco
    """
    print("\n########### TESTE SIMPLES (WINE - DEFAULT) ###########")

    # 1. Carrega dataset
    data = load_wine()
    X = data.data
    y = data.target

    # 2. Split hold-out
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # 3. Instancia usando apenas defaults
    auto = AutoEnsembleClassifier()

    # 4. Hold-out: treino + avaliação
    resultados_holdout = auto.fit_evaluate(
        X_train,
        y_train,
        X_test,
        y_test,
        media="macro",
    )

    print("\n===================== HOLD-OUT (WINE - DEFAULT) =====================")
    print("Resultados dos modelos ensemble (hold-out):")
    print(resultados_holdout)

    print("\nResumo interpretado (hold-out):")
    print(auto.summarize_results(top_k=3))

    # 5. Validação cruzada simples (sem salvar)
    resultados_cv = auto.fit_evaluate_cv(
        X,
        y,
        cv=5,
        media="macro",
        random_state_cv=42,
        shuffle=True,
    )

    print("\n================= VALIDAÇÃO CRUZADA (WINE - DEFAULT) =================")
    print("Resultados médios (cross-validation, 5-fold):")
    print(resultados_cv)


def teste_parametros_wine():
    """
    Teste com parâmetros no dataset Wine:
    - Define primary_metric = 'f1'
    - Sobrescreve hiperparâmetros de RF e ExtraTrees
    - Define pasta de saída e nome de experimento
    - Salva resultados de hold-out e CV em CSV
    """
    print("\n########### TESTE COM PARÂMETROS (WINE) ###########")

    # 1. Carrega dataset
    data = load_wine()
    X = data.data
    y = data.target

    # 2. Split hold-out
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=123,  # outro seed só para variar
        stratify=y,
    )

    # 3. Instancia com parâmetros avançados
    auto = AutoEnsembleClassifier(
        metricas=["accuracy", "f1", "recall", "precision"],
        primary_metric="f1",           # ranking guiado por F1
        add_rank_columns=True,
        model_overrides={              # sobrescrita de hiperparâmetros
            "RandomForest": {"n_estimators": 500, "max_depth": None},
            "ExtraTrees": {"n_estimators": 800},
        },
        output_dir="resultados_bench",   # pasta onde salvar CSV
        experiment_name="wine_ensembles",  # prefixo dos arquivos
        save_on_evaluate=True,            # salva hold-out
        save_on_cv=True,                  # salva CV
    )

    # 4. Hold-out: treino + avaliação
    resultados_holdout = auto.fit_evaluate(
        X_train,
        y_train,
        X_test,
        y_test,
        media="macro",
    )

    print("\n===================== HOLD-OUT (WINE - PARAMETRIZADO) =====================")
    print("Resultados dos modelos ensemble (hold-out):")
    print(resultados_holdout)

    print("\nResumo interpretado (hold-out):")
    print(auto.summarize_results(top_k=3))

    # 5. Validação cruzada (com salvamento)
    resultados_cv = auto.fit_evaluate_cv(
        X,
        y,
        cv=5,
        media="macro",
        random_state_cv=123,
        shuffle=True,
    )

    print("\n=========== VALIDAÇÃO CRUZADA (WINE - PARAMETRIZADO) ===========")
    print("Resultados médios (cross-validation, 5-fold):")
    print(resultados_cv)

    print(
        "\nArquivos CSV devem ter sido salvos em 'resultados_bench/' "
        "com o prefixo 'wine_ensembles_'."
    )


def teste_dataset_dificil():
    """
    Teste com um dataset sintético mais difícil (make_classification):
    - Binário, com desbalanceamento, ruído de rótulo e baixa separação de classes.
    - Aqui é esperado que as métricas NÃO cheguem em 1.0.
    - Inclui:
        * avaliação hold-out
        * cross-validation
        * matriz de confusão e classification report do melhor modelo
    """
    print("\n########### TESTE EM DATASET MAIS DIFÍCIL (make_classification) ###########")

    # 1. Gera dataset sintético mais complexo
    X, y = make_classification(
        n_samples=2000,
        n_features=30,
        n_informative=10,
        n_redundant=5,
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[0.7, 0.3],   # desbalanceado
        flip_y=0.05,          # 5% de ruído nos rótulos
        class_sep=0.8,        # separação moderada (não muito fácil)
        random_state=42,
    )

    # 2. Split hold-out
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # 3. Instancia o avaliador com F1 como métrica principal (problema binário)
    auto = AutoEnsembleClassifier(
        metricas=["accuracy", "f1", "recall", "precision"],
        primary_metric="f1",
        add_rank_columns=True,
        output_dir="resultados_bench_dificil",
        experiment_name="hard_dataset",
        save_on_evaluate=False,  # pode ligar se quiser gerar CSV
        save_on_cv=False,
    )

    # 4. Hold-out
    resultados_holdout = auto.fit_evaluate(
        X_train,
        y_train,
        X_test,
        y_test,
        media="binary",  # problema binário; métrica binária faz sentido
    )

    print("\n===================== HOLD-OUT (DATASET DIFÍCIL) =====================")
    print("Resultados dos modelos ensemble (hold-out):")
    print(resultados_holdout)

    print("\nResumo interpretado (hold-out):")
    print(auto.summarize_results(top_k=3))

    # 5. Matriz de confusão + classification report do melhor modelo
    best_model = auto.best_model_
    if best_model is not None:
        y_pred_best = best_model.predict(X_test)

        print("\nMatriz de confusão (melhor modelo - hold-out):")
        print(confusion_matrix(y_test, y_pred_best))

        print("\nClassification report (melhor modelo - hold-out):")
        print(classification_report(y_test, y_pred_best, digits=4))
    else:
        print("\n[AVISO] Nenhum best_model_ definido. Verifique a lógica de avaliação.")

    # 6. Validação cruzada nesse dataset mais difícil
    resultados_cv = auto.fit_evaluate_cv(
        X,
        y,
        cv=5,
        media="binary",
        random_state_cv=42,
        shuffle=True,
    )

    print("\n=========== VALIDAÇÃO CRUZADA (DATASET DIFÍCIL) ===========")
    print("Resultados médios (cross-validation, 5-fold):")
    print(resultados_cv)


def main():
    teste_simples_wine()
    teste_parametros_wine()
    teste_dataset_dificil()
    print("\nTodos os testes foram executados com sucesso.")


if __name__ == "__main__":
    main()
