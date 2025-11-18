"""
model_zoo.py

Contém funções para criação de coleções de modelos ensemble
com hiperparâmetros padrão ou customizados.
"""

from typing import Dict, Optional, Mapping
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
)


def criar_modelos_padrao(
    random_state: int = 42,
    n_jobs: int = -1,
    overrides: Optional[Mapping[str, Mapping]] = None,
) -> Dict[str, object]:
    """
    Cria um dicionário de modelos ensemble com hiperparâmetros padrão,
    permitindo ao usuário sobrescrever parâmetros de forma opcional.

    Parâmetros
    ----------
    random_state : int
        Semente aleatória para reprodutibilidade.
    n_jobs : int
        Número de núcleos para paralelização nos modelos que suportam (ex.: RandomForest).
    overrides : dict[str, dict], opcional
        Dicionário com sobrescritas de hiperparâmetros por modelo.
        Exemplo:
            overrides = {
                "RandomForest": {"n_estimators": 500, "max_depth": 10},
                "ExtraTrees":   {"n_estimators": 1000}
            }

    Retorno
    -------
    dict
        Dicionário {nome_modelo: instancia_modelo}.
    """
    modelos = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
        "Bagging_RandomForestBase": BaggingClassifier(
            # Em versões mais antigas do sklearn o parâmetro era base_estimator
            estimator=RandomForestClassifier(
                n_estimators=50,
                random_state=random_state,
                n_jobs=n_jobs,
            ),
            n_estimators=10,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200,
            random_state=random_state,
        ),
    }

    # Aplica sobrescritas de hiperparâmetros, se o usuário forneceu
    if overrides is not None:
        for nome_modelo, params in overrides.items():
            if nome_modelo in modelos:
                modelos[nome_modelo].set_params(**params)

    return modelos
