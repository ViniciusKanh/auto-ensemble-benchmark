"""
core.py

Implementa a classe principal AutoEnsembleClassifier,
responsável por treinar e avaliar automaticamente múltiplos
modelos ensemble de classificação.

Fluxos de uso típicos
---------------------

Avaliação simples (hold-out):

    from auto_ensemble_benchmark import AutoEnsembleClassifier
    auto = AutoEnsembleClassifier()
    auto.fit(X_train, y_train)
    resultados = auto.evaluate(X_test, y_test)

Ou em uma única chamada:

    resultados = auto.fit_evaluate(X_train, y_train, X_test, y_test)

Avaliação com validação cruzada (k-fold):

    resultados_cv = auto.fit_evaluate_cv(X, y, cv=5)
"""

from typing import Dict, List, Optional, Mapping
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from .model_zoo import criar_modelos_padrao


class AutoEnsembleClassifier:
    """
    Classe para treinar e avaliar automaticamente múltiplos modelos ensemble.

    Objetivos principais:
    ---------------------
    - Receber conjuntos de treino e teste.
    - Treinar vários modelos ensemble (RandomForest, ExtraTrees, etc.).
    - Avaliar cada modelo nas métricas:
        * Accuracy
        * F1-Score
        * Recall
        * Precision
    - Retornar um DataFrame com os resultados, com:
        * ranking por métrica (opcional)
        * ranking global (overall_rank) baseado na soma das posições.
    - Opcionalmente salvar resultados em disco (CSV) em uma pasta definida.

    Atributos principais:
    ---------------------
    modelos : dict
        Dicionário de modelos a serem avaliados.
    modelos_treinados_ : dict
        Modelos já ajustados (após chamada de .fit()).
    resultados_ : pd.DataFrame ou None
        Resultados da última chamada de .evaluate() / .fit_evaluate().
    resultados_cv_ : pd.DataFrame ou None
        Resultados da última chamada de .fit_evaluate_cv().
    best_model_name_ : str ou None
        Nome do melhor modelo segundo a métrica principal.
    best_model_ : objeto estimador ou None
        Instância do melhor modelo treinado (após .fit()).
    """

    def __init__(
        self,
        modelos: Optional[Dict[str, object]] = None,
        metricas: Optional[List[str]] = None,
        primary_metric: str = "accuracy",
        add_rank_columns: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
        model_overrides: Optional[Mapping[str, Mapping]] = None,
        output_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        save_on_evaluate: bool = False,
        save_on_cv: bool = False,
    ):
        """
        Parâmetros
        ----------
        modelos : dict, opcional
            Dicionário {nome_modelo: instancia_modelo}. Se None, usa modelos padrão.
        metricas : list[str], opcional
            Lista de métricas a serem calculadas.
            Suporta:
                ["accuracy", "f1", "recall", "precision"]
            Se None, usa todas.
        primary_metric : str
            Métrica principal usada para:
                - ordenar a tabela de resultados
                - definir o "melhor modelo"
            Deve estar contida em `metricas`.
        add_rank_columns : bool
            Se True, adiciona colunas de ranking por métrica (`rank_<métrica>`)
            e um ranking global (`overall_rank`).
        random_state : int
            Semente aleatória padrão (para criação dos modelos default).
        n_jobs : int
            Número de núcleos para paralelização nos modelos que suportam.
        model_overrides : dict[str, dict], opcional
            Sobrescritas de hiperparâmetros a serem passadas para
            `criar_modelos_padrao`, se `modelos` não for fornecido.
        output_dir : str, opcional
            Caminho para pasta onde salvar resultados (CSV). Se None,
            não salva nada automaticamente.
        experiment_name : str, opcional
            Nome do experimento usado como prefixo nos arquivos de saída.
            Se None, usa "auto_ensemble_experiment".
        save_on_evaluate : bool
            Se True, salva resultados de hold-out em CSV ao final de .evaluate().
        save_on_cv : bool
            Se True, salva resultados de cross-validation em CSV ao final de .fit_evaluate_cv().
        """
        if modelos is None:
            modelos = criar_modelos_padrao(
                random_state=random_state,
                n_jobs=n_jobs,
                overrides=model_overrides,
            )

        if metricas is None:
            metricas = ["accuracy", "f1", "recall", "precision"]

        self.modelos = modelos
        self.metricas = metricas
        self.primary_metric = primary_metric
        self.add_rank_columns = add_rank_columns
        self.random_state = random_state
        self.n_jobs = n_jobs

        # I/O de resultados
        self.output_dir: Optional[Path] = Path(output_dir) if output_dir is not None else None
        self.experiment_name: str = experiment_name or "auto_ensemble_experiment"
        self.save_on_evaluate = save_on_evaluate
        self.save_on_cv = save_on_cv

        # Atributos preenchidos após o fit/evaluate
        self.modelos_treinados_: Dict[str, object] = {}
        self.resultados_: Optional[pd.DataFrame] = None
        self.resultados_cv_: Optional[pd.DataFrame] = None
        self.classes_: Optional[np.ndarray] = None
        self.best_model_name_: Optional[str] = None
        self.best_model_: Optional[object] = None

        self._validar_metricas()
        self._preparar_pasta_saida()

    # ============================================================
    # Utilitários internos
    # ============================================================

    def _validar_metricas(self) -> None:
        """
        Valida se as métricas fornecidas são suportadas
        e se a primary_metric é válida.
        """
        metricas_suportadas = {"accuracy", "f1", "recall", "precision"}
        desconhecidas = set(self.metricas) - metricas_suportadas
        if desconhecidas:
            raise ValueError(
                f"Métricas desconhecidas: {desconhecidas}. "
                f"Métricas suportadas: {metricas_suportadas}."
            )

        if self.primary_metric not in self.metricas:
            raise ValueError(
                f"primary_metric='{self.primary_metric}' não está em metricas={self.metricas}."
            )

    def _preparar_pasta_saida(self) -> None:
        """
        Cria a pasta de saída se `output_dir` foi definido.
        """
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _detectar_media(
        self,
        y_true: np.ndarray,
        media: str = "auto",
    ) -> str:
        """
        Decide qual tipo de média usar para F1/Recall/Precision.

        Parâmetros
        ----------
        y_true : array-like
            Rótulos verdadeiros (necessário para detectar binário vs multiclasse).
        media : str
            - 'auto' (padrão):
                * binário => 'binary'
                * multiclasse => 'macro'
            - outro valor: retornado diretamente.

        Retorno
        -------
        str
            Tipo de média a ser utilizada.
        """
        if media != "auto":
            return media

        classes_unicas = np.unique(y_true)
        if len(classes_unicas) == 2:
            return "binary"
        return "macro"

    @staticmethod
    def _clonar_modelo(modelo):
        """
        Clona um modelo sklearn usando sklearn.base.clone.

        Vantagens:
        ----------
        - Funciona com estimadores aninhados (ex.: BaggingClassifier).
        - Mantém os mesmos hiperparâmetros.
        """
        return clone(modelo)

    def _salvar_df(self, df: pd.DataFrame, suffix: str) -> None:
        """
        Salva um DataFrame em CSV na pasta de saída, se configurada.

        Parâmetros
        ----------
        df : DataFrame
            DataFrame a ser salvo.
        suffix : str
            Sufixo do nome do arquivo (ex.: 'holdout_results').
        """
        if self.output_dir is None:
            return

        filename = f"{self.experiment_name}_{suffix}.csv"
        path = self.output_dir / filename
        df.to_csv(path, index=True)

    # ============================================================
    # Treino e avaliação em hold-out
    # ============================================================

    def fit(self, X_train, y_train) -> "AutoEnsembleClassifier":
        """
        Treina todos os modelos no conjunto de treino.

        Parâmetros
        ----------
        X_train : array-like ou DataFrame
            Atributos de treino.
        y_train : array-like
            Rótulos de treino.

        Retorno
        -------
        self : AutoEnsembleClassifier
        """
        self.modelos_treinados_.clear()
        self.classes_ = np.unique(y_train)
        self.best_model_name_ = None
        self.best_model_ = None

        for nome, modelo in self.modelos.items():
            modelo_clone = self._clonar_modelo(modelo)
            modelo_clone.fit(X_train, y_train)
            self.modelos_treinados_[nome] = modelo_clone

        return self

    def evaluate(
        self,
        X_test,
        y_test,
        media: str = "auto",
    ) -> pd.DataFrame:
        """
        Avalia todos os modelos treinados no conjunto de teste.

        Parâmetros
        ----------
        X_test : array-like ou DataFrame
            Atributos de teste.
        y_test : array-like
            Rótulos verdadeiros de teste.
        media : str
            Tipo de média para F1/Recall/Precision.
            - 'auto' (default):
                * 'binary' se problema binário,
                * 'macro' se multiclasse.
            - 'binary', 'macro', 'micro', 'weighted', etc.

        Retorno
        -------
        DataFrame
            DataFrame com linhas = modelos e colunas = métricas,
            podendo incluir colunas de ranking por métrica e ranking global.
        """
        if not self.modelos_treinados_:
            raise RuntimeError("Nenhum modelo foi treinado. Chame .fit() antes de .evaluate().")

        media_real = self._detectar_media(y_test, media)
        registros = []

        for nome, modelo in self.modelos_treinados_.items():
            y_pred = modelo.predict(X_test)

            registro = {"modelo": nome}

            if "accuracy" in self.metricas:
                registro["accuracy"] = accuracy_score(y_test, y_pred)

            if "f1" in self.metricas:
                registro["f1"] = f1_score(y_test, y_pred, average=media_real)

            if "recall" in self.metricas:
                registro["recall"] = recall_score(y_test, y_pred, average=media_real)

            if "precision" in self.metricas:
                registro["precision"] = precision_score(y_test, y_pred, average=media_real)

            registros.append(registro)

        df = pd.DataFrame(registros).set_index("modelo")

        # Ranking por métrica e ranking global (overall_rank)
        if self.add_rank_columns:
            for m in self.metricas:
                if m in df.columns:
                    df[f"rank_{m}"] = df[m].rank(
                        ascending=False,
                        method="min",
                    ).astype(int)

            # Ranking global = soma das posições em cada métrica
            rank_cols = [f"rank_{m}" for m in self.metricas if f"rank_{m}" in df.columns]
            if rank_cols:
                df["overall_rank"] = df[rank_cols].sum(axis=1).astype(float)
                # quanto menor o overall_rank, melhor

        # Ordena pela primary_metric (depois pelo overall_rank se existir)
        if "overall_rank" in df.columns:
            df = df.sort_values(
                by=[self.primary_metric, "overall_rank"],
                ascending=[False, True],
            )
        else:
            df = df.sort_values(by=self.primary_metric, ascending=False)

        # Guarda melhor modelo segundo a primary_metric (e desempate pelo overall_rank)
        self.best_model_name_ = df.index[0]
        self.best_model_ = self.modelos_treinados_[self.best_model_name_]

        self.resultados_ = df

        # Salva em disco se configurado
        if self.save_on_evaluate:
            self._salvar_df(df, "holdout_results")

        return df

    def fit_evaluate(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        media: str = "auto",
    ) -> pd.DataFrame:
        """
        Atalho para treinar e avaliar em uma única chamada (hold-out).

        Parâmetros
        ----------
        X_train, y_train : ver .fit()
        X_test, y_test   : ver .evaluate()
        media : str
            Tipo de média para F1/Recall/Precision.

        Retorno
        -------
        DataFrame
            DataFrame de resultados com métricas e ranking.
        """
        self.fit(X_train, y_train)
        return self.evaluate(X_test, y_test, media=media)

    # ============================================================
    # Avaliação com validação cruzada (k-fold)
    # ============================================================

    def fit_evaluate_cv(
        self,
        X,
        y,
        cv: int = 5,
        media: str = "auto",
        random_state_cv: int = 42,
        shuffle: bool = True,
    ) -> pd.DataFrame:
        """
        Avalia os modelos usando validação cruzada estratificada.

        Para cada modelo:
        - Treina em (k-1) folds
        - Avalia no fold restante
        - Repete para todos os folds
        - Calcula média e desvio-padrão das métricas

        Parâmetros
        ----------
        X : array-like ou DataFrame
            Atributos (todas as amostras).
        y : array-like
            Rótulos verdadeiros.
        cv : int
            Número de folds da validação cruzada.
        media : str
            Tipo de média para F1/Recall/Precision (mesma lógica de .evaluate()).
        random_state_cv : int
            Random state do StratifiedKFold.
        shuffle : bool
            Se True, embaralha os dados antes de criar os folds.

        Retorno
        -------
        DataFrame
            DataFrame com linhas = modelos e colunas:
            - <métrica>_mean
            - <métrica>_std
        """
        X = np.asarray(X)
        y = np.asarray(y)

        skf = StratifiedKFold(
            n_splits=cv,
            shuffle=shuffle,
            random_state=random_state_cv,
        )

        media_real = self._detectar_media(y, media)
        registros = []

        for nome, modelo in self.modelos.items():
            metricas_por_fold = {m: [] for m in self.metricas}

            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                modelo_clone = self._clonar_modelo(modelo)
                modelo_clone.fit(X_train, y_train)
                y_pred = modelo_clone.predict(X_test)

                if "accuracy" in self.metricas:
                    metricas_por_fold["accuracy"].append(
                        accuracy_score(y_test, y_pred)
                    )

                if "f1" in self.metricas:
                    metricas_por_fold["f1"].append(
                        f1_score(y_test, y_pred, average=media_real)
                    )

                if "recall" in self.metricas:
                    metricas_por_fold["recall"].append(
                        recall_score(y_test, y_pred, average=media_real)
                    )

                if "precision" in self.metricas:
                    metricas_por_fold["precision"].append(
                        precision_score(y_test, y_pred, average=media_real)
                    )

            # Agrega média e desvio-padrão por métrica
            registro = {"modelo": nome}
            for m in self.metricas:
                valores = np.array(metricas_por_fold[m])
                registro[f"{m}_mean"] = float(np.mean(valores))
                registro[f"{m}_std"] = float(np.std(valores, ddof=1))  # desvio-padrão amostral

            registros.append(registro)

        df_cv = pd.DataFrame(registros).set_index("modelo")

        # Ordena pela métrica principal (média)
        col_mean = f"{self.primary_metric}_mean"
        if col_mean in df_cv.columns:
            df_cv = df_cv.sort_values(by=col_mean, ascending=False)

        self.resultados_cv_ = df_cv

        # Salva em disco se configurado
        if self.save_on_cv:
            self._salvar_df(df_cv, "cv_results")

        return df_cv

    # ============================================================
    # Métodos auxiliares de inspeção
    # ============================================================

    def get_results(self) -> pd.DataFrame:
        """
        Retorna o DataFrame de resultados da última avaliação hold-out.
        """
        if self.resultados_ is None:
            raise RuntimeError(
                "Nenhuma avaliação hold-out foi feita ainda. "
                "Chame .evaluate() ou .fit_evaluate()."
            )
        return self.resultados_

    def get_results_cv(self) -> pd.DataFrame:
        """
        Retorna o DataFrame de resultados da última avaliação com cross-validation.
        """
        if self.resultados_cv_ is None:
            raise RuntimeError(
                "Nenhuma avaliação com validação cruzada foi feita ainda. "
                "Chame .fit_evaluate_cv()."
            )
        return self.resultados_cv_

    def summarize_results(self, top_k: int = 3) -> str:
        """
        Gera um resumo textual dos resultados hold-out, destacando
        o melhor modelo e diferenças de desempenho.

        Parâmetros
        ----------
        top_k : int
            Número de melhores modelos a destacar no resumo.

        Retorno
        -------
        str
            Resumo textual das métricas e ranking dos modelos.
        """
        if self.resultados_ is None:
            raise RuntimeError(
                "Nenhuma avaliação hold-out foi feita ainda. "
                "Chame .evaluate() ou .fit_evaluate()."
            )

        df = self.resultados_
        pm = self.primary_metric

        linhas = []

        # Melhor modelo
        best_name = df.index[0]
        best_val = df.iloc[0][pm]
        linhas.append(
            f"Melhor modelo (segundo {pm}): {best_name} "
            f"com {pm} = {best_val:.4f}."
        )

        # Top-k modelos
        top_k = min(top_k, len(df))
        if top_k > 1:
            linhas.append("\nTop modelos (hold-out):")
            for i in range(top_k):
                nome = df.index[i]
                vals = df.iloc[i]
                metricas_str = ", ".join(
                    f"{m}={vals[m]:.4f}" for m in self.metricas if m in df.columns
                )
                if "overall_rank" in df.columns:
                    metricas_str += f", overall_rank={vals['overall_rank']:.1f}"
                linhas.append(f"  {i+1}. {nome} -> {metricas_str}")

        # Diferença do primeiro para o segundo, se existir
        if len(df) >= 2:
            segundo_val = df.iloc[1][pm]
            diff = best_val - segundo_val
            linhas.append(
                f"\nDiferença entre o 1º e o 2º em {pm}: {diff:.4f}."
            )

        return "\n".join(linhas)
