"""
Pacote auto_ensemble_benchmark

Fornece uma interface simples para avaliação automática
de modelos ensemble de classificação.
"""

from .core import AutoEnsembleClassifier
from .model_zoo import criar_modelos_padrao

__all__ = ["AutoEnsembleClassifier", "criar_modelos_padrao"]
