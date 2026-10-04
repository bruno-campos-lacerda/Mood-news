import importlib
import re
import sys
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import joblib

from .classifier import classificar_noticia


BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PATH_TIPO = BACKEND_DIR / "models" / "modelo_noticias_tipo.pkl"
MODEL_PATH_QUALIDADE = BACKEND_DIR / "models" / "modelo_noticias.pkl"
VECTORIZER_PATH_QUALIDADE = BACKEND_DIR / "models" / "vectorizer_noticias.pkl"

TIPO_MAPA = {
    "desmatamento": "Desmatamento e Queimadas",
    "poluição": "Poluição e Degradação",
    "preservação": "Preservação Ambiental",
    "acidente": "Desastres Ambientais",
    "geral": "Temas Gerais de Meio Ambiente",
    "polui": "Poluição e Degradação",
    "preserva": "Preservação Ambiental",
}


@dataclass(frozen=True)
class ClassificationResult:
    response: str
    positive_probability: float
    negative_probability: float


@contextmanager
def _sklearn_loss_pickle_compatibility():
    previous_module = sys.modules.get("_loss")
    sys.modules["_loss"] = importlib.import_module("sklearn._loss._loss")
    try:
        yield
    finally:
        if previous_module is None:
            sys.modules.pop("_loss", None)
        else:
            sys.modules["_loss"] = previous_module


def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower().strip())
    texto = "".join(caractere for caractere in texto if unicodedata.category(caractere) != "Mn")
    texto = re.sub(r"[^\w\s]", "", texto)
    return re.sub(r"\s+", " ", texto)


class MoodNewsService:
    def __init__(self, modelo_tipo, modelo_qualidade, vectorizer_qualidade):
        self.modelo_tipo = modelo_tipo
        self.modelo_qualidade = modelo_qualidade
        self.vectorizer_qualidade = vectorizer_qualidade

    @classmethod
    def load(cls):
        model_paths = (
            MODEL_PATH_TIPO,
            MODEL_PATH_QUALIDADE,
            VECTORIZER_PATH_QUALIDADE,
        )
        for model_path in model_paths:
            if not model_path.is_file():
                raise FileNotFoundError(f"Arquivo de modelo não encontrado: {model_path}")

        with _sklearn_loss_pickle_compatibility():
            modelo_tipo = joblib.load(MODEL_PATH_TIPO)
            modelo_qualidade = joblib.load(MODEL_PATH_QUALIDADE)
            vectorizer_qualidade = joblib.load(VECTORIZER_PATH_QUALIDADE)
        return cls(modelo_tipo, modelo_qualidade, vectorizer_qualidade)

    def classify_result(self, texto: str) -> ClassificationResult:
        texto = texto.strip()
        if not texto:
            raise ValueError("O texto da notícia não pode estar vazio.")

        texto_normalizado = normalizar_texto(texto)
        tipo_probabilidades = self.modelo_tipo.predict_proba([texto_normalizado])[0]
        tipo_classes = self.modelo_tipo.classes_
        tipo_label = tipo_classes[max(range(len(tipo_probabilidades)), key=tipo_probabilidades.__getitem__)]

        tipo_final = "Temas Gerais de Meio Ambiente"
        tipo_str = str(tipo_label).lower()
        for chave, categoria in TIPO_MAPA.items():
            if chave in tipo_str:
                tipo_final = categoria
                break

        qualidade = classificar_noticia(
            texto_normalizado,
            self.modelo_qualidade,
            self.vectorizer_qualidade,
        )
        probabilidade_boa = qualidade["probabilidades"]["boa"]
        porcentagem_boa = round(probabilidade_boa * 100)
        porcentagem_ruim = 100 - porcentagem_boa
        confianca = int(qualidade["confianca"] * 100)
        if qualidade["rotulo"] == 1:
            response = (
                "Que ótima notícia! 🌱\n\n"
                f"Essa notícia tem um impacto positivo no meio ambiente e está relacionada a {tipo_final}.\n\n"
                f"Confiança da IA: {confianca}%"
            )
        else:
            response = (
                "Infelizmente, esta notícia traz um impacto negativo. ⚠️\n\n"
                f"Ela aborda problemas de {tipo_final}, o que é preocupante para a nossa natureza.\n\n"
                f"Confiança da IA: {confianca}%"
            )
        return ClassificationResult(
            response=response,
            positive_probability=porcentagem_boa / 100,
            negative_probability=porcentagem_ruim / 100,
        )

    def classify(self, texto: str) -> str:
        return self.classify_result(texto).response
