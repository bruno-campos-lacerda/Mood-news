"""Treina o modelo de classificação de TIPO de notícia.

Usa Pipeline(TfidfVectorizer + SGDClassifier) e salva em models/.
"""

import sys
from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

BACKEND_DIR = Path(__file__).resolve().parent.parent
ARQUIVO_DATASET = BACKEND_DIR / "data" / "noticias.csv"
ARQUIVO_MODELO_TIPO = BACKEND_DIR / "models" / "modelo_noticias_tipo.pkl"


def main():
    print("Treinando modelo de Tipo de Notícia...")

    ARQUIVO_MODELO_TIPO.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ARQUIVO_DATASET, encoding="utf-8")

    # remove linhas inválidas
    df = df.dropna(subset=["typetarget", "text"])
    df = df[df["text"].str.strip() != ""]

    X = df["text"]
    y = df["typetarget"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo_tipo = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", max_features=20000)),
        ("clf", SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42))
    ])

    modelo_tipo.fit(X_train, y_train)
    y_pred = modelo_tipo.predict(X_test)
    
    print("Acurácia (Tipo):", accuracy_score(y_test, y_pred))

    joblib.dump(modelo_tipo, ARQUIVO_MODELO_TIPO)
    print("Modelo de tipo salvo em:", ARQUIVO_MODELO_TIPO)


if __name__ == "__main__":
    main()
