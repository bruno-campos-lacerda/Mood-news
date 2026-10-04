"""Treina o modelo incremental de qualidade (0=ruim, 1=boa).

Usa o dataset em data/noticias.csv e salva artefatos em models/.
"""

import logging
import sys
import argparse
from pathlib import Path

# Adiciona o diretório Backend ao path para importar classifier
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from classifier import DEFAULT_DATASET, treinar_modelo_inicial
from Backend.training_chart import generate_training_chart


MODELS_DIR = BACKEND_DIR / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o classificador ou gera o gráfico de distribuição.")
    parser.add_argument(
        "--grafico",
        action="store_true",
        help="Gera o gráfico de notícias boas e ruins em cinco dias sem alterar os modelos.",
    )
    args = parser.parse_args()

    if args.grafico:
        chart_path = generate_training_chart(DEFAULT_DATASET)
        print(f"Gráfico gerado sem modificar os modelos: {chart_path}")
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    resultado = treinar_modelo_inicial(
        dataset_path=DEFAULT_DATASET,
        modelo_path=MODELS_DIR / "modelo_noticias.pkl",
        vectorizer_path=MODELS_DIR / "vectorizer_noticias.pkl",
        historico_path=MODELS_DIR / "historico_noticias.csv",
    )

    metricas = resultado["metricas"]
    print("Modelo incremental de qualidade treinado e salvo.")
    print(f"Total treino: {metricas.get('total_treino', 0)}")
    print(f"Total teste: {metricas.get('total_teste', 0)}")
    print(f"Distribuicao: {metricas.get('distribuicao', {})}")
    if "acuracia" in metricas:
        print(f"Acuracia: {metricas['acuracia']:.4f}")
        print("Matriz de confusao:")
        print(metricas["matriz_confusao"])
        print("Relatorio:")
        print(metricas["relatorio"])


if __name__ == "__main__":
    main()
