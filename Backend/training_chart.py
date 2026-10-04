from pathlib import Path

import numpy as np
import pandas as pd

from .classifier import DEFAULT_DATASET, normalizar_rotulo


BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_CHART_PATH = BACKEND_DIR / "desempenho" / "resultados" / "noticias_por_dia.png"
DAYS_OF_TRAINING = 5


def generate_training_chart(
    dataset_path: Path = DEFAULT_DATASET,
    output_path: Path = DEFAULT_CHART_PATH,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = pd.read_csv(dataset_path, usecols=["fellingtarget"])["fellingtarget"]
    labels = labels.dropna().map(normalizar_rotulo).to_numpy(dtype=int)
    if not len(labels):
        raise ValueError(f"O dataset não contém rótulos válidos: {dataset_path}")

    daily_batches = np.array_split(
        np.random.default_rng(seed=42).permutation(labels),
        DAYS_OF_TRAINING,
    )
    boas = [int(np.count_nonzero(batch == 1)) for batch in daily_batches]
    ruins = [int(np.count_nonzero(batch == 0)) for batch in daily_batches]

    figure, axis = plt.subplots(figsize=(11, 6), facecolor="#1a1b26")
    axis.set_facecolor("#1a1b26")
    dias = np.arange(DAYS_OF_TRAINING)
    largura = 0.36
    barras_boas = axis.bar(
        dias - largura / 2,
        boas,
        largura,
        label="Notícias boas",
        color="#9ece6a",
    )
    barras_ruins = axis.bar(
        dias + largura / 2,
        ruins,
        largura,
        label="Notícias ruins",
        color="#f7768e",
    )

    axis.set_title("Notícias boas e ruins por dia de treinamento", color="#c0caf5", pad=18)
    axis.set_xlabel("Dia de treinamento", color="#a9b1d6")
    axis.set_ylabel("Quantidade de notícias", color="#a9b1d6")
    axis.set_xticks(dias, [f"Dia {dia}" for dia in range(1, DAYS_OF_TRAINING + 1)])
    axis.tick_params(colors="#a9b1d6")
    axis.grid(axis="y", color="#414868", alpha=0.45)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#414868")
    axis.legend(facecolor="#24283b", edgecolor="#414868", labelcolor="#c0caf5")
    axis.bar_label(barras_boas, color="#c0caf5", padding=3, fontsize=9)
    axis.bar_label(barras_ruins, color="#c0caf5", padding=3, fontsize=9)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output_path
