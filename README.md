# MoodNews

Aplicativo desktop em PySide6 para classificar notícias ambientais localmente com modelos de Machine Learning. A análise é feita diretamente no processo do aplicativo: não há servidor, API HTTP ou conexão externa durante o uso.

## Executar

Com Python 3.10 ou superior:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r Backend/requirements.txt -r app_pyside6/requirements.txt
python app_pyside6/main.py
```

O aplicativo carrega os modelos incluídos em `Backend/models/` ao iniciar e os executa localmente. Se os arquivos de modelo estiverem ausentes, treine-os primeiro:

```bash
python Backend/scripts/treinar_qualidade.py
python Backend/scripts/treinar_tipo.py
```

## Modelos

| Arquivo | Função |
|---|---|
| `modelo_noticias.pkl` e `vectorizer_noticias.pkl` | Classificam o impacto como BOA ou RUIM |
| `modelo_noticias_tipo.pkl` | Classifica a categoria ambiental |

O classificador de qualidade também oferece aprendizado incremental pelo terminal:

```bash
python Backend/classifier.py --interactive
```

O gerador opcional de dataset (`Backend/scripts/build_dataset.py`) busca notícias na internet quando executado. Ele não é necessário para instalar ou usar o aplicativo; a classificação da aplicação usa os modelos e dados locais.

## Dependências

- `PySide6` — interface desktop
- `scikit-learn`, `numpy`, `pandas` e `joblib` — modelos e treinamento

*Projeto educacional — 2026*
