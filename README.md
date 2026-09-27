# MoodNews

Um projeto educacional com uma aplicação desktop PySide6 e uma API Python para classificação de notícias ambientais usando Machine Learning.

## 📁 Estrutura do Projeto

```
MOODNEWS/
├── app_pyside6/              # Aplicação desktop PySide6
│   ├── main.py
│   ├── models/               # Modelos de dados da interface
│   ├── screens/              # Telas da aplicação
│   ├── assets/               # Assets do app
│   └── requirements.txt
│
├── Backend/                  # Backend Python
│   ├── app.py               # API REST (FastAPI)
│   ├── classifier.py        # Classificador incremental (TF-IDF + SGD)
│   ├── requirements.txt     # Dependências do backend
│   ├── data/
│   │   └── noticias.csv     # Dataset de notícias ambientais
│   ├── models/
│   │   ├── modelo_noticias.pkl          # Modelo de qualidade (boa/ruim)
│   │   ├── modelo_noticias_tipo.pkl     # Modelo de tipo de notícia
│   │   ├── vectorizer_noticias.pkl      # TF-IDF vectorizer
│   │   └── historico_noticias.csv       # Histórico de feedback
│   └── scripts/
│       ├── build_dataset.py             # Coleta notícias via Google News
│       ├── treinar_qualidade.py         # Treina modelo de qualidade
│       └── treinar_tipo.py              # Treina modelo de tipo
│
├── .gitignore
└── README.md
```

## 🚀 Como Rodar

### Aplicativo desktop (Windows PowerShell)

```bash
# Criar o ambiente e instalar as dependências da interface e do backend
uv venv
uv pip install --python .venv\Scripts\python.exe -r Backend/requirements.txt -r app_pyside6/requirements.txt

# Iniciar o aplicativo (a API local é iniciada automaticamente)
.venv\Scripts\python.exe app_pyside6/main.py
```

O aplicativo reutiliza uma API já ativa; caso contrário, inicia a API local e aguarda os modelos carregarem. Ao fechar o aplicativo, somente a API iniciada por ele é encerrada. Para usar outro backend, defina `MOODNEWS_BACKEND_URL` com a URL completa do endpoint `/chat`.

## 🧠 Modelos de IA

O projeto usa dois classificadores:

| Modelo | Arquivo | Função |
|---|---|---|
| **Qualidade** | `modelo_noticias.pkl` | Classifica notícia como BOA (1) ou RUIM (0) |
| **Tipo** | `modelo_noticias_tipo.pkl` | Classifica categoria (desmatamento, poluição, preservação, etc.) |

### Aprendizado Incremental

O classificador de qualidade usa `TfidfVectorizer` + `SGDClassifier(loss="log_loss")`.
O TF-IDF é ajustado no treino inicial e o classificador é atualizado com `partial_fit`.

```bash
# Treinar modelo de qualidade
python Backend/scripts/treinar_qualidade.py

# Treinar modelo de tipo
python Backend/scripts/treinar_tipo.py

# Coletar mais notícias para o dataset
python Backend/scripts/build_dataset.py

# Classificar via CLI (modo interativo)
python Backend/classifier.py --interactive
```

## 📡 Endpoints da API

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Status dos modelos |
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Classifica uma notícia |
| `POST` | `/feedback` | Atualiza o modelo com correção do usuário |

### Exemplos

**Classificar notícia:**
```json
POST /chat
{ "text": "Desmatamento na Amazônia atinge recorde" }
```

**Enviar feedback:**
```json
POST /feedback
{ "text": "Texto da notícia", "label": 1 }
```

Use `label=0` para notícia ruim e `label=1` para notícia boa.

## ⚙️ Dependências Principais

### Python
- `fastapi` + `uvicorn` — API REST
- `scikit-learn` — Machine Learning
- `pandas` — Manipulação de dados
- `joblib` — Serialização de modelos

### Aplicação desktop
- `PySide6` — Interface gráfica Qt
- `requests` — Cliente HTTP para a API

---

*Projeto educacional — 2026*
