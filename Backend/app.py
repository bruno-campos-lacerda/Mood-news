from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import threading
import unicodedata
from pathlib import Path

from classifier import atualizar_modelo, carregar_modelo, classificar_noticia

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PATH_TIPO = BACKEND_DIR / "models" / "modelo_noticias_tipo.pkl"
MODEL_PATH_QUALIDADE = BACKEND_DIR / "models" / "modelo_noticias.pkl"
VECTORIZER_PATH_QUALIDADE = BACKEND_DIR / "models" / "vectorizer_noticias.pkl"
HISTORICO_PATH_QUALIDADE = BACKEND_DIR / "models" / "historico_noticias.csv"

modelo_tipo = None
modelo_qualidade = None
vectorizer_qualidade = None
historico_qualidade = None
modelos_erro = None
modelos_carregando = False
modelos_lock = threading.Lock()

# Mapear tipos de notícia para categorias desejadas
TIPO_MAPA = {
    "desmatamento": "Desmatamento e Queimadas",
    "poluição": "Poluição e Degradação",
    "preservação": "Preservação Ambiental",
    "acidente": "Desastres Ambientais",
    "geral": "Temas Gerais de Meio Ambiente",
    "polui": "Poluição e Degradação",
    "preserva": "Preservação Ambiental",
}

# Mapear qualidade numérica para texto
QUALIDADE_MAPA = {
    "1": "BOA",
    "0": "RUIM",
    "-1": "RUIM",
    "1.0": "BOA",
    "0.0": "RUIM",
    "-1.0": "RUIM",
}


class MessageInput(BaseModel):
    text: str


class FeedbackInput(BaseModel):
    text: str
    label: int


def normalizar_texto(texto):
    texto = str(texto).lower().strip()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto


def _carregar_modelos_em_background():
    global modelo_tipo, modelo_qualidade, vectorizer_qualidade, historico_qualidade
    global modelos_erro, modelos_carregando

    try:
        print(f"Tentando carregar modelos...")
        print(f"Modelo TIPO existe: {MODEL_PATH_TIPO.exists()}")
        print(f"Modelo QUALIDADE existe: {MODEL_PATH_QUALIDADE.exists()}")
        print(f"Vectorizer QUALIDADE existe: {VECTORIZER_PATH_QUALIDADE.exists()}")
        print(f"Histórico QUALIDADE existe: {HISTORICO_PATH_QUALIDADE.exists()}")

        if not MODEL_PATH_TIPO.exists():
            raise FileNotFoundError(f"Modelo TIPO não encontrado em {MODEL_PATH_TIPO}")
        if not MODEL_PATH_QUALIDADE.exists():
            raise FileNotFoundError(f"Modelo QUALIDADE não encontrado em {MODEL_PATH_QUALIDADE}")
        if not VECTORIZER_PATH_QUALIDADE.exists():
            raise FileNotFoundError(f"Vectorizer QUALIDADE não encontrado em {VECTORIZER_PATH_QUALIDADE}")

        # Carregar modelo de TIPO
        modelo_tipo_carregado = joblib.load(MODEL_PATH_TIPO)
        # Carregar classificador, vectorizer e histórico de QUALIDADE
        (
            modelo_qualidade_carregado,
            vectorizer_qualidade_carregado,
            historico_qualidade_carregado,
        ) = carregar_modelo(
            modelo_path=MODEL_PATH_QUALIDADE,
            vectorizer_path=VECTORIZER_PATH_QUALIDADE,
            historico_path=HISTORICO_PATH_QUALIDADE,
        )

        with modelos_lock:
            modelo_tipo = modelo_tipo_carregado
            modelo_qualidade = modelo_qualidade_carregado
            vectorizer_qualidade = vectorizer_qualidade_carregado
            historico_qualidade = historico_qualidade_carregado
            modelos_erro = None

        print(f"Modelo TIPO carregado: {type(modelo_tipo_carregado).__name__}")
        print(f"Modelo QUALIDADE carregado: {type(modelo_qualidade_carregado).__name__}")
        print(f"Histórico QUALIDADE: {len(historico_qualidade_carregado)} exemplos")
        
        # Contar notícias boas e ruins
        if len(historico_qualidade_carregado) > 0:
            noticias_boas = (historico_qualidade_carregado['label'] == 1).sum()
            noticias_ruins = (historico_qualidade_carregado['label'] == 0).sum()
            print(f"\n📊 Estatísticas do Histórico:")
            print(f"   ✅ Notícias BOA: {noticias_boas}")
            print(f"   ❌ Notícias RUIM: {noticias_ruins}")
            print(f"   📈 Total: {noticias_boas + noticias_ruins}\n")
    except Exception as e:
        with modelos_lock:
            modelo_tipo = None
            modelo_qualidade = None
            vectorizer_qualidade = None
            historico_qualidade = None
            modelos_erro = str(e)
        print(f"Erro ao carregar modelos: {e}")
        import traceback
        traceback.print_exc()
    finally:
        with modelos_lock:
            modelos_carregando = False


def iniciar_carregamento_modelos():
    global modelos_carregando, modelos_erro

    with modelos_lock:
        modelos_prontos = (
            modelo_tipo is not None
            and modelo_qualidade is not None
            and vectorizer_qualidade is not None
            and historico_qualidade is not None
        )
        if modelos_prontos or modelos_carregando:
            return

        modelos_carregando = True
        modelos_erro = None

    threading.Thread(target=_carregar_modelos_em_background, daemon=True).start()


def obter_modelos():
    with modelos_lock:
        modelos_prontos = (
            modelo_tipo is not None
            and modelo_qualidade is not None
            and vectorizer_qualidade is not None
            and historico_qualidade is not None
        )
        if modelos_prontos:
            return modelo_tipo, modelo_qualidade, vectorizer_qualidade, historico_qualidade

        erro = modelos_erro
        carregando = modelos_carregando

    if erro:
        raise HTTPException(status_code=500, detail=f"Modelos indisponíveis: {erro}")

    if carregando:
        raise HTTPException(
            status_code=503,
            detail="O backend ainda está carregando os modelos. Tente novamente em alguns segundos.",
        )

    iniciar_carregamento_modelos()
    raise HTTPException(
        status_code=503,
        detail="O backend iniciou o carregamento dos modelos. Tente novamente em alguns segundos.",
    )


@app.on_event("startup")
def startup_event():
    iniciar_carregamento_modelos()


@app.get("/")
def read_root():
    with modelos_lock:
        models_loaded = (
            modelo_tipo is not None
            and modelo_qualidade is not None
            and vectorizer_qualidade is not None
            and historico_qualidade is not None
        )
        loading = modelos_carregando
        error = modelos_erro

    status = "ready" if models_loaded else "loading" if loading else "error" if error else "idle"
    return {
        "status": status,
        "models_loaded": models_loaded,
        "loading": loading,
        "error": error,
    }


@app.get("/health")
def health():
    return read_root()


@app.post("/chat")
def chat(message: MessageInput):
    (
        modelo_tipo_atual,
        modelo_qualidade_atual,
        vectorizer_qualidade_atual,
        _,
    ) = obter_modelos()

    texto = message.text.strip()
    if not texto:
        return {"response": "Envie um texto válido."}

    try:
        # normalizar texto
        texto_normalizado = normalizar_texto(texto)

        # Predição de TIPO de notícia
        tipo_pred = modelo_tipo_atual.predict([texto_normalizado])[0]
        tipo_probas = modelo_tipo_atual.predict_proba([texto_normalizado])[0]
        tipo_classes = modelo_tipo_atual.classes_
        
        tipo_max_prob = max(tipo_probas)
        tipo_max_idx = list(tipo_probas).index(tipo_max_prob)
        tipo_label = tipo_classes[tipo_max_idx]
        
        # Mapear tipo para categoria desejada
        tipo_str = str(tipo_label).lower()
        tipo_final = "Temas Gerais de Meio Ambiente"
        for k, v in TIPO_MAPA.items():
            if k in tipo_str:
                tipo_final = v
                break

        # Predição de QUALIDADE com o modelo incremental.
        qualidade_predicao = classificar_noticia(
            texto_normalizado,
            modelo_qualidade_atual,
            vectorizer_qualidade_atual,
        )
        qualidade_label = str(qualidade_predicao["rotulo"])
        qualidade_max_prob = qualidade_predicao["confianca"]
        qualidade_final = QUALIDADE_MAPA.get(qualidade_label, qualidade_predicao["classe"])
        
        # Montar resposta amigável e intuitiva para o usuário
        if qualidade_final == "BOA":
            resposta = f"Que ótima notícia! 🌱\n\nEssa notícia tem um impacto positivo no meio ambiente e está relacionada a {tipo_final}.\n\nConfiança da IA: {int(qualidade_max_prob*100)}%"
        else:
            resposta = f"Infelizmente, esta notícia traz um impacto negativo. ⚠️\n\nEla aborda problemas de {tipo_final}, o que é preocupante para a nossa natureza.\n\nConfiança da IA: {int(qualidade_max_prob*100)}%"
        
        return {
            "tipo": tipo_final,
            "qualidade": qualidade_final,
            "rotulo": int(qualidade_predicao["rotulo"]),
            "probabilidades": qualidade_predicao.get("probabilidades", []),
            "confidence": float(qualidade_max_prob),
            "response": resposta,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


@app.post("/feedback")
def feedback(message: FeedbackInput):
    """Recebe a correcao do usuario e atualiza o modelo com partial_fit."""
    global modelo_qualidade, vectorizer_qualidade, historico_qualidade

    texto = message.text.strip()
    if not texto:
        return {"response": "Envie um texto válido."}
    if message.label not in (0, 1):
        raise HTTPException(status_code=400, detail="Rótulo inválido. Use 0=ruim ou 1=boa.")

    # Garante que os artefatos foram carregados antes de tentar atualizar.
    obter_modelos()

    try:
        texto_normalizado = normalizar_texto(texto)
        with modelos_lock:
            resultado = atualizar_modelo(
                texto_normalizado,
                message.label,
                modelo_qualidade,
                vectorizer_qualidade,
                historico_qualidade,
            )
            modelo_qualidade = resultado["modelo"]
            vectorizer_qualidade = resultado["vectorizer"]
            historico_qualidade = resultado["historico"]

        return {
            "status": "updated",
            "rotulo": resultado["rotulo_aprendido"],
            "classe": resultado["classe_aprendida"],
            "total_historico": resultado["total_historico"],
            "total_feedback": resultado.get("total_feedback", 0),
            "replay_executado": resultado.get("replay_executado", False),
            "response": "Modelo atualizado com aprendizado incremental.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao atualizar modelo: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Bind to 0.0.0.0 so the server is reachable from other machines on the network
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
