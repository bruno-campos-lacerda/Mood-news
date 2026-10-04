"""Classificador incremental de noticias usando TF-IDF + SGDClassifier.

O TfidfVectorizer cria o vocabulário inicial, enquanto o SGDClassifier suporta
atualizações contínuas (online learning) usando partial_fit.
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "data" / "noticias.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "modelo_noticias.pkl"
DEFAULT_VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer_noticias.pkl"
DEFAULT_HISTORY_PATH = BASE_DIR / "models" / "historico_noticias.csv"

CLASSES = np.array([0, 1])
LABEL_TEXT = {0: "RUIM", 1: "BOA"}
HISTORY_COLUMNS = ["text", "label", "source", "created_at"]


def normalizar_rotulo(rotulo) -> int:
    """Converte rótulos textuais ou numéricos para o padrão binário: 0=ruim, 1=boa."""
    if isinstance(rotulo, str):
        valor = rotulo.strip().lower()
        if valor in ["0", "0.0", "-1", "-1.0", "ruim", "regular", "negativa", "negativo"]:
            return 0
        if valor in ["1", "1.0", "boa", "bom", "positiva", "positivo"]:
            return 1

    try:
        return 1 if float(rotulo) > 0 else 0
    except (TypeError, ValueError):
        raise ValueError("Rótulo inválido. Use 0=ruim ou 1=boa.")


def carregar_dados_dataset(dataset_path: Path = DEFAULT_DATASET, coluna_texto="text", coluna_rotulo="fellingtarget") -> pd.DataFrame:
    """Carrega o dataset e converte os rótulos para o formato padrão."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=[coluna_texto, coluna_rotulo])
    
    # Extrai as colunas importantes e limpa o texto
    dados = pd.DataFrame()
    dados["text"] = df[coluna_texto].astype(str).str.strip()
    
    # Ignora erros na aplicação do normalizar_rotulo, convertendo para NaN e depois removendo
    def tentar_normalizar(r):
        try:
            return normalizar_rotulo(r)
        except ValueError:
            return np.nan
            
    dados["label"] = df[coluna_rotulo].apply(tentar_normalizar)
    dados = dados.dropna(subset=["label"])
    dados["label"] = dados["label"].astype(int)
    
    # Remove textos vazios
    dados = dados[dados["text"] != ""]
    return dados


def salvar_modelo(modelo, vectorizer, historico=None, modelo_path=DEFAULT_MODEL_PATH, vectorizer_path=DEFAULT_VECTORIZER_PATH, historico_path=DEFAULT_HISTORY_PATH):
    """Salva o classificador, vectorizer e o histórico em arquivos."""
    modelo_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(modelo, modelo_path)
    joblib.dump(vectorizer, vectorizer_path)

    if historico is not None:
        historico.to_csv(historico_path, index=False)


def carregar_modelo(modelo_path=DEFAULT_MODEL_PATH, vectorizer_path=DEFAULT_VECTORIZER_PATH, historico_path=DEFAULT_HISTORY_PATH):
    """Carrega o classificador, vectorizer e o histórico, se existirem."""
    if not modelo_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError("Modelo ou vectorizer não encontrados.")

    modelo = joblib.load(modelo_path)
    vectorizer = joblib.load(vectorizer_path)

    if historico_path.exists():
        historico = pd.read_csv(historico_path)
    else:
        historico = pd.DataFrame(columns=HISTORY_COLUMNS)

    return modelo, vectorizer, historico


def treinar_modelo_inicial(dataset_path=DEFAULT_DATASET, salvar=True):
    """Treina o modelo pela primeira vez usando o dataset base."""
    # 1. Carrega e prepara os dados
    print("Carregando dataset...")
    dados = carregar_dados_dataset(dataset_path)
    
    X = dados["text"]
    y = dados["label"].to_numpy()

    # Divisão entre treino e teste (estratificada se houver exemplos suficientes)
    if len(dados) >= 2 and dados["label"].nunique() >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = X, pd.Series(dtype=str), y, np.array([])

    # 2. Configura e treina o Vectorizer (converte texto em números)
    print("Criando vocabulário (TfidfVectorizer)...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), lowercase=True, strip_accents="unicode")
    X_train_vec = vectorizer.fit_transform(X_train)

    # 3. Configura o Modelo SGDClassifier
    modelo = SGDClassifier(
        loss="log_loss", # Regressão logística (permite calcular probabilidades)
        penalty="l2",
        alpha=1e-5,
        learning_rate="optimal",
        max_iter=1,      # Usamos 1 pois faremos o loop de épocas manualmente com partial_fit
        random_state=42
    )

    # 4. Treinamento em lotes (Mini-batch Training)
    print("Treinando o modelo SGDClassifier em lotes...")
    epocas = 5
    batch_size = 128
    
    # Calcula pesos balanceados manualmente para dar a mesma importância a classes minoritárias
    pesos_por_classe = {classe: len(y_train) / (2 * np.sum(y_train == classe)) if np.sum(y_train == classe) > 0 else 1.0 for classe in CLASSES}
    pesos = np.array([pesos_por_classe[rotulo] for rotulo in y_train])

    primeira_vez = True
    for epoca in range(epocas):
        # Embaralha os dados em cada época
        indices = np.random.permutation(len(y_train))
        for inicio in range(0, len(indices), batch_size):
            lote = indices[inicio:inicio + batch_size]
            
            kwargs = {"sample_weight": pesos[lote]}
            if primeira_vez:
                kwargs["classes"] = CLASSES
                primeira_vez = False
                
            modelo.partial_fit(X_train_vec[lote], y_train[lote], **kwargs)

    # 5. Avaliação do modelo
    metricas = {
        "total_treino": len(X_train),
        "total_teste": len(X_test),
        "distribuicao": dados["label"].value_counts().to_dict(),
    }

    if len(X_test) > 0:
        X_test_vec = vectorizer.transform(X_test)
        y_pred = modelo.predict(X_test_vec)
        metricas["acuracia"] = accuracy_score(y_test, y_pred)
        metricas["matriz_confusao"] = confusion_matrix(y_test, y_pred, labels=CLASSES).tolist()
        metricas["relatorio"] = classification_report(y_test, y_pred, labels=CLASSES, target_names=["RUIM", "BOA"], zero_division=0)
        
        print(f"Treino concluído. Acurácia: {metricas['acuracia']:.4f}")
        print("Relatório:\n", metricas["relatorio"])

    # 6. Salva o histórico inicial
    historico = pd.DataFrame({
        "text": dados["text"],
        "label": dados["label"],
        "source": "initial_dataset",
        "created_at": datetime.now(timezone.utc).isoformat()
    })

    if salvar:
        salvar_modelo(modelo, vectorizer, historico)

    return {"modelo": modelo, "vectorizer": vectorizer, "historico": historico, "metricas": metricas}


def classificar_noticia(texto: str, modelo=None, vectorizer=None):
    """Classifica uma notícia retornando se é Boa ou Ruim e a confiança."""
    if not texto.strip():
        raise ValueError("O texto da notícia não pode estar vazio.")

    if modelo is None or vectorizer is None:
        modelo, vectorizer, _ = carregar_modelo()

    X_vec = vectorizer.transform([texto])
    rotulo = int(modelo.predict(X_vec)[0])

    probabilidades_por_classe = {0: 0.0, 1: 0.0}
    if hasattr(modelo, "predict_proba"):
        probas = modelo.predict_proba(X_vec)[0]
        for classe, probabilidade in zip(modelo.classes_, probas):
            classe = int(classe)
            if classe in probabilidades_por_classe:
                probabilidades_por_classe[classe] = float(probabilidade)
    else:
        probabilidades_por_classe[rotulo] = 1.0

    return {
        "rotulo": rotulo,
        "classe": LABEL_TEXT[rotulo],
        "confianca": probabilidades_por_classe[rotulo],
        "probabilidades": {
            "ruim": probabilidades_por_classe[0],
            "boa": probabilidades_por_classe[1],
        },
    }


def atualizar_modelo(texto: str, rotulo_correto: int, modelo=None, vectorizer=None, historico=None, salvar=True):
    """Atualiza o modelo em tempo real com um novo exemplo (feedback)."""
    if modelo is None or vectorizer is None or historico is None:
        modelo, vectorizer, historico = carregar_modelo()

    rotulo = normalizar_rotulo(rotulo_correto)
    X_novo = vectorizer.transform([texto])
    
    # partial_fit ensina o novo exemplo ao modelo online
    kwargs = {} if hasattr(modelo, "classes_") else {"classes": CLASSES}
    modelo.partial_fit(X_novo, np.array([rotulo]), **kwargs)

    # Adiciona a nova notícia ao histórico
    novo_registro = pd.DataFrame([{
        "text": texto,
        "label": rotulo,
        "source": "feedback",
        "created_at": datetime.now(timezone.utc).isoformat()
    }])
    historico = pd.concat([historico, novo_registro], ignore_index=True)

    # Replay: a cada 10 feedbacks, re-treina rapidamente com uma amostra do histórico para não esquecer o passado
    total_feedback = len(historico[historico["source"] == "feedback"])
    if total_feedback > 0 and total_feedback % 10 == 0:
        amostra = historico.sample(min(len(historico), 1000), random_state=42)
        X_hist = vectorizer.transform(amostra["text"])
        y_hist = amostra["label"].to_numpy()
        
        pesos_hist = [len(y_hist) / (2 * np.sum(y_hist == c)) if np.sum(y_hist == c) > 0 else 1.0 for c in y_hist]
        modelo.partial_fit(X_hist, y_hist, sample_weight=pesos_hist)

    if salvar:
        salvar_modelo(modelo, vectorizer, historico)

    return {
        "modelo": modelo,
        "vectorizer": vectorizer,
        "historico": historico,
        "rotulo_aprendido": rotulo,
        "classe_aprendida": LABEL_TEXT[rotulo],
        "total_historico": len(historico)
    }


def loop_interativo():
    """Fluxo para classificar e ensinar o modelo manualmente no terminal."""
    try:
        modelo, vectorizer, historico = carregar_modelo()
    except FileNotFoundError:
        print("Modelo não encontrado. Iniciando treinamento inicial...")
        resultado = treinar_modelo_inicial()
        modelo, vectorizer, historico = resultado["modelo"], resultado["vectorizer"], resultado["historico"]

    print("\n--- MODO INTERATIVO ---")
    print("Digite uma notícia para classificar. Deixe em branco para sair.")
    
    while True:
        texto = input("\nNotícia: ").strip()
        if not texto:
            break

        predicao = classificar_noticia(texto, modelo, vectorizer)
        print(f"-> Predição: {predicao['classe']} (Confiança: {predicao['confianca']:.2%})")

        correcao = input("Rótulo correto (0=Ruim, 1=Boa, Enter para pular): ").strip()
        if correcao:
            try:
                resultado = atualizar_modelo(texto, int(correcao), modelo, vectorizer, historico)
                modelo = resultado["modelo"]
                historico = resultado["historico"]
                print(f"-> Modelo atualizado! Histórico agora tem {resultado['total_historico']} exemplos.")
            except ValueError as e:
                print(f"Erro ao processar o rótulo: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Classificador de Notícias - Didático")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Caminho para o CSV de treino")
    parser.add_argument("--train", action="store_true", help="Força o treinamento inicial")
    parser.add_argument("--interactive", action="store_true", help="Abre o modo interativo")
    parser.add_argument("--text", type=str, help="Classifica um texto via linha de comando")
    parser.add_argument("--label", type=int, choices=[0, 1], help="Rótulo correto para o --text")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if args.train:
        treinar_modelo_inicial(dataset_path=args.dataset)

    if args.text:
        modelo, vectorizer, historico = carregar_modelo()
        predicao = classificar_noticia(args.text, modelo, vectorizer)
        print(f"Predição: {predicao['classe']} (Confiança: {predicao['confianca']:.2%})")
        
        if args.label is not None:
            atualizar_modelo(args.text, args.label, modelo, vectorizer, historico)
            print("Modelo atualizado com o novo rótulo.")

    if args.interactive or (not args.train and not args.text):
        loop_interativo()


if __name__ == "__main__":
    main()
