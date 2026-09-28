"""
build_dataset.py — Gerador de dataset de notícias ambientais do Brasil
Coleta notícias via Google News RSS, resolve URLs com googlenewsdecoder,
extrai conteúdo completo, classifica sentimento e salva CSV para treino de IA.
"""

import csv
import os
import re
import time
import random
import urllib.parse
from datetime import datetime
from pathlib import Path

import feedparser
from googlenewsdecoder import new_decoderv1
from newspaper import Article
from tqdm import tqdm

# ── Configurações ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
ARQUIVO_CSV = DATA_DIR / "../desempenho/noticias_terca.csv"
MIN_NOTICIAS = 700
MIN_TEXT_LENGTH = 150
DELAY_BETWEEN_DECODE = 2       # segundos entre decode de URLs (rate limit)
DELAY_BETWEEN_EXTRACT = (0.5, 1.2)  # segundos entre extração de artigos

# ── Termos de busca ambientais (organizados por categoria) ────
SEARCH_QUERIES = {
    "desmatamento": [
        "desmatamento Brasil",
        "desmatamento Amazônia",
        "desmatamento ilegal Brasil",
        "desmatamento Cerrado",
        "desmatamento Mata Atlântica",
        "queimadas Amazônia",
        "queimadas Brasil floresta",
        "incêndios florestais Brasil",
        "madeira ilegal Brasil",
        "grilagem terras Brasil",
        "desflorestamento Brasil 2025",
        "queimadas Pantanal",
    ],
    "poluição": [
        "poluição rios Brasil",
        "poluição ar Brasil cidades",
        "poluição oceano praias Brasil",
        "contaminação água Brasil",
        "lixo plástico mar Brasil",
        "poluição industrial Brasil",
        "emissão gases efeito estufa Brasil",
        "mudanças climáticas Brasil",
        "aquecimento global Brasil",
        "agrotóxico contaminação Brasil",
        "saneamento básico esgoto Brasil",
        "lixão aterro sanitário Brasil",
        "reciclagem resíduos Brasil",
        "carbono emissões Brasil",
        "poluição sonora Brasil",
    ],
    "preservação": [
        "preservação ambiental Brasil",
        "conservação biodiversidade Brasil",
        "unidade conservação parque nacional Brasil",
        "reflorestamento Brasil",
        "restauração florestal Brasil",
        "espécies ameaçadas extinção Brasil",
        "reserva ambiental Brasil",
        "energia renovável Brasil",
        "energia solar eólica Brasil",
        "sustentabilidade ambiental Brasil",
        "bioeconomia Brasil",
        "economia verde Brasil",
        "transição energética Brasil",
        "corredor ecológico Brasil",
        "proteção fauna flora Brasil",
        "biocombustível Brasil",
    ],
    "acidente": [
        "desastre ambiental Brasil",
        "vazamento óleo Brasil",
        "rompimento barragem ambiental Brasil",
        "acidente ambiental Brasil",
        "crime ambiental Brasil",
        "enchente inundação impacto ambiental Brasil",
        "seca estiagem impacto ambiental Brasil",
        "mortandade peixes rio Brasil",
        "contaminação solo mineração Brasil",
    ],
    "geral": [
        "meio ambiente Brasil notícias",
        "política ambiental Brasil governo",
        "legislação ambiental Brasil",
        "Ibama fiscalização operação",
        "ICMBio Brasil conservação",
        "Ministério Meio Ambiente Brasil",
        "COP clima Brasil negociação",
        "Pantanal preservação conservação",
        "Caatinga meio ambiente bioma",
        "bioma brasileiro ameaça",
        "água potável crise Brasil",
        "agronegócio meio ambiente impacto",
        "mineração impacto ambiental",
        "Amazônia proteção floresta",
        "mercado carbono Brasil",
    ],
}

# ── Palavras-chave para classificação de sentimento ───────────
POSITIVE_KEYWORDS = [
    "preservação", "conservação", "proteção", "proteger", "preservar",
    "restauração", "restaurar", "recuperação", "recuperar", "reflorestar",
    "reflorestamento", "replantio", "plantar árvores",
    "avanço", "conquista", "vitória", "sucesso", "progresso",
    "melhoria", "melhora", "melhorar", "evolução", "inovação",
    "redução", "reduziu", "diminuição", "diminuiu", "queda no desmatamento",
    "queda na poluição", "queda nas emissões", "menor taxa",
    "menor índice", "recuo",
    "energia renovável", "energia limpa", "energia solar", "energia eólica",
    "sustentável", "sustentabilidade", "bioeconomia", "biocombustível",
    "reciclagem", "reutilização", "economia circular", "economia verde",
    "fiscalização", "autuação", "multa ambiental", "apreensão",
    "combate ao desmatamento", "operação contra",
    "acordo ambiental", "compromisso", "meta climática",
    "plano nacional", "programa de proteção", "criação de reserva",
    "nova unidade de conservação", "área protegida",
    "nova espécie descoberta", "população cresceu", "reintrodução",
]

NEGATIVE_KEYWORDS = [
    "desmatamento", "desmatou", "devastação", "destruição", "destruir",
    "queimada", "incêndio florestal", "fogo na floresta",
    "derrubada", "supressão vegetal", "perda florestal",
    "poluição", "poluir", "contaminação", "contaminar", "contaminado",
    "vazamento", "derramamento", "tóxico", "veneno", "agrotóxico",
    "esgoto", "lixão", "aterro irregular",
    "desastre", "tragédia", "catástrofe", "acidente ambiental",
    "rompimento", "enchente", "inundação", "deslizamento", "erosão",
    "seca severa", "estiagem", "colapso",
    "extinção", "ameaçada", "risco de extinção", "em perigo",
    "mortandade", "morte de animais", "morte de peixes",
    "crime ambiental", "ilegal", "irregular", "clandestino",
    "grilagem", "invasão", "garimpo ilegal", "madeira ilegal",
    "tráfico de animais", "caça ilegal", "pesca ilegal",
    "aquecimento global", "efeito estufa", "emissões aumentaram",
    "recorde de calor", "temperatura recorde", "nível do mar subiu",
    "derretimento", "onda de calor",
    "retrocesso", "piora", "aumento do desmatamento",
    "recorde negativo", "pior índice",
]


def classify_sentiment(title: str, text: str) -> int:
    """
    Classifica sentimento da notícia:
      1 = boa (positiva), 0 = neutra, -1 = ruim (negativa)
    """
    combined = f"{title} {text}".lower()

    pos_score = sum(combined.count(kw.lower()) for kw in POSITIVE_KEYWORDS)
    neg_score = sum(combined.count(kw.lower()) for kw in NEGATIVE_KEYWORDS)

    if pos_score > 0 and neg_score > 0:
        ratio = max(pos_score, neg_score) / max(min(pos_score, neg_score), 1)
        if ratio < 1.5:
            return 0

    if pos_score > neg_score and pos_score >= 2:
        return 1
    elif neg_score > pos_score and neg_score >= 2:
        return -1
    elif pos_score > neg_score:
        return 1
    elif neg_score > pos_score: 
        return -1
    return 0


def resolve_gnews_url(google_url: str) -> str | None:
    """Resolve URL do Google News para a URL real do artigo."""
    try:
        result = new_decoderv1(google_url, interval=DELAY_BETWEEN_DECODE)
        if result.get("status"):
            return result["decoded_url"]
    except Exception:
        pass
    return None


def extract_article(url: str) -> dict | None:
    """Extrai título e texto completo de um artigo."""
    try:
        article = Article(url, language="pt")
        article.download()
        article.parse()

        title = (article.title or "").strip()
        text = (article.text or "").strip()
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        if len(text) < MIN_TEXT_LENGTH:
            return None

        return {"title": title, "text": text, "url": url}
    except Exception:
        return None


def load_existing_data(filepath: str) -> list[dict]:
    """Carrega dados existentes do CSV."""
    data = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    return data


def save_csv(data: list[dict], filepath: str):
    """Salva dados no CSV."""
    fieldnames = ["typetarget", "fellingtarget", "title", "text", "url"]
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def main():
    print("=" * 60)
    print("🌿 Gerador de Dataset - Notícias Ambientais do Brasil")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carregar dados existentes
    existing_data = load_existing_data(str(ARQUIVO_CSV))
    existing_urls = {row["url"].strip() for row in existing_data if row.get("url")}
    print(f"\n📂 Notícias existentes no CSV: {len(existing_data)}")

    # 2. Coletar URLs do Google News RSS
    print("\n🔍 Buscando notícias no Google News RSS...")
    all_candidates = []

    for category, queries in SEARCH_QUERIES.items():
        for query in tqdm(queries, desc=f"  📡 [{category}]", leave=False):
            encoded = urllib.parse.quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
            try:
                feed = feedparser.parse(rss_url)
                for entry in feed.entries:
                    link = entry.get("link", "")
                    title = entry.get("title", "")
                    if link:
                        all_candidates.append({
                            "google_url": link,
                            "rss_title": title,
                            "category": category,
                        })
            except Exception:
                pass
            time.sleep(random.uniform(0.3, 0.8))

    # Remover duplicatas por Google URL
    seen = set()
    unique = []
    for c in all_candidates:
        if c["google_url"] not in seen:
            seen.add(c["google_url"])
            unique.append(c)

    print(f"\n📊 Candidatas únicas encontradas: {len(unique)}")
    needed = max(0, MIN_NOTICIAS - len(existing_data))
    print(f"📊 Notícias necessárias: {needed}")

    if needed == 0:
        print("✅ Dataset já possui notícias suficientes!")
        return

    random.shuffle(unique)

    # 3. Resolver URLs + Extrair artigos
    print(f"\n📰 Resolvendo URLs e extraindo artigos...")
    new_data = []
    errors = 0
    decode_fails = 0

    pbar = tqdm(total=needed, desc="  📄 Coletando")

    for candidate in unique:
        if len(new_data) >= needed:
            break

        # Resolver URL do Google News
        real_url = resolve_gnews_url(candidate["google_url"])
        if not real_url:
            decode_fails += 1
            continue

        # Pular se já existe
        if real_url in existing_urls:
            continue

        # Extrair artigo
        article = extract_article(real_url)
        if article is None:
            errors += 1
            continue

        # Classificar sentimento
        sentiment = classify_sentiment(article["title"], article["text"])

        # Truncar texto longo
        text = article["text"]
        if len(text) > 3000:
            text = text[:3000] + "..."

        new_data.append({
            "typetarget": candidate["category"],
            "fellingtarget": sentiment,
            "title": article["title"],
            "text": text,
            "url": real_url,
        })
        existing_urls.add(real_url)
        pbar.update(1)

        time.sleep(random.uniform(*DELAY_BETWEEN_EXTRACT))

    pbar.close()

    print(f"\n✅ Notícias extraídas com sucesso: {len(new_data)}")
    print(f"🔗 Falhas na resolução de URL: {decode_fails}")
    print(f"❌ Falhas na extração: {errors}")

    # 4. Combinar + Salvar
    all_data = existing_data + new_data
    print(f"\n📊 Total final do dataset: {len(all_data)}")

    # Estatísticas
    sentiments = {"1": 0, "0": 0, "-1": 0}
    categories = {}
    for row in all_data:
        s = str(row.get("fellingtarget", "0"))
        sentiments[s] = sentiments.get(s, 0) + 1
        c = row.get("typetarget", "geral")
        categories[c] = categories.get(c, 0) + 1

    print("\n📈 Distribuição de sentimento:")
    print(f"   👍 Positivas (1):  {sentiments.get('1', 0)}")
    print(f"   😐 Neutras (0):   {sentiments.get('0', 0)}")
    print(f"   👎 Negativas (-1): {sentiments.get('-1', 0)}")

    print("\n📂 Distribuição por categoria:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"   {cat}: {count}")

    save_csv(all_data, str(ARQUIVO_CSV))
    print(f"\n💾 Dataset salvo em: {ARQUIVO_CSV}")

    if len(all_data) >= MIN_NOTICIAS:
        print(f"\n🎉 Meta de {MIN_NOTICIAS} notícias atingida!")
    else:
        remaining = MIN_NOTICIAS - len(all_data)
        print(f"\n⚠️  Coletadas {len(all_data)} notícias. Faltam {remaining}.")
        print("   Execute novamente para tentar coletar mais.")


if __name__ == "__main__":
    main()
