"""
消化器分野 最新知見ダイジェスト Bot
====================================
- PubMed から消化器分野の直近の注目論文を取得
- Gemini API で臨床的に興味深い知見を選定し日本語で紹介
- Discord Webhook で1日4回投稿
- 重複排除のためPMIDを記録
"""

import os
import json
import random
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import requests
import google.generativeai as genai

# ============================================================
# 設定
# ============================================================
GI_DIGEST_WEBHOOK_URL = os.environ["GI_DIGEST_WEBHOOK_URL"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.0-flash"

# 通知済みPMID記録
NOTIFIED_FILE = Path(__file__).parent / "notified_digest_pmids.json"

# 直近何日分から選ぶか（広めに取って良い論文を選定）
SEARCH_DAYS = 7

# 1回の投稿で紹介する論文数
ARTICLES_PER_POST = 1

# PubMed検索の最大取得件数（候補プール）
MAX_RESULTS = 50

# ============================================================
# 検索クエリ一覧（消化器分野を幅広くカバー）
# ランダムに1つ選んで検索し、多様なトピックを紹介する
# ============================================================
SEARCH_QUERIES = [
    # 総合 - 消化器病学
    '"Gastroenterology"[MeSH] AND "humans"[MeSH]',
    # 炎症性腸疾患
    '"Inflammatory Bowel Diseases"[MeSH] AND "humans"[MeSH]',
    # 肝臓
    '"Liver Diseases"[MeSH] AND "humans"[MeSH]',
    # 膵臓
    '"Pancreatic Diseases"[MeSH] AND "humans"[MeSH]',
    # 消化管腫瘍
    '"Gastrointestinal Neoplasms"[MeSH] AND "humans"[MeSH]',
    # 食道・胃
    '"Esophageal Diseases"[MeSH] OR "Stomach Diseases"[MeSH]',
    # 腸内細菌叢
    '"Gastrointestinal Microbiome"[MeSH] AND "humans"[MeSH]',
    # 内視鏡
    '"Endoscopy, Gastrointestinal"[MeSH] AND "humans"[MeSH]',
    # 肝細胞癌
    '"Carcinoma, Hepatocellular"[MeSH] AND "humans"[MeSH]',
    # NAFLD/MASLD
    '"Non-alcoholic Fatty Liver Disease"[MeSH] AND "humans"[MeSH]',
    # 胆道疾患
    '"Biliary Tract Diseases"[MeSH] AND "humans"[MeSH]',
    # 機能性消化管障害
    '"Irritable Bowel Syndrome"[MeSH] OR "Functional Gastrointestinal Disorders"[MeSH]',
]

# ============================================================
# PubMed E-utilities
# ============================================================
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed(query: str, reldate: int) -> list[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": MAX_RESULTS,
        "datetype": "edat",
        "reldate": reldate,
        "retmode": "json",
        "sort": "relevance",  # 関連度順（引用数等を考慮）
    }
    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_articles(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    resp = requests.get(EFETCH_URL, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    articles = []

    for article_elem in root.findall(".//PubmedArticle"):
        pmid = _text(article_elem, ".//PMID")
        title = _full_text(article_elem, ".//ArticleTitle")

        # Abstract
        abstract_parts = []
        for at in article_elem.findall(".//AbstractText"):
            label = at.get("Label", "")
            text = "".join(at.itertext()).strip()
            if label:
                abstract_parts.append(f"[{label}] {text}")
            else:
                abstract_parts.append(text)
        abstract = "\n".join(abstract_parts)

        if not abstract:
            abstract_node = article_elem.find(".//Abstract")
            if abstract_node is not None:
                abstract = "".join(abstract_node.itertext()).strip()

        if not abstract:
            continue

        journal = _full_text(article_elem, ".//Journal/Title")

        authors = []
        for author in article_elem.findall(".//Author")[:3]:
            last = _text(author, "LastName")
            fore = _text(author, "ForeName")
            if last:
                authors.append(f"{last} {fore}".strip())
        if len(article_elem.findall(".//Author")) > 3:
            authors.append("et al.")

        doi = ""
        for aid in article_elem.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text or ""

        # 出版年
        pub_year = _text(article_elem, ".//PubDate/Year")

        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "authors": ", ".join(authors),
            "doi": doi,
            "year": pub_year,
        })

    return articles


def _text(elem, path: str) -> str:
    node = elem.find(path)
    if node is not None and node.text:
        return node.text.strip()
    return ""


def _full_text(elem, path: str) -> str:
    node = elem.find(path)
    if node is not None:
        return "".join(node.itertext()).strip()
    return ""


# ============================================================
# Gemini API で知見紹介文を生成
# ============================================================
def generate_digest(article: dict) -> dict:
    """
    論文から臨床的に面白い知見を抽出し、
    読みやすい日本語の紹介文を生成する
    """
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""あなたは消化器内科の専門医向けに最新の医学知見を紹介する医学ライターです。
以下の論文のAbstractを読み、臨床的に興味深いポイントを1つ選んで紹介してください。

## 出力フォーマット（厳守）
以下の3つのセクションを出力してください。それ以外は出力しないでください。

HEADLINE: （知見の核心を1行で。例: 「JAK阻害薬の逐次投与、UC患者で有効性を維持」）

BODY: （3〜5文で。なぜ重要なのか、従来の知見と何が違うのか、臨床へのインパクトは何かを含める。専門医が読んで「へぇ」と思うような切り口で。）

CATEGORY: （以下から1つ選択: IBD / 肝臓 / 膵胆道 / 消化管腫瘍 / 上部消化管 / 下部消化管 / 腸内細菌叢 / 内視鏡 / 機能性疾患 / その他）

## 論文情報
タイトル: {article['title']}
ジャーナル: {article['journal']}
著者: {article['authors']}

Abstract:
{article['abstract']}
"""

    response = model.generate_content(prompt)
    text = response.text

    # レスポンスをパース
    headline = ""
    body = ""
    category = "その他"

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("HEADLINE:"):
            headline = line.replace("HEADLINE:", "").strip()
        elif line.startswith("BODY:"):
            body = line.replace("BODY:", "").strip()
        elif line.startswith("CATEGORY:"):
            category = line.replace("CATEGORY:", "").strip()

    # BODY が1行で取れなかった場合（複数行にまたがる場合）
    if not body:
        lines = text.split("\n")
        capture = False
        body_lines = []
        for line in lines:
            if line.strip().startswith("BODY:"):
                body_lines.append(line.strip().replace("BODY:", "").strip())
                capture = True
            elif line.strip().startswith("CATEGORY:"):
                capture = False
            elif capture:
                body_lines.append(line.strip())
        body = " ".join(body_lines).strip()

    return {
        "headline": headline or article["title"],
        "body": body or "（要約生成に失敗しました）",
        "category": category,
    }


# ============================================================
# カテゴリごとの絵文字マッピング
# ============================================================
CATEGORY_EMOJI = {
    "IBD": "🔥",
    "肝臓": "🫁",
    "膵胆道": "💛",
    "消化管腫瘍": "🎗️",
    "上部消化管": "🔴",
    "下部消化管": "🔵",
    "腸内細菌叢": "🦠",
    "内視鏡": "🔬",
    "機能性疾患": "🧠",
    "その他": "📋",
}


# ============================================================
# Discord 通知
# ============================================================
def send_discord_digest(article: dict, digest: dict):
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/"
    doi_url = f"https://doi.org/{article['doi']}" if article["doi"] else ""

    emoji = CATEGORY_EMOJI.get(digest["category"], "📋")

    links = f"[PubMed]({pubmed_url})"
    if doi_url:
        links += f"  |  [Full Text]({doi_url})"

    embed = {
        "title": f"{emoji} {digest['headline']}"[:256],
        "url": pubmed_url,
        "color": 0x2ECC71,  # 緑色（ダイジェストは緑で区別）
        "description": digest["body"][:2048],
        "fields": [
            {
                "name": "📄 原著論文",
                "value": f"**{article['title'][:200]}**\n"
                         f"_{article['journal']}_  |  {article['authors']}",
                "inline": False,
            },
            {
                "name": "🔗 リンク",
                "value": links,
                "inline": False,
            },
        ],
        "footer": {
            "text": f"{digest['category']}  |  PMID: {article['pmid']}",
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

    payload = {
        "username": "GI Digest Bot",
        "avatar_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/US-NLM-PubMed-Logo.svg/200px-US-NLM-PubMed-Logo.svg.png",
        "embeds": [embed],
    }

    resp = requests.post(GI_DIGEST_WEBHOOK_URL, json=payload, timeout=15)
    resp.raise_for_status()
    print(f"[Discord] ダイジェスト送信: PMID {article['pmid']}")


# ============================================================
# 重複排除
# ============================================================
def load_notified_pmids() -> set[str]:
    if NOTIFIED_FILE.exists():
        data = json.loads(NOTIFIED_FILE.read_text())
        return set(data.get("pmids", []))
    return set()


def save_notified_pmids(pmids: set[str]):
    recent = sorted(pmids)[-2000:]
    NOTIFIED_FILE.write_text(json.dumps({"pmids": recent}, indent=2))


# ============================================================
# メイン処理
# ============================================================
def main():
    print(f"=== GI Digest Bot 実行: {datetime.now().isoformat()} ===")

    notified = load_notified_pmids()

    # ランダムに検索クエリを2つ選び、候補を広げる
    selected_queries = random.sample(SEARCH_QUERIES, min(2, len(SEARCH_QUERIES)))

    all_pmids = []
    for query in selected_queries:
        print(f"[Search] {query[:60]}...")
        pmids = search_pubmed(query, reldate=SEARCH_DAYS)
        all_pmids.extend(pmids)
        time.sleep(1)

    # 重複除去 & 未通知のみ
    seen = set()
    new_pmids = []
    for p in all_pmids:
        if p not in notified and p not in seen:
            new_pmids.append(p)
            seen.add(p)

    print(f"[Filter] 候補 {len(new_pmids)} 件")

    if not new_pmids:
        print("新規論文なし。終了。")
        return

    # 候補からランダムに選んでabstract取得
    selected_pmids = random.sample(new_pmids, min(10, len(new_pmids)))
    articles = fetch_articles(selected_pmids)

    if not articles:
        print("abstract付き論文なし。終了。")
        return

    # 1件を選んでダイジェスト生成 & 投稿
    article = random.choice(articles)
    try:
        digest = generate_digest(article)
        send_discord_digest(article, digest)
        notified.add(article["pmid"])
        save_notified_pmids(notified)
        print(f"=== 完了: {digest['category']} / PMID {article['pmid']} ===")
    except Exception as e:
        print(f"[Error] PMID {article['pmid']}: {e}")


if __name__ == "__main__":
    main()
