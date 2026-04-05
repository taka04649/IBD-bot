"""
PubMed Ulcerative Colitis 新着論文監視 & Discord通知 Bot
======================================================
- PubMed E-utilities で直近の新着論文を取得
- Google Gemini API で abstract を日本語要約
- Discord Webhook で通知
- 重複排除のためPMIDを記録
"""

import os
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import requests
import google.generativeai as genai

# ============================================================
# 設定 (環境変数から読み込み)
# ============================================================
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# Gemini の設定
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"

# 検索クエリ (MeSH + フリーテキスト)
SEARCH_QUERY = (
    '("Colitis, Ulcerative"[MeSH] OR "ulcerative colitis"[Title/Abstract]) OR '
    '("Crohn Disease"[MeSH] OR "Crohn disease"[Title/Abstract] OR "Crohn\'s disease"[Title/Abstract])'
)

# 取得する日数
SEARCH_DAYS = 1

# 通知済みPMID記録ファイル
NOTIFIED_FILE = Path(__file__).parent / "notified_pmids.json"

# 1回あたりの最大取得件数
MAX_RESULTS = 20

# ============================================================
# PubMed E-utilities
# ============================================================
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed(query: str, reldate: int = 1) -> list[str]:
    """ESearch で新着PMIDを取得"""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": MAX_RESULTS,
        "datetype": "edat",
        "reldate": reldate,
        "retmode": "json",
        "sort": "date",
    }
    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    pmids = data.get("esearchresult", {}).get("idlist", [])
    print(f"[PubMed] {len(pmids)} 件の新着論文を検出")
    return pmids


def fetch_articles(pmids: list[str]) -> list[dict]:
    """EFetch でPMIDからタイトル・abstract等を取得"""
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

    # デバッグ: 取得したXMLの先頭を表示
    print(f"[Debug] XML response length: {len(resp.content)} bytes")
    print(f"[Debug] XML preview: {resp.content[:500].decode('utf-8', errors='replace')}")

    root = ET.fromstring(resp.content)
    articles = []

    for article_elem in root.findall(".//PubmedArticle"):
        pmid = _text(article_elem, ".//PMID")
        title = _full_text(article_elem, ".//ArticleTitle")

        # Abstract は複数の AbstractText 要素を結合
        abstract_parts = []
        for at in article_elem.findall(".//AbstractText"):
            label = at.get("Label", "")
            text = "".join(at.itertext()).strip()
            if label:
                abstract_parts.append(f"[{label}] {text}")
            else:
                abstract_parts.append(text)
        abstract = "\n".join(abstract_parts)

        # AbstractText が見つからなかった場合、Abstract 要素全体から取得を試みる
        if not abstract:
            abstract_node = article_elem.find(".//Abstract")
            if abstract_node is not None:
                abstract = "".join(abstract_node.itertext()).strip()

        # デバッグログ
        if abstract:
            print(f"  [OK] PMID {pmid}: abstract {len(abstract)} 文字")
        else:
            print(f"  [SKIP] PMID {pmid}: abstractなし - {title[:60]}")
            continue  # abstractがない論文はスキップ

        # ジャーナル名
        journal = _full_text(article_elem, ".//Journal/Title")

        # 著者 (先頭3名)
        authors = []
        for author in article_elem.findall(".//Author")[:3]:
            last = _text(author, "LastName")
            fore = _text(author, "ForeName")
            if last:
                authors.append(f"{last} {fore}".strip())
        if len(article_elem.findall(".//Author")) > 3:
            authors.append("et al.")

        # DOI
        doi = ""
        for aid in article_elem.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text or ""

        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "authors": ", ".join(authors),
            "doi": doi,
        })

    print(f"[PubMed] {len(articles)} 件のabstract付き論文を取得")
    return articles


def _text(elem, path: str) -> str:
    """XMLから直接のテキストを取得"""
    node = elem.find(path)
    if node is not None and node.text:
        return node.text.strip()
    return ""


def _full_text(elem, path: str) -> str:
    """XMLから子要素含む全テキストを取得（<i>タグ等に対応）"""
    node = elem.find(path)
    if node is not None:
        return "".join(node.itertext()).strip()
    return ""


# ============================================================
# Gemini API で要約生成
# ============================================================
def summarize_abstract(title: str, abstract: str) -> str:
    """Gemini APIでabstractを日本語要約"""
    if not abstract:
        return "（Abstractなし）"

    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""以下の医学論文のAbstractを日本語で簡潔に要約してください。

## フォーマット
- **研究デザイン**: （1文）
- **主要結果**: （2〜3文）
- **臨床的意義**: （1文）

## 論文
タイトル: {title}

Abstract:
{abstract}
"""

    response = model.generate_content(prompt)
    return response.text


# ============================================================
# Discord Webhook 通知
# ============================================================
def send_discord_notification(article: dict, summary: str):
    """Discord Webhook で Embed 付き通知を送信"""
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/"
    doi_url = f"https://doi.org/{article['doi']}" if article["doi"] else ""

    links = f"[PubMed]({pubmed_url})"
    if doi_url:
        links += f"  |  [DOI]({doi_url})"

    embed = {
        "title": article["title"][:256],
        "url": pubmed_url,
        "color": 0x3498DB,
        "fields": [
            {
                "name": "📖 ジャーナル / 著者",
                "value": f"*{article['journal']}*\n{article['authors']}",
                "inline": False,
            },
            {
                "name": "📝 要約",
                "value": summary[:1024],
                "inline": False,
            },
            {
                "name": "🔗 リンク",
                "value": links,
                "inline": False,
            },
        ],
        "footer": {
            "text": f"PMID: {article['pmid']}",
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

    payload = {
        "username": "PubMed UC Bot",
        "avatar_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/US-NLM-PubMed-Logo.svg/200px-US-NLM-PubMed-Logo.svg.png",
        "embeds": [embed],
    }

    resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=15)
    resp.raise_for_status()
    print(f"[Discord] 通知送信: PMID {article['pmid']}")


# ============================================================
# 重複排除
# ============================================================
def load_notified_pmids() -> set[str]:
    if NOTIFIED_FILE.exists():
        data = json.loads(NOTIFIED_FILE.read_text())
        return set(data.get("pmids", []))
    return set()


def save_notified_pmids(pmids: set[str]):
    recent = sorted(pmids)[-1000:]
    NOTIFIED_FILE.write_text(json.dumps({"pmids": recent}, indent=2))


# ============================================================
# メイン処理
# ============================================================
def main():
    print(f"=== PubMed UC Bot 実行: {datetime.now().isoformat()} ===")

    # 1. 新着PMID検索
    pmids = search_pubmed(SEARCH_QUERY, reldate=SEARCH_DAYS)

    # 2. 重複排除
    notified = load_notified_pmids()
    new_pmids = [p for p in pmids if p not in notified]
    print(f"[Filter] 新規 {len(new_pmids)} 件 (既通知除外)")

    if not new_pmids:
        print("新着論文なし。終了。")
        return

    # 3. Abstract取得
    articles = fetch_articles(new_pmids)

    # 4. 要約 & 通知
    for article in articles:
        try:
            summary = summarize_abstract(article["title"], article["abstract"])
            send_discord_notification(article, summary)
            notified.add(article["pmid"])
            time.sleep(2)  # rate limit 対策
        except Exception as e:
            print(f"[Error] PMID {article['pmid']}: {e}")

    # 5. 通知済みPMID保存
    save_notified_pmids(notified)
    print(f"=== 完了: {len(articles)} 件通知 ===")


if __name__ == "__main__":
    main()
