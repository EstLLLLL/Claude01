#!/usr/bin/env python3
"""Fetch daily clothing-brand news from Google News RSS and summarize it.

Standard library only, so it runs anywhere without `pip install`.
Reads brands from config.json, fetches each article, and uses the DeepSeek
API (OpenAI-compatible) to write a short Chinese summary per item, then
writes a Markdown digest to news/<date>.md.

Set the DEEPSEEK_API_KEY environment variable to enable summaries. Without
it the digest still lists every item, just without summaries.
"""

import datetime
import html
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.json")
NEWS_DIR = os.path.join(ROOT, "news")
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
USER_AGENT = "Mozilla/5.0 (compatible; halara-news-bot/1.0)"

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
ARTICLE_TEXT_LIMIT = 4000  # chars of article body sent to the model


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def build_url(brand, language, country, window):
    query = f'"{brand}" when:{window}'
    params = {
        "q": query,
        "hl": language,
        "gl": country,
        "ceid": f"{country}:{language.split('-')[0]}",
    }
    return f"{GOOGLE_NEWS_RSS}?{urllib.parse.urlencode(params)}"


def http_get(url, timeout=30):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read(), response.geturl(), charset


def fetch_feed(url):
    body, _, _ = http_get(url)
    return body


def parse_items(raw_xml, limit):
    root = ET.fromstring(raw_xml)
    items = []
    for item in root.iterfind(".//item"):
        title = item.findtext("title", default="").strip()
        link = item.findtext("link", default="").strip()
        pub_date = item.findtext("pubDate", default="").strip()
        source_el = item.find("source")
        source = source_el.text.strip() if source_el is not None and source_el.text else ""
        if not title or not link:
            continue
        items.append(
            {
                "title": html.unescape(title),
                "link": link,
                "pub_date": pub_date,
                "source": source,
            }
        )
        # limit <= 0 means "no limit": include every item from the feed.
        if limit > 0 and len(items) >= limit:
            break
    return items


def extract_text(raw_html, charset):
    try:
        text = raw_html.decode(charset, errors="replace")
    except (LookupError, TypeError):
        text = raw_html.decode("utf-8", errors="replace")
    text = re.sub(r"(?is)<(script|style|noscript|template)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_article_text(url):
    """Resolve the Google News redirect and pull readable body text."""
    raw_html, _, charset = http_get(url, timeout=25)
    return extract_text(raw_html, charset)


def deepseek_summarize(title, article_text, summary_language, model, base_url):
    if not DEEPSEEK_API_KEY:
        return None

    if article_text and len(article_text) > 200:
        source_block = article_text[:ARTICLE_TEXT_LIMIT]
        instruction = (
            f"用{summary_language}为下面这篇新闻写 2-3 句话的摘要，"
            f"说清楚发生了什么、涉及哪个品牌、有何影响。只输出摘要本身。"
        )
    else:
        source_block = title
        instruction = (
            f"下面只有新闻标题，正文无法获取。请用{summary_language}基于标题"
            f"写 1-2 句话说明它大概在讲什么，并在结尾注明“（仅据标题）”。只输出摘要本身。"
        )

    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是一个简洁、客观的新闻摘要助手。"},
                {"role": "user", "content": f"{instruction}\n\n标题：{title}\n\n内容：{source_block}"},
            ],
            "temperature": 0.3,
            "stream": False,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def build_markdown(date_str, results, summarized):
    lines = [f"# Clothing Brand News - {date_str}", ""]
    total = sum(len(v) for v in results.values())
    note = "with DeepSeek summaries" if summarized else "no summaries (DEEPSEEK_API_KEY unset)"
    lines.append(
        f"_Auto-generated digest. {total} item(s) across {len(results)} brand(s); {note}._"
    )
    lines.append("")
    for brand, items in results.items():
        lines.append(f"## {brand}")
        lines.append("")
        if not items:
            lines.append("_No fresh news found in the selected window._")
            lines.append("")
            continue
        for idx, it in enumerate(items, 1):
            meta = " · ".join(p for p in (it["source"], it["pub_date"]) if p)
            lines.append(f"{idx}. [{it['title']}]({it['link']})")
            if meta:
                lines.append(f"   - {meta}")
            if it.get("summary"):
                lines.append(f"   - 摘要：{it['summary']}")
            elif summarized:
                lines.append("   - 摘要：（生成失败）")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    config = load_config()
    brands = config.get("brands", [])
    if not brands:
        print("No brands configured in config.json", file=sys.stderr)
        return 1

    limit = int(config.get("max_items_per_brand", 0))
    language = config.get("language", "en-US")
    country = config.get("country", "US")
    window = str(config.get("time_window", "1d"))
    summary_language = config.get("summary_language", "Chinese")
    model = config.get("deepseek_model", "deepseek-chat")
    base_url = config.get("deepseek_base_url", "https://api.deepseek.com")

    if not DEEPSEEK_API_KEY:
        print("[warn] DEEPSEEK_API_KEY not set; listing items without summaries.", file=sys.stderr)

    results = {}
    for brand in brands:
        try:
            raw = fetch_feed(build_url(brand, language, country, window))
            items = parse_items(raw, limit)
        except Exception as exc:  # one brand failing shouldn't kill the run
            print(f"[warn] {brand}: feed failed: {exc}", file=sys.stderr)
            results[brand] = []
            continue

        for it in items:
            article_text = ""
            try:
                article_text = fetch_article_text(it["link"])
            except Exception as exc:
                print(f"[warn] {brand}: article fetch failed: {exc}", file=sys.stderr)
            try:
                it["summary"] = deepseek_summarize(
                    it["title"], article_text, summary_language, model, base_url
                )
            except Exception as exc:
                print(f"[warn] {brand}: summarize failed: {exc}", file=sys.stderr)
                it["summary"] = None

        results[brand] = items
        print(f"[ok] {brand}: {len(items)} item(s)")

    date_str = datetime.date.today().isoformat()
    os.makedirs(NEWS_DIR, exist_ok=True)
    out_path = os.path.join(NEWS_DIR, f"{date_str}.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(build_markdown(date_str, results, summarized=bool(DEEPSEEK_API_KEY)))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
