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


def build_url(query, language, country, window):
    query = f"{query} when:{window}"
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


def resolve_google_news_url(url):
    """Turn a news.google.com/rss/articles/<id> link into the publisher URL.

    Google no longer 302-redirects these links; the real URL must be
    obtained from the internal `batchexecute` endpoint, using a signature
    and timestamp embedded in the article page.
    """
    host = urllib.parse.urlparse(url).netloc
    if "news.google.com" not in host:
        return url  # already a direct publisher link

    article_id = urllib.parse.urlparse(url).path.rstrip("/").split("/")[-1]
    page, _, charset = http_get(
        f"https://news.google.com/rss/articles/{article_id}", timeout=25
    )
    page_text = page.decode(charset or "utf-8", errors="replace")

    def attr(name):
        m = re.search(name + r'="([^"]+)"', page_text)
        return m.group(1) if m else None

    signature = attr("data-n-a-sg")
    timestamp = attr("data-n-a-ts")
    inner_id = attr("data-n-a-id") or article_id
    if not signature or not timestamp:
        raise RuntimeError("could not find batchexecute signature")

    inner = json.dumps(
        [
            "garturlreq",
            [
                ["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1,
                 None, None, None, None, None, 0, 1],
                "X", "X", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0,
            ],
            inner_id,
            int(timestamp),
            signature,
        ]
    )
    freq = json.dumps([[["Fbv4je", inner, None, "generic"]]])
    body = urllib.parse.urlencode({"f.req": freq}).encode("utf-8")
    request = urllib.request.Request(
        "https://news.google.com/_/DotsSplashUi/data/batchexecute",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        raw = response.read().decode("utf-8", errors="replace")

    m = re.search(r'garturlres\\",\\"(.*?)\\"', raw)
    if not m:
        raise RuntimeError("batchexecute returned no URL")
    return json.loads('"' + m.group(1) + '"')


MIN_BODY_CHARS = 500  # below this a page is likely a JS/consent shell


def fetch_article_text(url):
    """Resolve the Google News redirect and pull readable body text.

    Sites like MSN serve a JS shell to bots, so when the direct fetch
    yields too little text we retry through the r.jina.ai reader proxy,
    which renders the page and returns plain readable text.
    """
    real_url = resolve_google_news_url(url)
    text = ""
    try:
        raw_html, _, charset = http_get(real_url, timeout=25)
        text = extract_text(raw_html, charset)
    except Exception as exc:
        print(f"[warn] direct fetch failed: {exc}", file=sys.stderr)

    if len(text) < MIN_BODY_CHARS:
        try:
            raw, _, charset = http_get(f"https://r.jina.ai/{real_url}", timeout=40)
            alt = raw.decode(charset or "utf-8", errors="replace")
            alt = re.sub(r"\s+", " ", alt).strip()
            if len(alt) > len(text):
                text = alt
        except Exception as exc:
            print(f"[warn] reader-proxy fallback failed: {exc}", file=sys.stderr)
    return text


def is_relevant(brand, title, body):
    """Keep items where the brand is in the title or clearly central to the
    body (>= 2 mentions), dropping articles that only name-drop it once."""
    b = brand.lower()
    if b in title.lower():
        return True
    return body.lower().count(b) >= 2


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
    lines = [f"# AI Company News - {date_str}", ""]
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
    window = str(config.get("time_window", "1d"))
    locales = config.get("locales") or [
        {"language": config.get("language", "en-US"), "country": config.get("country", "US")}
    ]
    summary_language = config.get("summary_language", "Chinese")
    model = config.get("deepseek_model", "deepseek-chat")
    base_url = config.get("deepseek_base_url", "https://api.deepseek.com")

    if not DEEPSEEK_API_KEY:
        print("[warn] DEEPSEEK_API_KEY not set; listing items without summaries.", file=sys.stderr)

    results = {}
    for entry in brands:
        if isinstance(entry, str):
            name, query, match = entry, entry, entry.lower()
        else:
            name = entry["name"]
            query = entry.get("query", name)
            match = entry.get("match", query).lower()

        merged = []
        seen = set()
        for loc in locales:
            try:
                raw = fetch_feed(
                    build_url(query, loc["language"], loc["country"], window)
                )
                for it in parse_items(raw, 0):
                    key = it["title"].strip().lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(it)
            except Exception as exc:  # one locale failing shouldn't kill the run
                print(f"[warn] {name} [{loc.get('language')}]: feed failed: {exc}", file=sys.stderr)

        if limit > 0:
            merged = merged[:limit]

        kept = []
        dropped = 0
        for it in merged:
            article_text = ""
            try:
                article_text = fetch_article_text(it["link"])
            except Exception as exc:
                print(f"[warn] {name}: article fetch failed: {exc}", file=sys.stderr)

            if not is_relevant(match, it["title"], article_text):
                dropped += 1
                continue

            try:
                it["summary"] = deepseek_summarize(
                    it["title"], article_text, summary_language, model, base_url
                )
            except Exception as exc:
                print(f"[warn] {name}: summarize failed: {exc}", file=sys.stderr)
                it["summary"] = None
            kept.append(it)

        results[name] = kept
        print(f"[ok] {name}: {len(kept)} kept, {dropped} dropped (from {len(merged)} merged)")

    date_str = datetime.date.today().isoformat()
    os.makedirs(NEWS_DIR, exist_ok=True)
    out_path = os.path.join(NEWS_DIR, f"{date_str}.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(build_markdown(date_str, results, summarized=bool(DEEPSEEK_API_KEY)))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
