#!/usr/bin/env python3
"""Fetch daily clothing-brand news from Google News RSS and summarize it.

Standard library only, so it runs anywhere without `pip install`.
Reads brands from config.json, fetches each article, and uses the DeepSeek
API (OpenAI-compatible) to write a short Chinese summary per item, then
writes a Markdown digest to news/<date>.md.

Set the DEEPSEEK_API_KEY environment variable to enable summaries. Without
it the digest still lists every item, just without summaries.
"""

import concurrent.futures
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


def is_relevant(match, title, body):
    """Keep items where any alias is in the title, or aliases are clearly
    central to the body (>= 2 total mentions). `match` is a lowercase
    string of aliases separated by '|'."""
    aliases = [a.strip() for a in match.lower().split("|") if a.strip()]
    t = title.lower()
    if any(a in t for a in aliases):
        return True
    x = body.lower()
    return sum(x.count(a) for a in aliases) >= 2


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


def build_issue_body(date_str, results, repo, issue_items):
    """Condensed digest for a GitHub Issue notification."""
    file_url = f"https://github.com/{repo}/blob/main/news/{date_str}.md" if repo else ""
    lines = [f"AI 公司新闻日报 · {date_str}", ""]
    if file_url:
        lines.append(f"完整摘要：{file_url}")
        lines.append("")
    for brand, items in results.items():
        if not items:
            continue
        lines.append(f"### {brand}（{len(items)} 条）")
        for it in items[:issue_items]:
            lines.append(f"- [{it['title']}]({it['link']})")
            summary = it.get("summary")
            if summary:
                lines.append(f"  {summary[:120]}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def create_github_issue(title, body):
    """Open an Issue using the Actions-provided GITHUB_TOKEN. No-op when
    the token/repository is unavailable (e.g. local runs)."""
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if not token or not repo:
        print("[warn] GITHUB_TOKEN/REPOSITORY unset; skipping Issue.", file=sys.stderr)
        return
    payload = json.dumps({"title": title, "body": body}).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    print(f"Opened issue #{data.get('number')}: {data.get('html_url')}")


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
    concurrency = max(1, int(config.get("concurrency", 8)))

    if not DEEPSEEK_API_KEY:
        print("[warn] DEEPSEEK_API_KEY not set; listing items without summaries.", file=sys.stderr)

    companies = []
    for entry in brands:
        if isinstance(entry, str):
            companies.append({"name": entry, "query": entry, "match": entry.lower()})
        else:
            q = entry.get("query", entry["name"])
            companies.append(
                {
                    "name": entry["name"],
                    "query": q,
                    "match": entry.get("match", q).lower(),
                }
            )

    # Phase 1: fetch every (company, locale) feed in parallel.
    def grab_feed(task):
        company, loc = task
        try:
            raw = fetch_feed(build_url(company["query"], loc["language"], loc["country"], window))
            return company["name"], parse_items(raw, 0)
        except Exception as exc:  # one locale failing shouldn't kill the run
            print(f"[warn] {company['name']} [{loc.get('language')}]: feed failed: {exc}", file=sys.stderr)
            return company["name"], []

    feed_tasks = [(c, loc) for c in companies for loc in locales]
    merged = {c["name"]: [] for c in companies}
    seen = {c["name"]: set() for c in companies}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for cname, items in pool.map(grab_feed, feed_tasks):
            for it in items:
                key = it["title"].strip().lower()
                if key in seen[cname]:
                    continue
                seen[cname].add(key)
                merged[cname].append(it)

    # Process a few extra candidates so relevance filtering still leaves
    # roughly `limit` items; the final list is capped to `limit` below.
    if limit > 0:
        for cname in merged:
            merged[cname] = merged[cname][: limit * 3]

    # Phase 2: resolve + fetch + summarize every item in parallel.
    match_by = {c["name"]: c["match"] for c in companies}

    def process(task):
        cname, it = task
        article_text = ""
        try:
            article_text = fetch_article_text(it["link"])
        except Exception as exc:
            print(f"[warn] {cname}: article fetch failed: {exc}", file=sys.stderr)
        if not is_relevant(match_by[cname], it["title"], article_text):
            return cname, None
        try:
            it["summary"] = deepseek_summarize(
                it["title"], article_text, summary_language, model, base_url
            )
        except Exception as exc:
            print(f"[warn] {cname}: summarize failed: {exc}", file=sys.stderr)
            it["summary"] = None
        return cname, it

    item_tasks = [(c["name"], it) for c in companies for it in merged[c["name"]]]
    results = {c["name"]: [] for c in companies}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for cname, it in pool.map(process, item_tasks):
            if it is not None:
                results[cname].append(it)

    if limit > 0:
        for cname in results:
            results[cname] = results[cname][:limit]

    for c in companies:
        name = c["name"]
        print(f"[ok] {name}: {len(results[name])} kept (from {len(merged[name])} merged)")

    date_str = datetime.date.today().isoformat()
    os.makedirs(NEWS_DIR, exist_ok=True)
    out_path = os.path.join(NEWS_DIR, f"{date_str}.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(build_markdown(date_str, results, summarized=bool(DEEPSEEK_API_KEY)))
    print(f"Wrote {out_path}")

    if config.get("create_issue", True) and sum(len(v) for v in results.values()):
        try:
            body = build_issue_body(
                date_str,
                results,
                os.environ.get("GITHUB_REPOSITORY", ""),
                int(config.get("issue_items_per_brand", 5)),
            )
            create_github_issue(f"AI 公司新闻日报 · {date_str}", body)
        except Exception as exc:
            print(f"[warn] issue creation failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
