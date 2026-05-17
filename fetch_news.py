#!/usr/bin/env python3
"""Fetch daily clothing-brand news from Google News RSS.

Standard library only, so it runs anywhere without `pip install`.
Reads brands from config.json and writes a Markdown digest to news/<date>.md.
"""

import datetime
import html
import json
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.json")
NEWS_DIR = os.path.join(ROOT, "news")
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
USER_AGENT = "Mozilla/5.0 (compatible; halara-news-bot/1.0)"


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def build_url(brand, language, country):
    # Restrict to the last day so the digest stays "today's news".
    query = f'"{brand}" when:1d'
    params = {
        "q": query,
        "hl": language,
        "gl": country,
        "ceid": f"{country}:{language.split('-')[0]}",
    }
    return f"{GOOGLE_NEWS_RSS}?{urllib.parse.urlencode(params)}"


def fetch_feed(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


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
        if len(items) >= limit:
            break
    return items


def build_markdown(date_str, results):
    lines = [f"# Clothing Brand News - {date_str}", ""]
    total = sum(len(v) for v in results.values())
    lines.append(f"_Auto-generated digest. {total} item(s) across {len(results)} brand(s)._")
    lines.append("")
    for brand, items in results.items():
        lines.append(f"## {brand}")
        lines.append("")
        if not items:
            lines.append("_No fresh news found in the last 24h._")
            lines.append("")
            continue
        for it in items:
            meta = " · ".join(p for p in (it["source"], it["pub_date"]) if p)
            lines.append(f"- [{it['title']}]({it['link']})")
            if meta:
                lines.append(f"  - {meta}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    config = load_config()
    brands = config.get("brands", [])
    if not brands:
        print("No brands configured in config.json", file=sys.stderr)
        return 1

    limit = int(config.get("max_items_per_brand", 8))
    language = config.get("language", "en-US")
    country = config.get("country", "US")

    results = {}
    for brand in brands:
        try:
            raw = fetch_feed(build_url(brand, language, country))
            results[brand] = parse_items(raw, limit)
            print(f"[ok] {brand}: {len(results[brand])} item(s)")
        except Exception as exc:  # network/parse failure for one brand shouldn't kill the run
            print(f"[warn] {brand}: {exc}", file=sys.stderr)
            results[brand] = []

    date_str = datetime.date.today().isoformat()
    os.makedirs(NEWS_DIR, exist_ok=True)
    out_path = os.path.join(NEWS_DIR, f"{date_str}.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(build_markdown(date_str, results))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
