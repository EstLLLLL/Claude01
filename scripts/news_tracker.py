"""Daily yoga/athleisure brand news tracker.

Pipeline:
  1. Fetch the last 24h of Google News RSS results for each brand.
  2. Ask Claude to summarize each brand's headlines in 3-5 bullets.
  3. Build an HTML email digest.
  4. Drop it into the configured Gmail account's Drafts folder
     (or print to stdout when --dry-run is passed).

Run locally:
  ANTHROPIC_API_KEY=... python scripts/news_tracker.py --dry-run

Run in CI: provide all the env vars listed in README.md.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import os
import sys
import time
from dataclasses import dataclass
from email.mime.text import MIMEText
from typing import Iterable
from urllib.parse import quote_plus

import feedparser
from anthropic import Anthropic
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build as build_gmail


BRANDS: list[str] = [
    "Halara",
    "CRZ Yoga",
    "Vuori",
    "Alo Yoga",
    "Lululemon",
]

MAX_ARTICLES_PER_BRAND = 15
CLAUDE_MODEL = "claude-sonnet-4-6"
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]

SYSTEM_PROMPT = (
    "You are a brand-news analyst covering the athleisure and activewear industry. "
    "Given a list of headlines for a single brand, produce 3-5 tight bullets focused on: "
    "product launches, financial/business updates, partnerships and collabs, controversies, "
    "and notable press. Cite each bullet inline as (Source — date). Be concrete. "
    "If the headlines contain nothing material — only SEO spam, unrelated matches, or "
    "duplicates of older news — reply with exactly: No notable news today."
)


@dataclass
class Article:
    title: str
    source: str
    link: str
    published: str
    snippet: str


def fetch_brand_news(brand: str) -> list[Article]:
    """Fetch the last 24h of Google News results for a brand."""
    query = quote_plus(f'"{brand}" when:1d')
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    articles: list[Article] = []
    for entry in feed.entries[:MAX_ARTICLES_PER_BRAND]:
        articles.append(
            Article(
                title=entry.get("title", "").strip(),
                source=(entry.get("source", {}) or {}).get("title", "")
                or entry.get("author", "")
                or "Unknown",
                link=entry.get("link", ""),
                published=entry.get("published", ""),
                snippet=entry.get("summary", "").strip(),
            )
        )
    return articles


def summarize_brand(client: Anthropic, brand: str, articles: list[Article]) -> str:
    """Ask Claude to summarize a brand's day. Skips the API call if no articles."""
    if not articles:
        return "No notable news today."

    headline_block = "\n".join(
        f"- {a.title} ({a.source} — {a.published})\n  {a.snippet}" for a in articles
    )
    user_prompt = (
        f"Brand: {brand}\n\nHeadlines from the last 24 hours:\n\n{headline_block}"
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def render_html(today: dt.date, sections: list[tuple[str, str, list[Article]]]) -> str:
    parts = [
        "<html><body style=\"font-family:-apple-system,Segoe UI,Helvetica,sans-serif;"
        "max-width:720px;margin:0 auto;padding:16px;color:#222;\">",
        f"<h1 style=\"margin-bottom:4px;\">📰 Brand News Digest</h1>",
        f"<p style=\"color:#666;margin-top:0;\">{today.isoformat()}</p>",
    ]
    for brand, summary, articles in sections:
        parts.append(f"<h2 style=\"margin-top:28px;border-bottom:1px solid #eee;padding-bottom:4px;\">{html.escape(brand)}</h2>")
        # Render Claude's bullets — it returns markdown-ish text; convert simple bullets.
        for line in summary.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(("- ", "* ")):
                parts.append(f"<li>{html.escape(line[2:])}</li>")
            else:
                parts.append(f"<p>{html.escape(line)}</p>")
        if articles:
            parts.append("<details><summary style=\"cursor:pointer;color:#666;font-size:13px;margin-top:8px;\">"
                         f"All {len(articles)} headlines</summary><ul style=\"font-size:13px;color:#555;\">")
            for a in articles:
                title = html.escape(a.title)
                src = html.escape(a.source)
                parts.append(f'<li><a href="{html.escape(a.link)}">{title}</a> — {src}</li>')
            parts.append("</ul></details>")
    parts.append("</body></html>")
    return "\n".join(parts)


def render_text(today: dt.date, sections: list[tuple[str, str, list[Article]]]) -> str:
    out = [f"Brand News Digest — {today.isoformat()}", ""]
    for brand, summary, articles in sections:
        out.append(f"## {brand}")
        out.append(summary)
        if articles:
            out.append("")
            out.append("Headlines:")
            for a in articles:
                out.append(f"  - {a.title} ({a.source}) {a.link}")
        out.append("")
    return "\n".join(out)


def gmail_credentials_from_env() -> Credentials:
    client_id = os.environ["GMAIL_CLIENT_ID"]
    client_secret = os.environ["GMAIL_CLIENT_SECRET"]
    refresh_token = os.environ["GMAIL_REFRESH_TOKEN"]
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=GMAIL_SCOPES,
    )
    creds.refresh(GoogleAuthRequest())
    return creds


def create_gmail_draft(subject: str, html_body: str, text_body: str, to_addr: str) -> str:
    creds = gmail_credentials_from_env()
    service = build_gmail("gmail", "v1", credentials=creds, cache_discovery=False)

    msg = MIMEText(html_body, "html", "utf-8")
    msg["To"] = to_addr
    msg["From"] = to_addr
    msg["Subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    draft = service.users().drafts().create(
        userId="me", body={"message": {"raw": raw}}
    ).execute()
    return draft["id"]


def build_digest(client: Anthropic | None, brands: Iterable[str]) -> list[tuple[str, str, list[Article]]]:
    sections: list[tuple[str, str, list[Article]]] = []
    for brand in brands:
        print(f"[fetch] {brand}", file=sys.stderr)
        articles = fetch_brand_news(brand)
        print(f"  {len(articles)} articles", file=sys.stderr)
        if client is None:
            summary = "(dry-run: skipping Claude summary)"
        else:
            summary = summarize_brand(client, brand, articles)
        sections.append((brand, summary, articles))
        time.sleep(0.5)  # gentle pacing for both Google News and the API
    return sections


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily brand news tracker")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the digest to stdout instead of creating a Gmail draft. "
             "Skips the Claude API call too if ANTHROPIC_API_KEY is unset.",
    )
    args = parser.parse_args()

    today = dt.date.today()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        client = Anthropic(api_key=api_key)
    elif args.dry_run:
        client = None  # raw fetch only
    else:
        print("ERROR: ANTHROPIC_API_KEY is required (or pass --dry-run).", file=sys.stderr)
        return 2

    sections = build_digest(client, BRANDS)

    subject = f"📰 Brand News Digest — {today.isoformat()}"
    html_body = render_html(today, sections)
    text_body = render_text(today, sections)

    if args.dry_run:
        print(text_body)
        return 0

    to_addr = os.environ["RECIPIENT_EMAIL"]
    draft_id = create_gmail_draft(subject, html_body, text_body, to_addr)
    print(f"Created Gmail draft {draft_id} for {to_addr}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
