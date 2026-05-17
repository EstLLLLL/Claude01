# Claude01 — Daily Brand News

Fetches daily news for clothing brands (Halara and peers) from Google News
RSS and writes a Markdown digest to `news/<date>.md`.

## Usage

```bash
python3 fetch_news.py
```

Standard library only — no `pip install` required. Output goes to
`news/YYYY-MM-DD.md`.

## Configuration

Edit `config.json`:

| Key | Meaning |
| --- | --- |
| `brands` | List of brand names to track (add/remove freely) |
| `max_items_per_brand` | Max headlines kept per brand |
| `language` | Google News `hl`, e.g. `en-US` |
| `country` | Google News `gl`/`ceid`, e.g. `US` |

## Automation

`.github/workflows/daily-news.yml` runs daily at 06:00 UTC (and on manual
`workflow_dispatch`), then commits the new digest back to the repository.

> Note: outbound HTTP is blocked in some sandboxed environments, so the
> script may produce an empty digest there. It runs normally on GitHub
> Actions runners, which have open network egress.
