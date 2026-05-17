# Claude01 — Daily AI Company News

Fetches daily news for AI companies (OpenAI, Anthropic, xAI, Google
Gemini, DeepSeek, ByteDance, MiniMax, Zhipu, Moonshot, Qwen, …) from
Google News RSS in both Chinese and English, fetches each article,
summarizes it with the DeepSeek API, and writes a Markdown digest to
`news/<date>.md`. Every relevant item found that day is listed (no
truncation), each with a short Chinese summary.

## Usage

```bash
export DEEPSEEK_API_KEY=sk-...   # optional; without it items are listed without summaries
python3 fetch_news.py
```

Standard library only — no `pip install` required. Output goes to
`news/YYYY-MM-DD.md`.

## Configuration

Edit `config.json`:

| Key | Meaning |
| --- | --- |
| `brands` | Companies to track. Each entry is `{ "name": display, "query": google-news search, "match": relevance keyword }` (a plain string also works) |
| `max_items_per_brand` | Items kept per company; `0` = no limit (list all) |
| `time_window` | Google News recency, e.g. `1d`, `7d` |
| `locales` | List of `{ "language": hl, "country": gl }`; results from all locales are merged and de-duplicated by title |
| `summary_language` | Language for summaries, e.g. `Chinese` |
| `deepseek_model` | DeepSeek model, e.g. `deepseek-chat` |
| `deepseek_base_url` | DeepSeek API base, default `https://api.deepseek.com` |

Summaries require a `DEEPSEEK_API_KEY`. For the daily workflow, add it
under the repo's **Settings → Secrets and variables → Actions** as
`DEEPSEEK_API_KEY`.

## Automation

`.github/workflows/daily-news.yml` runs daily at 06:00 UTC (and on manual
`workflow_dispatch`), then commits the new digest back to the repository.

> Note: outbound HTTP is blocked in some sandboxed environments, so the
> script may produce an empty digest there. It runs normally on GitHub
> Actions runners, which have open network egress.
