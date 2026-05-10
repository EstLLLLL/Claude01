# Yoga Brand News Tracker

A daily news digest for athleisure / yoga brands — **Halara, CRZ Yoga, Vuori, Alo Yoga, Lululemon** — that runs on GitHub Actions, summarizes headlines with Claude, and emails the result to your inbox each morning.

## How it works

```
GitHub Actions cron (13:00 UTC daily)
  ──► scripts/news_tracker.py
        ├─ Google News RSS  (last 24h, per brand)
        ├─ Anthropic Claude (per-brand 3–5 bullet summary)
        └─ Gmail SMTP       (sends the digest email)
```

## Setup (one time, ~5 minutes)

### 1. Get a Claude API key

[console.anthropic.com](https://console.anthropic.com/) → **API Keys** → Create. Save it for step 3.

### 2. Create a Gmail App Password

You need 2-Step Verification turned on first (Google Account → Security → 2-Step Verification).

Then go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords):

- App name: `news-tracker` (anything works)
- Click **Create**. Google shows a 16-character password — copy it (spaces are fine, they're stripped on use).

### 3. Add GitHub secrets

Repo → **Settings → Secrets and variables → Actions → New repository secret**. Add four secrets:

| Name | Value |
|---|---|
| `ANTHROPIC_API_KEY` | from step 1 |
| `GMAIL_USER` | the Gmail address that will send the digest, e.g. `you@gmail.com` |
| `GMAIL_APP_PASSWORD` | the 16-character app password from step 2 |
| `RECIPIENT_EMAIL` | where to deliver the digest (often the same as `GMAIL_USER`) |

### 4. Run it

- **Manually**: repo → **Actions → Daily Brand News → Run workflow**.
- **Scheduled**: it runs daily at 13:00 UTC (≈ 21:00 Beijing / 6am PT / 9am ET). Edit the cron in `.github/workflows/daily-news.yml` to change the time.

The digest arrives titled `📰 Brand News Digest — YYYY-MM-DD`.

## Local dry-run

Print the digest to your terminal without sending email:

```bash
# With Claude summarization:
ANTHROPIC_API_KEY=sk-ant-... python scripts/news_tracker.py --dry-run

# Or just see the raw headlines (no API key needed):
python scripts/news_tracker.py --dry-run
```

## Customizing

- **Brands** — edit the `BRANDS` list at the top of `scripts/news_tracker.py`.
- **Schedule** — edit the `cron` line in `.github/workflows/daily-news.yml`.
- **Model** — change `CLAUDE_MODEL` in `scripts/news_tracker.py` (default is `claude-sonnet-4-6`).
- **Headlines per brand** — change `MAX_ARTICLES_PER_BRAND`.

## Files

| Path | Purpose |
|---|---|
| `scripts/news_tracker.py` | The main pipeline (fetch → summarize → email) |
| `.github/workflows/daily-news.yml` | Daily cron + manual trigger |
| `requirements.txt` | Python deps (`feedparser`, `anthropic`) |
