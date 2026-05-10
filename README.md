# Yoga Brand News Tracker

A daily news digest for athleisure / yoga brands — **Halara, CRZ Yoga, Vuori, Alo Yoga, Lululemon** — that runs on GitHub Actions, summarizes headlines with Claude, and lands the result in your Gmail Drafts folder each morning.

## How it works

```
GitHub Actions cron (13:00 UTC daily)
  ──► scripts/news_tracker.py
        ├─ Google News RSS  (last 24h, per brand)
        ├─ Anthropic Claude (per-brand 3–5 bullet summary)
        └─ Gmail API        (creates a draft in your inbox)
```

## Setup (one time, ~10 minutes)

### 1. Get a Claude API key

[console.anthropic.com](https://console.anthropic.com/) → **API Keys** → Create. Save it for step 4.

### 2. Set up Gmail API access

1. Go to [console.cloud.google.com](https://console.cloud.google.com/), create a project.
2. **APIs & Services → Library →** enable **Gmail API**.
3. **APIs & Services → OAuth consent screen →** External, add yourself as a test user.
4. **APIs & Services → Credentials →** Create Credentials → **OAuth client ID** → **Desktop app**. Download the JSON.
5. Save the JSON as `credentials.json` at the repo root (it's gitignored).

### 3. Mint a refresh token

```bash
pip install -r requirements.txt
python scripts/auth_setup.py
```

A browser window opens — sign in with the Gmail account that should receive the digests. The script prints three values: `GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`, `GMAIL_REFRESH_TOKEN`. Copy them.

Delete `credentials.json` afterwards.

### 4. Add GitHub secrets

Repo → **Settings → Secrets and variables → Actions → New repository secret**. Add five secrets:

| Name | Value |
|---|---|
| `ANTHROPIC_API_KEY` | from step 1 |
| `GMAIL_CLIENT_ID` | from step 3 |
| `GMAIL_CLIENT_SECRET` | from step 3 |
| `GMAIL_REFRESH_TOKEN` | from step 3 |
| `RECIPIENT_EMAIL` | the Gmail address you authenticated with |

### 5. Run it

- **Manually**: repo → **Actions → Daily Brand News → Run workflow**.
- **Scheduled**: it runs daily at 13:00 UTC. Edit the cron in `.github/workflows/daily-news.yml` to change the time.

Drafts appear in your Gmail Drafts folder titled `📰 Brand News Digest — YYYY-MM-DD`.

## Local dry-run

Print the digest to your terminal without touching Gmail:

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
| `scripts/news_tracker.py` | The main pipeline (fetch → summarize → draft) |
| `scripts/auth_setup.py` | One-time helper to mint the Gmail refresh token |
| `.github/workflows/daily-news.yml` | Daily cron + manual trigger |
| `requirements.txt` | Python deps |
