# Claude01 — Daily AI Company News

Fetches daily news for AI companies (OpenAI, Anthropic, xAI, Google
Gemini, DeepSeek, ByteDance, MiniMax, Zhipu, Moonshot, Qwen, …) from
Google News RSS in both Chinese and English, fetches each article,
summarizes it with DeepSeek hosted on Volcengine Ark, and writes a Markdown digest to
`news/<date>.md`. Every relevant item found that day is listed (no
truncation), each with a short Chinese summary.

## Usage

```bash
export ARK_API_KEY=...
export ARK_MODEL=deepseek-v4-pro-ga-260813
python3 ark_client.py  # verify credentials/model access
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

## Ark configuration and failures

Configure repository **Secret** `ARK_API_KEY` and repository **Variable**
`ARK_MODEL=deepseek-v4-pro-ga-260813`. The client uses
`https://ark.cn-beijing.volces.com/api/v3`; it never falls back to the
DeepSeek direct API. Model versions are selected explicitly after checking
Ark's model list and making a successful test call. V4.1 is not assumed to
be available just because DeepSeek's own API offers it.

Every run tests model access first. Authentication, billing, unavailable-model,
and invalid-response errors fail the workflow. They cannot be interpreted as
"no news" or overwrite an existing digest. Temporary throttling/server errors
are retried up to three times. RSS candidates and the retrieved text sent to
the model are retained in the `news-materials-<run-id>` artifact for 30 days,
including when analysis fails after fetching.

## Automation

The workflow retains its existing three daily scheduling slots. Manual
`dry_run=true` creates an artifact preview without committing a digest or
creating a GitHub Issue. Runs are serialized to prevent concurrent writes.
Regression tests run for code changes and before each daily job.

Explicit Ark content-filter refusals are not quality judgments. Such candidates
remain in the raw artifact and appear under the digest’s "模型未处理的候选"
section with their original links. They are never retried to bypass filtering.
Malformed JSON may be retried; exhausted errors still fail publication.
