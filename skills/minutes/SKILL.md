---
name: minutes
description: Full pipeline to turn a meeting or interview recording into an Obsidian meeting memo, and to enroll on-device speaker voiceprints from named voice clips into the user's iCloud-shared private profile library. Transcribes audio with Doubao (Volcengine) Seed ASR, generates a Chinese summary and Q&A transcript with DeepSeek via Ark (`deepseek-v4-flash` by default), and uses enrolled voiceprints as high-confidence speaker hints. Doubao is the only ASR provider; use ElevenLabs only when the user explicitly asks for it by name. For English meetings, also converts the Polished Transcript section to bilingual format (English + Chinese per speaker turn). Chinese meetings get Chinese-only output. Use this skill whenever the user gives you recording files (.m4a, .mp3, .wav, .qta, etc.) and wants meeting notes, minutes, a transcript saved to Obsidian, or says "录入声纹" / "记住这个人的声音" / "同步声纹库".
---

# Minutes

End-to-end pipeline: audio recording → Obsidian meeting memo.

## Shared voiceprint enrollment

When the user supplies named voice clips and asks to enroll the voice:

- Treat a voiceprint as sensitive biometric data. Compute it on-device. Store only the derived profile in the user-authorized private iCloud directory at `~/Library/Mobile Documents/com~apple~CloudDocs/Codex/Private/Voiceprints/profiles`; never send clips or embeddings to ASR, LLM, or other third-party services.
- Require at least two independent clips. Reject enrollment if either has under two seconds of usable speech or their embedding consistency is below the built-in threshold.
- Do not transcribe the enrollment clips. Do not copy raw clips, the model, or the Python runtime into iCloud; keep the model/runtime under `~/.codex/private/voiceprints` on each Mac.
- Run:

```bash
"$HOME/.codex/private/voiceprints/runtime/bin/python" \
  "$HOME/.codex/skills/minutes/scripts/voiceprint.py" enroll \
  --name "<speaker name>" \
  --audio "<clip 1>" \
  --audio "<clip 2>"
```

To migrate profiles made by this skill on another Mac from the old local directory into the shared iCloud library, run:

```bash
"$HOME/.codex/private/voiceprints/runtime/bin/python" \
  "$HOME/.codex/skills/minutes/scripts/voiceprint.py" migrate-local
```

Never overwrite a same-name profile silently. The migration command skips an identical file, reports a conflict for a different same-name profile, and rejects vectors produced by a different model/schema. Both Macs must use the same profile model/format; copy or migrate profiles only after the compatibility check passes.

Report the sample-to-sample consistency, active iCloud profile path, and whether raw audio was stored. Future `fish_minutes_pipeline.py` runs automatically score each chunk-local diarization label against the shared profiles. Only scores above the conservative threshold become name aliases; uncertain matches remain role labels. An explicit `--speaker-alias` always overrides a voiceprint suggestion.

## What this skill does

1. Asks three quick questions (if not already provided)
2. Transcribes the audio using Doubao Seed ASR via `fish_minutes_pipeline.py` (legacy filename — it calls Doubao, never Fish Audio)
3. Reconciles chunk-local diarization labels into one global speaker roster, using names when supported and roles when names are uncertain
4. Corrects high-confidence proper-noun ASR errors occurrence by occurrence; never blanket-replaces ambiguous words
5. Generates a detailed Chinese summary with five fixed sections and validates that it is not a placeholder or shallow recap
6. Verifies that the transcript lost no wording: only speaker labels, adjacent-turn merging, and audited exact-substring term fixes may change
7. **English meeting only**: converts the Polished Transcript to bilingual format (EN + ZH per turn)
8. Saves the memo under the matching quarter folder inside `$VOLCENGINE_OBSIDIAN_MEETING_MEMO_DIR`, using the recording date, and cleans up final JSON/SRT intermediates by default

## Step 1 — Gather inputs

Before running anything, ask the user (skip any already answered in the same message):

- **会议语言是中文还是英文？** (Is this a Chinese or English meeting?) — determines whether bilingual conversion runs
- **参会人有哪些？** (Who attended? Names or roles)
- **关键词有哪些？** (What are the key topics or keywords?)

If the language is obvious from the filename or context (e.g., "Nike专家" suggests an English expert call), make a reasonable assumption and confirm briefly rather than asking.

If the user provided a speaker alias mapping (e.g., "Speaker 1 = Joe"), collect those too via `--speaker-alias "1=Joe"`.

## Step 2 — Check file duration

Use the bundled Swift script to get the duration:

```bash
swift ~/.codex/skills/volcengine-transcribe/scripts/audio_duration.swift "<audio_file>"
```

- Use Doubao Seed ASR for both short and long files. There is no provider choice to make — do not run a preflight and do not ask which ASR to use.
- Use ElevenLabs (`transcribe_volcengine.py` / `transcribe_volcengine_full.py` with `TRANSCRIBE_PROVIDER=elevenlabs`) **only** if the user names ElevenLabs explicitly.

If the file is a `.qta` (Voice Memos), convert it to `.m4a` first — the volcengine-transcribe skill handles this automatically.

## Step 3 — Run transcription

Always `source ~/.zshrc` first so environment variables are loaded in the shell.

**Default Doubao path** (the script name is legacy; its ASR is Doubao `volc.bigasr.auc_turbo`):
```bash
source ~/.zshrc && python3 "$HOME/Library/Mobile Documents/com~apple~CloudDocs/Codex/.codex-bin/fish_minutes_pipeline.py" \
  "<audio_file>" \
  --language zh \
  --chunk-seconds 300 \
  --participant "<comma-separated participants>" \
  --keyword "<comma-separated keywords>" \
  --out-dir "$VOLCENGINE_OBSIDIAN_MEETING_MEMO_DIR"
```

**ElevenLabs — only when the user names it explicitly:**
```bash
source ~/.zshrc && TRANSCRIBE_PROVIDER=elevenlabs python3 ~/.codex/skills/volcengine-transcribe/scripts/transcribe_volcengine_full.py \
  "<audio_file>" \
  --language zh \
  --chunk-seconds 300 \
  --participant "<comma-separated participants>" \
  --keyword "<comma-separated keywords>" \
  --out-dir "$VOLCENGINE_OBSIDIAN_MEETING_MEMO_DIR"
```

Add `--speaker-alias "1=Name"` flags if the user provided speaker mappings. For chunked Doubao output, treat these as strong hints; the pipeline still reconciles labels across the full meeting because diarization IDs restart in each audio chunk.

The default pipeline must finish its built-in quality pass before the memo is considered complete:

- Build a real roster from self-introductions, direct forms of address, question/answer roles, and supplied participant hints. Never turn a third party merely discussed in the meeting into a participant.
- Assign every transcript turn to the constrained roster. Prefer `访谈者A` / `受访者` over inventing a name.
- Discover and review proper-noun corrections in context. Apply only exact-substring swaps and retain the audit list in `quality_report.json` under the hidden work directory.
- Produce exactly `会议概览 / 主要观点 / 关键事实与数据 / 风险与未决问题 / 待办事项`. Preserve figures, periods, denominators when stated, examples, causal chains, disagreements, and verification gaps. Attribute company/guest claims rather than presenting them as independently verified facts.
- Fail the memo step instead of writing a shallow/placeholder summary, losing transcript text, or leaving more than 25% of turns with unresolved generic speaker labels. ASR checkpoints remain reusable after such a failure.

### If transcription fails

The Doubao and ElevenLabs full CLIs save chunk checkpoints automatically. If the run fails partway through (e.g., network or DeepSeek timeout), re-run the exact same command — it will skip completed chunks and only redo what failed.

## Step 4 — Bilingual conversion (English meetings only)

Skip this step entirely for Chinese meetings — the memo is already in Chinese and needs no further processing.

For English meetings, once the `*_meeting-memo.md` file is generated, convert the **Polished Transcript** section to bilingual:

- Find the generated memo file in `$VOLCENGINE_OBSIDIAN_MEETING_MEMO_DIR`
- Replace generic labels `Speaker 0` / `Speaker 1` with real names if the user provided them (or infer from the transcript content)
- Add Chinese translation under each English speaker turn
- For any turns already in Chinese (sometimes the LLM switches mid-transcript), add English above them
- Keep everything before `## Polished Transcript` (summary, key decisions, etc.) unchanged
- Save in-place; keep a `.backup.md` copy of the original

Format for each turn:

```
**Speaker Name**

English text

中文翻译
```

## Step 4.5 — Post-process (cleanup + reconcile)

The pipeline prints `[quality]` and `[term-check]` lines. Require `integrity=passed`; inspect the detected speaker roster, speaker count, term-fix count, and any protected-term miss before reporting done. A protected term may legitimately be absent if it was not spoken, so check its context rather than blindly inserting it.

Then collapse the manual cleanup chores into one command (dry-run first, add
`--apply` once the listed deletions look right):

```bash
# Meeting Memo run:
python3 "$HOME/Library/Mobile Documents/com~apple~CloudDocs/Codex/.codex-bin/postprocess.py" --mode minute --quarter <YYYY-Qn>
# Podcast Transcript run (also reconciles _people/*.md episode_count):
python3 "$HOME/Library/Mobile Documents/com~apple~CloudDocs/Codex/.codex-bin/postprocess.py" --mode podcast
```

Deletions of synced-vault files need user confirmation — show the dry-run
output and ask before `--apply`.

## Step 5 — Report

Tell the user:
- The memo file path
- Whether bilingual conversion ran (English meeting) or was skipped (Chinese meeting)
- The number of speakers detected and whether speaker labels were applied
- That final `_result.json` and `_transcript.srt` files were cleaned up (unless `VOLCENGINE_KEEP_INTERMEDIATE_OUTPUTS=true`)
- Any `[term-check]` misses you had to fix
- Any caveats (e.g., "some turns in the second half were already in Chinese from the DeepSeek step; I added English above those")

## Environment variables (expected in ~/.zshrc)

| Variable | Purpose |
|---|---|
| `VOLCENGINE_SPEECH_API_KEY` | Doubao Seed ASR key — the only ASR key this pipeline needs |
| `VOLCENGINE_SPEECH_RESOURCE_ID` | Doubao resource id, default `volc.seedasr.auc`; the minutes pipeline pins `volc.bigasr.auc_turbo` |
| `HTTPS_PROXY` / `HTTP_PROXY` | Leave UNSET. Doubao and Ark are both reached directly; a global proxy would strangle Ark's long requests. |
| `TRANSCRIBE_PROVIDER` | `volcengine` (default). Set `elevenlabs` only when the user names ElevenLabs. |
| `ELEVENLABS_API_KEY` | ElevenLabs STT key; only used on explicit request |
| `ELEVENLABS_STT_MODEL` | `scribe_v2` |
| `ELEVENLABS_LANGUAGE_CODE` | `zh` |
| `ELEVENLABS_DIARIZE` | `true` |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `DEEPSEEK_BASE_URL` | DeepSeek API base URL, default `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | DeepSeek model, default `deepseek-v4-flash`; set `deepseek-v4-pro` only when quality matters more than cost |
| `DEEPSEEK_THINKING` | `enabled` by default for `deepseek-v4-*`; set `disabled` for faster non-reasoning output |
| `DEEPSEEK_REASONING_EFFORT` | `high` by default when thinking is enabled |
| `DEEPSEEK_REQUEST_TIMEOUT_SEC` | LLM request timeout, default `300` |
| `ARK_API_KEY` / `ARK_MODEL` / `ARK_BASE_URL` | Automatic fallback when direct DeepSeek fails (including HTTP 402); ASR checkpoints and LLM quality-cache are reused |
| `VOLCENGINE_OBSIDIAN_MEETING_MEMO_DIR` | Output folder in Obsidian |
| `VOLCENGINE_KEEP_INTERMEDIATE_OUTPUTS` | Optional debug flag; set `true` to keep final `_result.json` and `_transcript.srt` |
| `MINUTES_VOICEPRINT_PROFILE_DIR` | Optional override for the shared profile directory; default is private iCloud Drive under `Codex/Private/Voiceprints/profiles` |
| `MINUTES_VOICEPRINT_LOCAL_DIR` | Optional override for this Mac's local model/runtime root; default is `~/.codex/private/voiceprints` |

Never ask the user to paste these keys. If the new `DEEPSEEK_*` variables are missing, report which variable name is missing without printing secret values.

## CLI paths

```
~/.codex/skills/volcengine-transcribe/scripts/transcribe_volcengine.py      # single file
~/.codex/skills/volcengine-transcribe/scripts/transcribe_volcengine_full.py # chunked
~/.codex/skills/volcengine-transcribe/scripts/audio_duration.swift          # get duration
~/Library/Mobile Documents/com~apple~CloudDocs/Codex/.codex-bin/fish_minutes_pipeline.py  # DEFAULT: Doubao ASR + DeepSeek-on-Ark memo (name is legacy, no Fish inside)
~/Library/Mobile Documents/com~apple~CloudDocs/Codex/.codex-bin/minutes_quality.py  # global speakers + contextual terms + summary + integrity gates
~/Library/Mobile Documents/com~apple~CloudDocs/Codex/.codex-bin/podcast_bilingual_pipeline.py  # EN podcast -> bilingual md
~/Library/Mobile Documents/com~apple~CloudDocs/Codex/.codex-bin/postprocess.py    # byproduct cleanup + episode_count reconcile (run after)
# asr_preflight.py is DEPRECATED — it picked between ElevenLabs and Fish wallets. Doubao is now the only provider; do not run it.
```

## Output files

Saved to `$VOLCENGINE_OBSIDIAN_MEETING_MEMO_DIR` with a timestamp prefix:

| File | Contents |
|---|---|
| `<YYYY-Qn>/<YYYY-MM-DD memo title>.md` | Meeting memo (bilingual transcript for EN meetings; Chinese-only for ZH meetings) |

The memo title is inferred from the recording filename; if the recording name is generic (for example `Team Meeting`), the first few supplied keywords are appended. Normalize the final memo filename to exactly one leading recording date in `YYYY-MM-DD` form: strip every date/time prefix returned by the title model or inherited from the source name before composing the final path. By default, final `_result.json` and `_transcript.srt` files are deleted after the memo is written because the user does not use them. Keep hidden chunk checkpoint JSON files under `.volcengine-transcribe/` for long-recording resume support. If debugging requires final JSON/SRT outputs, set `VOLCENGINE_KEEP_INTERMEDIATE_OUTPUTS=true` before running.
