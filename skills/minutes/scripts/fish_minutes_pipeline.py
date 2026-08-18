#!/usr/bin/env python3
"""Transcribe a daily meeting with Doubao ASR, then polish with DeepSeek V4."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib import error, request

import minutes_quality


SKILL_DIR = Path.home() / ".codex" / "skills" / "volcengine-transcribe" / "scripts"
MAIN_SCRIPT = SKILL_DIR / "transcribe_volcengine.py"
VOICEPRINT_RUNTIME = Path.home() / ".codex" / "private" / "voiceprints" / "runtime" / "bin" / "python"
VOICEPRINT_SCRIPT = Path.home() / ".codex" / "skills" / "minutes" / "scripts" / "voiceprint.py"
DOUBAO_ASR_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
DOUBAO_RESOURCE_ID = "volc.bigasr.auc_turbo"

# Both Doubao ASR and Ark/DeepSeek use the existing Volcengine credentials.
# Requests are sent directly so a desktop proxy cannot truncate long uploads
# or long DeepSeek responses.

# Wiki names that are commonly spoken differently from their current page
# names. Most aliases are discovered from Wiki frontmatter; this small table
# covers legacy pages that do not yet have frontmatter aliases.
KNOWN_WIKI_ALIASES = {
    "Company/Fish": ["Fish Audio", "Fish", "冷月", "39AI", "39 AI"],
    "Company/Humanlaya": ["Humanlaya", "Human AI", "喜马拉雅", "Jasper"],
    "Company/Kimi": ["Kimi", "Moonshot", "月之暗面"],
    "Themes/AI语音": ["AI语音", "AI 语音", "TTS", "语音合成", "voice agent"],
    "Themes/Portfolio": ["Portfolio", "基金仓位", "资产配置", "pro-rata", "pro rata"],
    "Themes/短剧": ["短剧", "漫剧", "ReelShort", "DramaBox", "ShortMax"],
}
SAFE_SHORT_ALIASES = {"tts", "asr", "nlp", "gpu", "llm"}

# A title model is explicitly told not to return a date, but models and source
# filenames occasionally still do.  Consume every leading date (and an
# optional recording time) before composing the one canonical date prefix.
LEADING_DATE_PREFIX_RE = re.compile(
    r"""
    ^\s*[\[(（【]?\s*
    (?:
        20\d{2}[-_. /]?(?:0[1-9]|1[0-2])[-_. /]?(?:0[1-9]|[12]\d|3[01])
        |
        20\d{2}年(?:0?[1-9]|1[0-2])月(?:0?[1-9]|[12]\d|3[01])日
    )
    (?:
        [T _-]+
        (?!20\d{2}[-_. /]?(?:0[1-9]|1[0-2])[-_. /]?(?:0[1-9]|[12]\d|3[01]))
        (?:
            (?:[01]\d|2[0-3])[:._-][0-5]\d(?:[:._-][0-5]\d)?
            |
            (?:[01]\d|2[0-3])[0-5]\d(?:[0-5]\d)?(?!\d|[-_.]\d)
        )
    )?
    \s*[\])）】]?\s*[-_—–:：|·]*\s*[\[(（【]?\s*
    """,
    re.VERBOSE,
)


def strip_leading_date_prefixes(text: str) -> str:
    """Remove one or more date/time prefixes from a generated memo title."""
    value = text.strip()
    while value:
        cleaned = LEADING_DATE_PREFIX_RE.sub("", value, count=1).strip()
        if cleaned == value:
            break
        value = cleaned
    return value


class LocalMinutesModule:
    """Compatibility layer for the retired volcengine-transcribe helper.

    The meeting pipeline historically imported formatting and Ark helpers from
    that skill. Keeping these small functions local prevents a removed skill
    from breaking all future meeting memos.
    """

    @staticmethod
    def ms_to_clock(ms: int) -> str:
        total = max(0, int(ms)) // 1000
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"

    @staticmethod
    def transcript_plain_text(result: dict, speaker_prefix: str = "Speaker") -> str:
        lines = []
        for item in result.get("utterances", []):
            start = LocalMinutesModule.ms_to_clock(int(item.get("start_time", 0)))
            speaker = str(item.get("speaker", "0"))
            label = speaker if speaker.lower().startswith("speaker") else f"{speaker_prefix} {speaker}"
            lines.append(f"[{start}] {label}: {str(item.get('text', '')).strip()}")
        return "\n\n".join(x for x in lines if x.rstrip().endswith(":") is False)

    @staticmethod
    def render_srt(result: dict, speaker_prefix: str = "Speaker") -> str:
        def stamp(ms: int) -> str:
            ms = max(0, int(ms)); h, rem = divmod(ms, 3600000); m, rem = divmod(rem, 60000); s, milli = divmod(rem, 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"
        blocks = []
        for index, item in enumerate(result.get("utterances", []), 1):
            speaker = str(item.get("speaker", "0"))
            label = speaker if speaker.lower().startswith("speaker") else f"{speaker_prefix} {speaker}"
            blocks.append(f"{index}\n{stamp(item.get('start_time', 0))} --> {stamp(item.get('end_time', 1))}\n{label}: {item.get('text', '')}")
        return "\n\n".join(blocks) + "\n"

    @staticmethod
    def llm_chat(messages: list[dict]) -> str:
        api_key = os.environ.get("ARK_API_KEY")
        model = os.environ.get("ARK_MODEL")
        if not api_key or not model:
            raise RuntimeError("Missing ARK_API_KEY or ARK_MODEL")
        if "v4" not in model.casefold():
            raise RuntimeError("ARK_MODEL must point to DeepSeek V4 for the meeting memo workflow")
        base = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
        payload = json.dumps({"model": model, "messages": messages, "temperature": 0.2}).encode("utf-8")
        req = request.Request(
            f"{base}/chat/completions", data=payload, method="POST",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        # Ark should connect directly; see the network policy above.
        opener = request.build_opener(request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek V4/Ark HTTP {exc.code}: {detail[:500]}") from exc
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"Ark returned no choices: {str(body)[:500]}")
        return str(choices[0].get("message", {}).get("content", "")).strip()

    @staticmethod
    def final_output_paths(out_dir: Path, source_label: str, stamp: str, keywords: list[str]):
        date_match = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", source_label)
        date = "-".join(date_match.groups()) if date_match else stamp[:10]
        label = Path(source_label).name
        label = re.sub(r"\.(?:m4a|mp3|wav|qta|mp4|aac|caf|flac|ogg|opus)$", "", label, flags=re.IGNORECASE)
        label = strip_leading_date_prefixes(label)
        label = sanitize_title(label) or "待补充会议主题"
        base = f"{date} {label}"
        return out_dir / f"{base}_result.json", out_dir / f"{base}.md", out_dir / f"{base}_transcript.srt"

    @staticmethod
    def build_meeting_memo(source_label: str, stamp: str, polished: str, summary: str) -> str:
        date_match = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", source_label)
        date = "-".join(date_match.groups()) if date_match else stamp[:10]
        return (
            f"- Date: {date}\n- Original file: {source_label}\n\n"
            f"## Summary\n\n{summary.strip()}\n\n## Polished Transcript\n\n{polished.strip()}\n"
        )

    @staticmethod
    def cleanup_intermediate_outputs(paths: list[Path]) -> None:
        for path in paths:
            if path and path.exists():
                path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Doubao ASR + DeepSeek V4 meeting memo pipeline.")
    parser.add_argument("audio_file")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--chunk-seconds", type=int, default=900)
    parser.add_argument("--out-dir", default=os.environ.get("VOLCENGINE_OBSIDIAN_MEETING_MEMO_DIR", "."))
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--participant", action="append", default=[])
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument(
        "--speaker-alias",
        action="append",
        default=[],
        help="Optional raw speaker mapping such as '1=Joe' or 'Speaker 2=投资方'. Can be repeated.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--skip-memo", action="store_true")
    parser.add_argument("--request-timeout-sec", type=int, default=900)
    return parser.parse_args()


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def load_minutes_module():
    if not MAIN_SCRIPT.is_file():
        print(f"[compat] legacy helper missing; using local minutes adapter", flush=True)
        return LocalMinutesModule
    spec = importlib.util.spec_from_file_location("volc_transcribe", MAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_items(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        for part in re.split(r"[,，]", value):
            cleaned = part.strip()
            if cleaned:
                items.append(cleaned)
    return items


def parse_speaker_aliases(values: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --speaker-alias entry: {value!r}; expected LABEL=NAME")
        raw, name = value.split("=", 1)
        key = re.sub(r"^Speaker\s*", "", raw.strip(), flags=re.IGNORECASE)
        if not key or not name.strip():
            raise SystemExit(f"Invalid --speaker-alias entry: {value!r}; expected LABEL=NAME")
        aliases[key] = name.strip()
    return aliases


def safe_stem(path: Path) -> str:
    stem = re.sub(r"[^\w.-]+", "_", path.stem, flags=re.UNICODE).strip("._")
    return stem or "doubao_transcribe"


def duration_seconds(audio_file: Path) -> float:
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=noprint_wrappers=1:nokey=1", str(audio_file)]))


def match_local_voiceprints(audio_file: Path, utterances: list[dict]) -> tuple[dict[str, str], dict]:
    """Return high-confidence local voiceprint aliases without uploading audio."""
    if not VOICEPRINT_RUNTIME.is_file() or not VOICEPRINT_SCRIPT.is_file():
        return {}, {"status": "unavailable", "aliases": {}}
    command = [
        str(VOICEPRINT_RUNTIME), str(VOICEPRINT_SCRIPT), "match-utterances",
        "--audio", str(audio_file), "--utterances-json", "-",
    ]
    try:
        result = subprocess.run(
            command,
            input=json.dumps({"utterances": utterances}, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, {"status": "error", "error": str(exc), "aliases": {}}
    if result.returncode != 0:
        return {}, {
            "status": "error",
            "error": (result.stderr or result.stdout).strip()[:500],
            "aliases": {},
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}, {"status": "error", "error": "invalid voiceprint JSON", "aliases": {}}
    aliases = payload.get("aliases", {}) if isinstance(payload, dict) else {}
    if not isinstance(aliases, dict):
        aliases = {}
    return {str(key): str(value) for key, value in aliases.items()}, payload


def export_chunk(audio_file: Path, output_file: Path, start_sec: int, duration_sec: int) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Doubao flash accepts MP3/WAV/OGG, not M4A. Mono 32 kbps speech keeps each
    # 15-minute upload comfortably below the recommended 20 MB binary size.
    run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", str(start_sec),
         "-t", str(duration_sec), "-i", str(audio_file), "-vn", "-ac", "1",
         "-ar", "16000", "-b:a", "32k", str(output_file)])


def latest_doubao_json(dir_path: Path) -> Path | None:
    matches = sorted(dir_path.glob("*_doubao_result.json"))
    return matches[-1] if matches else None


def doubao_asr(chunk_file: Path, language: str, timeout_sec: int) -> dict:
    api_key = os.environ.get("VOLCENGINE_SPEECH_API_KEY")
    if not api_key:
        raise SystemExit("Missing required environment variable: VOLCENGINE_SPEECH_API_KEY")
    body = {
        "user": {"uid": "esther-meeting-memo"},
        "audio": {"data": base64.b64encode(chunk_file.read_bytes()).decode("ascii")},
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": True,
            "enable_ddc": True,
            "enable_speaker_info": True,
        },
    }
    req = request.Request(
        DOUBAO_ASR_URL,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": DOUBAO_RESOURCE_ID,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
        },
    )
    opener = request.build_opener(request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout_sec) as resp:
            status = resp.headers.get("X-Api-Status-Code", "")
            message = resp.headers.get("X-Api-Message", "")
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Doubao ASR HTTP {exc.code}: {raw[:500]}") from exc
    if status != "20000000":
        raise RuntimeError(f"Doubao ASR failed ({status} {message}): {raw[:500]}")
    parsed = json.loads(raw) if raw.strip() else {}
    if not isinstance(parsed, dict):
        raise RuntimeError("Doubao ASR returned a non-object JSON response.")
    return parsed


def result_to_utterances(
    body: dict,
    offset_ms: int,
    fallback_duration_sec: float,
    chunk_index: int | None = None,
) -> list[dict]:
    result = body.get("result") if isinstance(body.get("result"), dict) else body
    segments = result.get("utterances") or []
    utterances: list[dict] = []
    if isinstance(segments, list) and segments:
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            start = int(segment.get("start_time", 0)) + offset_ms
            end = int(segment.get("end_time", 0)) + offset_ms
            additions = segment.get("additions") if isinstance(segment.get("additions"), dict) else {}
            speaker = str(
                segment.get(
                    "speaker",
                    segment.get("speaker_id", additions.get("speaker", "0")),
                )
            )
            # Doubao diarization restarts for each exported audio chunk. Prefix
            # the local speaker id so later global reconciliation never assumes
            # that Speaker 1 in two different chunks is necessarily the same person.
            if chunk_index is not None and not re.match(r"^\d+-", speaker):
                speaker = f"{chunk_index + 1}-{speaker}"
            utterances.append({"start_time": start, "end_time": max(end, start + 1), "speaker": speaker, "text": text})
        return utterances

    text = str(result.get("text", "")).strip()
    if not text:
        return []
    duration_ms = body.get("audio_info", {}).get("duration") if isinstance(body.get("audio_info"), dict) else None
    duration_sec = (float(duration_ms) / 1000) if duration_ms else fallback_duration_sec
    return [
        {
            "start_time": offset_ms,
            "end_time": offset_ms + int(round(duration_sec * 1000)),
            "speaker": "0",
            "text": text,
        }
    ]


def chunk_text(text: str, max_chars: int = 12000) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            chunks.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def clean_model_output(text: str) -> str:
    cleaned_lines: list[str] = []
    skip_patterns = [
        r"^好的[，,].*(整理|记录|纪要|会议分析助手).*$",
        r"^以下是.*(整理稿|问答记录|访谈记录).*$",
        r"^###\s*访谈记录（第\s*\d+/\d+\s*段）$",
        r"^---+$",
    ]
    for line in text.strip().splitlines():
        stripped = line.strip()
        if any(re.match(pattern, stripped) for pattern in skip_patterns):
            continue
        cleaned_lines.append(line.rstrip())
    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def sanitize_title(text: str, max_chars: int = 48) -> str:
    """Turn a one-line model title into a safe, stable filename segment."""
    title = clean_model_output(text).splitlines()[0] if text.strip() else ""
    title = re.sub(r"^(标题|会议标题)\s*[:：]\s*", "", title.strip())
    title = title.strip("#*`'\"《》[]【】 ")
    title = strip_leading_date_prefixes(title)
    title = re.sub(r"[\\/:*?\"<>|]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip(" ._-")
    return title[:max_chars].rstrip(" ._-")


def generate_memo_title(module, summary: str, source_label: str) -> str:
    """Generate a semantic title after the model has understood the meeting."""
    prompt = (
        "请只输出一个会议文件标题，不要解释、不要 Markdown、不要日期。"
        "标题应让人仅看文件名就知道会议对象和核心内容，优先格式为："
        "公司/项目/人物 + 会议类型或核心议题。控制在 12-32 个中文字符；"
        "不要使用‘会议纪要’‘录音’‘讨论’等没有索引价值的泛词。\n\n"
        f"原录音名：{source_label}\n\n会议摘要：\n{summary[:12000]}"
    )
    try:
        raw = module.llm_chat(
            [
                {"role": "system", "content": "你是知识库文件命名编辑，输出必须只有一行标题。"},
                {"role": "user", "content": prompt},
            ]
        )
        title = sanitize_title(raw)
        if len(title) >= 4:
            return title
    except Exception as exc:
        print(f"[title] model generation failed, using fallback: {exc}", flush=True)

    # A useful failure mode: retain a compact piece of the summary instead of
    # silently falling all the way back to a timestamp-only recording name.
    plain = re.sub(r"[#*_>`\[\]]", "", summary)
    candidates = [re.sub(r"^[-\d.、\s]+", "", x).strip() for x in plain.splitlines()]
    candidates = [x for x in candidates if len(x) >= 8 and not x.startswith(("会议概览", "Summary"))]
    return sanitize_title(candidates[0] if candidates else source_label) or "待补充会议主题"


def find_vault_root(out_dir: Path) -> Path | None:
    for candidate in [out_dir, *out_dir.parents]:
        if (candidate / "Wiki").is_dir():
            return candidate
    return None


def parse_wiki_aliases(path: Path) -> list[str]:
    """Read simple inline or multiline aliases without requiring PyYAML."""
    text = path.read_text(encoding="utf-8", errors="replace")
    aliases = [path.stem]
    front = text.split("---", 2)[1] if text.startswith("---") and text.count("---") >= 2 else ""
    inline = re.search(r"^aliases:\s*\[([^]]*)\]", front, flags=re.MULTILINE)
    if inline:
        aliases.extend(x.strip(" '\"") for x in inline.group(1).split(","))
    else:
        block = re.search(r"^aliases:\s*\n((?:\s+-\s+.*\n?)*)", front, flags=re.MULTILINE)
        if block:
            aliases.extend(re.sub(r"^\s+-\s+", "", x).strip(" '\"") for x in block.group(1).splitlines())
    heading = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if heading:
        aliases.append(re.sub(r"\([^)]*\)", "", heading.group(1)).strip())
    return [x for x in dict.fromkeys(aliases) if x]


def primary_wiki_pages(vault: Path) -> list[Path]:
    """Return one canonical page per company plus all top-level theme pages."""
    company_dir = vault / "Wiki" / "Company"
    theme_dir = vault / "Wiki" / "Themes"
    pages = list(company_dir.glob("*.md")) + list(theme_dir.glob("*.md"))
    for folder in company_dir.iterdir() if company_dir.is_dir() else []:
        if not folder.is_dir():
            continue
        same_name = folder / f"{folder.name}.md"
        index = folder / "_Index.md"
        if same_name.is_file():
            pages.append(same_name)
        elif index.is_file():
            pages.append(index)
    return sorted(set(pages))


def wiki_backlinks(out_dir: Path, title: str, summary: str) -> list[str]:
    vault = find_vault_root(out_dir)
    if vault is None:
        return []
    haystack = f"{title}\n{summary}".casefold()
    links: list[str] = []
    for page in primary_wiki_pages(vault):
        rel = page.relative_to(vault / "Wiki").with_suffix("")
        rel_key = rel.as_posix()
        aliases = parse_wiki_aliases(page) + KNOWN_WIKI_ALIASES.get(rel_key, [])
        matched = False
        for alias in aliases:
            needle = alias.strip().casefold()
            # Short Latin abbreviations (AI, LI, APP, etc.) are too ambiguous.
            if not needle or (needle.isascii() and len(needle) < 4 and needle not in SAFE_SHORT_ALIASES):
                continue
            if needle in haystack:
                matched = True
                break
        if matched:
            links.append(f"[[Wiki/{rel_key}|{page.stem}]]")
    return links


def add_wiki_links(memo: str, links: list[str]) -> str:
    if not links:
        return memo
    block = "- Wiki: " + " · ".join(links) + "\n"
    marker = "\n## Summary"
    if marker in memo:
        return memo.replace(marker, "\n" + block + marker, 1)
    return block + "\n" + memo


def available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 100):
        candidate = path.with_name(f"{path.stem} ({index}){path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available output name for {path}")


def guidance(participants: list[str], keywords: list[str]) -> str:
    parts = []
    if participants:
        parts.append("参会人/角色提示：" + "、".join(participants))
    if keywords:
        parts.append("关键词提示：" + "、".join(keywords))
    locked = [t for t in participants + keywords if t]
    if locked:
        parts.append(
            "【专有名词保护】以下名称是已确认的人名/产品/术语，必须在输出中逐字原样保留，"
            "严禁改写、缩写、翻译或归一化成发音/拼写相近的其他词"
            "（例如绝不能把 OpenClaw 写成 OpenCL）：" + "、".join(locked)
        )
    return "\n".join(parts)


def verify_terms(label: str, text: str, terms: list[str]) -> None:
    """Automated replacement for the manual post-run keyword grep.

    DeepSeek occasionally normalizes a rare proper noun into a known
    confusable. A term may also legitimately be absent if it was never
    spoken, so this is a warning, not a failure — it just surfaces the
    list to eyeball instead of grepping by hand every run.
    """
    checked = [t for t in terms if t]
    if not checked:
        return
    missing = [t for t in checked if t not in text]
    if missing:
        print(
            f"[term-check] {label}: MISSING {len(missing)}/{len(checked)} -> "
            f"{', '.join(missing)} (可能被 DeepSeek 归一化，请人工核对/replace_all 修回)",
            flush=True,
        )
    else:
        print(f"[term-check] {label}: all {len(checked)} protected terms preserved", flush=True)


def polish_transcript(module, transcript: str, participants: list[str], keywords: list[str]) -> str:
    chunks = chunk_text(transcript)
    if not chunks:
        return ""
    ctx = guidance(participants, keywords)
    outputs: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        messages = [
            {
                "role": "system",
                "content": (
                    "你是专业会议纪要编辑。下面是豆包 ASR 生成的中文会议转写，可能没有可靠说话人标签。"
                    "请在不改变事实的前提下，把内容整理成适合阅读的问答式会议记录。"
                    "尽量根据上下文区分不同发言人；如果无法可靠判断，就使用通用 speaker 标签，不要强行猜姓名。"
                    "修正明显 ASR 错字、口语冗余和不通顺表达，"
                    "但不要新增原文没有的信息，不要把完整问答压缩得过短。输出 Markdown。"
                    "直接输出问答正文，不要写“以下是整理稿”“好的”等过程说明。"
                    f"\n{ctx}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"这是整场会议转写的第 {index}/{len(chunks)} 段。请只整理这一段，不要补写前后内容。\n\n"
                    f"{chunk}"
                ),
            },
        ]
        print(f"[deepseek] polish chunk {index}/{len(chunks)}", flush=True)
        outputs.append(clean_model_output(module.llm_chat(messages)))
    return clean_model_output("\n\n".join(part for part in outputs if part))


def summarize(module, polished: str, source_label: str, participants: list[str], keywords: list[str]) -> str:
    summary_input = polished
    chunks = chunk_text(polished, max_chars=22000)
    if len(chunks) > 2:
        partials: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            print(f"[deepseek] summarize part {index}/{len(chunks)}", flush=True)
            partials.append(
                module.llm_chat(
                    [
                        {
                            "role": "system",
                            "content": "你是资深会议分析助手。请基于片段提炼事实性要点，保留数据、例子、产品、商业模式和判断，不要编造。",
                        },
                        {
                            "role": "user",
                            "content": f"会议《{source_label}》片段 {index}/{len(chunks)}：\n\n{chunk}",
                        },
                    ]
                ).strip()
            )
        summary_input = "\n\n".join(partials)

    ctx = guidance(participants, keywords)
    print("[deepseek] final summary", flush=True)
    return module.llm_chat(
        [
            {
                "role": "system",
                "content": (
                    "你是资深会议分析助手。请生成详尽、结构化的中文会议纪要。"
                    "重点总结主要观点、关键事实、商业判断、风险、未决问题和可跟进事项。"
                    "语言直接、克制、像正式 memo。禁止编造。"
                    "直接输出 memo 正文，不要写“好的”“根据记录整理如下”等过程说明。"
                    f"\n{ctx}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"请为录音文件《{source_label}》生成会议纪要摘要。使用 Markdown，至少包含：\n"
                    "## 会议概览\n## 主要观点\n## 关键事实与数据\n## 风险与未决问题\n## 待办事项\n\n"
                    f"整理后的会议记录如下：\n{summary_input}"
                ),
            },
        ]
    )


def meeting_date(audio_file: Path, explicit_date: str | None) -> str:
    if explicit_date:
        try:
            return datetime.strptime(explicit_date, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError as exc:
            raise SystemExit("--date must use YYYY-MM-DD") from exc
    match = re.search(r"(20\d{2})[-_. ]?([01]\d)[-_. ]?([0-3]\d)", audio_file.stem)
    if match:
        return "-".join(match.groups())
    return datetime.now().strftime("%Y-%m-%d")


def build_quality_memo(
    title: str,
    date: str,
    original_file: str,
    roster: list[dict],
    summary: str,
    transcript: str,
) -> str:
    participants = "、".join(str(row.get("name", "")).strip() for row in roster if row.get("name"))
    return (
        f"# {title}\n\n"
        f"- Date: {date}\n"
        f"- Original file: {original_file}\n"
        f"- Participants: {participants or '未确认'}\n\n"
        f"## Summary\n\n{summary.strip()}\n\n"
        f"## Polished Transcript\n\n{transcript.strip()}\n"
    )


def main() -> int:
    args = parse_args()
    audio_file = Path(args.audio_file).expanduser().resolve()
    if not audio_file.exists():
        raise SystemExit(f"Audio file not found: {audio_file}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else out_dir / ".doubao-transcribe" / safe_stem(audio_file)
    work_dir.mkdir(parents=True, exist_ok=True)

    module = load_minutes_module()
    participants = split_items(args.participant)
    keywords = split_items(args.keyword)
    speaker_aliases = parse_speaker_aliases(args.speaker_alias)
    source_label = args.title or audio_file.name
    actual_date = meeting_date(audio_file, args.date)
    total_duration = duration_seconds(audio_file)
    total_chunks = int((total_duration + args.chunk_seconds - 1) // args.chunk_seconds)
    if args.max_chunks:
        total_chunks = min(total_chunks, args.max_chunks)
    print(f"[audio] duration={total_duration:.3f}s chunks={total_chunks} chunk_seconds={args.chunk_seconds}", flush=True)

    all_utterances: list[dict] = []
    raw_parts: list[str] = []
    doubao_results: list[dict] = []
    for idx in range(total_chunks):
        start_sec = idx * args.chunk_seconds
        chunk_duration = min(args.chunk_seconds, max(1, total_duration - start_sec))
        chunk_file = work_dir / f"chunk_{idx:03d}.mp3"
        chunk_out = work_dir / f"out_{idx:03d}"
        chunk_out.mkdir(parents=True, exist_ok=True)

        if not chunk_file.exists() or chunk_file.stat().st_size == 0:
            print(f"[chunk {idx + 1}/{total_chunks}] export {start_sec}s", flush=True)
            export_chunk(audio_file, chunk_file, start_sec, args.chunk_seconds)
        else:
            print(f"[chunk {idx + 1}/{total_chunks}] reuse audio", flush=True)

        doubao_json = latest_doubao_json(chunk_out)
        if doubao_json is None:
            for attempt in range(1, 4):
                try:
                    print(f"[chunk {idx + 1}/{total_chunks}] doubao asr attempt {attempt}", flush=True)
                    body = doubao_asr(chunk_file, args.language, args.request_timeout_sec)
                    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    doubao_json = chunk_out / f"{stamp}_doubao_result.json"
                    doubao_json.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    result_body = body.get("result") if isinstance(body.get("result"), dict) else body
                    (work_dir / f"chunk_{idx:03d}.txt").write_text(str(result_body.get("text", "")).strip() + "\n", encoding="utf-8")
                    break
                except Exception as exc:
                    if attempt == 3:
                        raise
                    print(f"[retry] chunk {idx + 1} failed: {exc}", flush=True)
                    time.sleep(3)
        else:
            print(f"[chunk {idx + 1}/{total_chunks}] reuse doubao result", flush=True)

        assert doubao_json is not None
        body = json.loads(doubao_json.read_text(encoding="utf-8"))
        doubao_results.append({"chunk": idx, "start_sec": start_sec, "result": body})
        result_body = body.get("result") if isinstance(body.get("result"), dict) else body
        text = str(result_body.get("text", "")).strip()
        if text:
            raw_parts.append(f"[{module.ms_to_clock(start_sec * 1000)}] {text}")
        utterances = result_to_utterances(body, start_sec * 1000, chunk_duration, idx)
        print(f"[chunk {idx + 1}/{total_chunks}] chars={len(text)} utterances={len(utterances)}", flush=True)
        all_utterances.extend(utterances)

    voiceprint_aliases, voiceprint_report = match_local_voiceprints(audio_file, all_utterances)
    # Explicit user mappings override biometric suggestions, including a
    # chunk-agnostic suffix such as "1=Name" matching raw label "3-1".
    for raw_label in list(voiceprint_aliases):
        suffix = raw_label.rsplit("-", 1)[-1]
        if raw_label in speaker_aliases or suffix in speaker_aliases:
            voiceprint_aliases.pop(raw_label)
    voiceprint_report["applied_aliases"] = dict(voiceprint_aliases)
    speaker_aliases = {**voiceprint_aliases, **speaker_aliases}
    matched_display = ", ".join(f"{label}={name}" for label, name in voiceprint_aliases.items())
    print(
        f"[voiceprint] status={voiceprint_report.get('status', 'unknown')} "
        f"matches={matched_display or 'none'}",
        flush=True,
    )

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"-{uuid.uuid4().hex[:6]}"
    result_obj = {
        "provider": "doubao",
        "source": audio_file.name,
        "result": {"utterances": all_utterances},
        "doubao_chunks": doubao_results,
        "voiceprint": voiceprint_report,
    }
    raw_transcript = "\n\n".join(raw_parts).strip()

    if args.skip_memo:
        result_path = out_dir / f"{stamp}_result.json"
        srt_path = out_dir / f"{stamp}_transcript.srt"
        memo_path = None
    else:
        result_path, memo_path, srt_path = module.final_output_paths(out_dir, source_label, stamp, keywords)
    raw_path = work_dir / "raw_transcript.txt"
    result_path.write_text(json.dumps(result_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    srt_path.write_text(module.render_srt({"utterances": all_utterances}, "Speaker"), encoding="utf-8")
    raw_path.write_text(raw_transcript + "\n", encoding="utf-8")
    print(f"[write] result={result_path}", flush=True)
    print(f"[write] srt={srt_path}", flush=True)
    print(f"[write] raw={raw_path}", flush=True)

    if args.skip_memo:
        return 0

    print("[quality] global speaker reconciliation + contextual term correction + summary", flush=True)
    quality = minutes_quality.reconcile_and_summarize(
        module,
        all_utterances,
        title=source_label,
        participants=participants,
        keywords=keywords,
        speaker_aliases=speaker_aliases,
        work_dir=work_dir,
    )
    polished = quality["transcript"]
    summary = quality["summary"]
    roster = quality["roster"]
    quality_report_path = work_dir / "quality_report.json"
    quality["report"]["voiceprint"] = voiceprint_report
    quality_report_path.write_text(
        json.dumps(quality["report"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    verify_terms("polished", polished, participants + keywords + list(speaker_aliases.values()))
    verify_terms("summary", summary, participants + keywords)
    semantic_title = generate_memo_title(module, summary, source_label)
    links = wiki_backlinks(out_dir, semantic_title, summary)
    dated_semantic_label = f"{actual_date} {semantic_title}"
    _, semantic_memo_path, _ = module.final_output_paths(out_dir, dated_semantic_label, stamp, keywords)
    memo_path = available_path(semantic_memo_path)
    print(f"[title] {semantic_title}", flush=True)
    print(f"[wiki] {', '.join(links) if links else 'no confident match'}", flush=True)
    polished_path = work_dir / "polished.md"
    summary_path = work_dir / "summary.md"
    polished_path.write_text(polished.strip() + "\n", encoding="utf-8")
    summary_path.write_text(summary.strip() + "\n", encoding="utf-8")
    memo = build_quality_memo(
        semantic_title,
        actual_date,
        audio_file.name,
        roster,
        summary,
        polished,
    )
    memo_path.write_text(add_wiki_links(memo, links), encoding="utf-8")
    module.cleanup_intermediate_outputs([result_path, srt_path])
    print(f"[write] polished={polished_path}", flush=True)
    print(f"[write] summary={summary_path}", flush=True)
    print(f"[write] quality_report={quality_report_path}", flush=True)
    print(
        f"[quality] speakers={len(roster)} term_fixes={len(quality['report']['term_changes'])} "
        f"integrity=passed",
        flush=True,
    )
    print(f"[write] memo={memo_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
