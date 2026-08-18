#!/usr/bin/env python3
"""Preservation-first quality pass for daily meeting memos.

The ASR transcript is treated as the source of truth. The LLM may classify
speakers, propose exact proper-noun substitutions, and summarize, but it may
not freely rewrite transcript wording. Every accepted transcript change is
recorded and replay-verified before a memo can be written.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


REQUIRED_SUMMARY_HEADINGS = (
    "### 会议概览",
    "### 主要观点",
    "### 关键事实与数据",
    "### 风险与未决问题",
    "### 待办事项",
)

PLACEHOLDER_PATTERNS = (
    "本录音包含约",
    "请根据实际内容补充",
    "待补充会议",
    "无法生成摘要",
    "未提供具体会议内容",
)

# These are candidates, not blanket replacements. Every occurrence is still
# judged in its own context before an exact-substring swap is accepted.
COMMON_TERM_RULES = (
    ("cloud code", "Claude Code", "Anthropic coding product; keep cloud computing uses unchanged"),
    ("cloudcode", "Claude Code", "joined ASR rendering of Claude Code"),
    ("cloudco", "Claude Code", "truncated ASR rendering of Claude Code"),
    ("cloud", "Claude", "only when referring to Anthropic/Claude, never generic cloud computing"),
    ("Cloud", "Claude", "only when referring to Anthropic/Claude, never generic cloud computing"),
    ("克劳德", "Claude", "Chinese phonetic rendering of Claude"),
    ("OPPO", "Opus", "only in Anthropic model context"),
    ("OPEX", "Opus", "only in model context; keep operating-expense uses unchanged"),
    ("OPS", "Opus", "only in Anthropic model context"),
    ("ops", "Opus", "only in Anthropic model context"),
    ("欧克斯", "Opus", "phonetic ASR rendering in Anthropic model context"),
    ("HiQ", "Haiku", "Anthropic Haiku model"),
    ("Hi Q", "Haiku", "Anthropic Haiku model"),
    ("海酷", "Haiku", "phonetic ASR rendering of Haiku"),
    ("SOL", "Sonnet", "only in Anthropic model context"),
    ("斐波那契", "Fable", "only when the conversation is about Anthropic model names"),
    ("斐波", "Fable", "only when the conversation is about Anthropic model names"),
    ("Fibonacci", "Fable", "only when the conversation is about Anthropic model names"),
    ("CDEX", "Codex", "OpenAI Codex product"),
    ("co- tokens", "code tokens", "ASR split in token-usage context"),
    ("kersa", "Cursor", "only in AI coding product context"),
    ("Workbody", "WorkBuddy", "Tencent WorkBuddy product, not CodeBuddy"),
    ("workbody", "WorkBuddy", "Tencent WorkBuddy product, not CodeBuddy"),
    ("Trade", "Trae", "only for ByteDance AI coding product"),
    ("tree", "Trae", "only for ByteDance AI coding product"),
)


class QualityError(RuntimeError):
    """Raised when a memo cannot safely pass the quality gate."""


def _json_from_text(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.rsplit("```", 1)[0] if "```" in text else text
    text = text.strip()
    starts = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
    if starts:
        text = text[min(starts):]
    end = max(text.rfind("}"), text.rfind("]"))
    if end >= 0:
        text = text[: end + 1]
    return json.loads(text)


def _cached_chat(module: Any, messages: list[dict[str, str]], cache_dir: Path, tag: str) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "model": os.environ.get("DEEPSEEK_MODEL") or os.environ.get("ARK_MODEL") or "unknown",
        "messages": messages,
        "version": 3,
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:32]
    path = cache_dir / f"{tag}-{digest}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload["content"])
    content = str(module.llm_chat(messages)).strip()
    if not content:
        raise QualityError(f"LLM returned empty content for {tag}")
    path.write_text(
        json.dumps({"tag": tag, "content": content}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return content


def _cached_json(module: Any, messages: list[dict[str, str]], cache_dir: Path, tag: str) -> Any:
    raw = _cached_chat(module, messages, cache_dir, tag)
    try:
        return _json_from_text(raw)
    except json.JSONDecodeError as exc:
        raise QualityError(f"LLM returned invalid JSON for {tag}: {raw[:300]}") from exc


def _split_blocks(lines: list[str], max_chars: int = 18000) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        line_size = len(line) + 2
        if current and size + line_size > max_chars:
            blocks.append("\n\n".join(current))
            current, size = [], 0
        current.append(line)
        size += line_size
    if current:
        blocks.append("\n\n".join(current))
    return blocks


def _normalize_alias_key(value: str) -> str:
    cleaned = re.sub(r"^Speaker\s*", "", value.strip(), flags=re.IGNORECASE)
    return cleaned


def _alias_for_label(label: str, aliases: dict[str, str]) -> str | None:
    raw = _normalize_alias_key(label)
    if raw in aliases:
        return aliases[raw]
    suffix = raw.rsplit("-", 1)[-1]
    return aliases.get(suffix)


def _prepare_turns(utterances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for item in utterances:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        label = str(item.get("speaker", "0")).strip() or "0"
        if turns and turns[-1]["label"] == label:
            turns[-1]["text"] = turns[-1]["text"].rstrip() + " " + text.lstrip()
            turns[-1]["end_time"] = int(item.get("end_time", turns[-1]["end_time"]))
        else:
            turns.append(
                {
                    "label": label,
                    "text": text,
                    "start_time": int(item.get("start_time", 0)),
                    "end_time": int(item.get("end_time", 0)),
                }
            )
    if not turns:
        raise QualityError("ASR returned no non-empty utterances")
    source_text = re.sub(
        r"\s+",
        " ",
        " ".join(str(item.get("text", "")).strip() for item in utterances if str(item.get("text", "")).strip()),
    ).strip()
    grouped_text = re.sub(r"\s+", " ", " ".join(turn["text"] for turn in turns)).strip()
    if source_text != grouped_text:
        raise QualityError("Transcript integrity check failed while grouping raw ASR utterances")
    return turns


def _sample_turns(turns: list[dict[str, Any]], limit: int = 80) -> list[tuple[int, dict[str, Any]]]:
    if len(turns) <= limit:
        return list(enumerate(turns))
    important: set[int] = set()
    self_intro = re.compile(
        r"(我叫|我的名字|my name is|I am |I'm |我是.{0,16}(创始人|负责人|工程师|合伙人|投资))",
        flags=re.IGNORECASE,
    )
    for index, turn in enumerate(turns):
        if self_intro.search(turn["text"]):
            important.add(index)
    step = max(1, len(turns) // max(1, limit - len(important)))
    important.update(range(0, len(turns), step))
    return [(i, turns[i]) for i in sorted(important)[:limit]]


def _discover_roster(
    module: Any,
    turns: list[dict[str, Any]],
    title: str,
    participants: list[str],
    keywords: list[str],
    aliases: dict[str, str],
    cache_dir: Path,
) -> list[dict[str, str]]:
    samples = _sample_turns(turns)
    lines = [f"[{i}] (原标签 {t['label']}) {t['text'][:320]}" for i, t in samples]
    confirmed = list(dict.fromkeys([*participants, *aliases.values()]))
    reply = _cached_json(
        module,
        [
            {
                "role": "system",
                "content": (
                    "你是会议说话人识别编辑。ASR 按音频分块后会重置说话人编号，"
                    "同一个人可能对应很多原标签。根据自我介绍、被称呼方式、问答关系和业务经历，"
                    "判断真实参会人。被讨论的第三方人物绝不是参会人。只输出 JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"会议标题：{title}\n"
                    f"已知参会人/角色（可能为空）：{'、'.join(confirmed) or '无'}\n"
                    f"关键词：{'、'.join(keywords) or '无'}\n\n"
                    "以下是全场抽样发言：\n\n" + "\n".join(lines) +
                    "\n\n输出：{\"roster\":[{\"name\":\"简短标签\",\"role\":\"访谈者/受访者/公司方/其他\","
                    "\"evidence\":\"识别依据\"}]}。人数要克制，通常 2-6 人。"
                    "只有对话明确支持时才用真实姓名；否则用『访谈者A』『受访者』等角色标签。"
                    "不要把对话中提到的创始人、客户、同事或名人列为参会人。"
                ),
            },
        ],
        cache_dir,
        "roster",
    )
    payload = reply.get("roster", []) if isinstance(reply, dict) else []
    roster: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if not name or name in seen or len(name) > 32:
            continue
        roster.append(
            {
                "name": name,
                "role": str(row.get("role", "")).strip(),
                "evidence": str(row.get("evidence", row.get("desc", ""))).strip(),
            }
        )
        seen.add(name)
    for name in confirmed:
        if name and name not in seen and len(name) <= 32:
            roster.append({"name": name, "role": "已知参会人/角色", "evidence": "用户提供"})
            seen.add(name)
    if not 1 <= len(roster) <= 10:
        raise QualityError(f"Unreliable speaker roster size: {len(roster)}")
    return roster


def _assign_speakers(
    module: Any,
    turns: list[dict[str, Any]],
    roster: list[dict[str, str]],
    title: str,
    aliases: dict[str, str],
    cache_dir: Path,
    window: int = 28,
) -> list[str]:
    names = [row["name"] for row in roster]
    roster_text = "\n".join(
        f"- {row['name']}（{row.get('role', '')}）：{row.get('evidence', '')}" for row in roster
    )
    assigned = [""] * len(turns)
    label_votes: dict[str, Counter[str]] = defaultdict(Counter)
    for index, turn in enumerate(turns):
        explicit = _alias_for_label(turn["label"], aliases)
        if explicit:
            assigned[index] = explicit
            label_votes[turn["label"]][explicit] += 1

    for start in range(0, len(turns), window):
        end = min(start + window, len(turns))
        pending = [i for i in range(start, end) if not assigned[i]]
        if not pending:
            continue
        context = [
            f"[已定 {i}] {assigned[i]}：{turns[i]['text'][:180]}"
            for i in range(max(0, start - 5), start)
            if assigned[i]
        ]
        current = [
            f"[{i}] (原标签 {turns[i]['label']}) {turns[i]['text'][:700]}" for i in pending
        ]
        reply = _cached_json(
            module,
            [
                {
                    "role": "system",
                    "content": (
                        "你逐条判断会议转录的说话人。原标签形如 分块-段内编号；同一分块内相同标签"
                        "通常是同一人，但跨分块编号不能直接类比。必须依据发言内容、称呼、问答关系"
                        "和前后文判断。名单之外的人不能新造。只输出 JSON。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"会议：{title}\n\n只能从以下名单选择：\n{roster_text}\n\n"
                        + (("上文：\n" + "\n".join(context) + "\n\n") if context else "")
                        + "待判断发言：\n" + "\n".join(current)
                        + "\n\n输出 {\"a\":{\"序号\":\"名单中的说话人\"}}，覆盖每个待判断序号。"
                        "提问和串场通常属于访谈方；长篇讲述本人经历、业务、数据通常属于受访方。"
                    ),
                },
            ],
            cache_dir,
            f"speaker-{start:05d}",
        )
        mapping = reply.get("a", {}) if isinstance(reply, dict) else {}
        for i in pending:
            value = str(mapping.get(str(i), "")).strip()
            if value not in names:
                prior = label_votes[turns[i]["label"]].most_common(1)
                value = prior[0][0] if prior else ""
            if value not in names:
                raise QualityError(f"Speaker assignment missing at turn {i}")
            assigned[i] = value
            label_votes[turns[i]["label"]][value] += 1
    return assigned


def _discover_term_rules(
    module: Any,
    turns: list[dict[str, Any]],
    title: str,
    protected_terms: list[str],
    cache_dir: Path,
) -> list[tuple[str, str, str]]:
    lines = [turn["text"] for turn in turns]
    blocks = _split_blocks(lines, max_chars=18000)
    discovered: list[tuple[str, str, str]] = []
    for index, block in enumerate(blocks, 1):
        reply = _cached_json(
            module,
            [
                {
                    "role": "system",
                    "content": (
                        "你只负责发现 ASR 中高度确定的专有名词误听，包括人名、公司、产品、模型和技术术语。"
                        "不要润色句子，不要修正常规口语，不要猜不确定的词。只输出 JSON。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"会议：{title}\n已确认专名：{'、'.join(protected_terms) or '无'}\n\n"
                        f"转录片段 {index}/{len(blocks)}：\n{block}\n\n"
                        "找出片段中高度确定的专名误听。输出 {\"rules\":[{\"wrong\":\"原文精确子串\","
                        "\"right\":\"正确写法\",\"confidence\":\"high/medium/low\",\"reason\":\"依据\"}]}。"
                        "只给 high；若无则 rules 为空。wrong 必须是片段里实际出现的连续原文，right 只写替换词。"
                    ),
                },
            ],
            cache_dir,
            f"term-discovery-{index:03d}",
        )
        rows = reply.get("rules", []) if isinstance(reply, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            wrong = str(row.get("wrong", "")).strip()
            right = str(row.get("right", "")).strip()
            confidence = str(row.get("confidence", "")).strip().lower()
            reason = str(row.get("reason", "")).strip()
            if confidence != "high" or not wrong or not right or wrong == right:
                continue
            if len(wrong) > 80 or len(right) > 80 or "\n" in wrong or "\n" in right:
                continue
            if len(wrong) < 2 and wrong not in {"K3", "R1", "o3"}:
                continue
            if wrong not in block:
                continue
            discovered.append((wrong, right, reason or "LLM high-confidence term discovery"))
    return list(dict.fromkeys(discovered))


def _occurrence_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    if re.fullmatch(r"[A-Za-z0-9_.+-]+", term):
        return re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])")
    return re.compile(escaped)


def _correct_terms(
    module: Any,
    turns: list[dict[str, Any]],
    title: str,
    protected_terms: list[str],
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, str, str]]]:
    rules = list(COMMON_TERM_RULES)
    rules.extend(_discover_term_rules(module, turns, title, protected_terms, cache_dir))
    rules = list(dict.fromkeys(rules))
    sites: list[dict[str, Any]] = []
    occupied: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for wrong, right, reason in sorted(rules, key=lambda row: len(row[0]), reverse=True):
        pattern = _occurrence_pattern(wrong)
        for turn_index, turn in enumerate(turns):
            for match in pattern.finditer(turn["text"]):
                start, end = match.span()
                if any(start < other_end and end > other_start for other_start, other_end in occupied[turn_index]):
                    continue
                occupied[turn_index].append((start, end))
                sites.append(
                    {
                        "id": len(sites),
                        "turn": turn_index,
                        "start": start,
                        "end": end,
                        "wrong": match.group(0),
                        "right": right,
                        "reason": reason,
                        "context": (
                            turn["text"][max(0, start - 80):start]
                            + "《" + match.group(0) + "》"
                            + turn["text"][end:end + 80]
                        ),
                    }
                )
    accepted: set[int] = set()
    for batch_start in range(0, len(sites), 50):
        group = sites[batch_start: batch_start + 50]
        if not group:
            continue
        payload = "\n".join(
            f"{site['id']}\t{site['wrong']}→{site['right']}\t{site['reason']}\t{site['context']}"
            for site in group
        )
        reply = _cached_json(
            module,
            [
                {
                    "role": "system",
                    "content": (
                        "你校对 ASR 专名。每行是一处候选替换，必须逐处结合《》前后文判断。"
                        "只决定是否接受给定替换，不能改写句子、不能提出第三种写法。拿不准就拒绝。只输出 JSON。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"会议：{title}\n已确认专名：{'、'.join(protected_terms) or '无'}\n\n"
                        "格式：编号<TAB>候选替换<TAB>依据<TAB>上下文\n" + payload
                        + "\n\n输出 {\"accept\":[编号,...]}，只列语境明确支持的替换。"
                    ),
                },
            ],
            cache_dir,
            f"term-decisions-{batch_start:05d}",
        )
        values = reply.get("accept", []) if isinstance(reply, dict) else []
        for value in values:
            try:
                accepted.add(int(value))
            except (TypeError, ValueError):
                continue

    output = deepcopy(turns)
    changes: list[dict[str, Any]] = []
    by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for site in sites:
        if site["id"] in accepted and site["wrong"] != site["right"]:
            by_turn[site["turn"]].append(site)
    for turn_index, group in by_turn.items():
        text = output[turn_index]["text"]
        for site in sorted(group, key=lambda row: row["start"], reverse=True):
            if text[site["start"]:site["end"]] != site["wrong"]:
                raise QualityError(f"Term offsets shifted at turn {turn_index}")
            text = text[:site["start"]] + site["right"] + text[site["end"]:]
            changes.append(dict(site))
        output[turn_index]["text"] = text

    # Replay all accepted exact swaps from the original and prove that the term
    # stage made no other transcript edits.
    replay = [turn["text"] for turn in turns]
    for turn_index, group in by_turn.items():
        text = replay[turn_index]
        for site in sorted(group, key=lambda row: row["start"], reverse=True):
            text = text[:site["start"]] + site["right"] + text[site["end"]:]
        replay[turn_index] = text
    if replay != [turn["text"] for turn in output]:
        raise QualityError("Transcript integrity check failed after term correction")
    return output, changes, rules


def _merge_turns(turns: list[dict[str, Any]], speakers: list[str]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for turn, speaker in zip(turns, speakers):
        if merged and merged[-1]["speaker"] == speaker:
            merged[-1]["text"] = merged[-1]["text"].rstrip() + " " + turn["text"].lstrip()
            merged[-1]["end_time"] = turn["end_time"]
        else:
            merged.append(
                {
                    "speaker": speaker,
                    "text": turn["text"],
                    "start_time": turn["start_time"],
                    "end_time": turn["end_time"],
                }
            )
    before = re.sub(r"\s+", " ", " ".join(turn["text"] for turn in turns)).strip()
    after = re.sub(r"\s+", " ", " ".join(turn["text"] for turn in merged)).strip()
    if before != after:
        raise QualityError("Transcript integrity check failed while merging adjacent turns")
    return merged


def _normalize_summary(summary: str) -> str:
    text = summary.strip()
    text = re.sub(r"^##\s+(会议概览|主要观点|关键事实与数据|风险与未决问题|待办事项)\s*$", r"### \1", text, flags=re.MULTILINE)
    text = re.sub(r"^####+\s+(会议概览|主要观点|关键事实与数据|风险与未决问题|待办事项)\s*$", r"### \1", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _summary_problems(summary: str, transcript_chars: int) -> list[str]:
    problems: list[str] = []
    for heading in REQUIRED_SUMMARY_HEADINGS:
        if heading not in summary:
            problems.append(f"missing heading: {heading}")
    for pattern in PLACEHOLDER_PATTERNS:
        if pattern in summary:
            problems.append(f"placeholder text: {pattern}")
    minimum = 300 if transcript_chars < 5000 else 650 if transcript_chars < 20000 else 1200
    if len(summary) < minimum:
        problems.append(f"summary too short: {len(summary)} < {minimum}")
    if summary.count("### ") != len(REQUIRED_SUMMARY_HEADINGS):
        problems.append("summary must contain exactly the five required sections")
    return problems


def _summarize(
    module: Any,
    merged: list[dict[str, Any]],
    title: str,
    roster: list[dict[str, str]],
    keywords: list[str],
    cache_dir: Path,
) -> tuple[str, list[str]]:
    lines = [f"**{turn['speaker']}：** {turn['text']}" for turn in merged]
    blocks = _split_blocks(lines, max_chars=18000)
    roster_text = "、".join(f"{row['name']}（{row.get('role', '')}）" for row in roster)
    evidence: list[str] = []
    for index, block in enumerate(blocks, 1):
        evidence.append(
            _cached_chat(
                module,
                [
                    {
                        "role": "system",
                        "content": (
                            "你为投资人制作会议事实卡。按发言人归属提取该片段中的观点、数字（含主体/口径/期间）、"
                            "案例、因果链、分歧、明确待办与不确定信息。区分会议陈述、推测和待核事实。"
                            "保留专名与原数字，禁止编造，禁止写空泛总结。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"会议《{title}》片段 {index}/{len(blocks)}，参会人：{roster_text}\n\n{block}",
                    },
                ],
                cache_dir,
                f"summary-evidence-{index:03d}",
            ).strip()
        )
    evidence_text = "\n\n".join(evidence)
    prompt = (
        f"请为《{title}》生成可直接入库的中文会议摘要。参会人：{roster_text}。"
        f"关键词：{'、'.join(keywords) or '无'}。\n\n"
        "必须且只能使用以下五个三级标题：\n"
        + "\n".join(REQUIRED_SUMMARY_HEADINGS)
        + "\n\n要求：\n"
        "1. 会议概览交代对象、目的、角色与背景；主要观点保留论点、证据、反方或条件；\n"
        "2. 关键事实与数据逐条写清主体、数值、期间/口径（若会议未披露口径就明确写未披露）；\n"
        "3. 把公司方/受访者的表述写成『其称/其判断』，不要升级成外部已证实事实；\n"
        "4. 保留具体案例、产品名、商业模式、因果链、分歧和仍需核实的专名/数字；\n"
        "5. 待办事项只写会议明确形成或自然需要跟进的事项；没有明确待办就写『会议未形成明确待办』；\n"
        "6. 语言直接、克制、信息密度高，不写过程说明，不编造。\n\n"
        f"分片事实卡如下：\n{evidence_text}"
    )
    summary = _normalize_summary(
        _cached_chat(
            module,
            [
                {"role": "system", "content": "你是资深投资研究会议纪要编辑。输出正式 Markdown memo 摘要。"},
                {"role": "user", "content": prompt},
            ],
            cache_dir,
            "summary-final",
        )
    )
    problems = _summary_problems(summary, sum(len(turn["text"]) for turn in merged))
    if problems:
        summary = _normalize_summary(
            _cached_chat(
                module,
                [
                    {
                        "role": "system",
                        "content": "你修复会议摘要的结构和信息缺口。不得新增事实，只输出修复后的完整摘要。",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"原摘要存在问题：{'；'.join(problems)}\n\n"
                            f"必须使用五个指定三级标题。原摘要：\n{summary}\n\n事实卡：\n{evidence_text}"
                        ),
                    },
                ],
                cache_dir,
                "summary-repair",
            )
        )
        problems = _summary_problems(summary, sum(len(turn["text"]) for turn in merged))
    if problems:
        raise QualityError("Summary quality gate failed: " + "; ".join(problems))
    return summary, evidence


def _render_transcript(merged: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"**{turn['speaker']}：** {turn['text']}" for turn in merged).strip()


def reconcile_and_summarize(
    module: Any,
    utterances: list[dict[str, Any]],
    *,
    title: str,
    participants: list[str],
    keywords: list[str],
    speaker_aliases: dict[str, str],
    work_dir: Path,
) -> dict[str, Any]:
    """Return a verified transcript, summary, roster, and audit report."""
    cache_dir = work_dir / "quality-cache"
    turns = _prepare_turns(utterances)
    roster = _discover_roster(
        module, turns, title, participants, keywords, speaker_aliases, cache_dir
    )
    speakers = _assign_speakers(
        module, turns, roster, title, speaker_aliases, cache_dir
    )
    protected_terms = list(dict.fromkeys([*participants, *keywords, *speaker_aliases.values()]))
    corrected, changes, discovered_rules = _correct_terms(
        module, turns, title, protected_terms, cache_dir
    )
    merged = _merge_turns(corrected, speakers)
    summary, evidence = _summarize(module, merged, title, roster, keywords, cache_dir)
    transcript = _render_transcript(merged)

    generic = sum(1 for name in speakers if re.search(r"^(Speaker|说话人|未确认)", name, re.I))
    if generic / max(1, len(speakers)) > 0.25:
        raise QualityError(f"Too many unresolved speaker turns: {generic}/{len(speakers)}")
    report = {
        "quality_version": 3,
        "utterances_in": len(utterances),
        "turns_in": len(turns),
        "raw_labels": sorted({turn["label"] for turn in turns}),
        "turns_out": len(merged),
        "roster": roster,
        "speaker_counts": Counter(speakers).most_common(),
        "unresolved_speaker_turns": generic,
        "term_changes": changes,
        "term_rules_considered": [
            {"wrong": wrong, "right": right, "reason": reason}
            for wrong, right, reason in discovered_rules
        ],
        "summary_chars": len(summary),
        "summary_evidence_blocks": len(evidence),
        "integrity": "exact transcript text preserved except audited exact-substring term replacements",
    }
    return {
        "transcript": transcript,
        "summary": summary,
        "roster": roster,
        "merged_turns": merged,
        "report": report,
    }
