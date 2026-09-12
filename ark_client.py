"""Ark JSON chat client. Configuration/auth/billing errors must fail the job."""

import json
import os
import re
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class ArkError(RuntimeError):
    pass


def settings():
    key = os.environ.get("ARK_API_KEY", "").strip()
    model = os.environ.get("ARK_MODEL", "").strip()
    base = os.environ.get("ARK_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    if not key or not model:
        raise ArkError("ARK_API_KEY and ARK_MODEL are required; no fallback to DeepSeek direct API")
    if base != DEFAULT_BASE_URL:
        raise ArkError("ARK_BASE_URL must be the Ark Beijing inference endpoint")
    return key, model, base


def chat_json(messages, *, temperature=0.2, max_tokens=1600):
    key, model, base = settings()
    # Ark's V4 models do not advertise response_format=json_object support.
    # Ask for JSON in the prompt and validate the returned object ourselves.
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "stream": False,
    }).encode("utf-8")
    for attempt in range(3):
        request = urllib.request.Request(
            base + "/chat/completions", data=payload, method="POST",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.load(response)
            choice = data["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ArkError("Ark response was truncated; refusing incomplete JSON")
            content = choice["message"]["content"]
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
            result = json.loads(cleaned)
            if not isinstance(result, dict):
                raise ArkError("Ark response must be a JSON object")
            return result
        except urllib.error.HTTPError as exc:
            # Never log the request, Authorization header, or full response body.
            code = ""
            try:
                error = json.loads(exc.read()).get("error", {})
                code = str(error.get("code", ""))
            except (ValueError, AttributeError):
                pass
            detail = {401: "invalid API key", 402: "insufficient balance/payment required",
                      403: "access denied/model not enabled", 404: "model or endpoint not found"}.get(exc.code, "request rejected")
            message = f"Ark HTTP {exc.code}: {detail} ({code.replace(key, '[redacted]')[:120]})"
            if exc.code != 429 and exc.code < 500:
                raise ArkError(message) from None
            if attempt == 2:
                raise ArkError(message + "; retries exhausted") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 2:
                raise ArkError(f"Ark network error ({type(exc).__name__}); retries exhausted") from None
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise ArkError("Ark returned invalid JSON or an invalid chat response") from None
        time.sleep(2 ** attempt)


def preflight():
    _, model, base = settings()
    result = chat_json([{"role": "user", "content": 'Return only this JSON object: {"ok": true}'}], max_tokens=64)
    if result.get("ok") is not True:
        raise ArkError("Ark preflight did not return the expected JSON")
    print(f"Ark preflight OK: {model} at {base}")


if __name__ == "__main__":
    preflight()
