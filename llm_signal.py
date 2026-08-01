"""Optional 4th detection signal: an LLM's judgment on the posting, via
Groq's free API (serves open-weight models like Llama 3.3).

Entirely optional and fails soft: if GROQ_API_KEY isn't set, or the API
call fails/times out for any reason, get_llm_signal() returns None and
detection.py falls back to the ML + rules + vagueness combination it
already used before this existed. A flaky third-party API should degrade
the feature, never take down predictions.
"""
import json
import os
import sys

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TIMEOUT_SECONDS = 6
MAX_INPUT_CHARS = 4000

SYSTEM_PROMPT = (
    "You are a fraud-detection specialist reviewing job postings for signs "
    "of being a scam - e.g. requests for money/fees, unrealistic pay for "
    "the effort described, missing or unverifiable employer details, "
    "high-pressure urgency language, requests to contact via informal "
    "channels like personal WhatsApp numbers, or promises that are too "
    "good to be true. Judge only what's in the text; don't assume a "
    "posting is fake just because it's short or informally written if "
    "nothing else is actually suspicious about it. "
    'Respond with strict JSON only, no other text: '
    '{"fraud_score": <integer 0-100, confidence this is a fraudulent posting>, '
    '"reasoning": "<one or two sentence plain-English explanation>"}'
)

try:
    import requests
except Exception:  # requests not installed
    requests = None


def get_llm_signal(text: str):
    """Returns {"fraud_probability": float 0-1, "reasoning": str} or None
    if the LLM signal is unavailable for any reason (no API key, network
    failure, malformed response, timeout)."""
    if not GROQ_API_KEY or requests is None or not text or not text.strip():
        return None

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text[:MAX_INPUT_CHARS]},
                ],
                "temperature": 0.2,
                "max_tokens": 220,
                "response_format": {"type": "json_object"},
            },
            timeout=GROQ_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        score = float(parsed.get("fraud_score", 0))
        score = max(0.0, min(100.0, score))
        reasoning = str(parsed.get("reasoning") or "").strip()[:400]

        return {"fraud_probability": score / 100.0, "reasoning": reasoning}
    except Exception as exc:
        print(f"[llm_signal] Groq call failed, skipping LLM signal: {exc!r}", file=sys.stderr)
        return None
