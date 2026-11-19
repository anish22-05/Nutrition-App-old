"""Tiny helper: one chat call that must return a JSON object."""
import json

from app import llm


def chat_json(system: str, user: str, model: str) -> dict:
    resp = llm.get_client().chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return json.loads(resp.choices[0].message.content or "{}")
