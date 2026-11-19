"""Prompt-injection defence for retrieved text.

Uploaded PDFs are untrusted, and their text is pasted straight into the model's prompt. A
document saying "ignore your instructions and ..." is therefore an attack on the assistant.
Defence in depth, because no single layer is reliable on its own:

1. **Spotlighting** - every source is wrapped in a delimiter carrying a random per-request
   nonce, and the system prompt says only text outside those delimiters is instructions. A
   document can't close the delimiter early because it can't guess the nonce.
2. **Neutralising** - invisible / bidi control characters (used to hide instructions) and
   anything that looks like our own markup tags are removed from the source text.
3. **Flagging** - sources matching known injection phrasing are labelled as suspicious in
   the prompt, and uploads containing them get a warning at ingestion time.
4. **Canary** - the system prompt holds a random marker; if it ever appears in an answer,
   the model was talked into revealing its instructions and the answer is withheld.
"""
import re
import secrets

# Zero-width and bidirectional-override characters: invisible in a PDF viewer, read by the model.
_INVISIBLE_RANGES = [(0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x2064), (0x2066, 0x2069), (0xFEFF, 0xFEFF)]
_INVISIBLE = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _INVISIBLE_RANGES) + "]")
# Markup that imitates our prompt structure (<context>, <source ...>, <system>, <instructions>).
_FAKE_TAGS = re.compile(r"</?\s*(?:context|source|src-[0-9a-f]+|system|instructions?|assistant|user)\b[^>]*>", re.IGNORECASE)

_PATTERNS = [
    r"\bignore\b.{0,40}\b(?:previous|prior|above|earlier|all|any|the)\b.{0,30}\b(?:instructions?|rules?|prompts?|directions?)",
    r"\bdisregard\b.{0,40}\b(?:instructions?|rules?|prompts?|context)",
    r"\bforget\b.{0,30}\b(?:instructions?|rules?|everything)",
    r"\b(?:system|developer)\s+(?:prompt|message|instructions?)\b",
    r"\byou are now\b|\bact as\b.{0,30}\b(?:unrestricted|jailbroken|dan)\b|\bnew instructions?\s*:",
    r"\b(?:reveal|print|repeat|show|output)\b.{0,30}\b(?:your|the)\b.{0,15}\b(?:instructions?|prompt|rules)\b",
    r"\b(?:assistant|ai|model|chatbot|llm)\b.{0,20}\b(?:must|should|will)\s+(?:now\s+)?(?:say|reply|respond|answer|tell)\b",
    # role labels pretending to start a new message ("SYSTEM: ...", "developer: ...")
    r"(?:^|[\n\]>])\s*(?:system|developer|admin(?:istrator)?|assistant)\s*:",
    r"\bnew (?:rules?|instructions?|directives?)\s*(?:from (?:the )?(?:developer|admin\w*|system|operator)|:)",
    r"\b(?:end|start|begin|finish|close|sign)\b.{0,10}\b(?:every|each|all|your)\s+(?:answers?|responses?|replies|messages?)\b",
    r"\b(?:note|message|instructions?)\s+(?:to|for)\s+(?:the\s+)?(?:ai|assistant|model|chatbot)\b",
]
_INJECTION = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE | re.DOTALL)


def new_nonce() -> str:
    return secrets.token_hex(6)


def neutralize(text: str) -> str:
    text = _INVISIBLE.sub("", text)
    return _FAKE_TAGS.sub("[tag removed]", text)


def find_injection(text: str) -> str | None:
    """The suspicious phrase, if the text looks like it is addressing the AI."""
    m = _INJECTION.search(_INVISIBLE.sub("", text))
    return m.group(0)[:120] if m else None


_SENTENCE = re.compile(r"[^.!?\n]*(?:[.!?]+|\n|$)")


def redact_instructions(text: str) -> tuple[str, int]:
    """Remove each sentence/line that reads as an instruction to the AI; keep everything else.

    Prompt wording alone did not stop every attack in eval/injection.py (a fake "SYSTEM:" line
    still got obeyed 1 time in 3), so the model never gets to see the instruction at all. The
    factual sentences around it survive, so a mostly-genuine document stays useful."""
    out, removed = [], 0
    for sentence in _SENTENCE.findall(neutralize(text)):
        if sentence.strip() and _INJECTION.search(sentence):
            out.append("[instruction removed] ")
            removed += 1
        else:
            out.append(sentence)
    return "".join(out).strip(), removed


def count_suspicious(chunks: list[str]) -> int:
    return sum(1 for c in chunks if find_injection(c))


def leaked(answer: str, canary: str) -> bool:
    return bool(canary) and canary in answer
