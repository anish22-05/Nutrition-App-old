"""Safety layer for health questions. Deterministic rules, so behaviour is predictable and tested.

Three situations need different handling than an ordinary nutrition question:

* **emergency** - first-person signs of a medical or mental-health emergency (overdose, an
  allergic reaction, chest pain, suicidal thoughts). No document search, no model call: an
  immediate fixed message pointing to emergency help.
* **eating_disorder** - first-person intent towards extreme restriction or purging. The
  question is still answered from the sources, but the model is told not to give tips that
  would enable it, and a support notice is shown.
* **medical** - dosing, medication, diagnosis or treatment. Answered from the sources as
  general information only (no personalised dosing or diagnosis) with a clinician notice.

The rules look for *first-person intent*, so informational questions ("what are the health
effects of purging?", which the textbook covers) are answered normally.
"""
import re
from dataclasses import dataclass

_EMERGENCY = re.compile(
    r"\b(?:i|i've|i have|i'm|im|my (?:child|son|daughter|kid|baby|husband|wife|friend|mom|dad))\b.{0,60}"
    r"(?:\b(?:took|swallowed|ate|drank|taken|overdosed)\b.{0,40}\b(?:too many|too much|whole (?:bottle|pack)|pills|overdose)"
    r"|\bcan'?t breathe|\bcannot breathe|\bthroat (?:is )?(?:closing|swelling|tight)|\bchest pain"
    r"|\b(?:passed out|unconscious|fainting|seizure)|\bwant(?:s)? to (?:die|kill (?:my|him|her)self|end (?:it|my life)))"
    r"|\b(?:suicidal|kill myself|end my life|anaphyla)"
    # body-part phrasing ("my throat is swelling") has no person word before it
    r"|\bmy (?:throat|tongue|lips?|face) (?:is |are |feels? )?(?:closing|swelling|swollen|tight)",
    re.IGNORECASE | re.DOTALL,
)
_EATING_DISORDER = re.compile(
    r"\b(?:i|me|my|myself)\b.{0,50}"
    r"(?:\b(?:purge|purging|throw up|vomit|starve|starving)\b|\blaxatives?\b.{0,30}\b(?:lose|weight)"
    r"|\beat(?:ing)?\b.{0,20}\b(?:under|less than|below|only)\s*[1-7]\d{2}\s*(?:k?cal|calories)"
    r"|\b(?:stop|quit) eating\b)"
    r"|\b(?:pro[- ]?ana|thinspo)\b",
    re.IGNORECASE | re.DOTALL,
)
_MEDICAL = re.compile(
    r"\b(?:dose|dosage|dosing|overdose|mg (?:a|per) day|how (?:many|much) (?:mg|pills|tablets)"
    r"|medications?|medicines?|drugs?|prescri\w*|insulin|metformin|warfarin|statins?|antibiotics?"
    r"|diagnos\w*|do i have|symptoms? of|am i (?:deficient|sick)|interact(?:ion|s)? with"
    r"|cure|treat(?:ment|ing)? (?:for|my)|supplements? (?:for|to treat))\b",
    re.IGNORECASE,
)

EMERGENCY_MESSAGE = (
    "This sounds like it could be an emergency. Please call your local emergency number now "
    "(for example 911 in the US, 112 in the EU, 108/112 in India) or go to the nearest emergency "
    "department. For a possible poisoning or overdose you can also contact a poison control "
    "centre. If you are thinking about ending your life, please reach out to a crisis line "
    "right away (for example 988 in the US, or your country's equivalent). "
    "I can't help with this here, but people near you can, right now."
)
EATING_DISORDER_NOTICE = (
    "If you're struggling with eating, food or body image, you don't have to handle it alone. "
    "A doctor, a registered dietitian or an eating-disorder helpline can help."
)
MEDICAL_NOTICE = (
    "General information from the sources, not medical advice. Ask a doctor or pharmacist before "
    "changing medication, doses or supplements, or about any symptoms."
)

PROMPT_ADDITIONS = {
    "eating_disorder": (
        "The user may be at risk of an eating disorder. Do not give instructions, tips or targets "
        "for extreme calorie restriction, fasting to lose weight, purging or laxative misuse. "
        "Answer compassionately from the sources and encourage professional support."
    ),
    "medical": (
        "This is a medical question. Give general information from the sources only. Do not give a "
        "personalised dose, diagnosis or treatment plan; say a clinician should decide those."
    ),
}


@dataclass(frozen=True)
class SafetyCheck:
    category: str | None  # None | "emergency" | "eating_disorder" | "medical"
    notice: str | None

    @property
    def blocks_answer(self) -> bool:
        return self.category == "emergency"

    @property
    def prompt_addition(self) -> str | None:
        return PROMPT_ADDITIONS.get(self.category or "")

    def to_dict(self) -> dict | None:
        return {"category": self.category, "notice": self.notice} if self.category else None


def classify(*texts: str) -> SafetyCheck:
    """Most severe category across the user's message and its rewritten form."""
    joined = "\n".join(t for t in texts if t)
    if _EMERGENCY.search(joined):
        return SafetyCheck("emergency", EMERGENCY_MESSAGE)
    if _EATING_DISORDER.search(joined):
        return SafetyCheck("eating_disorder", EATING_DISORDER_NOTICE)
    if _MEDICAL.search(joined):
        return SafetyCheck("medical", MEDICAL_NOTICE)
    return SafetyCheck(None, None)
