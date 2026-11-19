"""Tools the answering model can call when a question needs numbers, not passages.

* energy_needs        - BMR (Mifflin-St Jeor) and daily energy needs for an activity level.
* recommended_intake  - RDA / AI for common nutrients by age, sex and life stage
                        (US National Academies DRIs as published by the NIH Office of Dietary
                        Supplements). Adults and teens (14+) only: outside that range the
                        tool says so instead of guessing.
* lookup_food         - nutrients of a food from USDA FoodData Central (live API).

Each tool returns a JSON-serialisable dict, including `source`, and never raises: errors come
back as {"error": ...} so the model can tell the user what went wrong.
"""
import json
import logging
import re
from functools import lru_cache

import requests

from app.config import get_settings

log = logging.getLogger(__name__)

DRI_SOURCE = "US National Academies Dietary Reference Intakes (NIH Office of Dietary Supplements fact sheets)"
FDC_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

ACTIVITY = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9}


def energy_needs(sex: str, age: float, weight_kg: float, height_cm: float, activity: str = "sedentary") -> dict:
    sex = sex.lower()
    if sex not in ("male", "female"):
        return {"error": "sex must be 'male' or 'female' (the equation is defined for those two)."}
    if not (18 <= age <= 100 and 30 <= weight_kg <= 300 and 120 <= height_cm <= 230):
        return {"error": "Inputs out of range: the equation is for adults (18-100 y, 30-300 kg, 120-230 cm)."}
    if activity not in ACTIVITY:
        return {"error": f"activity must be one of {sorted(ACTIVITY)}"}
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if sex == "male" else -161)
    return {
        "bmr_kcal_per_day": round(bmr),
        "estimated_energy_needs_kcal_per_day": round(bmr * ACTIVITY[activity]),
        "activity_factor": ACTIVITY[activity],
        "method": "Mifflin-St Jeor equation x activity factor",
        "note": "An estimate; individual needs vary by about +/-10%.",
        "source": "Mifflin MD, St Jeor ST et al. (1990), Am J Clin Nutr 51:241-7",
    }


# nutrient -> (unit, kind, rows). Each row: (sex, min_age, max_age, life_stage, value).
# sex "any" matches both; life_stage None = not pregnant/lactating.
_DRI: dict[str, tuple[str, str, list[tuple]]] = {
    "protein": ("g/day", "RDA", [("male", 14, 18, None, 52), ("female", 14, 18, None, 46), ("male", 19, 200, None, 56),
                                 ("female", 19, 200, None, 46), ("female", 14, 50, "pregnancy", 71), ("female", 14, 50, "lactation", 71)]),
    "iron": ("mg/day", "RDA", [("male", 14, 18, None, 11), ("male", 19, 200, None, 8), ("female", 14, 18, None, 15),
                               ("female", 19, 50, None, 18), ("female", 51, 200, None, 8), ("female", 14, 50, "pregnancy", 27),
                               ("female", 14, 18, "lactation", 10), ("female", 19, 50, "lactation", 9)]),
    "calcium": ("mg/day", "RDA", [("any", 14, 18, None, 1300), ("any", 19, 50, None, 1000), ("male", 51, 70, None, 1000),
                                  ("female", 51, 70, None, 1200), ("any", 71, 200, None, 1200),
                                  ("female", 14, 18, "pregnancy", 1300), ("female", 19, 50, "pregnancy", 1000),
                                  ("female", 14, 18, "lactation", 1300), ("female", 19, 50, "lactation", 1000)]),
    "vitamin d": ("mcg/day (40 IU = 1 mcg)", "RDA", [("any", 14, 70, None, 15), ("any", 71, 200, None, 20),
                                                    ("female", 14, 50, "pregnancy", 15), ("female", 14, 50, "lactation", 15)]),
    "vitamin c": ("mg/day (smokers: +35 mg)", "RDA", [("male", 14, 18, None, 75), ("female", 14, 18, None, 65), ("male", 19, 200, None, 90),
                                                     ("female", 19, 200, None, 75), ("female", 14, 18, "pregnancy", 80),
                                                     ("female", 19, 50, "pregnancy", 85), ("female", 14, 18, "lactation", 115),
                                                     ("female", 19, 50, "lactation", 120)]),
    "vitamin b12": ("mcg/day", "RDA", [("any", 14, 200, None, 2.4), ("female", 14, 50, "pregnancy", 2.6), ("female", 14, 50, "lactation", 2.8)]),
    "folate": ("mcg DFE/day", "RDA", [("any", 14, 200, None, 400), ("female", 14, 50, "pregnancy", 600), ("female", 14, 50, "lactation", 500)]),
    "vitamin a": ("mcg RAE/day", "RDA", [("male", 14, 200, None, 900), ("female", 14, 200, None, 700), ("female", 14, 18, "pregnancy", 750),
                                         ("female", 19, 50, "pregnancy", 770), ("female", 14, 18, "lactation", 1200),
                                         ("female", 19, 50, "lactation", 1300)]),
    "magnesium": ("mg/day", "RDA", [("male", 14, 18, None, 410), ("male", 19, 30, None, 400), ("male", 31, 200, None, 420),
                                    ("female", 14, 18, None, 360), ("female", 19, 30, None, 310), ("female", 31, 200, None, 320),
                                    ("female", 14, 18, "pregnancy", 400), ("female", 19, 30, "pregnancy", 350),
                                    ("female", 31, 50, "pregnancy", 360), ("female", 14, 18, "lactation", 360),
                                    ("female", 19, 30, "lactation", 310), ("female", 31, 50, "lactation", 320)]),
    "zinc": ("mg/day", "RDA", [("male", 14, 200, None, 11), ("female", 14, 18, None, 9), ("female", 19, 200, None, 8),
                               ("female", 14, 18, "pregnancy", 12), ("female", 19, 50, "pregnancy", 11),
                               ("female", 14, 18, "lactation", 13), ("female", 19, 50, "lactation", 12)]),
    "potassium": ("mg/day", "AI", [("male", 14, 18, None, 3000), ("male", 19, 200, None, 3400), ("female", 14, 18, None, 2300),
                                   ("female", 19, 200, None, 2600), ("female", 14, 18, "pregnancy", 2600),
                                   ("female", 19, 50, "pregnancy", 2900), ("female", 14, 18, "lactation", 2500),
                                   ("female", 19, 50, "lactation", 2800)]),
    "fiber": ("g/day", "AI", [("male", 14, 50, None, 38), ("male", 51, 200, None, 30), ("female", 14, 18, None, 26),
                              ("female", 19, 50, None, 25), ("female", 51, 200, None, 21), ("female", 14, 50, "pregnancy", 28),
                              ("female", 14, 50, "lactation", 29)]),
    "sodium": ("mg/day (reduce intake if above this)", "CDRR", [("any", 14, 200, None, 2300), ("female", 14, 50, "pregnancy", 2300),
                                                                ("female", 14, 50, "lactation", 2300)]),
    "water": ("L/day total water (food + drinks)", "AI", [("male", 14, 18, None, 3.3), ("female", 14, 18, None, 2.3), ("male", 19, 200, None, 3.7),
                                                         ("female", 19, 200, None, 2.7), ("female", 14, 50, "pregnancy", 3.0),
                                                         ("female", 14, 50, "lactation", 3.8)]),
    "vitamin k": ("mcg/day", "AI", [("any", 14, 18, None, 75), ("male", 19, 200, None, 120), ("female", 19, 200, None, 90),
                                    ("female", 14, 18, "pregnancy", 75), ("female", 19, 50, "pregnancy", 90),
                                    ("female", 14, 18, "lactation", 75), ("female", 19, 50, "lactation", 90)]),
    "vitamin e": ("mg/day (alpha-tocopherol)", "RDA", [("any", 14, 200, None, 15), ("female", 14, 50, "pregnancy", 15),
                                                       ("female", 14, 50, "lactation", 19)]),
    "iodine": ("mcg/day", "RDA", [("any", 14, 200, None, 150), ("female", 14, 50, "pregnancy", 220), ("female", 14, 50, "lactation", 290)]),
}
_ALIASES = {"fibre": "fiber", "dietary fiber": "fiber", "vit d": "vitamin d", "vit c": "vitamin c", "b12": "vitamin b12",
            "cobalamin": "vitamin b12", "folic acid": "folate", "salt": "sodium", "fluids": "water", "ascorbic acid": "vitamin c"}


def recommended_intake(nutrient: str, age: float, sex: str, life_stage: str | None = None) -> dict:
    key = _ALIASES.get(nutrient.lower().strip(), nutrient.lower().strip())
    if key not in _DRI:
        return {"error": f"No reference value for '{nutrient}'. Available: {', '.join(sorted(_DRI))}."}
    if age < 14:
        return {"error": "This table covers ages 14 and over; for children, use the documents or a paediatric reference."}
    sex = sex.lower()
    stage = (life_stage or "").lower() or None
    if stage not in (None, "pregnancy", "lactation"):
        return {"error": "life_stage must be 'pregnancy', 'lactation' or empty."}
    unit, kind, rows = _DRI[key]
    for row_sex, lo, hi, row_stage, value in rows:
        if row_stage == stage and row_sex in ("any", sex) and lo <= age <= hi:
            out = {"nutrient": key, "value": value, "unit": unit, "type": kind, "age": age, "sex": sex,
                   "life_stage": stage, "source": DRI_SOURCE}
            if key == "protein" and stage is None:
                out["note"] = "Equivalent to about 0.8 g per kg of body weight per day for adults."
            return out
    return {"error": f"No {key} value for sex={sex}, age={age}, life_stage={stage}."}


_FDC_NUTRIENTS = {  # FoodData Central nutrient numbers -> label
    "208": "energy_kcal", "203": "protein_g", "204": "fat_g", "205": "carbohydrate_g", "291": "fiber_g",
    "269": "sugars_g", "301": "calcium_mg", "303": "iron_mg", "307": "sodium_mg", "306": "potassium_mg",
    "401": "vitamin_c_mg", "328": "vitamin_d_mcg", "418": "vitamin_b12_mcg", "435": "folate_dfe_mcg",
}


@lru_cache(maxsize=256)
def _fdc_search(query: str) -> str:
    resp = requests.get(
        FDC_URL,
        params={"query": query, "pageSize": 5, "dataType": "Foundation,SR Legacy", "api_key": get_settings().usda_api_key},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.text


def lookup_food(food: str, amount_g: float = 100) -> dict:
    food = re.sub(r"\s+", " ", food).strip()[:80]
    if not food:
        return {"error": "food is empty"}
    if not (1 <= amount_g <= 2000):
        return {"error": "amount_g must be between 1 and 2000"}
    try:
        foods = json.loads(_fdc_search(food.lower())).get("foods") or []
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        return {"error": "The USDA food database refused the request (rate limit?)." if code in (403, 429) else f"USDA lookup failed ({code})."}
    except Exception:
        log.warning("USDA lookup failed", exc_info=True)
        return {"error": "The USDA food database could not be reached."}
    if not foods:
        return {"error": f"No match for '{food}' in USDA FoodData Central."}
    best = foods[0]
    per100 = {}
    for n in best.get("foodNutrients", []):
        label = _FDC_NUTRIENTS.get(str(n.get("nutrientNumber")))
        if label and n.get("value") is not None and label not in per100:
            per100[label] = n["value"]
    factor = amount_g / 100
    return {
        "food": best.get("description"),
        "fdc_id": best.get("fdcId"),
        "amount_g": amount_g,
        "nutrients": {k: round(v * factor, 2) for k, v in per100.items()},
        "other_matches": [f.get("description") for f in foods[1:4]],
        "source": f"USDA FoodData Central, fdcId {best.get('fdcId')} ({best.get('dataType')})",
    }


SCHEMAS = [
    {"type": "function", "function": {
        "name": "energy_needs",
        "description": "Estimate an adult's basal metabolic rate and daily calorie needs from sex, age, weight, height and activity level.",
        "parameters": {"type": "object", "properties": {
            "sex": {"type": "string", "enum": ["male", "female"]},
            "age": {"type": "number", "description": "years"},
            "weight_kg": {"type": "number"},
            "height_cm": {"type": "number"},
            "activity": {"type": "string", "enum": list(ACTIVITY)},
        }, "required": ["sex", "age", "weight_kg", "height_cm", "activity"]}}},
    {"type": "function", "function": {
        "name": "recommended_intake",
        "description": "Official recommended daily intake (RDA/AI) of a nutrient for a person aged 14+ by sex and life stage.",
        "parameters": {"type": "object", "properties": {
            "nutrient": {"type": "string", "description": f"one of: {', '.join(sorted(_DRI))}"},
            "age": {"type": "number"},
            "sex": {"type": "string", "enum": ["male", "female"]},
            "life_stage": {"type": "string", "enum": ["pregnancy", "lactation"], "description": "omit if neither"},
        }, "required": ["nutrient", "age", "sex"]}}},
    {"type": "function", "function": {
        "name": "lookup_food",
        "description": "Nutrient content (calories, protein, fat, carbs, fibre, minerals, vitamins) of a generic food from the USDA database.",
        "parameters": {"type": "object", "properties": {
            "food": {"type": "string", "description": "generic food name in English, e.g. 'banana, raw' or 'lentils, boiled'"},
            "amount_g": {"type": "number", "description": "grams; default 100"},
        }, "required": ["food"]}}},
]

_FUNCS = {"energy_needs": energy_needs, "recommended_intake": recommended_intake, "lookup_food": lookup_food}


def run(name: str, arguments: str) -> dict:
    """Execute a tool call from the model. Bad names/arguments become {"error": ...}."""
    fn = _FUNCS.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        args = json.loads(arguments or "{}")
        if not isinstance(args, dict):
            raise TypeError
        return fn(**args)
    except (ValueError, TypeError) as e:
        return {"error": f"bad arguments for {name}: {e}"[:200]}


_TOOLISH = re.compile(
    r"\b(?:calorie|kcal|bmr|metabolic|energy needs|tdee|how much|how many|grams?|mg|mcg|rda|recommended|"
    r"daily (?:intake|value|need)|should i (?:eat|get|have)|need per day|in (?:a|an|one|\d+)\b|contain|nutrients? in|protein in)\b",
    re.IGNORECASE,
)


def might_need_tools(question: str) -> bool:
    """Cheap pre-check: with no matching documents, only spend an LLM call if a tool could help."""
    return bool(_TOOLISH.search(question))
