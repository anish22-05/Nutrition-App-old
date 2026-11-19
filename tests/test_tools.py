"""Nutrition tools: deterministic calculators, the DRI table, and the USDA lookup (network mocked)."""
import json

import pytest
import requests

from app import tools


def test_energy_needs_mifflin_st_jeor():
    r = tools.energy_needs("male", 30, 80, 180, "sedentary")
    assert r["bmr_kcal_per_day"] == 1780  # 10*80 + 6.25*180 - 5*30 + 5
    assert r["estimated_energy_needs_kcal_per_day"] == 2136  # x 1.2
    f = tools.energy_needs("female", 30, 60, 165, "moderate")
    assert f["bmr_kcal_per_day"] == round(600 + 1031.25 - 150 - 161)


@pytest.mark.parametrize("args", [("other", 30, 80, 180, "light"), ("male", 10, 30, 140, "light"),
                                  ("male", 30, 80, 180, "couch")])
def test_energy_needs_rejects_bad_input(args):
    assert "error" in tools.energy_needs(*args)


@pytest.mark.parametrize(("nutrient", "age", "sex", "stage", "value"), [
    ("iron", 25, "female", None, 18),
    ("iron", 25, "female", "pregnancy", 27),
    ("iron", 60, "female", None, 8),
    ("iron", 40, "male", None, 8),
    ("calcium", 60, "female", None, 1200),
    ("calcium", 60, "male", None, 1000),
    ("vitamin d", 75, "male", None, 20),
    ("fibre", 30, "male", None, 38),           # alias
    ("folate", 30, "female", "pregnancy", 600),
    ("protein", 40, "male", None, 56),
    ("potassium", 30, "female", None, 2600),
])
def test_recommended_intake_values(nutrient, age, sex, stage, value):
    r = tools.recommended_intake(nutrient, age, sex, stage)
    assert r["value"] == value and "source" in r


def test_recommended_intake_refuses_children_and_unknowns():
    assert "error" in tools.recommended_intake("iron", 8, "female")
    assert "error" in tools.recommended_intake("unobtainium", 30, "male")
    assert "error" in tools.recommended_intake("iron", 30, "female", "retirement")


class FakeResp:
    def __init__(self, payload, status=200):
        self.text, self.status_code = json.dumps(payload), status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


def test_lookup_food_scales_nutrients(monkeypatch):
    tools._fdc_search.cache_clear()
    payload = {"foods": [{"description": "Lentils, mature seeds, cooked, boiled", "fdcId": 172421, "dataType": "SR Legacy",
                          "foodNutrients": [{"nutrientNumber": "203", "value": 9.02}, {"nutrientNumber": "208", "value": 116},
                                            {"nutrientNumber": "999", "value": 1}]}]}
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp(payload))
    r = tools.lookup_food("lentils boiled", 200)
    assert r["food"].startswith("Lentils") and r["nutrients"] == {"protein_g": 18.04, "energy_kcal": 232}
    assert "172421" in r["source"]


def test_lookup_food_errors_become_messages(monkeypatch):
    tools._fdc_search.cache_clear()
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp({}, 429))
    assert "rate limit" in tools.lookup_food("kale")["error"]
    tools._fdc_search.cache_clear()
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError()))
    assert "could not be reached" in tools.lookup_food("kale")["error"]
    tools._fdc_search.cache_clear()
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp({"foods": []}))
    assert "No match" in tools.lookup_food("zzz")["error"]


def test_run_dispatches_and_never_raises():
    assert tools.run("energy_needs", '{"sex":"male","age":30,"weight_kg":80,"height_cm":180,"activity":"sedentary"}')["bmr_kcal_per_day"] == 1780
    assert "error" in tools.run("nope", "{}")
    assert "error" in tools.run("energy_needs", "not json")
    assert "error" in tools.run("energy_needs", '{"unexpected": 1}')


def test_might_need_tools():
    assert tools.might_need_tools("How many calories do I need?")
    assert tools.might_need_tools("how much protein is in 100 g of tofu")
    assert not tools.might_need_tools("Who wrote the textbook?")
