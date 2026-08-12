# -*- coding: utf-8 -*-
"""
Acceptance tests for the v5 rule engine and the dose layer.

The internal logic is exercised offline: the engine is fed readings, as if they came from the
backend's GET, and the recommendations are checked without calling the real backend. No test uses
the random forest or the circular labelling.

Run with:  uv run --python 3.12 --with pytest pytest tests/ -v
"""
import csv
import json
import os
import re
from datetime import datetime, timedelta, timezone

import pytest

from coffeetech_rules_v5 import Reading, CopperLedger
from coffeetech_recommendations import (
    build_recommendations,
    build_anticipated_recommendations,
    combine_recommendations,
    build_window_from_records,
    find_conventional,
    render_description,
    map_to_azure_payload,
    recommendations_signature,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "data_sensors_san_ignacio_feb2026.csv")

# Pattern for a forbidden absolute N dose ("40 g N/planta", "30 g de nitrógeno"). It must not
# match an organic amendment dose ("2–5 kg/planta/año" of compost).
ABS_N_DOSE = re.compile(r"\d[\d.,]*\s*(mg|g|kg)\s*(de\s+)?(n\b|nitr)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Helpers for building synthetic windows
def make_window(n_hours, *, soil, temp=20.0, rh=70.0, rain_pattern=None, soil_pattern=None,
                N=34.0, P=12.0, K=130.0, stage="vegetativo", altitude=1450.0):
    """Builds a window of n_hours readings one hour apart. `rain_pattern` is a list of bools (or
    None for no rain) applied hour by hour. `soil_pattern` allows an hourly soil humidity series,
    which the engine needs to derive the wet-dry envelope."""
    # Starts at LOCAL MIDNIGHT (05:00 UTC in Peru). The engine groups by local day, so a window
    # starting at UTC midnight enters at 19:00 the previous day and splits the first day in two --
    # and the patterns below, written in 24 h blocks, would stop meaning what they say.
    base = datetime(2026, 2, 1, 5, 0, tzinfo=timezone.utc)
    win = []
    for i in range(n_hours):
        rain = bool(rain_pattern[i]) if rain_pattern else False
        s = soil_pattern[i] if soil_pattern else soil
        win.append(Reading(
            ts=base + timedelta(hours=i), N=N, P=P, K=K,
            air_temp=temp, air_rh=rh, soil_moist=s, rain=rain,
            stage=stage, altitude=altitude,
        ))
    return win


def drydown_window(stage, *, wet=78.0, dry=30.0, wet_hours=24, dry_hours=36, altitude=1450.0):
    """A realistic window for the water rules: it starts wet, just after rain, and dries out. That
    gives a wet-dry span the engine can derive its site-relative threshold from."""
    n = wet_hours + dry_hours
    soil_series = [wet] * wet_hours + [dry] * dry_hours
    rain = [True] * 2 + [False] * (n - 2)   # the opening rain falls outside the last 24 h
    return make_window(n, soil=dry, soil_pattern=soil_series, rain_pattern=rain,
                       stage=stage, altitude=altitude)


def borer_window(stage="fructificacion", altitude=1450.0):
    """A FULL dry day followed by a rainy, mild one: the transition that fires R3.

    It needs its own helper because the real february campaign contains no complete dry day -- all
    three whole days in the CSV have rain -- and the only candidate, the 17th, is a 19 h fragment,
    which the rule does not accept as evidence that a day was dry. The altitude modulation tests
    need a window that really carries the transition.
    """
    return make_window(48, soil=60.0, temp=22.0, rain_pattern=[False] * 24 + [True] * 24,
                       stage=stage, altitude=altitude)


def single_reading(**kwargs):
    """A single-reading window, for one-off NPK or stage diagnoses."""
    defaults = dict(soil=60.0, temp=20.0, rh=70.0)
    defaults.update(kwargs)
    return make_window(1, **defaults)


def rec_by_rule(recs, rule_id):
    for r in recs:
        if r.rule_id == rule_id:
            return r
    return None


def rule_ids(recs):
    return [r.rule_id for r in recs]


@pytest.fixture(scope="module")
def csv_window_fruct():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return build_window_from_records(rows, stage="fructificacion", altitude=1450.0)


# --------------------------------------------------------------------------- #
# CRITERION 1 — San Ignacio feb2026 CSV, fructificacion, 1450 masl
def test_c1_csv_fructificacion(csv_window_fruct):
    recs = build_recommendations(csv_window_fruct, stage="fructificacion", altitude=1450.0)
    ids = rule_ids(recs)

    # N provisional, no absolute N dose
    n = rec_by_rule(recs, "NPK_N")
    assert n is not None and n.provisional is True
    assert not (n.dose and ABS_N_DOSE.search(n.dose)), f"N no debe llevar dosis absoluta: {n.dose}"

    # P adequate (info)
    p = rec_by_rule(recs, "NPK_P")
    assert p is not None and p.severity == "info"

    # K high -> "do not apply potassium" (info, no fertiliser product)
    k = rec_by_rule(recs, "NPK_K")
    assert k is not None and k.severity == "info"
    assert k.product is None
    assert "potasio" in k.farmer_message.lower()

    # R2 (American leaf spot) fires.
    assert "R2_AMERICAN_LEAF_SPOT" in ids
    # R3 (borer flight) does NOT fire, and that is the correct result: this campaign contains no
    # COMPLETE dry day. All three whole days in the CSV (18, 19 and 20 feb) have rain, and the 17th
    # -- the only one without -- is a 19 h fragment missing the first five local hours. Rain here is
    # spread almost evenly through the day (the 00:00-04:59 block holds 20.9 % of the annual rain
    # against 20.8 % of the time), so those five hours are not a harmless gap. Accepting the
    # fragment as a dry day would fire the rule on an artifact, not on the plot.
    assert "R3_BERRY_BORER" not in ids
    assert "R3_BORER_SANITATION" in ids      # RE-RE does run all year, independent of the flight

    # The water rules do NOT fire: soil is wet in february
    assert not any(x.startswith("R10_") for x in ids)

    # Organic certification
    assert find_conventional(recs) == []


# --------------------------------------------------------------------------- #
# CRITERION 2 — domain overlay: K=70 -> out of range -> laboratory
def test_c2_fuera_de_dominio_k70():
    win = single_reading(K=70.0, stage="fructificacion")
    recs = build_recommendations(win, stage="fructificacion", altitude=1450.0)
    fuera = rec_by_rule(recs, "NPK_K_OUT_OF_DOMAIN")
    assert fuera is not None
    assert fuera.refer is True
    assert fuera.product is None and fuera.dose is None
    assert fuera.rec_type == "lab"
    # No parallel ordinary K band diagnosis may exist.
    assert rec_by_rule(recs, "NPK_K") is None


# --------------------------------------------------------------------------- #
# CRITERION 3 — differentiation by crop stage
def test_c3_fosforo_plantula_vs_fructificacion():
    p_plant = rec_by_rule(
        build_recommendations(single_reading(P=15.0, stage="plantula"),
                              stage="plantula", altitude=1450.0),
        "NPK_P",
    )
    p_fruct = rec_by_rule(
        build_recommendations(single_reading(P=15.0, stage="fructificacion"),
                              stage="fructificacion", altitude=1450.0),
        "NPK_P",
    )
    assert p_plant is not None and p_plant.severity == "warning"     # P's floor rises in plantula
    assert p_fruct is not None and p_fruct.severity == "info"        # adequate in production


def test_c3_potasio_maduracion_vs_vegetativo():
    k_madur = rec_by_rule(
        build_recommendations(single_reading(K=100.0, stage="maduracion"),
                              stage="maduracion", altitude=1450.0),
        "NPK_K",
    )
    k_veg = rec_by_rule(
        build_recommendations(single_reading(K=100.0, stage="vegetativo"),
                              stage="vegetativo", altitude=1450.0),
        "NPK_K",
    )
    assert k_madur is not None and k_madur.severity == "alert"       # zero tolerance in maduracion
    assert k_veg is not None and k_veg.severity == "warning"         # moderate in vegetativo


# --------------------------------------------------------------------------- #
# CRITERION 4 — altitude modulation (it does not change soil thresholds)
def test_c4_altitud_muy_alta_suprime_broca(csv_window_fruct):
    recs = build_recommendations(csv_window_fruct, stage="fructificacion", altitude=2000.0)
    r2 = rec_by_rule(recs, "R2_AMERICAN_LEAF_SPOT")
    assert r2 is not None and r2.severity == "alert"     # American leaf spot stays high
    # Borer suppression is checked on a window that WOULD fire: over the real CSV R3 does not come
    # out anyway -- there is no complete dry day -- and a test that passes for want of data proves
    # nothing about suppression.
    ids = rule_ids(build_recommendations(borer_window(altitude=2000.0),
                                         stage="fructificacion", altitude=2000.0))
    assert "R3_BERRY_BORER" not in ids                   # borer suppressed at 2000 masl


def test_c4_altitud_baja_refuerza_broca(csv_window_fruct):
    r3 = rec_by_rule(build_recommendations(borer_window(altitude=1000.0),
                                           stage="fructificacion", altitude=1000.0),
                     "R3_BERRY_BORER")
    r2 = rec_by_rule(build_recommendations(csv_window_fruct, stage="fructificacion",
                                           altitude=1000.0),
                     "R2_AMERICAN_LEAF_SPOT")
    assert r3 is not None and r3.severity == "alert"     # borer rises to alert
    assert r2 is not None and r2.severity == "warning"   # American leaf spot drops to warning


def test_bajo_riesgo_no_abre_con_parentesis_ni_baja_a_info():
    # At 1700 masl the borer's weight is 'low': the alert drops to WARNING -- not to info, which
    # would read as all clear -- and states the low risk at the end, with no opening parenthesis.
    recs = build_recommendations(borer_window(altitude=1700.0),
                                 stage="fructificacion", altitude=1700.0)
    r3 = rec_by_rule(recs, "R3_BERRY_BORER")
    assert r3 is not None
    assert r3.severity == "warning"                      # still a warning, not info
    assert not r3.farmer_message.startswith("(")         # no hedging parenthesis at the start
    assert "bajo" in r3.farmer_message.lower()           # the low risk is stated in the text


# --------------------------------------------------------------------------- #
# The borer is a FRUIT pest, so it only applies where there is grain: fructificacion, maduracion,
# cosecha
def test_broca_solo_en_etapas_con_fruto():
    # A dry day (24 h) plus a mild rainy one (24 h) -> a dry→rain transition at day scale.
    pattern = [False] * 24 + [True] * 24
    win_madur = make_window(48, soil=60.0, temp=22.0, rain_pattern=pattern, stage="maduracion")
    recs_m = build_recommendations(win_madur, stage="maduracion", altitude=1450.0)
    assert rec_by_rule(recs_m, "R3_BERRY_BORER") is not None     # there is grain -> it applies

    win_veg = make_window(48, soil=60.0, temp=22.0, rain_pattern=pattern, stage="vegetativo")
    recs_v = build_recommendations(win_veg, stage="vegetativo", altitude=1450.0)
    assert rec_by_rule(recs_v, "R3_BERRY_BORER") is None         # no fruit -> it does not apply


def test_broca_saneamiento_se_emite_todo_el_ano():
    # RE-RE is the key cultural control and the reservoir is the leftover berries (~140 d), so it
    # cannot be suppressed outside fruiting stages. It must be emitted in every productive stage,
    # reinforced to a warning during harvest and the between-harvest period (vegetativo).
    for stage in ("vegetativo", "floracion", "fructificacion", "maduracion", "cosecha"):
        recs = build_recommendations(single_reading(stage=stage), stage=stage, altitude=1450.0)
        san = rec_by_rule(recs, "R3_BORER_SANITATION")
        assert san is not None, f"falta el saneamiento de broca en {stage}"
        assert san.rec_type == "pest"
    # Reinforced where the second pass belongs and where the reservoir lives.
    for stage in ("cosecha", "vegetativo"):
        san = rec_by_rule(build_recommendations(single_reading(stage=stage), stage=stage,
                                                altitude=1450.0), "R3_BORER_SANITATION")
        assert san.severity == "warning"
    # It does not apply in plantula: no fruit and no leftovers.
    assert rec_by_rule(build_recommendations(single_reading(stage="plantula"), stage="plantula",
                                             altitude=1450.0), "R3_BORER_SANITATION") is None


def test_broca_saneamiento_es_accion_cultural_no_todo_en_orden():
    # Collecting leftover berries is an ACTION even with no input attached: it must not be
    # labelled all clear, which is reserved for an adequate nutritional level.
    recs = build_recommendations(single_reading(stage="vegetativo"), stage="vegetativo", altitude=1450.0)
    data = json.loads(map_to_azure_payload(recs, "hub")["recommendationDescription"])
    san = next(i for i in data["items"] if i["rule_id"] == "R3_BORER_SANITATION")
    assert san["actionability"] == "direct"
    assert san["actionability_label"] != "Todo en orden"


def test_recordatorios_de_laboratorio_separados():
    # Soil analysis and liming are two reminders with different cadences and anchors.
    recs = build_recommendations(single_reading(), stage="vegetativo", altitude=1450.0)
    suelo = rec_by_rule(recs, "LAB_SOIL_ANALYSIS")
    cal = rec_by_rule(recs, "LAB_LIMING")
    assert suelo is not None and cal is not None
    assert "dos años" in suelo.farmer_message or "cada dos" in suelo.farmer_message
    assert "lluvia" in cal.farmer_message.lower()      # anchored to the onset of the rains
    assert rec_by_rule(recs, "LAB_PH_AL_KAMPRATH") is None   # retired old rule


def test_roya_se_modula_por_carga_de_fruto():
    # A sustained wet, mild window (3 days) -> a rust infection period. At 1700 masl rust weighs
    # 'low'; with fruit load it goes up a step, so the alert is NOT degraded to a warning the way it
    # is in a stage without fruit.
    def humeda(stage):
        return make_window(72, soil=60.0, temp=22.0, rh=92.0, stage=stage, altitude=1700.0)
    r_fruto = rec_by_rule(build_recommendations(humeda("maduracion"), stage="maduracion", altitude=1700.0),
                          "R1_RUST")
    r_veg = rec_by_rule(build_recommendations(humeda("vegetativo"), stage="vegetativo", altitude=1700.0),
                        "R1_RUST")
    assert r_fruto is not None and r_veg is not None
    assert r_fruto.severity == "alert"      # reinforced by the fruit load
    assert r_veg.severity == "warning"      # degraded by altitude (low weight)
    assert "carga de fruto" in r_fruto.agronomist_message.lower()


def test_broca_no_dispara_por_llovizna_de_una_hora():
    # All within one day, with no preceding dry day: one rainy hour is not enough for the flight.
    pattern = [False] * 20 + [True] + [False] * 3
    win = make_window(24, soil=60.0, temp=22.0, rain_pattern=pattern, stage="maduracion")
    recs = build_recommendations(win, stage="maduracion", altitude=1450.0)
    assert rec_by_rule(recs, "R3_BERRY_BORER") is None           # no dry day→rainy day transition


# --------------------------------------------------------------------------- #
# CRITERION 5 — water management: flower induction against the sensitive phases
def test_c5_floracion_seca_recomienda_riego():
    recs = build_recommendations(drydown_window("floracion"), stage="floracion", altitude=1450.0)
    ids = rule_ids(recs)
    assert "R10_DEFICIT_CRITICAL" in ids
    r = rec_by_rule(recs, "R10_DEFICIT_CRITICAL")
    assert "acolchado" in r.farmer_message.lower() or "riega" in r.farmer_message.lower()


def test_c5_vegetativo_seco_no_riega():
    recs = build_recommendations(drydown_window("vegetativo"), stage="vegetativo", altitude=1450.0)
    ids = rule_ids(recs)
    assert "R10_DEFICIT_INDUCTION" in ids          # stress induces flowering: do NOT irrigate
    assert "R10_DEFICIT_CRITICAL" not in ids


def test_c5_sin_recorrido_humedo_seco_no_recomienda_riego():
    # A flat soil series gives the sensor no usable envelope, so no site-relative threshold can be
    # derived and the engine stays quiet instead of inventing an absolute cut, which means nothing
    # with an uncalibrated capacitive probe.
    win = make_window(48, soil=30.0, rain_pattern=[False] * 48, stage="floracion")
    ids = rule_ids(build_recommendations(win, stage="floracion", altitude=1450.0))
    assert not any(x.startswith("R10_") for x in ids)


def test_c5_la_floracion_ya_no_sale_de_una_transicion_cualquiera():
    """`R11_FLOWERING_EXPECTED` must NOT come out of the diagnosis over measured data.

    The published trigger is more than 10 mm of rain in a day after a dry period, and this sensor
    reports rain as a boolean, so the depth cannot be checked here at all. Built on what the sensor
    does give -- a dry hour followed by a rainy one -- the condition is true on nearly every day.
    The rule lives in `forecast_rules`, where there are millimetres; were it to come out of here
    again, it would be firing without the quantity that defines it.
    """
    pattern = [False] * 20 + [True]
    win = make_window(21, soil=30.0, rain_pattern=pattern, stage="vegetativo")
    recs = build_recommendations(win, stage="vegetativo", altitude=1450.0)
    assert "R11_FLOWERING_EXPECTED" not in rule_ids(recs)


# --------------------------------------------------------------------------- #
# CRITERION 6 — certification: no output carries a conventional synthetic product
def test_c6_sin_productos_convencionales(csv_window_fruct):
    escenarios = [
        build_recommendations(csv_window_fruct, stage="fructificacion", altitude=1450.0),
        # Deficiencies that activate the organic dose layer (compost, rock phosphate, SOP)
        build_recommendations(single_reading(N=18.0, P=5.0, K=95.0, stage="fructificacion"),
                              stage="fructificacion", altitude=1450.0),
        build_recommendations(make_window(24, soil=30.0, rain_pattern=[False] * 24, stage="floracion"),
                              stage="floracion", altitude=1450.0),
    ]
    for recs in escenarios:
        assert find_conventional(recs) == []
        # The final text sent to the backend must be free of synthetics too.
        assert text_is_organic(render_description(recs))


def text_is_organic(text):
    """True when the text contains no blocked synthetics."""
    from coffeetech_rules_v5 import CONVENTIONAL_BLOCKED
    low = text.lower()
    return not any(term.lower() in low for term in CONVENTIONAL_BLOCKED)


# --------------------------------------------------------------------------- #
# CRITERION 7 — nitrogen: no output prescribes an absolute N dose from the sensor
def test_c7_nitrogeno_sin_dosis_absoluta():
    escenarios = [
        build_recommendations(single_reading(N=18.0, stage="fructificacion"),   # severe
                              stage="fructificacion", altitude=1450.0),
        build_recommendations(single_reading(N=24.0, stage="vegetativo"),        # low
                              stage="vegetativo", altitude=1450.0),
        build_recommendations(single_reading(N=34.0, stage="fructificacion"),    # adequate
                              stage="fructificacion", altitude=1450.0),
    ]
    for recs in escenarios:
        for r in recs:
            # No dose, in any recommendation, expresses a mass of N.
            assert not (r.dose and ABS_N_DOSE.search(r.dose)), f"Dosis absoluta de N prohibida: {r.rule_id} -> {r.dose}"
        n = rec_by_rule(recs, "NPK_N")
        assert n is not None and n.provisional is True


# --------------------------------------------------------------------------- #
# CRITERION 8 — copper: over the cap it is replaced by cultural management
def test_c8_tope_de_cobre(csv_window_fruct):
    # Ledger with room left: it recommends authorised copper
    ledger_ok = CopperLedger()
    recs_ok = build_recommendations(csv_window_fruct, stage="fructificacion",
                                    altitude=1450.0, copper_ledger=ledger_ok)
    r2_ok = rec_by_rule(recs_ok, "R2_AMERICAN_LEAF_SPOT")
    assert r2_ok is not None
    assert r2_ok.product is not None and "cobre" in r2_ok.product.lower()

    # Ledger at the cap (4 kg Cu/ha/year applied): replaced by cultural management
    ledger_full = CopperLedger()
    ledger_full.applied = 4.0
    recs_full = build_recommendations(csv_window_fruct, stage="fructificacion",
                                      altitude=1450.0, copper_ledger=ledger_full)
    r2_full = rec_by_rule(recs_full, "R2_AMERICAN_LEAF_SPOT")
    assert r2_full is not None
    assert "cobre" not in (r2_full.product or "").lower()
    assert r2_full.method == "cultural"


# --------------------------------------------------------------------------- #
# Extra — the backend payload keeps the {recommendationDescription, deviceHubId} contract
def test_payload_contract(csv_window_fruct):
    recs = build_recommendations(csv_window_fruct, stage="fructificacion", altitude=1450.0)
    payload = map_to_azure_payload(recs, "hub_milagro_01")
    assert set(payload.keys()) == {"recommendationDescription", "deviceHubId"}
    assert payload["deviceHubId"] == "hub_milagro_01"
    assert isinstance(payload["recommendationDescription"], str) and payload["recommendationDescription"]


# --------------------------------------------------------------------------- #
# ANTICIPATED ALERTS, from a simulated forecast; sklearn is not needed
def a_reading(temp=20.0, rh=70.0, **kw):
    base = dict(ts=datetime(2026, 2, 1, tzinfo=timezone.utc), N=34.0, P=12.0, K=130.0,
                air_temp=temp, air_rh=rh, soil_moist=60.0, rain=False,
                stage="fructificacion", altitude=1450.0)
    base.update(kw)
    return Reading(**base)


def test_anticipada_termica():
    # A forecast of acute heat (>32 °C) -> anticipated thermal alert R9.
    recs = build_anticipated_recommendations(a_reading(), [{"horizon_h": 6, "air_temp": 35.0, "air_rh": 50.0}])
    r9 = rec_by_rule(recs, "R9_ACUTE_HEAT")
    assert r9 is not None
    assert r9.forecast is True and r9.horizon_h == 6
    assert r9.rec_type == "thermal"


def test_anticipada_fungica_probabilistica():
    # Forecast RH ≥85 % with T in 18-28 °C -> early fungal warning, probabilistic, with a check.
    recs = build_anticipated_recommendations(a_reading(), [{"horizon_h": 6, "air_temp": 22.0, "air_rh": 90.0}])
    f = rec_by_rule(recs, "FORECAST_FUNGAL_RISK")
    assert f is not None
    assert f.forecast is True and f.severity == "warning"
    assert f.verification is not None           # verification step is mandatory
    assert f.refer is False                       # a warning, not a diagnosis nor a forced referral


def test_anticipada_fungica_no_dispara_fuera_de_rango():
    # High RH but T outside 18-28 -> the anticipated fungal warning is not emitted.
    recs = build_anticipated_recommendations(a_reading(), [{"horizon_h": 6, "air_temp": 12.0, "air_rh": 95.0}])
    assert rec_by_rule(recs, "FORECAST_FUNGAL_RISK") is None


def test_combinar_deduplica_condicion_ya_activa():
    # If the heat is already active now, its forecast version is not repeated.
    win = make_window(1, soil=60.0, temp=25.0)   # temp>23 -> immediate R9_HEAT_QUALITY
    immediate = build_recommendations(win, stage="fructificacion", altitude=1450.0)
    assert rec_by_rule(immediate, "R9_HEAT_QUALITY") is not None
    anticipated = build_anticipated_recommendations(a_reading(), [{"horizon_h": 2, "air_temp": 25.0, "air_rh": 60.0}])
    combined = combine_recommendations(immediate, anticipated)
    r9s = [r for r in combined if r.rule_id == "R9_HEAT_QUALITY"]
    assert len(r9s) == 1 and r9s[0].forecast is False   # the immediate one is kept


def test_combinar_deduplica_horizontes_conserva_el_mas_temprano():
    anticipated = build_anticipated_recommendations(
        a_reading(),
        [{"horizon_h": 6, "air_temp": 35.0, "air_rh": 50.0},
         {"horizon_h": 2, "air_temp": 35.0, "air_rh": 50.0}],
    )
    combined = combine_recommendations([], anticipated)
    r9s = [r for r in combined if r.rule_id == "R9_ACUTE_HEAT"]
    assert len(r9s) == 1 and r9s[0].horizon_h == 2       # the earliest alert is kept


def test_formato_recommendation_description():
    # The text keeps its shape: header, technical voice, forecast and referral prefix.
    win = make_window(1, soil=60.0)
    immediate = build_recommendations(win, stage="fructificacion", altitude=1450.0)
    anticipated = build_anticipated_recommendations(a_reading(), [{"horizon_h": 6, "air_temp": 35.0, "air_rh": 50.0}])
    combined = combine_recommendations(immediate, anticipated)
    text = render_description(combined)
    assert "Técnico:" in text
    assert "Previsión: en ~6 h" in text
    assert text.lstrip().startswith("[")            # every block opens with [severity · type]
    # At least one recommendation with a referral must carry the consult-the-technician prefix.
    assert "⚠ Consultar técnico APROCASSI:" in text
    # The payload contract holds with the anticipated text included.
    payload = map_to_azure_payload(combined, "hub_milagro_01")
    assert set(payload.keys()) == {"recommendationDescription", "deviceHubId"}
    # Organic certification over the anticipated ones too.
    assert find_conventional(combined) == []


# --------------------------------------------------------------------------- #
# CONTENT DEDUPLICATION, so identical messages are not resent
def test_dedup_firma_estable_ante_decimales_del_pronostico():
    # The same thermal alert, with the forecast varying by tenths: that must not count as a new
    # diagnosis, so the signature matches and nothing is resent.
    win = make_window(1, soil=60.0)
    immediate = build_recommendations(win, stage="fructificacion", altitude=1450.0)
    a1 = combine_recommendations(immediate, build_anticipated_recommendations(
        a_reading(), [{"horizon_h": 6, "air_temp": 35.0, "air_rh": 50.0}]))
    a2 = combine_recommendations(immediate, build_anticipated_recommendations(
        a_reading(), [{"horizon_h": 6, "air_temp": 35.9, "air_rh": 51.0}]))
    assert recommendations_signature(a1) == recommendations_signature(a2)


def test_dedup_firma_cambia_si_aparece_alerta():
    # A new thermal alert appears -> the diagnosis changed -> the signature changes -> resend.
    win = make_window(1, soil=60.0)
    immediate = build_recommendations(win, stage="fructificacion", altitude=1450.0)
    sin_calor = combine_recommendations(immediate, build_anticipated_recommendations(
        a_reading(), [{"horizon_h": 6, "air_temp": 20.0, "air_rh": 50.0}]))   # does not fire R9
    con_calor = combine_recommendations(immediate, build_anticipated_recommendations(
        a_reading(), [{"horizon_h": 6, "air_temp": 35.0, "air_rh": 50.0}]))   # fires acute R9
    assert recommendations_signature(sin_calor) != recommendations_signature(con_calor)


# --------------------------------------------------------------------------- #
# INGESTION: the backend exposes the data-records timestamp as `updatedAt`
def test_ventana_reconoce_updatedAt():
    # Records as the backend delivers them: timestamp in 'updatedAt' and NPK in camelCase.
    records = [
        {"deviceHubId": "1C:69:20:31:4B:78", "updatedAt": f"2026-02-17T10:{m:02d}:00",
         "nitrogen": 35, "phosphorus": 15, "potassium": 140,
         "celciusGradeTemperature": 20, "airHumidityPercent": 70,
         "soilHumidityPercent": 65, "precipitationDetected": 0}
        for m in range(5)
    ]
    win = build_window_from_records(records, stage="maduracion", altitude=1700.0)
    assert len(win) == 5                        # empty unless `updatedAt` is accepted as the stamp
    recs = build_recommendations(win, stage="maduracion", altitude=1700.0)
    assert recs                                 # and so does not land in the "no readings" case
    assert rec_by_rule(recs, "LAB_SOIL_ANALYSIS") is not None


def test_mensaje_ventana_vacia_no_parece_diagnostico_sano():
    texto = render_description([])
    assert "no hay lecturas" in texto.lower()   # a clear "data missing" message
    assert "mantener el manejo" not in texto.lower()  # and must not read as all clear


# --------------------------------------------------------------------------- #
# The user-facing text must not leak internal English values (band, weight, method)
def test_texto_no_filtra_valores_en_ingles(csv_window_fruct):
    texto = render_description(build_recommendations(csv_window_fruct, stage="fructificacion", altitude=1450.0))
    # Internal English values that must not reach the text:
    assert "(banda 'adequate'" not in texto        # N's band
    assert "Peso por altitud: high" not in texto   # altitude weight
    assert "Peso por altitud: medium" not in texto
    assert "(soil," not in texto                    # application method
    # They must now come out in Spanish:
    assert "banda 'adecuado'" in texto
    assert "Peso por altitud: alto" in texto        # R2 at 1450 masl (medium band)
    assert "(suelo," in texto                       # dolomitic lime: soil method


# --------------------------------------------------------------------------- #
# The backend payload is structured JSON, so the frontend can filter by role
def test_payload_es_json_estructurado(csv_window_fruct):
    recs = build_recommendations(csv_window_fruct, stage="fructificacion", altitude=1450.0)
    payload = map_to_azure_payload(recs, "hub_milagro_01")
    # The backend contract is still two fields and the description is still a string…
    assert set(payload.keys()) == {"recommendationDescription", "deviceHubId"}
    assert isinstance(payload["recommendationDescription"], str)
    # …but that string is valid, structured JSON.
    data = json.loads(payload["recommendationDescription"])
    assert data["v"] == 1
    assert isinstance(data["items"], list) and data["items"]
    assert sum(data["summary"].values()) == len(data["items"])
    it = data["items"][0]
    for k in ("severity", "severity_label", "type", "type_label", "farmer_message", "agronomist_message",
              "kind", "category", "actionability", "actionability_label", "saving"):
        assert k in it
    assert it["severity_label"] in ("Crítico", "Alerta", "Aviso", "Info")
    assert it["actionability"] in ("direct", "verify", "consult", "structural", "monitor", "none")
    # When there is an action, the method comes translated for the frontend.
    con_accion = [i for i in data["items"] if i["action"]]
    assert con_accion, "se esperaba al menos una recomendación con acción/dosis"
    assert con_accion[0]["action"]["method_label"] in ("suelo", "foliar", "cultural")


def test_payload_json_ventana_vacia():
    data = json.loads(map_to_azure_payload([], "hub")["recommendationDescription"])
    assert data["items"] == []
    assert "no hay lecturas" in data["empty_message"].lower()


def test_payload_sin_citas_y_flags():
    # High K -> "do not apply" (a saving); LAB -> a reminder that refers to the technician.
    recs = build_recommendations(single_reading(K=200.0, stage="fructificacion"),
                                 stage="fructificacion", altitude=1450.0)
    data = json.loads(map_to_azure_payload(recs, "hub")["recommendationDescription"])
    blob = json.dumps(data, ensure_ascii=False)
    for cita in ("Cenicafé", "Kamprath", "FAO", "Bebber", "INIA"):
        assert cita not in blob, f"cita de literatura filtrada al usuario: {cita}"
    k = next(i for i in data["items"] if i["rule_id"] == "NPK_K")
    assert k["saving"] is True                    # high potassium: not applying is a saving
    assert k["actionability"] == "none"           # no input to apply, so not "do it now"
    lab = next(i for i in data["items"] if i["rule_id"] == "LAB_SOIL_ANALYSIS")
    assert lab["kind"] == "reminder"              # goes to its own section, not as an alert
    assert lab["actionability"] == "consult"      # needs a technician or a laboratory


# --------------------------------------------------------------------------- #
# ACTIONABILITY: the chip must not say "apply now" where there is nothing to apply
def _rule_item(items, rule_id):
    return next(i for i in items if i["rule_id"] == rule_id)


def test_accionabilidad_nivel_adecuado_es_todo_en_orden():
    # P and K at an adequate level (info, no product): no action -> "none/Todo en orden", never
    # "direct/Puedes aplicarlo ahora".
    recs = build_recommendations(single_reading(P=15.0, K=140.0, stage="vegetativo"),
                                 stage="vegetativo", altitude=1450.0)
    data = json.loads(map_to_azure_payload(recs, "hub")["recommendationDescription"])
    for rid in ("NPK_P", "NPK_K"):
        it = _rule_item(data["items"], rid)
        assert it["actionability"] == "none", f"{rid} debería ser 'none', no {it['actionability']}"
        assert it["actionability_label"] == "Todo en orden"


def test_accionabilidad_calor_agudo_es_medida_de_fondo():
    # Acute heat (>32 °C): the real lever is structural, shade, so "structural" and not "direct".
    # The message also offers an immediate palliative: no foliar sprays in the heat.
    recs = build_recommendations(single_reading(temp=34.0, stage="fructificacion"),
                                 stage="fructificacion", altitude=1450.0)
    data = json.loads(map_to_azure_payload(recs, "hub")["recommendationDescription"])
    heat = _rule_item(data["items"], "R9_ACUTE_HEAT")
    assert heat["actionability"] == "structural"
    assert heat["actionability_label"] == "Medida de fondo"
    assert "foliar" in heat["farmer_message"].lower()   # the immediate palliative is present


def test_accionabilidad_nitrogeno_bajo_deriva_al_tecnico():
    # Low N: advisory and provisional -> "consult", even when it carries an amendment dose.
    recs = build_recommendations(single_reading(N=22.0, stage="vegetativo"),
                                 stage="vegetativo", altitude=1450.0)
    data = json.loads(map_to_azure_payload(recs, "hub")["recommendationDescription"])
    n = _rule_item(data["items"], "NPK_N")
    assert n["actionability"] == "consult"
    assert n["provisional"] is True


def test_accionabilidad_producto_concreto_es_direct():
    # Low P (band 'low', no referral): there is rock phosphate to apply ->
    # "direct/Puedes aplicarlo ahora".
    recs = build_recommendations(single_reading(P=8.0, stage="vegetativo"),
                                 stage="vegetativo", altitude=1450.0)
    data = json.loads(map_to_azure_payload(recs, "hub")["recommendationDescription"])
    p = _rule_item(data["items"], "NPK_P")
    assert p["action"] is not None                # it carries a product and a dose
    assert p["refer"] is False
    assert p["actionability"] == "direct"
