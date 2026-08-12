# -*- coding: utf-8 -*-
"""
The crop stage: that the engine USES it where it should, and does not fake it when it lacks one.

The system receives each section's stage from the backend. A cross audit -- reading the syntax tree
and, separately, running the 34 rules across the six stages and comparing what they emit -- found
four things that did not add up:

1. `R9_HEAT_QUALITY` wrote the condition IN PROSE ("Si hay grano llenándose…") while holding the
   stage. It handed the grower a question the system could already answer.
2. `R1_RUST_FORECAST` did not apply the fruit load factor that `R1_RUST` does. Same disease, and the
   anticipated one matters most because copper works BEFORE the infection window.
3. The label said "[Peso por altitud: medio]" when altitude gave LOW and the fruit load had raised
   it. It attributed to altitude a step that was not its own.
4. An unrecognised stage was silently replaced by "vegetativo", which is NOT neutral: it is the
   flower induction stage, where the water rule says there is no need to irrigate. A section in
   grain filling received the opposite of the right advice.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from coffeetech_rules_v5 import (
    FORECAST, STAGE_UNKNOWN, VALID_STAGES, Reading, WATER, _bump_weight, _weight,
    env_alerts, evaluate_window, inferential_rules, water_rules,
)
from coffeetech_weather import HourlyForecast, PILOT_ELEVATION, PILOT_LAT, PILOT_LON

AHORA = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)
ALT = 1823.0


def _ventana(etapa, horas=200):
    """A real humidity envelope: it rises with the rain and then dries steadily."""
    return [Reading(ts=AHORA - timedelta(hours=h), air_temp=20.0, air_rh=95.0,
                    soil_moist=70.0 - min(45.0, (horas - h) * 0.30),
                    rain=((horas - h) < 20), N=36.4, P=11.7, K=163.0,
                    stage=etapa, altitude=ALT)
            for h in range(horas, 0, -1)]


@pytest.mark.parametrize("etapa", VALID_STAGES)
def test_el_calor_no_le_pregunta_al_agricultor_por_su_etapa(etapa):
    """The engine knows the stage: asking for it in the message hands the work back to the user."""
    a = next(x for x in env_alerts(Reading(ts=AHORA, air_temp=26.0, stage=etapa, altitude=ALT))
             if x.rule_id == "R9_HEAT_QUALITY")
    assert "Si hay grano" not in a.farmer_message
    con_grano = etapa in ("fructificacion", "maduracion")
    assert ("calidad de la taza" in a.farmer_message) == con_grano, (
        f"En {etapa} el mensaje {'debería' if con_grano else 'no debería'} hablar de la taza: "
        f"sin fruto en desarrollo no hay calidad que perder.")


#: Hours the built forecast covers. The past reaches beyond `flowering_gap_days` because
#: `R11_FLOWERING_EXPECTED` refuses to claim no inductive rain fell when it cannot see that far
#: back, and a short window would silence it for lack of data rather than by its threshold.
_PASADO_H = 24 * (FORECAST["flowering_gap_days"] + 1)
_FUTURO_H = 72
_RANGO_H = range(-_PASADO_H, _FUTURO_H)


def _pronostico_de_roya():
    """An hourly forecast holding a leaf wetness window long enough to fire `R1_RUST_FORECAST`.

    Every variable the rule does not read is left neutral, so no other rule fires by accident and
    muddies which alert is being measured. Wetness runs from +8 h to +30 h: 22 hours above the
    12 the rule asks for, at a temperature inside its 15-28 °C band.
    """
    def serie(f):
        return [f(i) for i in _RANGO_H]

    return HourlyForecast(
        latitude=PILOT_LAT, longitude=PILOT_LON, elevation=PILOT_ELEVATION,
        times=[AHORA + timedelta(hours=i) for i in _RANGO_H],
        values={
            "temperature_2m": serie(lambda i: 20.0),
            "relative_humidity_2m": serie(lambda i: 95.0 if 8 <= i < 30 else 70.0),
            "precipitation": serie(lambda i: 0.0),
            "precipitation_probability": serie(lambda i: 10.0),
            "shortwave_radiation": serie(lambda i: 200.0),
            "et0_fao_evapotranspiration": serie(lambda i: 0.1),
            "vapour_pressure_deficit": serie(lambda i: 0.30),
            "wind_speed_10m": serie(lambda i: 8.0 if 2 <= i < 6 else 2.0),
            "cloud_cover": serie(lambda i: 50.0),
        })


def test_la_roya_anticipada_escala_igual_que_la_medida():
    """Same disease, same factor: fruit load has to raise the anticipated rust the way it raises
    the measured one."""
    w = _pronostico_de_roya()
    from coffeetech_rules_v5 import forecast_rules
    sev = {}
    for etapa in ("vegetativo", "maduracion"):
        a = next((x for x in forecast_rules(_ventana(etapa), w, stage=etapa, altitude=ALT,
                                            now=AHORA) if x.rule_id == "R1_RUST_FORECAST"), None)
        assert a is not None, f"R1_RUST_FORECAST no disparó en {etapa}"
        sev[etapa] = a.severity
    assert sev["maduracion"] == "alert" and sev["vegetativo"] == "warning", (
        f"La carga de fruto debe subir la severidad de la roya anticipada: {sev}")


def test_la_etiqueta_de_peso_no_atribuye_a_la_altitud_lo_que_subio_el_fruto():
    """At 1823 m rust weighs LOW. In fructificacion it was published as MEDIUM, as if from altitude."""
    assert _weight(ALT, "rust") == "low"
    assert _bump_weight("low") == "medium"
    a = next(x for x in inferential_rules(_ventana("maduracion"), altitude=ALT, stage="maduracion")
             if x.rule_id == "R1_RUST")
    assert "Peso por altitud: medio" not in a.agronomist_message
    assert "carga de fruto" in a.agronomist_message, (
        "Si el peso subió por carga de fruto, la etiqueta tiene que decirlo: si no, el técnico lee "
        "que la finca está en otra banda de altitud.")


def test_una_etapa_desconocida_no_se_sustituye_por_una_concreta():
    """"vegetativo" is NOT a neutral value: it is the stage where irrigation is suppressed on
    purpose."""
    assert STAGE_UNKNOWN not in VALID_STAGES
    assert STAGE_UNKNOWN not in WATER["induction_stages"]
    assert STAGE_UNKNOWN not in WATER["critical_stages"]

    ids = {a.rule_id for a in water_rules(_ventana(STAGE_UNKNOWN), stage=STAGE_UNKNOWN,
                                          altitude=ALT)}
    assert "R10_DEFICIT_INDUCTION" not in ids, (
        "Sin saber la etapa, el sistema NO puede decir «no hace falta regar»: en llenado de grano "
        "eso es el consejo contrario al correcto.")
    assert "R10_DEFICIT_MILD" in ids, "Debe degradar a la rama neutra, no callarse del todo."


def test_la_etapa_desconocida_se_declara_al_tecnico():
    """Degrading silently turns a prudent decision into a false claim: the technician would believe
    the system evaluated the stage and decided that."""
    al = evaluate_window(_ventana(STAGE_UNKNOWN))
    assert al, "La ventana de prueba debería emitir algo."
    assert all("Etapa no registrada" in a.agronomist_message for a in al)
    # And with a valid stage the message stays clean.
    assert not any("Etapa no registrada" in a.agronomist_message
                   for a in evaluate_window(_ventana("maduracion")))
