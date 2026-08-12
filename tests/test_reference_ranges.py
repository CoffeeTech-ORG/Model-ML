# -*- coding: utf-8 -*-
"""
Tests for the publishable reference block (`reference_ranges`).

The Reports chart consumes these values instead of keeping a range table of its own, which would
fall out of step and call "out of range" a potassium the diagnosis calls adequate. So what is
tested here is NOT that the numbers are particular ones, but that they AGREE with what the
diagnosis applies. Change a band on one side only and these tests fail.

Run with:  python -m pytest tests/ -q
"""
from datetime import datetime, timezone

import pytest

from coffeetech_rules_v5 import (
    SENSOR_DOMAIN,
    VALID_STAGES,
    Reading,
    diagnose_npk,
    reference_ranges,
)


def _reading(stage, **npk):
    return Reading(ts=datetime(2026, 7, 18, tzinfo=timezone.utc),
                   N=npk.get("N"), P=npk.get("P"), K=npk.get("K"),
                   stage=stage, altitude=1450.0)


def _severity_from_engine(nut, value, stage):
    """The severity the engine really emits for that reading."""
    alerts = [a for a in diagnose_npk(_reading(stage, **{nut: value}))
              if a.rule_id == f"NPK_{nut}"]
    assert alerts, f"el motor no emitió alerta de {nut} para {value}"
    return alerts[0].severity


# --------------------------------------------------------------------------- #
# The essential point: the published reference equals what the diagnosis applies
@pytest.mark.parametrize("stage", VALID_STAGES)
@pytest.mark.parametrize("nut", ["N", "P", "K"])
def test_bandas_publicadas_coinciden_con_el_diagnostico(nut, stage):
    """At the midpoint of each published band, the engine must emit that same severity."""
    for band in reference_ranges(stage)["metrics"][nut]["bands"]:
        probe = (band["from"] + band["to"]) / 2.0
        assert _severity_from_engine(nut, probe, stage) == band["severity"], (
            f"{nut}={probe} en etapa {stage}: la referencia dice '{band['severity']}' "
            f"pero el motor emite otra severidad"
        )


@pytest.mark.parametrize("stage", VALID_STAGES)
@pytest.mark.parametrize("nut", ["N", "P", "K"])
def test_el_rango_optimo_no_dispara_alerta_ni_warning(nut, stage):
    """If the reference says optimal, the engine cannot be warning in that range."""
    optimal = reference_ranges(stage)["metrics"][nut]["optimal"]
    if optimal is None:
        return  # no adequate band in that stage; covered by its own test
    lo, hi = optimal
    for probe in (lo, (lo + hi) / 2.0, hi - 0.01):
        assert _severity_from_engine(nut, probe, stage) == "info"


# --------------------------------------------------------------------------- #
# Trimming to the sensor's domain
@pytest.mark.parametrize("nut", ["N", "P", "K"])
def test_las_bandas_no_salen_del_dominio_calibrado(nut):
    """Outside the domain the engine cuts before classifying; publishing a band there would be a
    lie."""
    dmin, dmax = SENSOR_DOMAIN[nut]
    for band in reference_ranges()["metrics"][nut]["bands"]:
        assert band["from"] >= dmin
        assert band["to"] <= dmax


def test_potasio_no_publica_banda_severa_inalcanzable():
    """Severe K is <78 mg/kg but the sensor starts at 90: in practice that band does not exist."""
    labels = [b["label"] for b in reference_ranges()["metrics"]["K"]["bands"]]
    assert "severe" not in labels


# --------------------------------------------------------------------------- #
# Stage adjustments: the ones the engine really applies
def test_potasio_endurece_en_maduracion():
    """The moderate band becomes an alert in fructificacion and maduracion (zero tolerance)."""
    def moderate_severity(stage):
        bands = reference_ranges(stage)["metrics"]["K"]["bands"]
        return next(b["severity"] for b in bands if b["from"] == 90.0)

    assert moderate_severity("vegetativo") == "warning"
    assert moderate_severity("maduracion") == "alert"
    assert moderate_severity("fructificacion") == "alert"


def test_fosforo_sin_banda_adecuada_en_plantula_queda_documentado():
    """In plantula the engine raises P's floor to 20, and 20+ already falls in 'high': no adequate
    band is left. That is a real edge of the engine; the reference has to declare it rather than
    invent an optimum the diagnosis would never confirm."""
    ref = reference_ranges("plantula")["metrics"]["P"]
    assert ref["optimal"] is None
    assert "plantula" in ref["note"]


# --------------------------------------------------------------------------- #
# What deliberately has NO band
def test_humedad_de_suelo_no_publica_banda_absoluta():
    """The capacitive probe is not calibrated in volume: the engine derives the threshold from each
    plot's wet-dry envelope. A fixed % would be invented."""
    soil = reference_ranges()["metrics"]["soil_humidity"]
    assert soil["kind"] == "relative"
    assert soil["optimal"] is None


def test_humedad_de_aire_publica_umbrales_no_banda():
    air = reference_ranges()["metrics"]["air_humidity"]
    assert air["kind"] == "threshold"
    assert air["optimal"] is None
    assert [t["above"] for t in air["thresholds"]] == [85.0, 90.0]


@pytest.mark.parametrize("metric", ["temperature", "air_humidity"])
def test_los_umbrales_traen_etiqueta_corta_para_el_grafico(metric):
    """The chart label cannot be the whole sentence: it does not fit and collides with the others."""
    for threshold in reference_ranges()["metrics"][metric]["thresholds"]:
        assert threshold["short_es"]
        assert len(threshold["short_es"]) <= 16


def test_temperatura_publica_bandas_que_coinciden_con_env_alerts():
    """Temperature was the only metric with a band and no `bands`: the interface could not issue a
    verdict and it was the only row without a chip. The bands have to come from the same rule that
    runs in the diagnosis."""
    from coffeetech_rules_v5 import env_alerts

    bands = reference_ranges()["metrics"]["temperature"]["bands"]
    assert bands, "temperatura debe publicar bandas"

    for band in bands:
        probe = (band["from"] + band["to"]) / 2.0
        alerts = env_alerts(_reading("vegetativo").__class__(
            ts=datetime(2026, 1, 1, tzinfo=timezone.utc), air_temp=probe))
        expected = alerts[0].severity if alerts else "info"
        assert band["severity"] == expected, f"T={probe}: {band['severity']} != {expected}"


def test_temperatura_adecuada_coincide_con_el_optimo():
    metric = reference_ranges()["metrics"]["temperature"]
    adequate = [b for b in metric["bands"] if b["label"] == "adequate"]
    assert len(adequate) == 1
    assert [adequate[0]["from"], adequate[0]["to"]] == metric["optimal"]


def test_nitrogeno_se_marca_provisional():
    """The engine treats all N as an untraceable proxy; the frontend must be able to draw it
    differently."""
    assert reference_ranges()["metrics"]["N"]["provisional"] is True


# --------------------------------------------------------------------------- #
def test_etapa_invalida_no_revienta_y_cae_a_la_base():
    ref = reference_ranges("etapa-que-no-existe")
    assert ref["stage"] is None
    assert ref["metrics"]["K"]["optimal"] == reference_ranges()["metrics"]["K"]["optimal"]


def test_unidades_son_las_del_sensor():
    """The sensor delivers mg/kg of soil, not mg/L. The unit travels in the payload so no consumer
    has to write it by hand and get it wrong."""
    metrics = reference_ranges()["metrics"]
    assert all(metrics[n]["unit"] == "mg/kg" for n in ("N", "P", "K"))
    assert metrics["temperature"]["unit"] == "°C"
