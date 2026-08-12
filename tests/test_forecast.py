# -*- coding: utf-8 -*-
"""
Tests for the microclimate forecast component (non-circular random forest).

With a temporal split and a persistence baseline, they check the forest beats persistence where it
should (temperature, and air humidity at 6 h) and does NOT beat it where the honest thing is to say
it adds nothing (air humidity at 1 h, and soil humidity).

Requires numpy, pandas and scikit-learn:
    uv run --python 3.12 --with numpy --with pandas --with scikit-learn --with pytest pytest tests/test_forecast.py -v
"""
import json
import os

import pytest

import numpy as np

from coffeetech_forecast import (
    DEFAULT_HORIZONS_H, SERVED_HORIZONS_H, TARGET_FORMULATION, VALIDATION_FILE, Forecaster,
    _aplicar_formulacion, _deshacer_formulacion,
    horizontes_servibles, load_session, evaluate, train_and_persist, horizon_steps, TARGET_KEY,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOV_CSV = os.path.join(PROJECT_ROOT, "data", "data_sensors_san_ignacio_nov2025.csv")
FEB_CSV = os.path.join(PROJECT_ROOT, "data", "data_sensors_san_ignacio_feb2026.csv")


@pytest.fixture(scope="module")
def sessions():
    nov = load_session(NOV_CSV, "nov")
    feb = load_session(FEB_CSV, "feb")
    return nov, feb


def _skill(sessions, target, hours):
    nov, feb = sessions
    _, res = evaluate(nov, feb, target, horizon_steps(hours), verbose=False)
    return res["skill_rmse"]


# --------------------------------------------------------------------------- #
# The forest beats persistence where the forecast is useful
def test_temperatura_skill_2h(sessions):
    # Mind how this figure is read: it is the HOLDOUT's, testing on february's final 40 %, a
    # regime the model has already seen. Outside its campaign the same model comes out negative,
    # which is why 2 h is NOT served -- see `test_el_horizonte_de_2h_no_se_sirve`. It is kept
    # because it documents the difference between the three validations, which is half a section of
    # the report.
    skill = _skill(sessions, "celcius_grade_temperature", 2)
    assert 0.08 <= skill <= 0.20      # ≈ +13 % (reference +14 %)


def test_temperatura_skill_6h(sessions):
    skill = _skill(sessions, "celcius_grade_temperature", 6)
    assert skill >= 0.35              # ≈ +43 %


def test_humedad_aire_skill_6h(sessions):
    skill = _skill(sessions, "air_humidity_percent", 6)
    assert skill >= 0.10              # ≈ +16 %


# --------------------------------------------------------------------------- #
# Honesty: where persistence wins, the forest is not presented as an achievement
def test_humedad_aire_1h_gana_persistencia(sessions):
    skill = _skill(sessions, "air_humidity_percent", 1)
    assert skill < 0                  # at 1 h persistence is better


def test_humedad_suelo_no_se_pronostica(sessions):
    # Soil humidity is too slow: persistence beats it at every horizon.
    for hours in (1, 2, 6):
        assert _skill(sessions, "soil_humidity_percent", hours) < 0


# --------------------------------------------------------------------------- #
# Persisting the operational models plus their metadata
def test_train_and_persist_escribe_artefactos(sessions, tmp_path):
    nov, feb = sessions
    art = tmp_path / "artifacts"
    train_and_persist({"nov": nov, "feb": feb}, horizons_h=(6,), artifacts_dir=str(art))

    for target in ("celcius_grade_temperature", "air_humidity_percent"):
        assert (art / f"forecast_{TARGET_KEY[target]}_h6.joblib").exists()
    assert (art / "forecast_metadata.json").exists()

    on_disk = json.loads((art / "forecast_metadata.json").read_text(encoding="utf-8"))
    assert len(on_disk["models"]) == 2
    for m in on_disk["models"]:
        assert m["feat_cols"] and "lag_0" in m["feat_cols"]
        # The served model trains on EVERY campaign: its performance does not come from an
        # internal holdout -- which would test on a regime already seen -- but from the LOCO folds,
        # computed separately and attached. The metadata records both.
        assert m["entrenado_con"] == ["feb", "nov"]
        assert m["objetivo"] == TARGET_FORMULATION
        assert "loco" in m
    # It is recorded that soil humidity is not forecast.
    assert "suelo" in on_disk["soil_note"].lower()


# --------------------------------------------------------------------------- #
# Trained is not the same as served
#
# The 2 h horizon is trained and NOT served: it loses to persistence outside its own campaign, and
# the artifact is kept because one row of the report's table comes from it. These tests pin that
# separation, which a single default value is enough to undo by accident.
def test_el_horizonte_de_2h_no_se_sirve():
    """The constant that decides what is served, and why 2 h is not in it.

    Measured with LOCO over the three campaigns: at 2 h humidity falls to −23.0 % leaving may out
    and −20.8 % leaving november -- worse than repeating the current value -- and temperature
    collapses in november's fold (−6.5 %). Its positive figures come from the temporal blocks
    (+29.9 % and +18.7 %), which test on regimes the model has already seen.

    At 6 h both targets pass in all three folds: temperature +13.5 % in the worst and humidity
    +19.7 %.
    """
    assert 2 not in SERVED_HORIZONS_H
    assert 6 in SERVED_HORIZONS_H
    # And it is still TRAINED: retiring it from service is not deleting the measurement.
    assert 2 in DEFAULT_HORIZONS_H


def test_el_criterio_rechaza_un_horizonte_que_pierde_fuera_de_su_campana():
    """It is not enough for the number to be set by hand: the criterion has to reject it on its
    own."""
    def fila(horizonte, minimo):
        return {"target": "t", "horizon_h": horizonte, "loco": {"skill_min": minimo}}

    assert horizontes_servibles([fila(6, 0.25), fila(6, 0.19)]) == [6]
    # A single negative fold is enough to retire it: a forecast that fails in one season is not a
    # forecast that half works, it is one that cannot be let loose unwatched. Which is why the
    # MINIMUM over the folds decides and not the mean, which would hide exactly that case.
    assert horizontes_servibles([fila(2, 0.10), fila(2, -0.29)]) == []
    assert horizontes_servibles([fila(2, -0.09), fila(6, 0.25)]) == [6]


def test_sin_loco_el_criterio_no_inventa_un_veredicto():
    """An old validation JSON must not cause something to be served without LOCO evidence.

    The criterion falls back to the cross between campaigns, which is what those files were written
    under; and if that is missing too, nothing is served. What must NOT happen is absence of
    measurement reading as approval.
    """
    viejo = [{"target": "t", "horizon_h": 6, "entre_campanas": {"skill_min": 0.25}}]
    assert horizontes_servibles(viejo) == [6]
    assert horizontes_servibles([{"target": "t", "horizon_h": 6}]) == []


# --------------------------------------------------------------------------- #
# The target formulation: the CHANGE is predicted, not the value
#
# A random forest averages leaves and then averages trees, so its output cannot leave the range it
# learned on. With an absolute target that ceiling is a ceiling IN DEGREES and the model cannot warn
# of an extreme the pilot has not already seen. Predicting `y − lag_0` moves the ceiling onto the
# change, and `lag_0` -- the real reading -- has no cap. These tests pin that the round trip is exact
# and that the return leg respects physics.
def test_la_formulacion_de_cambio_va_y_vuelve_sin_perder_nada():
    y = np.array([20.0, 25.0, 12.0])
    base = np.array([18.0, 26.0, 15.0])

    directo = _aplicar_formulacion(y, base, delta=False)
    assert np.allclose(_deshacer_formulacion(directo, base, "celcius_grade_temperature", False), y)

    cambio = _aplicar_formulacion(y, base, delta=True)
    assert np.allclose(cambio, [2.0, -1.0, -3.0])
    assert np.allclose(_deshacer_formulacion(cambio, base, "celcius_grade_temperature", True), y)


def test_la_prediccion_de_humedad_no_se_sale_de_la_fisica():
    """`base + change` can land outside [0, 100]; a humidity of 103 % is not published.

    The clamp is only for relative humidity, which has hard physical limits. Temperature is NOT
    clamped on purpose: there is no physical maximum that applies here, and inventing one would be
    exactly the ceiling this formulation exists to remove.
    """
    base = np.array([95.0, 10.0])
    fuera = _deshacer_formulacion(np.array([15.0, -20.0]), base, "air_humidity_percent", True)
    assert np.allclose(fuera, [100.0, 0.0])

    calor = _deshacer_formulacion(np.array([9.0]), np.array([28.0]),
                                  "celcius_grade_temperature", True)
    assert np.allclose(calor, [37.0])


def test_la_constante_no_se_ha_separado_de_la_medicion():
    """What was measured outranks what was declared, and here it is checked on every `pytest`.

    `python coffeetech_forecast.py validar` already fails when they diverge, but that only protects
    whoever remembers to run it. The engine's thresholds have this same net.
    """
    ruta = os.path.join(PROJECT_ROOT, VALIDATION_FILE)
    if not os.path.exists(ruta):
        pytest.skip("Sin forecast_validation.json: correr `python coffeetech_forecast.py validar`")
    medido = json.loads(open(ruta, encoding="utf-8").read())["servibles"]
    assert medido == [float(h) for h in sorted(SERVED_HORIZONS_H)], (
        f"La medición dice {medido} y SERVED_HORIZONS_H dice {sorted(SERVED_HORIZONS_H)}")


def test_el_forecaster_no_carga_lo_que_no_sirve(sessions, tmp_path):
    """Even with the 2 h artifact on disk, it must not come out of `horizons()`.

    That is what stops an anticipated alert being raised from it: `main.py` walks
    `forecaster.horizons()`, so what is not loaded cannot speak.
    """
    nov, feb = sessions
    art = tmp_path / "artifacts"
    train_and_persist({"nov": nov, "feb": feb}, horizons_h=(2, 6), artifacts_dir=str(art))
    assert (art / "forecast_temp_h2.joblib").exists()      # trained and saved

    f = Forecaster.load(artifacts_dir=str(art))
    assert f.available()
    assert f.horizons() == [6]                              # but not served

    on_disk = json.loads((art / "forecast_metadata.json").read_text(encoding="utf-8"))
    por_horizonte = {m["horizon_h"]: m["servido"] for m in on_disk["models"]}
    assert por_horizonte[2] is False and por_horizonte[6] is True
    assert on_disk["criterio_servicio"]
