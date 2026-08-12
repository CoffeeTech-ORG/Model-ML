# -*- coding: utf-8 -*-
"""The forecast client (`coffeetech_weather`).

No network: a genuinely captured Open-Meteo response is served. What is checked is not that the API
works -- that is not ours to control -- but that WHAT IS DONE with its response is correct, which is
where the failures that matter live: the timezone and the altitude.
"""
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from coffeetech_forecast import (
    DEFAULT_HORIZONS_H,
    LONG_HORIZONS_NOT_PUBLISHED,
    MIN_HORIZON_H_FOR_WEATHER,
    WEATHER_TARGETS,
    _feature_columns,
    join_weather,
    latest_feature_row,
)
from coffeetech_weather import HourlyForecast, _parse, clear_cache, get_forecast


# A real excerpt of the response for (-5.1225, -79.0492) asking for 1450 m.
# The stamps arrive WITHOUT a timezone suffix even when UTC is requested: that is the thing to
# handle.
RESPUESTA = {
    "latitude": -5.125,
    "longitude": -79.0,
    "elevation": 1450.0,
    "timezone": "GMT",
    "utc_offset_seconds": 0,
    "hourly": {
        "time": ["2026-08-03T00:00", "2026-08-03T01:00", "2026-08-03T02:00", "2026-08-03T03:00"],
        "temperature_2m": [16.0, 15.6, 15.3, 15.1],
        "relative_humidity_2m": [82, 84, 85, 86],
        "precipitation": [0.0, 0.0, 0.2, 0.1],
        "precipitation_probability": [3, 5, 18, 12],
        "shortwave_radiation": [0.0, 0.0, 0.0, 0.0],
        "et0_fao_evapotranspiration": [0.0, 0.0, 0.0, 0.0],
        "vapour_pressure_deficit": [0.26, 0.25, 0.28, 0.28],
        "wind_speed_10m": [1.7, 2.1, 4.4, 6.1],
        "cloud_cover": [64, 71, 88, 83],
    },
}


def test_las_marcas_quedan_en_utc_explicito():
    """A stamp without a timezone is the failure that costs most and shows least.

    Read as local where it is UTC, a hub silent since yesterday reads as receiving data; and here,
    crossing forecast with telemetry, five hours of offset displace the diurnal cycle with nothing
    visibly breaking.
    """
    f = _parse(RESPUESTA)
    assert all(t.tzinfo is timezone.utc for t in f.times)
    assert f.times[0] == datetime(2026, 8, 3, 0, 0, tzinfo=timezone.utc)


def test_conserva_la_altitud_que_devolvio_la_api():
    """If the downscaling is not applied, the temperature carries the cell's bias -- several degrees
    in steep relief -- and that has to be detectable."""
    assert _parse(RESPUESTA).elevation == 1450.0


def test_las_unidades_son_las_de_los_umbrales():
    """VPD in kPa and wind in km/h: the same as the rules. No conversion means no conversion error, so
    this pins the contract."""
    f = _parse(RESPUESTA)
    assert max(f.series("vapour_pressure_deficit")) < 5      # kPa, not Pa
    assert max(f.series("wind_speed_10m")) < 200             # km/h, not scaled m/s


def test_la_ventana_corta_por_reloj_y_no_por_indice():
    """The series starts at a different hour depending on when it was requested, so cutting by
    position would give a different window each time."""
    # The stamps are offset by half an hour on purpose. `window()` reads the clock again a few
    # microseconds after this line, so a sample landing EXACTLY on the 24 h boundary would fall in
    # or out depending on that drift and the test would fail intermittently. With the offset no
    # sample touches the edge and the result depends only on the trimming, which is what is being
    # measured.
    ahora = datetime.now(timezone.utc)
    f = HourlyForecast(
        latitude=0.0, longitude=0.0, elevation=0.0,
        times=[ahora + timedelta(hours=h, minutes=30) for h in range(72)],
        values={"temperature_2m": [float(h) for h in range(72)]},
    )
    w = f.window(hours_ahead=24, span_hours=12)
    assert len(w) == 12
    assert all(24 <= (t - ahora).total_seconds() / 3600 < 36 for t in w.times)


def test_sin_coordenada_no_se_consulta():
    """A farm with no location has nothing to point at. `None` is the correct answer: inventing a
    coordinate would give a forecast that looks exactly as convincing as a real one."""
    clear_cache()
    assert asyncio.run(get_forecast(None, -79.0, 1450)) is None
    assert asyncio.run(get_forecast(-5.12, None, 1450)) is None


def test_un_fallo_de_red_no_tumba_el_diagnostico():
    """The forecast is an extra. If the API goes down, what the sensor measured still has to produce a
    diagnosis, which is why `None` is returned rather than propagating."""
    class ClienteRoto:
        async def get(self, *a, **k):
            raise RuntimeError("sin red")

    clear_cache()
    assert asyncio.run(get_forecast(-5.12, -79.0, 1450, client=ClienteRoto())) is None


# ── Crossing with telemetry ─────────────────────────────────────────────────────────────


def _telemetria(horas=6, tz="UTC"):
    """Synthetic telemetry every 2 min, with the columns the model uses."""
    n = horas * 30
    t = pd.date_range("2026-08-03T00:00", periods=n, freq="2min", tz=tz)
    return pd.DataFrame({
        "t": t,
        "session": "x",
        "celcius_grade_temperature": np.linspace(15, 25, n),
        "air_humidity_percent": np.linspace(90, 60, n),
        "soil_humidity_percent": np.full(n, 40.0),
        "precipitation_detected": np.zeros(n),
    })


def test_el_cruce_exige_zona_horaria_en_la_telemetria():
    """Crossing a series without a timezone against one in UTC displaces the diurnal cycle by five
    hours and the model gets worse with nothing visibly breaking. Better to fail here than to
    degrade silently."""
    d = _telemetria(tz=None)
    with pytest.raises(ValueError, match="zona horaria"):
        join_weather(d, _parse(RESPUESTA))


def test_los_acumulados_no_se_interpolan():
    """An hour's rain is a total, not a state: 2 mm at 14:00 is not 1 mm at 14:30. The hour's value is
    carried across; interpolating would smear it."""
    d = _telemetria(horas=4)
    unido = join_weather(d, _parse(RESPUESTA))
    lluvia = unido["api_precipitation"].dropna().unique()
    assert set(lluvia).issubset({0.0, 0.2, 0.1})   # only the hourly values, no intermediates


def test_el_estado_continuo_si_se_interpola():
    """Regional temperature between two known hours is a continuous state."""
    d = _telemetria(horas=4)
    unido = join_weather(d, _parse(RESPUESTA))
    temps = unido["api_temperature_2m"].dropna()
    assert len(temps.unique()) > 4        # more values than hours: interpolation happened
    assert temps.min() >= 15.0 and temps.max() <= 16.1


def test_sin_meteorologia_el_dataframe_sale_intacto():
    """A plot without a coordinate still trains and predicts on sensor data alone."""
    d = _telemetria()
    assert join_weather(d, None).equals(d)
    assert not [c for c in _feature_columns(d, "celcius_grade_temperature").columns
                if c.startswith(("api_", "anom_"))]


def test_el_clima_solo_entra_a_partir_del_horizonte_medido():
    """At 2 h cloud cover enters no target: in the cross between campaigns its sign changes by cell
    (+1.6 and +10.9 on temperature against +2.3 and −13.8 on humidity). The threshold is not a
    preference: it comes from that measurement."""
    d = join_weather(_telemetria(horas=12), _parse(RESPUESTA))
    corto = _feature_columns(d, "air_humidity_percent",
                             horizon_h=MIN_HORIZON_H_FOR_WEATHER - 1)
    largo = _feature_columns(d, "air_humidity_percent",
                             horizon_h=MIN_HORIZON_H_FOR_WEATHER)

    assert not [c for c in corto.columns if c.startswith("api_")]
    assert [c for c in largo.columns if c.startswith("api_")]


def test_la_nubosidad_entra_a_la_humedad_y_no_a_la_temperatura():
    """The split by target, which is a RESULT and not a preference.

    Measured with LOCO over the three campaigns and five seeds per cell, at 6 h: dropping cloud
    cover raises the temperature minimum from +3.8 % to +13.3 % (0/5 seeds in favour of including
    it) and lowers humidity's from +19.5 % to +14.1 % (5/5 in favour). Seed noise runs ±0.1 to ±0.4
    points, so neither effect is one lucky forest.

    The coherent explanation -- not a demonstrated one -- is that a 9 km cell's cloud cover serves
    humidity, a regional-scale phenomenon, while for temperature it makes the model learn how much
    the sun heats IN THAT CAMPAIGN, and that does not transfer: the sensor − regional model offset
    is already measured and changes by campaign.
    """
    d = join_weather(_telemetria(horas=12), _parse(RESPUESTA))
    temp = _feature_columns(d, "celcius_grade_temperature",
                            horizon_h=MIN_HORIZON_H_FOR_WEATHER)
    hr = _feature_columns(d, "air_humidity_percent", horizon_h=MIN_HORIZON_H_FOR_WEATHER)

    assert not [c for c in temp.columns if c.startswith("api_")]
    assert [c for c in hr.columns if c.startswith("api_")]
    assert "celcius_grade_temperature" not in WEATHER_TARGETS


def test_solo_entra_la_nubosidad_y_ninguna_otra_variable():
    """Radiation, VPD and wind were each tested separately and none contributed: two of them
    subtracted. Put in together they diluted the one that does contribute. This test stops them
    slipping back in "just in case", which is how the block of seven columns came about."""
    d = join_weather(_telemetria(horas=12), _parse(RESPUESTA))
    f = _feature_columns(d, "air_humidity_percent", horizon_h=MIN_HORIZON_H_FOR_WEATHER)
    clima = [c for c in f.columns if c.startswith(("api_", "anom_"))]
    assert clima == ["api_cloud_cover"], clima


def test_la_nubosidad_llega_al_modelo_sin_transformar():
    """The API's value is passed straight through. Any rescaling here would be one more conversion to
    get wrong, and the unit contract is already pinned in the client."""
    d = join_weather(_telemetria(horas=12), _parse(RESPUESTA))
    f = _feature_columns(d, "air_humidity_percent", horizon_h=MIN_HORIZON_H_FOR_WEATHER)
    assert np.allclose(f["api_cloud_cover"].dropna(), d["api_cloud_cover"].dropna())


def test_un_modelo_de_seis_horas_sin_clima_no_inventa_un_numero():
    """If the API goes down or the farm has no coordinate, the humidity model loses the inputs it
    learned on. `None` is the correct answer; filling with zeros would give a forecast that looks
    exactly as convincing as a real one. Same rule as the silent hub.

    It only applies to the HUMIDITY model: cloud cover is split by target, so the temperature one
    does not carry it and keeps predicting without the API. The complete forecast
    (`forecast_climate`) still needs both, so the output stays "nothing" rather than "half a
    forecast", which is correct.
    """
    entrenado = join_weather(_telemetria(horas=12), _parse(RESPUESTA))
    columnas = list(_feature_columns(entrenado, "air_humidity_percent",
                                     horizon_h=MIN_HORIZON_H_FOR_WEATHER).columns)
    assert any(c.startswith("api_") for c in columnas)

    sin_clima = _telemetria(horas=12)                      # the same telemetry, not joined
    assert latest_feature_row(sin_clima, "air_humidity_percent", columnas) is None

    # And with weather a row does come out: the `None` is for want of data, not an assembly
    # failure.
    assert latest_feature_row(entrenado, "air_humidity_percent", columnas) is not None

    # Temperature, by contrast, gets through without the API because it does not use it.
    cols_t = list(_feature_columns(entrenado, "celcius_grade_temperature",
                                   horizon_h=MIN_HORIZON_H_FOR_WEATHER).columns)
    assert latest_feature_row(sin_clima, "celcius_grade_temperature", cols_t) is not None


def test_los_horizontes_largos_no_se_publican():
    """The phase 11 plan expected 24 and 72 h by correction over the regional forecast. It was built,
    measured with the three campaigns, and does not win. The plan's rule is explicit -- horizons
    that do not win are not published -- and this pins it in the code."""
    for H in LONG_HORIZONS_NOT_PUBLISHED:
        assert H not in DEFAULT_HORIZONS_H


def test_la_razon_de_no_publicar_los_horizontes_largos_coincide_con_lo_medido():
    """It is not enough that they are out: the DECLARED reason has to be the measured one.

    At 72 h there is span enough to measure -- 514 usable hours -- and the horizon is still not
    served, for a reason that has nothing to do with sample size: only ONE campaign covers it, so
    none can be left out and tested on another. If anyone reopens this, let it be with the
    measurement in front rather than the old
    note.
    """
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta = os.path.join(raiz, "docs", "horizontes_largos.json")
    if not os.path.exists(ruta):
        pytest.skip("Sin docs/horizontes_largos.json: correr "
                    "`python scripts/horizontes_largos.py`")
    d = json.loads(open(ruta, encoding="utf-8").read())

    for r in d["resultados"]:
        if r["loco_aplicable"]:
            # Where validation WAS possible it must have come out negative: if one day it comes
            # out positive, this horizon deserves revisiting and this test forces a look.
            assert r["loco"]["skill_min"] <= 0, (
                f"{r['etiqueta']} a {r['horizon_h']:.0f} h ya supera a la persistencia en todos "
                f"sus pliegues ({r['loco']['skill_min']:+.1%}). El motivo declarado para no "
                f"publicarlo ha dejado de ser cierto: hay que revisar la decisión, no la prueba.")
        else:
            # Where it was not possible, the reason has to be the lack of campaigns of that
            # span.
            assert len(r["campanas_con_envergadura"]) < 2, (
                f"{r['etiqueta']} a {r['horizon_h']:.0f} h dice que LOCO no es aplicable pero hay "
                f"{len(r['campanas_con_envergadura'])} campañas con envergadura suficiente.")
