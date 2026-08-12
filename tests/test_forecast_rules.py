# -*- coding: utf-8 -*-
"""Anticipated rules (`forecast_rules`).

No network: the forecast is built by hand to provoke exactly each rule's condition. What is pinned
here is not the numbers -- those come from the literature and from the frequency report -- but the
behaviour that makes the rules defensible: that they do not fire without a forecast, that they
cross sensor and forecast where they should, and that they declare themselves as forecast and say
how many hours ahead.
"""
from datetime import datetime, timedelta, timezone

from coffeetech_rules_v5 import FORECAST, Reading, forecast_rules
from coffeetech_weather import HourlyForecast

AHORA = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
VARS = ("temperature_2m", "relative_humidity_2m", "precipitation", "shortwave_radiation",
        "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", "cloud_cover")


def _pronostico(horas=72, pasadas=24, **series):
    """A flat forecast with whatever variables the test needs written over it.

    It includes `pasadas` hours BEFORE `AHORA`, as the real request does (`PAST_DAYS`): without
    that stretch there is no overlap with telemetry and the sensor − API offset cannot be
    estimated.
    """
    base = {"temperature_2m": 22.0, "relative_humidity_2m": 70.0, "precipitation": 0.0,
            "shortwave_radiation": 0.0, "et0_fao_evapotranspiration": 0.0,
            "vapour_pressure_deficit": 0.3, "wind_speed_10m": 30.0, "cloud_cover": 50.0}
    total = pasadas + horas
    valores = {}
    for v in VARS:
        dado = series.get(v)
        if dado is None:
            valores[v] = [base[v]] * total
        elif isinstance(dado, list):
            # What the test defines applies to the FUTURE; the past stretch keeps the base
            # value.
            valores[v] = [base[v]] * pasadas + (dado + [base[v]] * horas)[:horas]
        else:
            valores[v] = [base[v]] * pasadas + [dado] * horas
    return HourlyForecast(latitude=-5.12, longitude=-79.05, elevation=1450.0,
                          times=[AHORA + timedelta(hours=h) for h in range(-pasadas, horas)],
                          values=valores)


def _sensor(horas=72, temp=22.0, rh=70.0, soil=50.0, rain=False, stage="maduracion"):
    return [Reading(ts=AHORA - timedelta(hours=h), air_temp=temp, air_rh=rh,
                    soil_moist=soil, rain=rain, stage=stage, altitude=1450.0)
            for h in range(horas, 0, -1)]


def _ids(alertas):
    return {a.rule_id for a in alertas}


def test_sin_pronostico_no_hay_alertas_anticipadas():
    """A farm without a coordinate, or the API being down, is not a failure: the diagnosis over
    measured data still goes out by its own path and here there is simply nothing to anticipate."""
    assert forecast_rules(_sensor(), None, now=AHORA) == []
    assert forecast_rules(_sensor(), _pronostico(horas=0), now=AHORA) == []


def test_toda_alerta_anticipada_se_declara_como_tal_y_dice_su_horizonte():
    """What was measured happened; what is forecast may not. If the interface cannot tell them
    apart, the system is asserting as fact something that is a prediction."""
    alertas = forecast_rules(_sensor(soil=78.0), _pronostico(precipitation=3.0), now=AHORA)
    assert alertas
    for a in alertas:
        assert a.forecast is True
        assert a.horizon_hours and a.horizon_hours > 0
        assert "PRONÓSTICO" in str(a)


def test_la_lixiviacion_avisa_antes_de_abonar_no_despues():
    """This rule's entire value is its timing: warning that the fertiliser washed away is useless.
    The threshold is rain forecast within 24 h, not rain that fell."""
    poca = forecast_rules(_sensor(), _pronostico(precipitation=0.5), now=AHORA)
    mucha = forecast_rules(_sensor(), _pronostico(precipitation=2.0), now=AHORA)
    assert "R7_N_LEACHING_FORECAST" not in _ids(poca)
    assert "R7_N_LEACHING_FORECAST" in _ids(mucha)      # 2 mm/h × 24 h = 48 mm ≥ 20


def test_el_encharcamiento_cruza_el_suelo_con_la_lluvia_prevista():
    """The same rain on drained soil does not waterlog. The sensor supplies the soil state, which
    the forecast cannot know: without that cross the rule would fire in every storm."""
    lluvia = _pronostico(precipitation=2.0)
    assert "R8_WATERLOGGING_FORECAST" not in _ids(forecast_rules(_sensor(soil=40.0), lluvia, now=AHORA))
    assert "R8_WATERLOGGING_FORECAST" in _ids(forecast_rules(_sensor(soil=78.0), lluvia, now=AHORA))


def test_la_broca_exige_grano_susceptible_y_seco_previo():
    """Two conditions, neither redundant: without grain there is nowhere to breed, which is what the
    stage is for, and without a preceding dry run there is no colonising flight, only rain."""
    lluvia_tras_seco = _pronostico(precipitation=[0.0] * 10 + [3.0] * 10)
    seco = _sensor(rain=False)
    assert "R3_BERRY_BORER_FORECAST" in _ids(forecast_rules(seco, lluvia_tras_seco,
                                                            stage="maduracion", now=AHORA))
    # With no susceptible grain the remaining path is sanitation, which is another rule.
    assert "R3_BERRY_BORER_FORECAST" not in _ids(forecast_rules(seco, lluvia_tras_seco,
                                                                stage="vegetativo", now=AHORA))
    # It was already raining: no transition.
    assert "R3_BERRY_BORER_FORECAST" not in _ids(forecast_rules(_sensor(rain=True), lluvia_tras_seco,
                                                                stage="maduracion", now=AHORA))


def test_el_vpd_promedia_TODAS_las_horas_y_no_solo_las_diurnas():
    """Kath et al. 2022 average over every hour of the season. Averaging only daytime hours -- which
    looked reasonable, since at night there is no photosynthesis to limit -- runs +0.3 kPa high at
    these coordinates and compared one quantity against a threshold derived from another. With the
    24 h mean the threshold is exceeded on 2.0 % of feb-jul hours against 40.2 % before.

    This test pins the correct quantity: a profile whose sunlit hours sit well above the threshold
    but whose nights are low does not fire, because its 24 h mean does not reach it.
    """
    dia = [0.0] * 6 + [900.0] * 12 + [0.0] * 6        # radiation: 12 h of sun
    # High by day (1.2), low by night (0.1): a daytime mean of 1.2 but a 24 h mean of ≈0.65.
    solo_de_dia = [0.1] * 6 + [1.2] * 12 + [0.1] * 6
    assert "R12_VPD_FRUIT_FILL" not in _ids(forecast_rules(
        _sensor(), _pronostico(shortwave_radiation=dia * 3, vapour_pressure_deficit=solo_de_dia * 3),
        stage="maduracion", now=AHORA))
    # A 24 h mean above the threshold: it does fire.
    sostenido = [0.7] * 6 + [1.3] * 12 + [0.7] * 6    # mean ≈ 1.1
    assert "R12_VPD_FRUIT_FILL" in _ids(forecast_rules(
        _sensor(), _pronostico(shortwave_radiation=dia * 3, vapour_pressure_deficit=sostenido * 3),
        stage="maduracion", now=AHORA))


def test_el_vpd_no_se_evalua_como_pico_horario():
    """The other error, the easy one: a high peak lasting a few hours is not a dry period. Treated
    as a peak, the warning fired on 95.8 % of days and informed of nothing."""
    pico = [0.2] * 9 + [3.0] * 3 + [0.2] * 12         # huge peak, mean ≈ 0.55
    assert "R12_VPD_FRUIT_FILL" not in _ids(forecast_rules(
        _sensor(), _pronostico(vapour_pressure_deficit=pico * 3), stage="maduracion", now=AHORA))


def test_el_vpd_solo_aplica_en_llenado_de_fruto():
    """Outside fructificacion and maduracion the threshold means nothing: there is no grain to
    fill."""
    f = _pronostico(vapour_pressure_deficit=1.1)
    assert "R12_VPD_FRUIT_FILL" not in _ids(forecast_rules(_sensor(), f, stage="vegetativo", now=AHORA))
    assert "R12_VPD_FRUIT_FILL" in _ids(forecast_rules(_sensor(), f, stage="fructificacion", now=AHORA))


def test_la_ventana_de_aplicacion_solo_sale_si_hay_algo_que_aplicar():
    """Measured over a year: there is a suitable band on 82 % of days. Emitting it always is advice
    nobody asked for, daily. The rule answers a question -- can I spray now? -- and only makes
    sense once the system has just said something has to be applied."""
    buen_viento = _pronostico(wind_speed_10m=10.0)
    assert "R13_SPRAY_WINDOW" not in _ids(forecast_rules(_sensor(), buen_viento, now=AHORA))
    assert "R13_SPRAY_WINDOW" in _ids(forecast_rules(_sensor(), buen_viento, now=AHORA,
                                                     pending_application=True))


def test_la_ventana_descarta_el_viento_fuera_de_rango_y_la_lluvia_posterior():
    """Above ~16 km/h drift grows fast; below ~5 there is thermal inversion. And a spray that washes
    off within the hour is a day's labour and the product thrown away."""
    def franja(**kw):
        return _ids(forecast_rules(_sensor(), _pronostico(**kw), now=AHORA, pending_application=True))

    assert "R13_SPRAY_WINDOW_NONE" in franja(wind_speed_10m=25.0)     # drift
    assert "R13_SPRAY_WINDOW_NONE" in franja(wind_speed_10m=2.0)      # thermal inversion
    assert "R13_SPRAY_WINDOW_NONE" in franja(wind_speed_10m=10.0, precipitation=1.0)  # washes off
    assert "R13_SPRAY_WINDOW" in franja(wind_speed_10m=10.0)


def test_la_roya_anticipada_no_se_emite_sin_desfase_medible():
    """The temperature threshold lives on the SENSOR's scale, and the regional model's can differ by
    4 °C on this plot. Without overlap to estimate that offset no zero is invented: a zero would
    assert that the plot coincides with its cell, and that has been worth 4 °C."""
    mojado = _pronostico(relative_humidity_2m=95.0, temperature_2m=22.0)
    # Sensor with no temperature: there is nothing to estimate the offset from.
    ciego = [Reading(ts=AHORA - timedelta(hours=h), air_temp=None, air_rh=70.0,
                     soil_moist=50.0, stage="maduracion", altitude=1450.0)
             for h in range(72, 0, -1)]
    assert "R1_RUST_FORECAST" not in _ids(forecast_rules(ciego, mojado, now=AHORA))
    assert "R1_RUST_FORECAST" in _ids(forecast_rules(_sensor(), mojado, now=AHORA))


def test_la_roya_anticipada_corrige_la_temperatura_al_sensor():
    """If the sensor runs 8 °C above the regional model, a regional forecast of 22 °C is 30 on the
    plot: outside the germination range. Using the raw value would raise an alert the plot does not
    justify."""
    mojado = _pronostico(relative_humidity_2m=95.0, temperature_2m=22.0)
    assert "R1_RUST_FORECAST" in _ids(forecast_rules(_sensor(temp=22.0), mojado, now=AHORA))
    assert "R1_RUST_FORECAST" not in _ids(forecast_rules(_sensor(temp=30.0), mojado, now=AHORA))


def test_la_ventana_de_hoja_mojada_tiene_que_ser_continua():
    """Scattered humid hours across three days germinate nothing: the spore needs CONTINUOUS free
    water. The run is measured consecutively, not accumulated."""
    intermitente = [95.0, 50.0] * 36            # alternating: no run reaches 12 h
    assert "R1_RUST_FORECAST" not in _ids(forecast_rules(
        _sensor(), _pronostico(relative_humidity_2m=intermitente), now=AHORA))
    continuo = [95.0] * 14 + [50.0] * 58
    assert "R1_RUST_FORECAST" in _ids(forecast_rules(
        _sensor(), _pronostico(relative_humidity_2m=continuo), now=AHORA))


def test_la_duracion_de_hoja_mojada_es_calibracion_del_sitio_no_literatura():
    """The 12 h do not come from a paper: the literature runs from 4-6 h (De Jong 1987 / Avelino
    2004, with alternating temperatures that cannot be verified here) to 24-48 h of continuous
    wetness (Gichuru et al. 2021, which is about continuous wetness and does not license a 6 h
    window). With this engine's leaf wetness proxy, RH >= 90 % in a 25 km cell, 6 h fires on 43.6 %
    of the year's days against 20.3 % at 12 h. This test stops anyone lowering them "to match the
    citation" without seeing that the proxy is what forces them longer."""
    assert FORECAST["rust_wet_hours"] == 12
    # An 8 h run would satisfy the literature's fast case and still does NOT fire here.
    ocho = [95.0] * 8 + [50.0] * 64
    assert "R1_RUST_FORECAST" not in _ids(forecast_rules(
        _sensor(), _pronostico(relative_humidity_2m=ocho), now=AHORA))


def test_el_minimo_termico_de_roya_sigue_la_literatura():
    """15 °C, not 18. De Jong et al. 1987 (doi 10.1007/BF01998091) puts the lower germination limit
    at 13 °C and Diniz et al. 2012 at 15.5. An 18 °C floor has no source behind it and drops real
    infection: 1763 hours a year fall between 15 and 18 °C with wet leaves."""
    assert FORECAST["rust_temp_min"] == 15.0
    # The sensor is pinned at 22 °C, which is what the forecast's PAST stretch carries, so the
    # offset comes out zero and the test measures the threshold rather than the correction. (With
    # the sensor at 16 while the past runs at 22, an offset of −6 would take a forecast of 16 down
    # to 10: correct, but a different thing would be under test.)
    fresco = _pronostico(relative_humidity_2m=95.0, temperature_2m=16.0)
    assert "R1_RUST_FORECAST" in _ids(forecast_rules(_sensor(temp=22.0), fresco, now=AHORA))
    # Below the published limit it still does not fire.
    frio = _pronostico(relative_humidity_2m=95.0, temperature_2m=11.0)
    assert "R1_RUST_FORECAST" not in _ids(forecast_rules(_sensor(temp=22.0), frio, now=AHORA))


def test_los_umbrales_llevan_las_unidades_de_la_api():
    """No conversion means no conversion error: wind in km/h and VPD in kPa, exactly as Open-Meteo
    returns them. This pins the contract so nobody "fixes" a threshold by converting it."""
    assert 1.0 < FORECAST["spray_wind_min_kmh"] < FORECAST["spray_wind_max_kmh"] < 100
    assert 0.1 < FORECAST["vpd_fruit_fill_kpa"] < 5.0


# ── Wiring through to the recommendation ────────────────────────────────────────────────

def test_las_alertas_anticipadas_llegan_a_recomendacion_marcadas_como_prevision():
    """The rules can be impeccable and useless if nothing calls them. This walks the stretch from
    the rule to the recommendation that goes out to the backend."""
    from coffeetech_recommendations import build_forecast_recommendations

    alertas = forecast_rules(_sensor(soil=78.0), _pronostico(precipitation=3.0), now=AHORA)
    recs = build_forecast_recommendations(alertas)
    assert recs and len(recs) == len(alertas)
    for r in recs:
        assert r.forecast is True
        assert r.horizon_h and r.horizon_h > 0
        assert "Previsión" in (r.forecast_detail or "")


def test_una_regla_anticipada_conserva_el_tipo_de_su_version_medida():
    """Forecast rust is still disease and forecast borer is still a pest. If the `_FORECAST` suffix
    dropped them into the management bucket, the interface would group them wrong and the grower
    would lose track of which problem is in front of them."""
    from coffeetech_recommendations import _type_for

    assert _type_for("R1_RUST_FORECAST") == _type_for("R1_RUST") == "disease"
    assert _type_for("R3_BERRY_BORER_FORECAST") == _type_for("R3_BERRY_BORER") == "pest"
    assert _type_for("R7_N_LEACHING_FORECAST") == _type_for("R7_N_LEACHING") == "management"
    assert _type_for("R12_VPD_FRUIT_FILL") == "thermal"


def test_la_franja_de_aplicacion_es_continua_y_en_hora_local():
    """Two defects that only appear running the rule against a real forecast.

    (1) The band has to be CONSECUTIVE. With suitable wind in the morning and at night but not in
    between, keeping the first and last hour would announce the whole day and invite spraying in
    exactly the hours that were discarded.

    (2) The hour is given to the grower in their own. The system travels in UTC on purpose, and
    this is the only point where it converts: when writing the band.
    """
    viento = [3.0] * 6 + [10.0] * 4 + [25.0] * 6 + [10.0] * 8   # two good stretches, a bad afternoon
    alertas = forecast_rules(_sensor(), _pronostico(wind_speed_10m=viento * 3),
                             now=AHORA, pending_application=True)
    ventana = next(a for a in alertas if a.rule_id == "R13_SPRAY_WINDOW")

    # The longest stretch is the 8 h one, not the 18 from the first suitable hour to the last.
    assert "8 h continuas" in ventana.agronomist_message
    assert "hora local" in ventana.agronomist_message
    assert "UTC" not in ventana.farmer_message


def test_los_umbrales_en_milimetros_salen_del_regimen_local():
    """Percentiles measured over three years of history at the pilot's coordinates and at its real
    altitude of 1823 m (24 h: p90=6.4, p98=16.3; 48 h: p98=25.7), not round numbers picked by eye.
    Asking the same series at 1450 m gives 9 / 17 / 29 instead, so the elevation is part of the
    measurement and not a detail of it.

    This test does not validate the agronomy -- that is the agronomist's review and the frequency
    report -- it stops anyone "rounding" them back into an assumption."""
    assert FORECAST["leaching_mm_24h"] == 16.0        # p98 of 24 h
    assert FORECAST["waterlog_mm_24h"] == 26.0        # p98 of 48 h
    assert FORECAST["borer_rain_mm"] == 6.0           # p90 of 24 h
    # The borer's has to tell rain from drizzle. At 5 mm/72 h it was exceeded on 65.9 % of the
    # year's hours: a threshold that filters nothing is the same as no threshold.
    assert FORECAST["borer_rain_mm"] > 5.0


def test_la_ventana_de_aplicacion_no_se_promete_como_certeza():
    """The 5 km/h lower limit falls in the densest part of this farm's wind distribution, and the
    forecast has MAE 1.99 km/h right there: it catches 79 % of the genuinely suitable hours but
    announces as suitable 43 % that are not.

    Giving an exact hour with that margin would sell precision that does not exist, so the message
    says it is a forecast and the agronomist's carries BOTH rates. Publishing only the hit rate kept
    the flattering half."""
    a = next(x for x in forecast_rules(_sensor(), _pronostico(wind_speed_10m=10.0),
                                       now=AHORA, pending_application=True)
             if x.rule_id == "R13_SPRAY_WINDOW")
    assert "previsión" in a.farmer_message.lower()
    assert str(FORECAST["spray_window_hit_rate_pct"]) in a.agronomist_message
    assert str(FORECAST["spray_window_false_alarm_pct"]) in a.agronomist_message


def test_la_lluvia_esperada_solo_es_noticia_tras_un_periodo_seco():
    """861 mm fall here each year and almost every 72 h window has water in it: announcing it always
    fires on 87.5 % of days, which is a true fact and a useless warning. Requiring the sensor to
    come from 24 dry hours drops it to 13.0 % and concentrates it in may-october -- the dry season,
    when rain returning changes decisions -- leaving 0-3 % from january to april.

    The counterfactual reproduces with `python scripts/frecuencia_reglas.py --sin-puerta-seca`."""
    lluvia = _pronostico(precipitation=[0.0] * 6 + [1.0] * 12)
    assert "R14_RAIN_EXPECTED" in _ids(forecast_rules(_sensor(rain=False), lluvia, now=AHORA))
    assert "R14_RAIN_EXPECTED" not in _ids(forecast_rules(_sensor(rain=True), lluvia, now=AHORA))


def test_la_lluvia_esperada_lleva_su_tasa_de_acierto_y_no_promete():
    """78.6 % at one day and 67.3 % at three, verified over a year at these coordinates. A rain
    warning without its rate invites more trust than the data supports, and "it will rain" is a
    promise the forecast cannot keep.

    The message carries hit rate AND false alarm: with 29 % false alarms, giving only the hit rate
    would leave the agronomist with a wrong idea of how much the warning can be leaned on."""
    a = next(x for x in forecast_rules(_sensor(rain=False),
                                       _pronostico(precipitation=[0.0] * 6 + [1.0] * 12), now=AHORA)
             if x.rule_id == "R14_RAIN_EXPECTED")
    assert "se espera" in a.farmer_message.lower()
    assert "lloverá" not in a.farmer_message.lower()
    assert str(FORECAST["rain_hit_rate_24h_pct"]) in a.agronomist_message
    assert str(FORECAST["rain_false_alarm_24h_pct"]) in a.agronomist_message
    assert str(FORECAST["hit_rate_dias"]) in a.agronomist_message


def test_ninguna_alerta_de_pronostico_se_presenta_como_accion_directa():
    """The general rule, and the reason it exists.

    `_VERIFICATION` was looked up by exact `rule_id`, so no anticipated rule had a verification step
    and all of them landed on "direct" -- you can apply this now. The result was that the system
    asserted MORE about what it merely predicts than about what it measured: `R1_RUST` asked for the
    leaf underside to be checked before applying copper while `R1_RUST_FORECAST`, the less certain
    version of the same thing, said to apply. A forecast is checked before acting; if another
    anticipated rule is added tomorrow, this test forces it to declare how it is checked.
    """
    from coffeetech_recommendations import _actionability_for, build_forecast_recommendations

    # A forecast that fires everything that can fire at once.
    dia = [0.0] * 6 + [900.0] * 12 + [0.0] * 6
    f = _pronostico(precipitation=[0.0] * 8 + [2.0] * 40, wind_speed_10m=10.0,
                    relative_humidity_2m=95.0, shortwave_radiation=dia * 3,
                    vapour_pressure_deficit=([0.3] * 6 + [1.2] * 12 + [0.3] * 6) * 3)
    alertas = forecast_rules(_sensor(soil=78.0, rain=False), f, stage="maduracion",
                             now=AHORA, pending_application=True)
    recs = build_forecast_recommendations(alertas)
    assert len(recs) >= 4, [r.rule_id for r in recs]

    directas = [r.rule_id for r in recs if _actionability_for(r) == "direct"]
    assert not directas, f"se presentan como acción directa sin verificar: {directas}"


def _con_pasado(dias_secos=95, mm_evento=None, hora_evento=None):
    """A forecast with a DEEP past, which is what the flowering rule now needs.

    `mm_evento` injects a downpour into the past to check the gate detects it: if there was already
    a water shock in the last 90 days, there is no dormancy left to break.
    """
    pasadas = dias_secos * 24
    lluvia_futura = [1.0] * 24 + [0.0] * 48          # 24 mm in the first 24 h: an inductive event
    f = _pronostico(horas=72, pasadas=pasadas, precipitation=lluvia_futura)
    if mm_evento:
        i = hora_evento if hora_evento is not None else pasadas // 2
        f.values["precipitation"][i] = mm_evento
    return f


def test_la_floracion_exige_profundidad_de_lluvia_tras_un_hueco_largo():
    """It fired almost daily on "a dry hour followed by a rainy one". Anthesis needs real rain --
    ≥10 mm/24 h (Frontiers 2025, over Boreux et al. 2016 and Lara-Estrada et al. 2024) -- after a
    period with no water shock at all."""
    # Drizzle: 0.2 mm/h is 4.8 mm in 24 h and does not reach the threshold.
    llovizna = _pronostico(horas=72, pasadas=95 * 24, precipitation=0.2)
    assert "R11_FLOWERING_EXPECTED" not in _ids(forecast_rules(
        _sensor(rain=False), llovizna, stage="floracion", now=AHORA))
    assert "R11_FLOWERING_EXPECTED" in _ids(forecast_rules(
        _sensor(rain=False), _con_pasado(), stage="floracion", now=AHORA))


def test_un_aguacero_en_el_hueco_cancela_la_floracion():
    """This is the whole gate: if there was already an inductive event within the 90 days, the plant
    had its shock and no latency accumulated. This used not to be checked -- the gate was 24 h of
    sensor -- which is why the rule fired in scattered months instead of once after the dry
    season."""
    con_evento = _con_pasado(mm_evento=FORECAST["flowering_rain_mm"] + 1.0)
    assert "R11_FLOWERING_EXPECTED" not in _ids(forecast_rules(
        _sensor(rain=False), con_evento, stage="floracion", now=AHORA))


def test_sin_pasado_suficiente_la_floracion_NO_se_afirma():
    """"It has not rained in 90 days" cannot be claimed on three days of data.

    That is the difference between not knowing and knowing it did not. A false positive here sends
    the grower to prepare boron and calcium for a flowering that is not coming, so in doubt the rule
    stays quiet. This is what breaks if someone lowers `PAST_DAYS` without checking who uses it.
    """
    corto = _pronostico(horas=72, pasadas=48, precipitation=[1.0] * 24 + [0.0] * 48)
    assert "R11_FLOWERING_EXPECTED" not in _ids(forecast_rules(
        _sensor(rain=False), corto, stage="floracion", now=AHORA))


def test_la_floracion_no_tiene_puerta_edafica_y_es_una_decision_declarada():
    """The second agronomic review REQUIRES the dry period to be measured in soil humidity, and it is
    right: the plant responds to root water tension, not to the rain gauge. It is not implemented
    because the two requirements that would make it valid are out of scope -- calibrating the
    capacitive probe against gravimetry and observing flowering in the field -- and a branch that
    can neither run nor be validated is exactly what potassium's unreachable "severe" band was.

    This test stops it slipping back in half-done. If anyone reintroduces the soil threshold, they
    have to reintroduce its calibration and its validation too, not just the constant.
    """
    fantasmas = [k for k in FORECAST if "flowering_soil" in k]
    assert not fantasmas, (
        f"Han vuelto {fantasmas} sin que haya con qué calibrarlos ni con qué validarlos. Ver "
        f"`docs/DECISIONES_DESCARTADAS.md`: está declarado como trabajo futuro, no como pendiente "
        f"de programar.")


def test_el_mensaje_declara_que_mide_lluvia_y_no_suelo():
    """The technician has to know that what is served is NOT what the physiology asks for.

    Staying quiet would present an operational device as if it were the physiological threshold,
    which is exactly what the review marked with a "No" in its technical note.
    """
    a = next(x for x in forecast_rules(_sensor(rain=False), _con_pasado(),
                                       stage="floracion", now=AHORA)
             if x.rule_id == "R11_FLOWERING_EXPECTED")
    m = a.agronomist_message
    assert "LÍMITE DECLARADO" in m
    assert "se mide en LLUVIA, no en la humedad del suelo" in m
    assert "NO validado contra floración observada" in m, (
        "Sin esa frase, la regla aparenta una validación de campo que no existe.")
