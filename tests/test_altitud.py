# -*- coding: utf-8 -*-
"""The site's altitude and what it decides.

It exists because of an expensive mistake: the pilot's altitude was typed as 1450 m when the hub
sits at 1823. That number is not decorative -- it decides the altitude band, and the band decides
the weight of four pest and disease rules. These tests pin that behaviour so a change of altitude
cannot pass unnoticed again.
"""
from datetime import datetime, timedelta, timezone

from coffeetech_rules_v5 import (
    ALTITUDE_RISK, COFFEE_ALTITUDE_MAX_M, COFFEE_ALTITUDE_MIN_M, Reading, altitude_band,
    inferential_rules,
)
from coffeetech_weather import PILOT_ELEVATION, PILOT_LAT, PILOT_LON, check_elevation

AHORA = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


def _ventana(alt, stage):
    """Weather favourable to rust, so the rule fires and its severity can be inspected."""
    return [Reading(ts=AHORA - timedelta(hours=h), air_temp=21.0, air_rh=95.0, soil_moist=60.0,
                    rain=True, stage=stage, altitude=alt)
            for h in range(96, 0, -1)]


def test_el_piloto_esta_a_1823_no_a_1450():
    """The 1450 m were typed by hand when the campaigns were loaded and stayed in the `altitude_masl`
    column of both CSVs. The hub is at 1823. The CSVs are not corrected -- they are the record of
    what was loaded -- so the good value lives here, and this test stops it being reverted."""
    assert PILOT_ELEVATION == 1823.0


def test_la_banda_altitudinal_cambia_con_la_correccion():
    """1450 fell in "medium"; 1823 falls in "high". Not a nuance: the band decides the weight."""
    assert altitude_band(1450.0) == "medium"
    assert altitude_band(PILOT_ELEVATION) == "high"
    assert ALTITUDE_RISK["medium"]["rust"] == "medium"
    assert ALTITUDE_RISK["high"]["rust"] == "low"
    assert ALTITUDE_RISK["high"]["cold_phoma"] == "high"   # cold does gain weight up high


def test_en_altura_la_roya_baja_de_alerta_a_aviso_fuera_de_fructificacion():
    """The "low" weight degrades the alert, drops the referral to the technician and adds the
    low-risk note. That is what the literature supports: rust pressure declines with altitude."""
    r = [a for a in inferential_rules(_ventana(PILOT_ELEVATION, "vegetativo"),
                                      altitude=PILOT_ELEVATION, stage="vegetativo")
         if a.rule_id == "R1_RUST"]
    assert r, "la regla debería disparar con este clima"
    assert r[0].severity == "warning"
    assert r[0].refer is False
    assert "riesgo es bajo" in r[0].farmer_message


def test_con_carga_de_fruto_la_roya_conserva_la_alerta_pese_a_la_altura():
    """Rust severity rises with fruit load (López-Bravo, Virginio-Filho & Avelino 2012: +28.9 %
    incidence and +129.2 % severity at 500 fruiting nodes). That step offsets altitude's low weight,
    so in maduracion the alert holds. Without this test someone could "simplify" the bump and lose
    the interaction."""
    r = [a for a in inferential_rules(_ventana(PILOT_ELEVATION, "maduracion"),
                                      altitude=PILOT_ELEVATION, stage="maduracion")
         if a.rule_id == "R1_RUST"]
    assert r and r[0].severity == "alert" and r[0].refer is True


def test_la_altitud_declarada_manda_sobre_la_del_mapa():
    """The 90 m elevation model smooths the relief and the user knows where they put the equipment,
    so the declared one wins. The map's serves to WARN that coordinate and altitude do not describe
    the same point.

    On the pilot the check passes: the DEM gives 1809 m at the farm's pin and the hub is at 1823,
    14 m apart. A gap of the order of 100 m means the two describe different places -- at the
    village centre, 2.28 km away, the DEM differs by 121 m -- and not that the grower typed the
    altitude wrong."""
    assert check_elevation(PILOT_LAT, PILOT_LON, 1823.0) == 1823.0
    # With no coordinates there is nothing to check against, but the declared one still
    # serves.
    assert check_elevation(None, None, 1823.0) == 1823.0


def test_sin_altitud_no_se_inventa_una_banda():
    """`None` means unknown, and it is not replaced by a number.

    No signature carries a numeric default, and none can: every altitude lands in a band and every
    band modulates something. A mid-range guess falls in `medium`, which raises American leaf spot
    to weight `high`, so a farm without an altitude would receive an invented modulation wearing
    the face of data.
    """
    assert altitude_band(None) is None
    r = [a for a in inferential_rules(_ventana(None, "maduracion"), altitude=None,
                                      stage="maduracion")
         if a.rule_id == "R1_RUST"]
    assert r, "la regla debería disparar con este clima"
    # Unmodulated: it keeps the severity its rule declares, and the referral.
    assert r[0].severity == "alert"
    assert r[0].refer is True
    # And it says so, rather than recording a weight nobody measured.
    assert "Sin altitud registrada" in r[0].agronomist_message
    assert "Peso por altitud" not in r[0].agronomist_message


def test_el_cero_no_es_una_altitud():
    """`0` was the backend's "not filled in", and it falls in the MAXIMUM risk band.

    The `altitude` column was `NOT NULL`, so an uncharacterised farm stored 0 -- and
    `altitude_band(0)` gives `low`, which raises rust and borer to weight `high`. The farm least is
    known about received the engine's most aggressive modulation. The backend can no longer send it
    (the column is nullable and a script turns the zeros into NULL), but `main.py` keeps the guard
    for old rows and other clients.

    The window is WIDE on purpose: it catches the empty field and the coarse typo, it does not
    arbitrate where coffee grows.
    """
    assert altitude_band(0.0) == "low"          # which is why a 0 must not reach the engine
    assert ALTITUDE_RISK["low"]["rust"] == "high"
    assert ALTITUDE_RISK["low"]["borer"] == "high"

    assert COFFEE_ALTITUDE_MIN_M <= PILOT_ELEVATION <= COFFEE_ALTITUDE_MAX_M
    for implausible in (0.0, -10.0, 145.0, 3500.0):
        assert not (COFFEE_ALTITUDE_MIN_M <= implausible <= COFFEE_ALTITUDE_MAX_M), implausible
    for real in (900.0, 1450.0, 1823.0, 2000.0):
        assert COFFEE_ALTITUDE_MIN_M <= real <= COFFEE_ALTITUDE_MAX_M, real
