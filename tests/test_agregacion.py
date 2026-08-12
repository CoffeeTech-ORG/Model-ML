# -*- coding: utf-8 -*-
"""How hours are aggregated before being compared against a threshold.

Three rules are unmeasurable at 1823 m if the hours are aggregated wrongly, and in none of the
three cases is the threshold at fault:

  · `R1_RUST` measured the leaf wetness run WITHIN each natural day, so every humid night crossing
    midnight was split into two stretches of less than 12 h.
  · `R2_AMERICAN_LEAF_SPOT` compared the MEAN of the 96 h window against the 19-23 °C band. At
    1823 m that mean never exceeds 19.4 °C, although 23.2 % of the year's hours do fall in band.
  · `R3_BERRY_BORER` compared the day's MEAN against the 20-30 °C flight band. The daily mean
    reaches at most 19.5 °C, although 20.8 % of hours are in range.

An average is not a physiological band. These tests pin the aggregation with the case that tells
the two apart: mean outside the band, enough hours inside it.
"""
from datetime import datetime, timedelta, timezone

from coffeetech_rules_v5 import RULES, Reading, inferential_rules

# Local midnight in Peru (UTC−5). The engine's days are local, so these tests' windows start
# where the grower's day starts.
BASE = datetime(2026, 3, 1, 5, 0, tzinfo=timezone.utc)
ALT = 1823.0


def _win(horas, temp, rh, rain=lambda i: False, stage="maduracion", base=BASE):
    """An hourly window whose temperature and humidity depend on the local hour."""
    return [Reading(ts=base + timedelta(hours=i), air_temp=temp(i), air_rh=rh(i),
                    soil_moist=60.0, rain=rain(i), N=36.4, P=11.7, K=163.0,
                    stage=stage, altitude=ALT)
            for i in range(horas)]


def _ids(win, stage="maduracion"):
    return {a.rule_id for a in inferential_rules(win, altitude=ALT, stage=stage)}


# --------------------------------------------------------------------------- #
# R1 — the leaf wetness run is not cut at midnight
def test_roya_cuenta_la_racha_que_cruza_medianoche():
    """Wet from 20:00 to 08:00: 13 consecutive hours, but only 4 before midnight and 9 after.

    Measured by natural day, neither half reaches the 12 h required and the rule never fires. The
    spore does not know when the date changes.
    """
    def mojado(i):
        return (i % 24) >= 20 or (i % 24) < 8
    win = _win(96, temp=lambda i: 18.0,                       # inside 15-28 °C throughout
               rh=lambda i: 95.0 if mojado(i) else 60.0)      # rh_leafwet = 90 %
    assert "R1_RUST" in _ids(win)

    # Control: the same total humidity, spread so no run reaches 12 h.
    picado = _win(96, temp=lambda i: 18.0,
                  rh=lambda i: 95.0 if (i % 24) < 11 else 60.0)
    assert "R1_RUST" not in _ids(picado)


# --------------------------------------------------------------------------- #
# R2 — hours in band, not the window's mean
def test_ojo_de_gallo_con_la_media_fuera_de_banda():
    """8 h/day at 21 °C and 16 h/day at 12 °C: mean 15 °C, well below the 19-23 band.

    That is 32 hours in band over the 96 h window, above the 24 the rule asks for. With the mean the
    rule was unreachable at this altitude; with hours it measures what it says it measures.
    """
    def temp(i):
        return 21.0 if 10 <= (i % 24) < 18 else 12.0
    win = _win(96, temp=temp, rh=lambda i: 85.0,              # leaf_spot_rh = 80 %
               rain=lambda i: (i % 24) == 15)                 # one rainy hour each day
    tvals = [r.air_temp for r in win]
    assert sum(tvals) / len(tvals) < RULES["leaf_spot_temp_min"]   # the mean, outside the band
    assert "R2_AMERICAN_LEAF_SPOT" in _ids(win)

    # Control: same humidity and rain, but only 2 h/day in band -> 8 of 96, below 24.
    frio = _win(96, temp=lambda i: 21.0 if 10 <= (i % 24) < 12 else 12.0,
                rh=lambda i: 85.0, rain=lambda i: (i % 24) == 15)
    assert "R2_AMERICAN_LEAF_SPOT" not in _ids(frio)


# --------------------------------------------------------------------------- #
# R3 — flight hours, not the day's mean
def test_broca_con_la_media_diaria_fuera_de_banda():
    """A dry day and then a rainy one with 6 h at 22 °C and the rest at 14: daily mean 16 °C.

    The mean never reaches the flight band's 20 °C, but the female flies in the warm afternoon
    hours, not in the day's average.
    """
    def temp(i):
        return 22.0 if 11 <= (i % 24) < 17 else 14.0
    win = _win(48, temp=temp, rh=lambda i: 70.0, rain=lambda i: i >= 24)
    dia2 = [r.air_temp for r in win[24:]]
    assert sum(dia2) / len(dia2) < RULES["borer_flight_temp_min"]  # the mean, outside the band
    assert "R3_BERRY_BORER" in _ids(win)

    # Control: only 2 h in flight range, below the 4 the guard asks for.
    frio = _win(48, temp=lambda i: 22.0 if 11 <= (i % 24) < 13 else 14.0,
                rh=lambda i: 70.0, rain=lambda i: i >= 24)
    assert "R3_BERRY_BORER" not in _ids(frio)


def test_broca_no_declara_seco_un_dia_a_medio_observar():
    """"It did not rain yesterday" is a negative claim: it needs almost the whole day.

    That is what happens in the real february campaign, where the only day without rain is a 19 h
    fragment. Rain here is spread almost evenly through the day -- the 00:00-04:59 block holds
    20.9 % of the annual rain against 20.8 % of the time -- so the missing hours are not a harmless
    gap.
    """
    def temp(i):
        return 22.0 if 11 <= (i % 24) < 17 else 14.0
    completo = _win(48, temp=temp, rh=lambda i: 70.0, rain=lambda i: i >= 24)
    assert "R3_BERRY_BORER" in _ids(completo)

    # The same case, but only the dry day's last 19 h were observed.
    trozo = completo[5:]
    assert len([r for r in trozo if r.ts < BASE + timedelta(hours=24)]) == 19
    assert "R3_BERRY_BORER" not in _ids(trozo)
