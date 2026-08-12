# -*- coding: utf-8 -*-
"""
The nutrient bands have to line up with the sensor's domain. Both ends, always.

Two ways they come apart, and both are invisible from the band table alone.

A band BELOW the domain floor can never be entered: potassium's "severe" asks for <78 mg/kg while
the calibrated domain starts at 90, so any lower reading is cut as out of domain first. Blocking
that band at the frontend is not enough either -- it stays alive in the declaration, in its
message and in the dose layer, and the agronomic review document shows it to the reviewer as a
rule that works.

At the other end, a remainder that is an ERROR LABEL swallows valid readings: with
`(55.0, "high"), (None, "out_of_domain")` and a `val < ceiling` cut, exactly 55.0 -- inside the
domain, because the guard is `val > dmax` -- falls through to `out_of_domain`, and the technician
reads that for a perfectly good reading.

So these tests cover the CLASS rather than the two cases: they walk each nutrient's domain and
require every declared band to be reachable and no valid reading to go unclassified.
"""
from __future__ import annotations

import pytest

from coffeetech_doses import ORGANIC_PRESCRIPTIONS, prescription_key
from coffeetech_rules_v5 import (
    NPK_BANDS, SENSOR_DOMAIN, _BAND_ES, _classify, _stage_adjust_band,
)

NUTRIENTES = ("N", "P", "K")
ETAPAS = ("plantula", "vegetativo", "floracion", "fructificacion", "maduracion", "cosecha")


def _recorrido(nut):
    """Test values inside the domain, including the EXACT edges.

    The edges are explicit because N's defect lived right there, and a sweep with a decimal step
    misses them through floating point error -- the first version of this check did exactly that and
    passed the broken edge.
    """
    dmin, dmax = SENSOR_DOMAIN[nut]
    paso = (dmax - dmin) / 400.0
    valores = [dmin, dmax, dmax - 1e-9]
    valores += [dmin + paso * i for i in range(401)]
    return sorted(v for v in valores if dmin <= v <= dmax)


@pytest.mark.parametrize("nut", NUTRIENTES)
def test_toda_banda_declarada_es_alcanzable(nut):
    """A band that cannot occur is a false promise: to the agronomist, to the frontend and to whoever
    reads the code."""
    alcanzadas = {_classify(v, NPK_BANDS[nut]) for v in _recorrido(nut)}
    declaradas = {nombre for _techo, nombre in NPK_BANDS[nut]}
    muertas = declaradas - alcanzadas
    assert not muertas, (
        f"{nut}: las bandas {sorted(muertas)} no se pueden alcanzar dentro del dominio "
        f"{SENSOR_DOMAIN[nut]}. O sobra la banda, o el dominio está mal — y cuál de las dos es "
        f"una pregunta agronómica, no de programación.")


@pytest.mark.parametrize("nut", NUTRIENTES)
def test_ninguna_lectura_valida_se_clasifica_fuera_de_dominio(nut):
    """The domain guard and the last band have to close at the same point.

    If the last band caps at `dmax`, `dmax` itself falls into the remainder. That is why each
    nutrient's last band carries `None`: it covers as far as the domain reaches.
    """
    for v in _recorrido(nut):
        banda = _classify(v, NPK_BANDS[nut])
        assert banda != "out_of_domain", (
            f"{nut}={v:g} está dentro del dominio {SENSOR_DOMAIN[nut]} y sin embargo se clasifica "
            f"«out_of_domain». El techo de la última banda no llega hasta el máximo.")


@pytest.mark.parametrize("nut", NUTRIENTES)
def test_la_ultima_banda_es_el_resto(nut):
    """A structural check, and the one that stops N's defect being reintroduced in a single stroke."""
    techos = [techo for techo, _n in NPK_BANDS[nut]]
    assert techos[-1] is None, f"{nut}: la última banda debe ser el resto (`None`)."
    assert all(t is not None for t in techos[:-1]), f"{nut}: sólo la última puede ser el resto."


@pytest.mark.parametrize("nut", NUTRIENTES)
def test_toda_banda_que_el_motor_puede_emitir_tiene_traduccion(nut):
    """Including the ones the stage adjustment produces, which are not in `NPK_BANDS`.

    `low_ripening` is one of those: `_stage_adjust_band` produces it, not the split by value. If any
    lacks a translation, the technician's message shows the English identifier -- which is exactly
    what happened with "out_of_domain".
    """
    emitibles = {_stage_adjust_band(nut, v, _classify(v, NPK_BANDS[nut]), etapa)
                 for v in _recorrido(nut) for etapa in ETAPAS}
    sin_traducir = {b for b in emitibles if b not in _BAND_ES}
    assert not sin_traducir, (
        f"{nut}: las bandas {sorted(sin_traducir)} se pueden emitir y no están en `_BAND_ES`, "
        f"así que saldrían sin traducir en el mensaje.")


@pytest.mark.parametrize("nut", NUTRIENTES)
def test_la_capa_de_dosis_no_espera_bandas_que_no_existen(nut):
    """Prescribing for an unreachable band is dead code that looks like coverage.

    K's dose layer listed "severe" among those that dose. It never ran, so that branch's apparent
    coverage was false.
    """
    emitibles = {_stage_adjust_band(nut, v, _classify(v, NPK_BANDS[nut]), etapa)
                 for v in _recorrido(nut) for etapa in ETAPAS}

    # Every prescription key the dose layer returns for this nutrient has to exist in the
    # catalogue, and has to come from a band the engine can really emit.
    claves = {prescription_key(f"NPK_{nut}", b, etapa)
              for b in emitibles for etapa in ETAPAS}
    for clave in claves - {None}:
        assert clave in ORGANIC_PRESCRIPTIONS, (
            f"{nut}: la banda emitible produce la clave {clave!r}, que no está en el catálogo.")

    # And the other way round: no UNREACHABLE band may still yield a prescription, because that
    # is the trace "severe" left in K.
    inalcanzables = {b for b in _BAND_ES if b not in emitibles}
    for banda in inalcanzables:
        assert prescription_key(f"NPK_{nut}", banda, "vegetativo") is None, (
            f"{nut}: la capa de dosis prescribe para la banda {banda!r}, que este nutriente no "
            f"puede emitir. Es código muerto con apariencia de cobertura.")


# --------------------------------------------------------------------------- #
# What the NPK agronomic review ruled (august 2026).
#
# Three directives, one of them a NO: nitrogen is not banded by stage. The agronomist confirmed it
# with his reason -- mineral N loses 3 to 55 % to volatilisation and leaching, and the sensor
# measures conductivity, so hardening the threshold in grain filling would fill the system with
# false alarms every time it rains -- and called the asymmetry against P and K a sound biological
# safeguard. These tests pin all three so nobody undoes them without rereading it.
ETAPAS_LLENADO = ("fructificacion", "maduracion")


def test_el_nitrogeno_no_se_bandea_por_etapa():
    """A deliberate NO, not an oversight. If someone "fixes" the asymmetry, this goes red."""
    for etapa in ETAPAS:
        for val in _recorrido("N"):
            cruda = _classify(val, NPK_BANDS["N"])
            assert _stage_adjust_band("N", val, cruda, etapa) == cruda, (
                f"El nitrógeno cambió de banda en {etapa} con N={val:g}. La revisión agronómica lo "
                f"prohíbe expresamente: sobre una lectura de conductividad, bandear por etapa "
                f"produce falsas alarmas con cada lluvia que diluya el suelo.")


def test_el_fosforo_endurece_en_plantula_Y_en_floracion():
    """floracion was added by the review: breaking dormancy and sustaining anthesis demands a peak of
    ATP, and in Cajamarca's fixing soils a reading at the edge ends in flower abortion."""
    val = 15.0   # inside "adequate" by value, at the lower edge
    assert _classify(val, NPK_BANDS["P"]) == "adequate"
    for etapa in ("plantula", "floracion"):
        assert _stage_adjust_band("P", val, "adequate", etapa) == "low", (
            f"En {etapa} el fósforo debe endurecer su piso.")
    for etapa in ("vegetativo", "fructificacion", "maduracion", "cosecha"):
        assert _stage_adjust_band("P", val, "adequate", etapa) == "adequate", (
            f"En {etapa} NO debe endurecer: sólo plántula y floración están justificadas.")


def test_el_exceso_de_potasio_explica_el_antagonismo_y_nombra_la_ceniza():
    """Forbidding "potassium fertiliser" without saying "ash" leaves out exactly what the grower has
    to hand. And without the mechanism -- K blocking Ca and Mg uptake -- the instruction makes no
    sense."""
    from datetime import datetime, timezone
    from coffeetech_rules_v5 import Reading, diagnose_npk

    a = next(x for x in diagnose_npk(Reading(ts=datetime(2026, 3, 15, tzinfo=timezone.utc),
                                             N=36.4, P=11.7, K=245.0, stage="vegetativo",
                                             altitude=1823.0)) if x.rule_id == "NPK_K")
    assert "ceniza" in a.farmer_message
    assert "calcio y magnesio" in a.farmer_message
    assert "Antagonismo catiónico K-Ca-Mg" in a.agronomist_message


def test_el_mensaje_de_nitrogeno_cambia_de_consecuencia_segun_la_etapa():
    """The band does not move, but the cost of the same deficit does: in vegetativo next year's
    bandolas are lost; in grain filling the plant cannibalises its own leaves (die-back)."""
    from datetime import datetime, timezone
    from coffeetech_rules_v5 import Reading, diagnose_npk

    def _msg(etapa):
        return next(x for x in diagnose_npk(Reading(ts=datetime(2026, 3, 15, tzinfo=timezone.utc),
                                                    N=22.0, P=11.7, K=163.0, stage=etapa,
                                                    altitude=1823.0))
                    if x.rule_id == "NPK_N").farmer_message

    assert "ramas nuevas vigorosas" in _msg("vegetativo")
    for etapa in ETAPAS_LLENADO:
        assert "sacrificando sus propias ramas" in _msg(etapa), (
            f"En {etapa} el mensaje debe advertir del die-back.")
    assert _msg("vegetativo") != _msg("maduracion")
