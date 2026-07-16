# -*- coding: utf-8 -*-
"""
CoffeeTech — agronomic rule engine v5
=====================================
El Milagro pilot, San Ignacio (Cajamarca, Peru) · APROCASSI cooperative (USDA-Organic + Fairtrade)

Every alert speaks in TWO VOICES plus a referral flag:
  * farmer_message      : plain language, no jargon, aimed at an action.
  * agronomist_message  : technical and traceable (band, threshold, method, organic product,
                          citation).
  * refer / referral_note : raised ONLY when expert judgement or a laboratory is needed (out of
                          domain, severe deficiency, liming, probable outbreak), to keep alert
                          fatigue down.

Literature anchors: Cenicafé Avance Técnico 497 (Sadeghian 2018); INIA-MIDAGRI 2025 (Salgado &
Ottos); FAO Arabica manual; Jaramillo et al. 2009; Bebber et al. 2016. Laid over the sensor's
calibrated domain.

Site decisions, configurable: coffee berry borer action threshold 2 %; copper cap 4 kg Cu/ha/year
(EU).

Stage names ("plantula", "floracion", …) stay in Spanish because they are values arriving from the
backend, not the engine's internal identifiers. The messages stay in Spanish because a grower in
San Ignacio reads them.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, List, Dict
from datetime import datetime, timedelta

# --------------------------------------------------------------------------- #
# ACTUATION CATEGORIES
CAT_A = "A"  # Real time, from the sensor
CAT_B = "B"  # Laboratorio periódico (ingreso manual de análisis)
CAT_C = "C"  # Base management (static note)

# THE SENSOR'S CALIBRATED DOMAIN (mg/kg) — El Milagro
SENSOR_DOMAIN = {"N": (15.0, 55.0), "P": (4.0, 22.0), "K": (90.0, 250.0)}

# DIAGNOSTIC BANDS ANCHORED TO LITERATURE (mg/kg). (exclusive_upper_bound, label)
#: Interpretation bands, each with its ceiling. The last one carries `None`: it is the remainder,
#: so it has to cover up to the domain's maximum.
#:
#: Two things the bands have to satisfy against `SENSOR_DOMAIN`, and `tests/test_npk_bandas.py`
#: walks every nutrient's domain to enforce them:
#:
#: 1. NO VALID READING FALLS THROUGH. The cut is `val < ceiling` and the domain guard is
#:    `val > dmax`, so a nutrient whose last named band ends exactly at the domain maximum leaves
#:    that maximum classified by the remainder. The remainder must therefore be a real band -- N's
#:    is `high`, which is what [46, 55) already says -- and never an error label.
#:
#: 2. NO BAND IS UNREACHABLE. A band below the domain minimum can never be entered, because a
#:    reading under it is cut as out of domain before classification, and it stays alive in the
#:    message and in the dose layer with nothing able to trigger it.
#:
#: Where a band's edge and the domain's edge disagree, which of the two moves is a calibration and
#: agronomy question for the review, not a decision to take here for convenience.
NPK_BANDS = {
    "N": [(20.0, "severe"), (27.0, "low"), (46.0, "adequate"), (None, "high")],
    "P": [(6.0, "severe"), (10.0, "low"), (20.0, "adequate"), (None, "high")],
    "K": [(115.0, "moderate"), (156.0, "adequate"), (235.0, "high"), (None, "excess")],
}

# Coffea arabica thermal optimum 18-23 °C, with cup quality lost above 23 °C: DaMatta & Ramalho
# 2006 (Braz. J. Plant Physiol. 18(1):55-81) and DaMatta et al. 2018. Above 23 °C fruit development
# and ripening accelerate and the cup suffers.
ENV = {"temp_cold": 15.0, "temp_warn": 23.0, "temp_hot": 32.0,
       "rh_fungal": 85.0, "rh_leafwet": 90.0, "broca_action_pct": 2.0, "soil_saturation": 80.0}
#: Copper cap. The binding instrument is Commission Implementing Regulation (EU) 2018/1981, which
#: restricts copper to "28 kg/ha over 7 years, that is an average of 4 kg/ha/year", plus Regulation
#: (EU) 2018/848 for organic production. Before february 2019 the limit was 6 kg/ha/year.
#:
#: THIS IMPLEMENTATION IS STRICTER THAN THE REGULATION, on purpose. `CopperLedger` applies a hard
#: ANNUAL cap and does not allow the 7-year averaging the regulation does permit -- exceeding 4 kg
#: in a difficult year while the seven-year mean holds. Carrying that averaging would need seven
#: years of application history the system does not have, and an annual cap is the safe side for a
#: certified cooperative. Documented so it does not read as ignorance of the rule.
#:
#: With Demeter or Naturland certification the cap would drop to ~3 kg/ha/year.
COPPER_CAP_KG_HA_YR = 4.0
VALID_STAGES = ["plantula", "vegetativo", "floracion", "fructificacion", "maduracion", "cosecha"]

#: A stage the backend reported that the mapping did not recognise. Deliberately NOT a valid stage.
#:
#: Defaulting an unmapped name to a real stage is not a harmless fallback. Pick "vegetativo" and
#: the water rule says there is no need to irrigate, because the dry spell helps flowering set
#: evenly -- advice that is the exact opposite of what a section in grain filling needs, delivered
#: with nothing on screen to show a guess was made.
#:
#: With the sentinel, no stage-gated rule recognises it and all of them degrade to their neutral
#: branch -- water management lands on `R10_DEFICIT_MILD`, which recommends mulching and holds in
#: any stage -- instead of picking a direction at random. And it is DECLARED in the technician's
#: voice, the same way an unrecorded altitude is.
STAGE_UNKNOWN = "no_registrada"

# On differentiating by stage (Cenicafé Avance 497 for production against Avance 532 for the
# nursery): the interpretation CUTS of a soil analysis do not change between production stages.
# What changes is the FERTILISATION -- timing and emphasis -- and the urgency of acting. So only
# two differences exist: (a) plantula raises P's floor; (b) fructificacion and maduracion put K at
# zero tolerance. No separate cuts are invented for vegetativo or floracion; they share the band.

# --- RISK modulation by altitude band (does NOT affect soil thresholds) ---
# Evidence: coffee berry borer is inversely related to altitude, peaking below 1600 m (Constantino
# et al., Bull. Ent. Res. 110(2):207-218, Nariño study); rust and borer both decline above
# 1200-1500 m (climatic escape, SENASA/Redagrícola); American leaf spot optimum 1100-1550 m;
# Phoma and cold are limiting above 1600 m (AgroPerú/BASF). "weight" is the rule's priority.
#: Altitude band cuts, in metres. They modulate EVERY pest and disease rule -- a `very_low` weight
#: suppresses the rule outright -- so they are agronomic thresholds in their own right and belong
#: where the agronomist can see them, not buried inside the function.
ALTITUDE_BANDS = {"low_max": 1200.0, "medium_max": 1600.0, "high_max": 1900.0}

#: PLAUSIBILITY window for a coffee plot's altitude. Not an agronomic band: a data guard. Outside
#: it the value is treated as UNKNOWN and nothing is modulated.
#:
#: It exists because the database held farms with `altitude = 0` -- the column was NOT NULL and 0
#: meant "not filled in" -- and that 0 falls in band `low`, the profile of MAXIMUM weight for rust
#: and borer. An uncharacterised farm received the engine's most aggressive modulation.
#:
#: DELIBERATELY WIDE. Its job is to catch the empty field and the coarse typo (145 for 1450), not
#: to arbitrate where coffee can grow -- there is no citation for that, and a narrow window would
#: reject real farms. There are no coffee plots in Peru below 200 m, nor anywhere above 2800.
COFFEE_ALTITUDE_MIN_M = 200.0
COFFEE_ALTITUDE_MAX_M = 2800.0


def altitude_band(masl: Optional[float]) -> Optional[str]:
    """The site's altitude band, or `None` when its location is unknown.

    `None` is not the same as "some altitude or other", and there is no neutral value to fall back
    on: every altitude lands in a band, and every band modulates something -- `medium`, the one a
    mid-range guess would reach, raises American leaf spot to weight `high`. So no altitude means
    no band, and no band means no modulation.
    """
    if masl is None: return None
    if masl < ALTITUDE_BANDS["low_max"]: return "low"
    if masl <= ALTITUDE_BANDS["medium_max"]: return "medium"
    if masl <= ALTITUDE_BANDS["high_max"]: return "high"
    return "very_high"

# --------------------------------------------------------------------------- #
# THRESHOLDS OF THE INFERENTIAL RULES (R1-R8, over MEASURED data)
#
# If a number decides whether a rule fires, it has a name and lives here. The inventory that feeds
# the report walks named constants only, so a literal written inside the body of `inferential_rules`
# is a threshold the agronomist never sees and nobody audits -- and two copies of the same physical
# limit, one named and one buried, drift apart without anything looking wrong.
#
# `scripts/informe_motor.py` fails when any constant here lacks a declared provenance.
RULES = {
    # --- R1 rust, over measured data. Aligned with the anticipated version: same biology. ---
    "rust_temp_min": 15.0,        # De Jong et al. 1987; Diniz et al. 2012 (mín. 15,5)
    "rust_temp_max": 28.0,
    "rust_wet_hours": 12,         # site calibration, same as in FORECAST: see R1's note
    "rust_days_of_3": 2,          # ≥2 infection days out of 3 consecutive
    # --- R2 American leaf spot (Mycena citricolor) ---
    "leaf_spot_rh": 80.0,
    "leaf_spot_rh_hours": 24,
    "leaf_spot_rain_days": 3,
    "leaf_spot_temp_hours": 24,   # hours ACCUMULATED in band, not the window's mean
    "leaf_spot_temp_min": 19.0,
    "leaf_spot_temp_max": 23.0,
    # --- R3 coffee berry borer, colonising flight over measured data ---
    # All three describe the flight HOUR, not the day: renamed from `borer_day_temp_*`, which said
    # "daily mean" and measured something else.
    "borer_flight_hours": 4,      # hours of the day inside the flight range
    "borer_flight_temp_min": 20.0,
    "borer_flight_temp_max": 30.0,
    "borer_dry_day_min_hours": 20,  # a day is only declared DRY when observed almost in full
    # --- R4 cercospora ---
    # The N and K cuts do NOT live here: they are derived from `NPK_BANDS`, which is where the
    # engine defines what "low" means for each nutrient. Duplicating them was two copies of one
    # number, and copies diverge. Same for humidity, which comes from `ENV.rh_fungal`.
    "cercospora_wet_hours": 24,
    # --- R5 leaf miner ---
    "leaf_miner_dry_hours": 72,
    "leaf_miner_temp_min": 20.0,
    "leaf_miner_temp_max": 30.0,
    "leaf_miner_warm_hours": 24,
    # --- R6 cold / Phoma. Leaf wetness comes from `ENV.rh_leafwet`, not a private copy. ---
    # Lorenzetti, Pozza & de Souza 2015 (Coffee Science 10(1):1-9): in vivo infection is favoured
    # between 15 and 20 °C -- "Temperatures ranging from 15-20 °C significantly increased germ tube
    # length and provided favorable conditions for pathogen infection".
    "cold_phoma_temp_max": 20.0,
    # 60 h ACCUMULATED across the window, not 24. At 24 h the rule fires on
    # 52,4 % [medido:cf.phoma_24h] of days, which contradicts the literature: Phoma is limiting on
    # cold windy ridges, typically above 1600 m.
    #
    # Site calibration, not a published figure. The local distribution of cold-wet hours per 96 h
    # window at 1823 m is median 25, p95 60, p98 66, maximum 81, so 60 h is the p95 -- and the
    # pilot sits INSIDE the disease's limiting band with weight `high`, which is why the p95 and
    # not a stricter cut. It fires on 5,5 % [medido:cf.phoma_24h.servido] of days.
    "cold_phoma_hours": 60,
    # --- R7 leaching and R8 waterlogging, over measured data ---
    "leaching_hours": 24,
    "waterlog_hours": 72,
}

ALTITUDE_RISK = {  # banda -> {regla: peso}
    "low":       {"rust": "high",  "borer": "high",  "leaf_spot": "low",  "cold_phoma": "low"},
    "medium":    {"rust": "medium", "borer": "medium", "leaf_spot": "high",  "cold_phoma": "medium"},
    "high":      {"rust": "low",   "borer": "low",   "leaf_spot": "high",  "cold_phoma": "high"},
    "very_high": {"rust": "low",   "borer": "very_low", "leaf_spot": "high", "cold_phoma": "high"},
}
#: Weight when there is no altitude. Nothing is modulated: the alert goes out with the severity its
#: rule declares and the technician's voice says why, instead of recording a weight nobody
#: measured.
WEIGHT_UNKNOWN = "unknown"

def _weight(masl: Optional[float], risk: str) -> str:
    banda = altitude_band(masl)
    if banda is None:
        return WEIGHT_UNKNOWN
    return ALTITUDE_RISK[banda].get(risk, "medium")

# Weight scale, so a factor that aggravates the risk can raise it one step (fruit load on rust,
# for instance).
_WEIGHT_ORDER = ["very_low", "low", "medium", "high"]

def _bump_weight(weight: str) -> str:
    """Raises the weight one step, capped at 'high'."""
    try:
        return _WEIGHT_ORDER[min(_WEIGHT_ORDER.index(weight) + 1, len(_WEIGHT_ORDER) - 1)]
    except ValueError:
        return weight
# The weight modulates severity, and whether the rule is suppressed as irrelevant on this site.

# Spanish labels for the internal values that appear inside message text. Identifiers are English;
# what the user reads is Spanish.
_BAND_ES = {"severe": "severo", "low": "bajo", "adequate": "adecuado", "high": "alto",
            "moderate": "moderado", "excess": "exceso", "low_ripening": "bajo (maduración)"}
_WEIGHT_ES = {"high": "alto", "medium": "medio", "low": "bajo", "very_low": "muy bajo"}

# --- Water management (RELATIVE, uncalibrated capacitive sensor) ---
# An ABSOLUTE threshold such as "<45%" is not defensible with an uncalibrated capacitive probe: the
# reading is not volumetric content, it depends on the soil (worse in clay) and on temperature, and
# the baseline shifts between installations (El Milagro: ~44 % nov-2024 against ~53.6 % feb-2026
# with the same sensor). So the threshold is derived PER DEPLOYMENT from the observed wet-dry
# envelope, with the allowable depletion fraction taken from literature.
#
# ON p = 0.40. Table 22 of FAO-56 (Allen et al. 1998) assigns COFFEE p = 0.40 directly, with
# rooting depth Zr = 0.9-1.5 m, in the section on tropical fruits and trees. The 0.50 sometimes
# quoted is the generic remark that it "is commonly used for many crops", not coffee's value.
#
# The table is defined for ETc ≈ 5 mm/day. FAO-56 chapter 8 gives the correction:
#     p = p_table + 0.04·(5 − ETc)
# and warns that in hot dry climates with high ETc, p falls 10-25 % below the table. It is not
# applied here because the threshold is RELATIVE to the sensor's envelope rather than an absolute
# water content: there is no ETc to put in the formula without inventing the calibration.
WATER = {
    "deficit_hours": 12,          # minimum persistence below the threshold
    "depletion_frac": 0.40,       # COFFEE's p in FAO-56 table 22, not an adjustment on 0.50
    "envelope_min_hours": 48,     # minimum hours of history to derive the envelope
    "envelope_min_span": 8.0,     # minimum wet-dry span, in sensor points, for it to be usable
    "critical_stages": ("plantula", "floracion", "fructificacion"),  # most sensitive to deficit
    "induction_stages": ("vegetativo",),  # estrés moderado pre-floración es NORMAL/benéfico -> no regar
}

ORGANIC = {
    "N": "compost, bocashi o guano de isla",
    "P": "roca fosfórica con compost/bocashi y micorrizas",
    "K": "sulfato de potasio (SOP) o sul-po-mag, con pulpa de café compostada",
    "rust": "cobre autorizado (dentro del tope), con poda y ventilación",
    "borer": "Beauveria bassiana con trampas y recojo de granos caídos",
}
CONVENTIONAL_BLOCKED = {"urea", "nitrato de amonio", "DAP", "superfosfato triple", "KCl",
                        "cloruro de potasio", "nitrato de potasio", "nitrato de calcio", "mezcla NPK sintetica"}

# --------------------------------------------------------------------------- #
@dataclass
class Reading:
    ts: datetime
    N: Optional[float] = None
    P: Optional[float] = None
    K: Optional[float] = None
    air_temp: Optional[float] = None
    air_rh: Optional[float] = None
    soil_moist: Optional[float] = None
    rain: bool = False
    stage: str = "vegetativo"
    altitude: Optional[float] = None

@dataclass
class Alert:
    rule_id: str
    category: str
    severity: str                 # info / warning / alert / critical
    farmer_message: str           # voz sencilla
    agronomist_message: str       # voz técnica + trazabilidad
    refer: bool = False           # ¿escalar a especialista?
    referral_note: str = ""       # texto de derivación (si refer=True)
    provisional: bool = False
    # A warning about what is FORECAST is not the same as one about what was measured, and the
    # user has to be able to tell them apart: the measured thing happened, the forecast one may
    # not. `horizon_hours` says how far ahead, because heavy rain at 6 h and at 72 h call for
    # different actions.
    forecast: bool = False
    horizon_hours: Optional[int] = None

    def __str__(self):
        p = " [PROVISIONAL]" if self.provisional else ""
        if self.forecast:
            p += f" [PRONÓSTICO {self.horizon_hours} h]" if self.horizon_hours else " [PRONÓSTICO]"
        s = (f"  ({self.category}) {self.rule_id} · {self.severity.upper()}{p}\n"
             f"     👤 Agricultor: {self.farmer_message}\n"
             f"     🎓 Agrónomo:   {self.agronomist_message}")
        if self.refer:
            s += f"\n     📞 Derivación: {self.referral_note}"
        return s

def _A(**kw) -> Alert:  # atajo
    return Alert(**kw)

# --------------------------------------------------------------------------- #
# NPK: classification, domain overlay and stage gating
def _corte_banda(nut: str, banda: str) -> float:
    """A band's ceiling from `NPK_BANDS`, so no rule duplicates the number.

    `R4_CERCOSPORA` reads its N and K cuts through here rather than holding its own. Two copies of
    one value diverge the moment either is edited, and nothing marks which one the diagnosis used.
    """
    for techo, nombre in NPK_BANDS[nut]:
        if nombre == banda and techo is not None:
            return float(techo)
    raise KeyError(f"banda {banda!r} sin techo en NPK_BANDS[{nut!r}]")


def _classify(value: float, bands) -> str:
    for upper, label in bands:
        if upper is None or value < upper:
            return label
    return bands[-1][1]

def diagnose_npk(r: Reading) -> List[Alert]:
    out: List[Alert] = []
    names = {"N": "nitrógeno", "P": "fósforo", "K": "potasio"}
    for nut in ("N", "P", "K"):
        val = getattr(r, nut)
        if val is None or val == 0.0:
            continue
        dmin, dmax = SENSOR_DOMAIN[nut]
        if val < dmin or val > dmax:
            out.append(_A(rule_id=f"NPK_{nut}_OUT_OF_DOMAIN", category=CAT_B, severity="alert",
                farmer_message=(f"El sensor no puede medir bien el {names[nut]} en este momento "
                                f"(marcó {val:g}). No apliques nada todavía por esta lectura."),
                agronomist_message=(f"{nut}={val:g} mg/kg fuera del dominio calibrado {dmin:g}-{dmax:g}. "
                                    f"Se suprime prescripción autónoma."),
                refer=True,
                referral_note=(f"Pide a un técnico de APROCASSI un análisis de laboratorio para confirmar "
                               f"el nivel real de {names[nut]}.")))
            continue
        band = _stage_adjust_band(nut, val, _classify(val, NPK_BANDS[nut]), r.stage)
        out.append(_npk_alert(nut, val, band, r.stage))
    return out

def _stage_adjust_band(nut: str, val: float, band: str, stage: str) -> str:
    """Adjusts the raw band for the crop stage. It lives in ONE place because both `diagnose_npk` and
    `reference_ranges` consume it, so the bands the frontend draws cannot drift from the ones the
    diagnosis applies."""
    # P: the useful floor rises in two stages, not one.
    #
    # plantula, for new root demand: phosphorus governs cell division at the root apices, and
    # without deep anchoring there is no later tolerance to water stress.
    #
    # floracion was added by the august 2026 agronomic review, and its argument is energetic:
    # breaking bud dormancy, expanding it and sustaining anthesis without aborting the pollen tube
    # demands an abrupt peak of ATP. In the acid soils of Cajamarca, with iron and aluminium fixing
    # phosphate, a reading at the lower edge of "adequate" does not sustain mass fruit set and ends
    # in flower abscission. Same recategorisation, different stage.
    if nut == "P" and stage in ("plantula", "floracion") and val < 20.0 and band == "adequate":
        return "low"
    # K: zero tolerance in fructificacion and maduracion. That is the peak of K demand and
    # extraction, and a deficiency then damages grain and cup irreversibly (FAO ~53 kg K/tonne).
    if nut == "K" and stage in ("fructificacion", "maduracion") and band == "moderate":
        return "low_ripening"
    return band

def _npk_alert(nut: str, val: float, band: str, stage: str) -> Alert:
    # -------- NITROGEN: always advisory / provisional --------
    if nut == "N":
        if band in ("severe", "low"):
            # THE BAND DOES NOT MOVE; THE MESSAGE DOES. The august 2026 agronomic review
            # confirmed that nitrogen must NOT be banded by stage: mineral N is too volatile, with
            # 3 to 55 % lost to volatilisation and leaching, and the sensor measures conductivity,
            # so hardening the threshold in grain filling would fill the system with false alarms
            # every time rain dilutes the soil solution. It called that asymmetry against P and K a
            # sound biological safeguard rather than an oversight.
            #
            # What it did require is that the CONSEQUENCE explained changes, because the same
            # deficit does not cost the same in every stage: in vegetativo the bandolas that will
            # carry next year's flower buds are lost; in grain filling the plant cannibalises its
            # own leaves to fill the cherry -- up to 64 % of the plant's N sits in leaf tissue --
            # and that is post-harvest die-back.
            consecuencia = {
                "vegetativo": (" Sin nitrógeno, tu planta no sacará ramas nuevas vigorosas, y en esas "
                               "ramas es donde saldrán las flores del año que viene."),
                "fructificacion": (" Cuidado: el fruto está chupando la fuerza de la planta. Si ves "
                                   "amarillamiento, está sacrificando sus propias ramas para llenar "
                                   "las cerezas y después se secan. Urge aplicar abono orgánico."),
            }
            consecuencia["maduracion"] = consecuencia["fructificacion"]
            nota_etapa = {
                "vegetativo": " En vegetativo compromete la formación de nudos productivos del ciclo siguiente.",
                "fructificacion": " En llenado, riesgo de translocación foliar hacia el fruto y die-back.",
                "maduracion": " En llenado, riesgo de translocación foliar hacia el fruto y die-back.",
            }
            return _A(rule_id="NPK_N", category=CAT_A, severity="warning", provisional=True,
                farmer_message=("El nitrógeno se ve algo bajo, pero esta lectura cambia mucho con la lluvia. "
                                "Fíjate si las hojas viejas están verde claro o amarillentas."
                                + consecuencia.get(stage, "")
                                + " Si tienes compost, bocashi o guano de isla, este es buen momento "
                                  "para aplicarlo."),
                agronomist_message=(f"N sensor={val:g} mg/kg (banda '{_BAND_ES.get(band, band)}', proxy provisional edge; no trazable a "
                                    f"Kjeldahl/foliar). Sin dosis absoluta. Marco: enmiendas orgánicas ricas en N."
                                    + nota_etapa.get(stage, "")
                                    + " La banda NO se ajusta por etapa a propósito: el N mineral es "
                                      "demasiado volátil para bandearlo sobre una lectura de "
                                      "conductividad."),
                refer=True,
                referral_note=("Conviene un análisis foliar pre-floración (N foliar óptimo 2.5-3.0%) para "
                               "confirmar el estado real de nitrógeno."))
        return _A(rule_id="NPK_N", category=CAT_A, severity="info", provisional=True,
            farmer_message=("El nivel de nitrógeno se ve bien por ahora. Igual, guíate por el color de las "
                            "hojas: verde oscuro y parejo es señal de planta sana."),
            agronomist_message=(f"N sensor={val:g} mg/kg (banda '{_BAND_ES.get(band, band)}'). Lectura provisional de N mineral; "
                                f"referencia real = foliar/Kjeldahl. Mantener plan base."),
            refer=False)

    # -------- PHOSPHORUS --------
    if nut == "P":
        if band == "severe":
            return _A(rule_id="NPK_P", category=CAT_A, severity="alert",
                farmer_message=(f"El fósforo está muy bajo ({val:g}). Las raíces nuevas pueden crecer poco. "
                                "Aplica roca fosfórica mezclada con compost lo antes posible."),
                agronomist_message=(f"P Bray II={val:g} mg/kg < 6 (bajo). Roca fosfórica + MO; "
                                    "revisar acidez (fijación Fe/Al)."),
                refer=True,
                referral_note="Informa a un técnico para que revise la acidez del suelo y ajuste el encalado.")
        note = " (en almácigo/plántula el fósforo debe estar más alto)" if stage == "plantula" and band == "low" else ""
        if band == "low":
            return _A(rule_id="NPK_P", category=CAT_A, severity="warning",
                farmer_message=(f"El fósforo está en el límite bajo ({val:g}){note}. En estos suelos ácidos el "
                                "fósforo se 'traba' fácil. Aplica roca fosfórica con bastante compost."),
                agronomist_message=(f"P Bray II={val:g} mg/kg (bajo-marginal; óptimo café 6-14). Fijación probable. "
                                    f"Favorecer micorrizas + {ORGANIC['P']}.{note}"),
                refer=False)
        if band == "high":
            return _A(rule_id="NPK_P", category=CAT_A, severity="warning",
                farmer_message=(f"El fósforo está alto ({val:g}). No apliques más fósforo por ahora."),
                agronomist_message=(f"P Bray II={val:g} mg/kg > 20 (alto). Riesgo de antagonismo con Zn. "
                                    "Interrumpir fosfatados."),
                refer=False)
        return _A(rule_id="NPK_P", category=CAT_A, severity="info",
            farmer_message=(f"El fósforo está bien ({val:g}). Solo mantén tu abonado orgánico de siempre."),
            agronomist_message=(f"P Bray II={val:g} mg/kg (adecuado 10-20). Reposición basal orgánica."),
            refer=False)

    # -------- POTASSIUM --------
    #
    # There is no `severe` branch for K: the sensor's calibrated domain starts at 90 mg/kg and any
    # reading under it is cut as out of domain before classification, so such a branch could never
    # run.
    #
    # What stays open, and goes to the agronomic review, is whether a K reading below 90 should be
    # treated as a severe deficiency rather than as "the sensor cannot measure it". Those are two
    # different answers for the grower, and deciding is agronomy, not programming.
    if nut == "K":
        if band == "low_ripening":
            return _A(rule_id="NPK_K", category=CAT_A, severity="alert",
                farmer_message=(f"El potasio está justo ({val:g}) y las plantas están llenando grano: es "
                                "momento clave. Es cuando más conviene abonar con potasio para no perder calidad."),
                agronomist_message=(f"K={val:g} mg/kg banda moderada + etapa MADURACIÓN (tolerancia cero). "
                                    f"Intervenir: {ORGANIC['K']}."),
                refer=False)
        if band == "moderate":
            return _A(rule_id="NPK_K", category=CAT_A, severity="warning",
                farmer_message=(f"El potasio está en el límite ({val:g}). Ve planificando abono orgánico de "
                                "potasio antes de que las plantas empiecen a llenar grano."),
                agronomist_message=(f"K={val:g} mg/kg (moderado 78-115). Planificar {ORGANIC['K']} antes de llenado."),
                refer=False)
        if band == "high":
            return _A(rule_id="NPK_K", category=CAT_A, severity="info",
                farmer_message=(f"Tu suelo tiene buen potasio ({val:g}). No compres ni apliques abono de "
                                "potasio ahora: de más, estorba a otros nutrientes."),
                agronomist_message=(f"K={val:g} mg/kg (alto). NO aplicar K: antagonismo con Ca/Mg."),
                refer=False)
        if band == "excess":
            # The MECHANISM, not just the prohibition. The agronomic review asked for cationic
            # antagonism to be explained: overwhelming potassium floods the root's uptake sites and
            # blocks calcium and magnesium -- magnesium is the central atom of chlorophyll and
            # calcium the cement of the cell wall -- so excess K produces an INDUCED deficiency the
            # grower sees as deformed new leaves and interveinal chlorosis. And it names ash, which
            # is the potassium input actually to hand on the plot: forbidding "potassium fertiliser"
            # without saying "ash" leaves out exactly what was about to be applied.
            return _A(rule_id="NPK_K", category=CAT_A, severity="warning",
                farmer_message=(f"El suelo tiene demasiado potasio ({val:g}) y está bloqueando la "
                                "entrada de calcio y magnesio a la raíz. No apliques ceniza ni "
                                "abonos ricos en potasio por ahora, para no afectar el vigor de las "
                                "hojas nuevas."),
                agronomist_message=(f"K={val:g} mg/kg > dominio. Antagonismo catiónico K-Ca-Mg en la "
                                    "rizosfera: la saturación de K compite por los sitios de "
                                    "absorción y puede inducir deficiencia de Ca y Mg (clorosis "
                                    "intervenal, deformación apical). Suspender todo aporte "
                                    "potásico, incluida ceniza; confirmar el balance por laboratorio."),
                refer=True,
                referral_note="Pide un análisis de laboratorio para revisar el balance de potasio, calcio y magnesio.")
        return _A(rule_id="NPK_K", category=CAT_A, severity="info",
            farmer_message=(f"El potasio está bien ({val:g}). Mantén tu plan de abono de siempre."),
            agronomist_message=(f"K={val:g} mg/kg (adecuado 115-156). Mantener plan base."),
            refer=False)

# --------------------------------------------------------------------------- #
# COFFEE BERRY BORER — RE-RE SANITATION (category C), the path that runs ALL YEAR
#
# It cannot be gated by stage because the borer does not disappear between harvests. It survives
# and breeds in leftover berries -- fallen to the ground and dried on the branch -- which are the
# RESERVOIR for the next campaign. Cenicafé (Constantino, Peña-Quiñones, Benavides et al. 2021)
# measured infested berries on the ground remaining a reservoir for 140 ± 8.2 days, and ONE berry
# from the ground infesting on average 590.2 ± 142.2 berries on the tree at 1218 masl (959.0 ± 89.6
# in an El Niño year). Román-Ruiz et al. (2019, Bull. Ent. Res. 109(4):544-549) confirm survival in
# leftovers through the between-harvest period.
#
# RE-RE (REcolección + REpase, collection plus a second pass) is, per Cenicafé's integrated borer
# management, the cultural practice that lowers the borer most: it keeps infestation below the
# economic damage threshold and removes over 80 % of the population. So this path is emitted ALWAYS
# except in plantula, and reinforced during harvest and the between-harvest period.
def borer_sanitation_note(stage: str) -> List[Alert]:
    if stage == "plantula":
        return []   # nursery: no fruit and no leftovers to sanitise
    if stage == "cosecha":
        return [_A(rule_id="R3_BORER_SANITATION", category=CAT_C, severity="warning",
            farmer_message=("Al terminar cada pasada de cosecha, haz el REPASE: 2 a 3 semanas después "
                            "recoge todos los granos que queden (maduros, sobremaduros y secos), del "
                            "árbol y del suelo. Ahí es donde la broca se queda viviendo hasta la próxima "
                            "cosecha. Los granos recogidos, échalos en saco cerrado al sol."),
            agronomist_message=("RE-RE (MIB Cenicafé): repase 2-3 semanas tras la última pasada + recolección "
                                "recurrente. Elimina >80% de la población y sostiene la infestación bajo el "
                                "umbral de daño económico. Solarizar el material recogido."),
            refer=False)]
    if stage == "vegetativo":
        return [_A(rule_id="R3_BORER_SANITATION", category=CAT_C, severity="warning",
            farmer_message=("Aunque ahora no haya cosecha, la broca sigue viva en los granos que quedaron "
                            "en el suelo y en la rama: de ahí sale para atacar la próxima cosecha. Recoge "
                            "los granos residuales cada 15 a 20 días y pon trampas para ir midiendo."),
            agronomist_message=("Entrecosecha: los residuales son el reservorio (~140 d). Recolección cada "
                                "15-20 d + trampeo con alcohol (etanol-metanol), que es cuando más informa. "
                                "Es la ventana de mayor retorno del control cultural."),
            refer=False)]
    return [_A(rule_id="R3_BORER_SANITATION", category=CAT_C, severity="info",
        farmer_message=("Mantén el hábito de recoger los granos caídos y revisar las trampas: así la broca "
                        "no se acumula para la próxima cosecha."),
        agronomist_message=("Saneamiento de base: recolección de residuales + trampeo permanente. La presión "
                            "de broca de la campaña siguiente depende del reservorio que se deje ahora."),
        refer=False)]

# --------------------------------------------------------------------------- #
# STATIC STAGE NOTES (category C)
def stage_notes(stage: str) -> List[Alert]:
    out = []
    if stage == "floracion":
        out.append(_A(rule_id="STAGE_FLOWERING_BCA", category=CAT_C, severity="info",
            farmer_message=("Tus plantas están floreando: es una etapa delicada. Conviene aplicar abonos "
                            "orgánicos con boro y calcio para que la flor cuaje y no se caiga."),
            agronomist_message=("Floración/cuajado: demanda de B (tubo polínico) y Ca (pared celular). Sensor no mide "
                                "B/Ca -> nota Cat. C preventiva."),
            refer=True,
            referral_note="Pregunta a tu técnico de APROCASSI la dosis adecuada de boro y calcio para tu parcela."))
    if stage == "cosecha":
        out.append(_A(rule_id="STAGE_HARVEST_MAINTENANCE", category=CAT_C, severity="info",
            farmer_message=("Después de la cosecha la planta queda cansada. Haz podas de limpieza y devuélvele "
                            "materia orgánica (compost) para que se recupere para la próxima campaña."),
            agronomist_message=("Poscosecha: agotamiento de reservas; restitución de MO + poda sanitaria + muestreo "
                                "de suelo para reajuste basal."),
            refer=True,
            referral_note="Coordina con tu técnico un muestreo de suelo para planificar el abonado del próximo ciclo."))
    return out

# --------------------------------------------------------------------------- #
# THERMAL SIGNAL (R9)
def env_alerts(r: Reading) -> List[Alert]:
    out = []
    t = r.air_temp
    if t is None:
        return out
    if t < ENV["temp_cold"]:
        # Severity `info`, not `warning`, and the message is about WHEN to fertilise rather than
        # about waiting.
        #
        # The august 2026 agronomic review confirmed the 15 °C -- below that point root membrane
        # fluidity drops and uptake with it -- but rejected the framing. At 1823 m this happens on
        # 97.8 % of days, always before dawn: a daily warning produces alarm fatigue and the grower
        # stops looking at EVERY alert, not just this one.
        #
        # The threshold is untouched because it is correct. What changes is what is done with it:
        # from a crisis alert to a timing instruction, which is actionable and does not wear out.
        out.append(_A(rule_id="R9_COLD", category=CAT_A, severity="info",
            farmer_message=(f"Las madrugadas están frías ({t:g}°C) y las raíces amanecen dormidas. "
                            "Programa el abono al suelo o foliar de media mañana en adelante, "
                            "cuando el sol ya haya calentado la tierra, para no desperdiciarlo."),
            agronomist_message=(f"T={t:g}°C < 15: absorción radicular deprimida por caída de "
                                "fluidez de membrana. Desplazar aplicaciones edáficas/foliares al "
                                "tercio medio del día. Contexto climatológico del sitio, no evento."),
            refer=False))
    elif t > ENV["temp_hot"]:
        out.append(_A(rule_id="R9_ACUTE_HEAT", category=CAT_A, severity="alert",
            farmer_message=(f"Hace mucho calor ({t:g}°C) y la planta se estresa. Hoy, evita aplicar abonos "
                            "foliares en las horas de más calor (se quema la hoja y se aprovechan mal). "
                            "Como medida de fondo, conviene tener más sombra (árboles) para proteger el cafetal."),
            # NOT a photosynthetic cutoff. The idea that coffee's photosynthesis stops at 34 °C
            # comes from Nunes et al. 1968, and DaMatta 2025 (10.1002/ael2.70050) treats it as
            # outdated: coffee sustains high photosynthesis to ~37 °C, and the decline above ~30 °C
            # is mediated by VPD through stomatal closure rather than by temperature itself. So
            # 32 °C is used as a START MARKER for thermal and water stress, and the rule that
            # quantifies the stress is R12_VPD_FRUIT_FILL.
            agronomist_message=(f"T={t:g}°C > 32: marcador de inicio de estrés térmico. La limitación "
                                "fotosintética a estas temperaturas es mediada por VPD (DaMatta 2025), "
                                "no un corte térmico; ver R12. Evitar foliares en el pico (fitotoxicidad "
                                "y baja absorción). Medida estructural: sombra/agroforestería."),
            refer=False))
    elif t > ENV["temp_warn"]:
        # The stage is not something to ask the grower: the engine has it. A message phrased "if
        # there is grain filling…" hands back a condition the system already knows, and in plantula
        # or vegetativo there is no grain to lose, so half of it would be dead text.
        con_grano = r.stage in ("fructificacion", "maduracion")
        out.append(_A(rule_id="R9_HEAT_QUALITY", category=CAT_A, severity="warning",
            farmer_message=(f"El calor ({t:g}°C) estresa la planta. Vigila la sombra y la humedad."
                            + (" Además, con el grano llenándose, apura la maduración y baja la "
                               "calidad de la taza." if con_grano else "")),
            agronomist_message=(f"T={t:g}°C (23-32): mayor respiración y estrés."
                                + (" En llenado/maduración, riesgo de maduración prematura y pérdida "
                                   "de taza." if con_grano else
                                   f" Etapa {r.stage}: sin fruto en desarrollo, el riesgo de taza no "
                                   "aplica.")),
            refer=False))
    return out

# --------------------------------------------------------------------------- #
# INFERENTIAL RULES R1-R8 (persistence over the window)
def _to_hourly(win: List[Reading]) -> List[dict]:
    buckets: Dict[datetime, List[Reading]] = {}
    for r in win:
        h = r.ts.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(h, []).append(r)
    hours = []
    for h in sorted(buckets):
        g = buckets[h]
        def avg(a):
            v = [getattr(x, a) for x in g if getattr(x, a) is not None]
            return sum(v) / len(v) if v else None
        hours.append({"h": h, "temp": avg("air_temp"), "rh": avg("air_rh"),
                      "soil": avg("soil_moist"), "rain": any(x.rain for x in g)})
    return hours

def _max_run(flags: List[bool]) -> int:
    """Longest CONTINUOUS run. Only for conditions that really can hold without a break."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def _horas_en(flags: List[bool]) -> int:
    """ACCUMULATED hours meeting the condition within the window, consecutive or not.

    Persistence has to be counted this way, not as a continuous run. `R5`, `R6` and `R7` all
    depend on temperature or rain, and the diurnal cycle breaks any run of those before a full day
    goes by. Longest continuous run in a year of reanalysis at the pilot's coordinates:

        R5  20-30 °C                10 h
        R6  T<17 °C and RH>=90      29 h
        R7  rain                    18 h

    Two of the three cannot reach 24 h at all, so a rule demanding it fires on 0 % of days for
    reasons that have nothing to do with its threshold.

    Every rule that counts persistence goes through here, so none of them can drift back to the
    other form. Reading the thresholds: "24 h" means 24 hours ACCUMULATED within the window, and
    each affected constant declares that in its provenance.
    """
    return sum(1 for f in flags if f)

def _apply_weight(alert: Alert, weight: str, origen_altitud: Optional[str] = None) -> Optional[Alert]:
    """Modulates an inferential alert by the site's altitude weight.

    very_low suppresses it as not relevant; low degrades alert to warning -- but NOT to 'info',
    which would read as all clear -- and states the low risk at the end of the message rather than
    hedging at the start; high reinforces it. The weight is recorded in the technician's voice."""
    if weight == "very_low":
        return None
    if weight == WEIGHT_UNKNOWN:
        # With no altitude nothing is modulated, and that is said. Staying quiet would leave the
        # technician believing the severity already came adjusted to the site.
        alert.agronomist_message += (" [Sin altitud registrada: no se aplicó modulación por "
                                     "altitud. Registrar la altitud o las coordenadas de la finca.]")
        return alert
    # The label says where the weight comes from, not just its value. `_bump_weight` raises it for
    # FRUIT LOAD, and recording that as "altitude weight" made the technician read the farm as
    # sitting in an altitude band that is not its own: at 1823 m rust weighs LOW, and in
    # fructificacion it was published as MEDIUM without saying the fruit had raised it.
    if origen_altitud is not None and origen_altitud != weight:
        alert.agronomist_message += (
            f" [Peso {_WEIGHT_ES.get(weight, weight)}: altitud "
            f"{_WEIGHT_ES.get(origen_altitud, origen_altitud)}, elevado un escalón por carga de fruto]")
    else:
        alert.agronomist_message += f" [Peso por altitud: {_WEIGHT_ES.get(weight, weight)}]"
    if weight == "low":
        if alert.severity == "alert":
            alert.severity = "warning"
        alert.refer = False
        alert.farmer_message += (" En tu zona este riesgo es bajo por la altitud, así que por ahora "
                                 "conviene vigilar sin apresurar aplicaciones.")
    return alert

def inferential_rules(win: List[Reading], altitude: Optional[float] = None,
                      stage: str = "vegetativo") -> List[Alert]:
    out: List[Alert] = []
    H = _to_hourly(win)
    if not H:
        return out
    # The day is the LOCAL one, not UTC. Keyed by UTC the "day" ran from 19:00 to 19:00 Peru time,
    # so rain at eight in the evening counted as the next day's. Rules that talk about days -- the
    # borer's "yesterday was dry and today it rained", American leaf spot's rainy-day count -- have
    # to use the day the grower means.
    by_day: Dict = {}
    for hr in H:
        by_day.setdefault(_a_hora_local(hr["h"]).date(), []).append(hr)
    days = sorted(by_day)
    def push(alert, risk, weight=None, origen=None):
        a = _apply_weight(alert, weight or _weight(altitude, risk), origen)
        if a is not None:
            out.append(a)

    # R1 rust. Rust severity rises sharply with FRUIT LOAD: more sink demand, weaker leaf, more
    # severity and defoliation. López-Bravo, Virginio-Filho & Avelino 2012 (Crop Protection
    # 38:21-29, CATIE) measure +28.9 % incidence and +129.2 % severity at 500 fruiting nodes
    # against a plant with no fruit. The mechanism is in Eskes and Souza 1981, a citation supplied
    # by the august 2026 agronomic review: in the vegetative phase the leaf has reserves to mount
    # the hypersensitive response -- phytoalexins and ROS that necrose the entry point and kill the
    # fungus -- while with the drupe draining it of carbohydrates and nitrogen the parenchyma is
    # left without the ATP for that defence and Hemileia colonises the mesophyll unopposed. So in
    # fructificacion and maduracion the site weight goes up one step. A modulation, NOT a gate:
    # rust is a foliar risk all year.
    fruit_load = stage in ("fructificacion", "maduracion")
    rust_weight_altitud = _weight(altitude, "rust")
    rust_weight = _bump_weight(rust_weight_altitud) if fruit_load else rust_weight_altitud
    # The leaf wetness run is measured over the CONTINUOUS window, not inside each calendar day.
    # Measuring it per natural day split every night that crossed midnight: with real runs of up to
    # 17 h, none reached the 12 required within one calendar day and the rule was left with 9
    # infection days a year. The spore does not know when the date changes. The days a valid run
    # touches are the ones that count as infection days.
    mojado = [((hr["rh"] or 0) >= ENV["rh_leafwet"] or hr["rain"]) and
              (hr["temp"] is not None
               and RULES["rust_temp_min"] <= hr["temp"] <= RULES["rust_temp_max"])
              for hr in H]
    infection_days = []
    corrida = []
    for hr, ok in zip(H, mojado + [False]):
        if ok:
            corrida.append(hr)
            continue
        if len(corrida) >= RULES["rust_wet_hours"]:
            infection_days.extend({_a_hora_local(x["h"]).date() for x in corrida})
        corrida = []
    if len(corrida) >= RULES["rust_wet_hours"]:
        infection_days.extend({_a_hora_local(x["h"]).date() for x in corrida})
    infection_days = sorted(set(infection_days))
    for i in range(len(days) - 1):
        if len(set(days[i:i+3]) & set(infection_days)) >= RULES["rust_days_of_3"]:
            carga = (" Con la carga de fruto de esta etapa, la roya golpea más fuerte."
                     if fruit_load else "")
            push(_A(rule_id="R1_RUST", category=CAT_A, severity="alert",
                farmer_message=("El clima húmedo y templado de estos días es ideal para que aparezca la roya. "
                                "Revisa el envés (parte de abajo) de las hojas buscando polvillo anaranjado."
                                + carga),
                agronomist_message=(f"Ventanas de humedad foliar (HR≥{ENV['rh_leafwet']:g}%/lluvia) "
                                    f"≥{RULES['rust_wet_hours']} h con T {RULES['rust_temp_min']:g}-"
                                    f"{RULES['rust_temp_max']:g}°C en ≥{RULES['rust_days_of_3']} de 3 días. "
                                    "Periodo de infección probable."
                                    + (" Carga de fruto alta: severidad y defoliación esperadas mayores."
                                       if fruit_load else "")),
                refer=True,
                referral_note=("Avisa a tu técnico de APROCASSI para revisar la incidencia de roya y decidir si "
                               "aplicar cobre autorizado.")), "rust", weight=rust_weight, origen=rust_weight_altitud)
            break

    # R2 American leaf spot
    rh80 = sum(1 for hr in H if (hr["rh"] or 0) >= RULES["leaf_spot_rh"])
    rain_days = sum(1 for d in days if any(hr["rain"] for hr in by_day[d]))
    # HOURS inside the favourable band, not the window's MEAN. Comparing an average against a
    # physiological band is the same defect of form as the continuous run: at 1823 m the mean of a
    # 96 h window reaches at most 19,4 °C [medido:clima.max_media_movil_96h_c], barely grazing the
    # lower edge of the 19-23 band, so almost no window entered it, while
    # 23,2 % [medido:clima.horas_en_banda_ojo_gallo_pct] of the year's HOURS do fall inside. The
    # responds to the hours it spends in its range, not to the period's average.
    #
    # The "0.9 % of windows" once published was that mean-based aggregation's frequency. It cannot
    # be re-measured without resurrecting deleted code, so what supports the conclusion is the fact
    # above, which is reproducible (`scripts/contrafactuales.py`).
    horas_templadas = _horas_en([hr["temp"] is not None
                                 and RULES["leaf_spot_temp_min"] <= hr["temp"]
                                 <= RULES["leaf_spot_temp_max"] for hr in H])
    if (rh80 >= RULES["leaf_spot_rh_hours"] and rain_days >= RULES["leaf_spot_rain_days"]
            and horas_templadas >= RULES["leaf_spot_temp_hours"]):
        push(_A(rule_id="R2_AMERICAN_LEAF_SPOT", category=CAT_A, severity="alert",
            farmer_message=("La neblina y humedad constante favorecen el 'ojo de gallo', que hace caer las hojas. "
                            "Retira las hojas enfermas del suelo y deja entrar más luz al cafetal."),
            agronomist_message=("HR≥80% sostenida + lluvia intermitente ≥3 días + T 19-23°C; franja endémica "
                                "1100-1550 msnm de Mycena citricolor."),
            refer=True,
            referral_note="Informa a tu técnico para revisar el ojo de gallo y ajustar la sombra de la parcela."), "leaf_spot")

    # R3 borer — FLIGHT / COLONISATION ALERT. Only one of the borer's two paths; the other is
    # RE-RE sanitation, which runs ALL YEAR in `borer_sanitation_note`.
    #
    # Colonising females fly after the first rains that follow dry weather, but can only BREED in
    # berries above 20 % dry matter. Constantino et al. 2021 (J. Insect Sci., PMC8121740): the
    # female "waits for the endosperm to exceed 20 % dry matter before entering the seed". That
    # point is reached 120-150 days after flowering (Vega et al. 2015 and primary sources; some
    # place the cycle's start at 90-120 days). The WINDOW is used rather than a single 120 d point.
    #
    # LIMITATION, and it is this rule's main one: with no record of flowering dates, the crop STAGE
    # stands in as a proxy for "there is susceptible grain". Once the agronomic log records
    # flowerings, this gate should move to real days after flowering and be re-anchored each
    # campaign, because in northern Peru flowering follows the onset of the rains and shifts from
    # year to year. San Ignacio and Jaén have a single main harvest, june to october, which is what
    # anchors the window.
    #
    # Firing is evaluated at DAY scale -- a dry day followed by a rainy one -- not per hour.
    if stage in ("fructificacion", "maduracion", "cosecha"):
        day_rain = {d: any(hr["rain"] for hr in by_day[d]) for d in days}
        def _horas_de_vuelo(d):
            """Hours of the day inside the flight range, not the day's mean.

            The daily mean at 1823 m reaches at most 19,7 °C [medido:clima.max_media_diaria_c], so
            the 20-30 band was never reached and the rule could not fire on any day -- even though
            20,8 % [medido:clima.horas_en_banda_vuelo_broca_pct] of the year's hours are in range.
            The female flies in the warm hours, not in the average.

            (19.5 °C was published once: the same series aggregated by UTC day. Here the day is the
            local one, which is the agronomically meaningful choice. The conclusion does not depend
            on which is used -- neither reaches 20 °C.)"""
            return _horas_en([hr["temp"] is not None
                              and RULES["borer_flight_temp_min"] <= hr["temp"]
                              <= RULES["borer_flight_temp_max"] for hr in by_day[d]])
        # "Yesterday was dry" is a NEGATIVE claim, and it can only be made about a day observed
        # almost in full. The days at the window's edges are fragments: calling a five-hour stretch
        # without rain dry confuses absence of data with absence of rain, and that alone was enough
        # to invent a dry→rain transition inside a single day. The other two conditions are
        # POSITIVE -- it rained, it was warm -- and over a fragment they can only fall short, never
        # overreach, so they need no such guard.
        completo = {d: len(by_day[d]) >= RULES["borer_dry_day_min_hours"] for d in days}
        # This condition is a GUARD, not the trigger: it discards a day that never warms enough
        # for the female to fly. What fires the rule is the dry→rain transition.
        dry_then_rain_day = any(
            completo[days[i - 1]] and (not day_rain[days[i - 1]]) and day_rain[days[i]]
            and _horas_de_vuelo(days[i]) >= RULES["borer_flight_hours"]
            for i in range(1, len(days))
        )
        if dry_then_rain_day:
            push(_A(rule_id="R3_BERRY_BORER", category=CAT_A, severity="alert",
                farmer_message=("Tras el tiempo seco llegó la lluvia: es cuando la broca vuela y ataca los granos. "
                                "Pon trampas y revisa cuántos granos están perforados."),
                agronomist_message=(f"Transición seco→lluvia (día seco seguido de día lluvioso) con T 20-30°C: vuelo "
                                    f"de colonización. Umbral de acción {ENV['broca_action_pct']:g}% (orgánico estricto)."),
                refer=True,
                referral_note=(f"Si más del {ENV['broca_action_pct']:g}% de granos están brocados, coordina con tu "
                               "técnico la aplicación de Beauveria bassiana.")), "borer")

    # R4 cercospora (iron spot): a poorly nourished plant (low N or K) plus prolonged humidity
    # plus, as a proxy, full sun and little shade. A disease of weakness (Souza et al. 2015).
    nvals = [x.N for x in win if x.N not in (None, 0.0)]
    kvals = [x.K for x in win if x.K not in (None, 0.0)]
    n_low = bool(nvals) and (sum(nvals) / len(nvals)) < _corte_banda("N", "low")
    k_low = bool(kvals) and (sum(kvals) / len(kvals)) < _corte_banda("K", "moderate")
    wet_hours = sum(1 for hr in H if (hr["rh"] or 0) >= ENV["rh_fungal"] or hr["rain"])
    if (n_low or k_low) and wet_hours >= RULES["cercospora_wet_hours"]:
        out.append(_A(rule_id="R4_CERCOSPORA", category=CAT_A, severity="warning",
            farmer_message=("Tus plantas están un poco débiles de nutrición y hay mucha humedad: puede aparecer "
                            "la 'mancha de hierro'. Lo primero es nutrir bien la planta (abono orgánico) y dar "
                            "algo de sombra; el remedio no es fumigar de una."),
            agronomist_message=("Nutrición baja (N y/o K) + humedad sostenida: favorabilidad de Cercospora coffeicola "
                                "(enfermedad de debilidad; cercosporina fotoactivada a pleno sol). Corregir N/K + sombra 35-65%."),
            refer=False))

    # R5 leaf miner (Leucoptera coffeella): prolonged drought plus T 20-30 °C, with more pressure
    # in the heat and at lower altitude.
    # Drought IS measured as a continuous run: a drought interrupted by rain is not a drought.
    long_dry = _max_run([not hr["rain"] for hr in H]) >= RULES["leaf_miner_dry_hours"]
    # The heat is not: the night always drops below 20 °C, so demanding 24 consecutive hours in
    # range made the rule impossible (real maximum run of the year: 15 h). Hours are accumulated.
    warm = [hr["temp"] is not None
            and RULES["leaf_miner_temp_min"] <= hr["temp"] <= RULES["leaf_miner_temp_max"] for hr in H]
    if long_dry and _horas_en(warm) >= RULES["leaf_miner_warm_hours"]:
        push(_A(rule_id="R5_LEAF_MINER", category=CAT_A, severity="warning",
            farmer_message=("El tiempo seco y cálido favorece al minador, que hace caminitos en las hojas. "
                            "Revisa las hojas de arriba y cuida a los insectos buenos (avispitas) que lo controlan."),
            agronomist_message=("Sequía prolongada + T 20-30°C: acumulación generacional de minador (mayor a menor "
                                "altitud). Conservar parasitoides; evitar amplio espectro."),
            refer=False), "borer")  # comparte el gradiente altitudinal de la broca

    # R6 cold / Phoma, relevant mostly at altitude: T < 17 °C sustained plus RH ≥ 90 % for ≥ 24 h.
    # Accumulated hours, not consecutive: the maximum continuous run observed in a year was 15 h,
    # so demanding 24 straight left the rule dead.
    cold_wet = [(hr["temp"] is not None and hr["temp"] < RULES["cold_phoma_temp_max"])
                and ((hr["rh"] or 0) >= ENV["rh_leafwet"] or hr["rain"]) for hr in H]
    if _horas_en(cold_wet) >= RULES["cold_phoma_hours"]:
        push(_A(rule_id="R6_COLD_PHOMA", category=CAT_A, severity="warning",
            farmer_message=("Frío húmedo persistente: puede aparecer 'muerte descendente' (Phoma) en las ramas. "
                            "Instala cortavientos y evita hacer heridas a la planta en estos días."),
            agronomist_message=("T<17°C + HR≥90%/lluvia ≥24 h: favorabilidad de Phoma/Botrytis (limitante >1600 msnm)."),
            refer=False), "cold_phoma")

    # R7 nitrogen leaching.
    # Accumulated hours: it does not rain 24 h straight here (real maximum run of the year: 23 h),
    # and nitrate washing depends on the water that percolates in total, not on it falling without
    # pause.
    if (_horas_en([hr["rain"] and (hr["soil"] or 0) >= ENV["soil_saturation"] for hr in H])
            >= RULES["leaching_hours"]):
        out.append(_A(rule_id="R7_N_LEACHING", category=CAT_A, severity="warning",
            farmer_message=("Con tanta lluvia, el abono se lava y se pierde. En vez de abonar todo de una vez, "
                            "reparte el abono en 3 o 4 aplicaciones pequeñas durante las lluvias."),
            agronomist_message=("Lluvia + suelo saturado ≥24 h: lixiviación de NO3- (correlación ~99% con lluvia en "
                                "Andosoles). Fraccionar N en 3-4 aplicaciones."),
            refer=False))

    # R8 waterlogging
    if (_max_run([(hr["soil"] or 0) > ENV["soil_saturation"] for hr in H])
            >= RULES["waterlog_hours"]):
        out.append(_A(rule_id="R8_WATERLOGGING", category=CAT_A, severity="alert",
            farmer_message=("El suelo lleva varios días encharcado y las raíces se pueden ahogar. Abre zanjas "
                            "para que drene el agua, sobre todo en las partes bajas."),
            agronomist_message=("Suelo > saturación ≥3 días: hipoxia radicular, caída de Fv/Fm. Drenaje/zanjas."),
            refer=True,
            referral_note="Si el encharcamiento persiste, pide a tu técnico evaluar el drenaje de la parcela."))
    return out

# --------------------------------------------------------------------------- #
# WATER MANAGEMENT RULES (soil humidity plus rain plus crop stage)
#   The key nuance (Cenicafé; SciELO Venezuela/Cuba): pre-flowering water stress INDUCES flowering
#   and rain breaks the latency, so do NOT irrigate inside the induction window. floracion and
#   grain filling are the phases MOST sensitive to deficit, and there water and mulching do come
#   first.
def deficit_threshold(H: List[dict]) -> Optional[float]:
    """SITE-RELATIVE deficit threshold, derived from the wet-dry envelope this deployment has
    observed. Not an absolute %, which means nothing with an uncalibrated capacitive probe.

    The observed maximum stands in for the wet extreme (post-rain, roughly field capacity after
    draining) and the sustained minimum for the dry one, and the rule fires on depleting
    `depletion_frac` (~40 %) of that span. Returns None without enough history or span: in that
    case no irrigation recommendation is issued, because saying nothing is more honest than
    inventing a threshold.

    THERE IS NO "days of water left" PROJECTION, and that is a measured decision. Projecting when
    the soil will reach this threshold and crossing it with the first forecast rain is the right
    question for rain-fed farming. Three measurements ruled it out:

      1. ET0 does not help project. The correlation between the observed drying rate and the
         regional model's ET0 is −0.06 and +0.08 across the two campaigns, and using it improves
         the error by 1.1 % and 0.5 %: noise. Drying of the layer the capacitive probe measures is
         governed by shade, mulch and soil structure, not by regional atmospheric demand.
      2. There is nothing to validate it against. In the pilot the soil crosses this threshold ONCE
         per campaign, and the longest dry run is 29 h. Projecting days from data whose longest
         drought lasts one would be invention.
      3. The phenomenon barely happens here. Over three years of reanalysis at these coordinates
         there are 7 dry runs of ≥3 days, with a median of 2-7 h. A rule for that would fire twice
         a year.

    What it would take to revisit: a telemetry campaign holding at least one multi-day dry run, and
    the three measurements above repeated with it in. May 2026 brings 24.5 days of the driest
    regime on record but no long run inside it, so the precondition is not met and this stays
    unbuilt."""
    vals = [hr["soil"] for hr in H if hr["soil"] is not None]
    if len(vals) < WATER["envelope_min_hours"]:
        return None
    lo, hi = min(vals), max(vals)
    if (hi - lo) < WATER["envelope_min_span"]:
        return None
    # Depleting 40 % of the span means recharging on falling to 60 % of it above the dry end.
    return lo + (1.0 - WATER["depletion_frac"]) * (hi - lo)


def water_rules(win: List[Reading], stage: str = "vegetativo",
                altitude: Optional[float] = None) -> List[Alert]:
    out: List[Alert] = []
    H = _to_hourly(win)
    if not H:
        return out
    thr = deficit_threshold(H)
    low = [(hr["soil"] is not None and thr is not None and hr["soil"] < thr) for hr in H]
    recent_rain = any(hr["rain"] for hr in H[-24:])  # it rained in the last ~24 h
    dry_then_rain = any((not H[i-1]["rain"]) and H[i]["rain"] for i in range(1, len(H)))

    # R10 — water deficit and irrigation, suppressed inside the flower induction window
    if thr is not None and _max_run(low) >= WATER["deficit_hours"] and not recent_rain:
        if stage in WATER["induction_stages"]:
            out.append(_A(rule_id="R10_DEFICIT_INDUCTION", category=CAT_C, severity="info",
                farmer_message=("El suelo está seco, pero en esta etapa un poco de sequía ayuda a que el café "
                                "florezca parejo cuando lleguen las lluvias. No es necesario regar ahora; sí "
                                "conviene poner acolchado con lo que ya tienes en la parcela: restos de la "
                                "poda o del deshierbe. No gastes jornales trayendo material de lejos."),
                agronomist_message=("Suelo bajo sostenido en etapa de inducción: el estrés moderado favorece "
                                    "floración sincronizada; SUPRIMIR recomendación de riego. Acolchado orgánico."),
                refer=False))
        elif stage in WATER["critical_stages"]:
            out.append(_A(rule_id="R10_DEFICIT_CRITICAL", category=CAT_A, severity="alert",
                farmer_message=("El suelo está seco y las plantas están en una etapa delicada (flor o llenado de "
                                "grano). Pon acolchado con los restos de tu poda o deshierbe para retener "
                                "humedad y, si tienes riego, riega "
                                "de forma moderada."),
                agronomist_message=("Suelo < umbral relativo sostenido en fase sensible (floración/llenado): déficit "
                                    "reduce cuajado/llenado y calidad. Acolchado (1ª respuesta orgánica) + riego si disponible."),
                refer=True,
                referral_note="Consulta con tu técnico un plan de riego/acolchado si la sequía se prolonga."))
        else:
            out.append(_A(rule_id="R10_DEFICIT_MILD", category=CAT_C, severity="info",
                farmer_message=("El suelo está algo seco. Pon acolchado con los restos de tu poda "
                                "o deshierbe alrededor de las "
                                "plantas para conservar la humedad."),
                agronomist_message=("Suelo bajo sostenido en fase tolerante: acolchado como conservación de humedad."),
                refer=False))

    # R11 is NOT here, and cannot be: anthesis is triggered by rain DEPTH and this sensor reports
    # rain as a boolean. Built on what it can see -- a dry hour followed by a rainy one -- the rule
    # fires nearly every day and predicts nothing. It lives in `forecast_rules`, where there are
    # millimetres: see `R11_FLOWERING_EXPECTED`.
    return out

# --------------------------------------------------------------------------- #
# COPPER ACCUMULATOR
class CopperLedger:
    def __init__(self, cap=COPPER_CAP_KG_HA_YR):
        self.cap = cap
        self.applied = 0.0
    def remaining(self):
        return max(0.0, self.cap - self.applied)
    def request(self, kg_ha: float) -> Alert:
        if self.applied + kg_ha > self.cap:
            return _A(rule_id="CU_CAP", category=CAT_A, severity="alert",
                farmer_message=("Ya usaste el máximo de cobre permitido para este año en tu parcela orgánica. "
                                "No apliques más cobre; mejor haz poda y mejora la sombra."),
                agronomist_message=(f"Aplicar {kg_ha:g} kg Cu/ha superaría el tope {self.cap:g} "
                                    f"(acumulado {self.applied:g}). Recomendación cúprica deshabilitada."),
                refer=True,
                referral_note="Consulta a tu técnico alternativas al cobre (variedades resistentes, saneamiento).")
        self.applied += kg_ha
        return _A(rule_id="CU_OK", category=CAT_A, severity="info",
            farmer_message=(f"Aplicación de cobre registrada. Te queda {self.remaining():g} kg por hectárea "
                            "para este año."),
            agronomist_message=(f"Cu +{kg_ha:g} kg/ha. Acumulado {self.applied:g}/{self.cap:g} kg/ha/año."),
            refer=False)

# --------------------------------------------------------------------------- #
# LABORATORY REMINDERS (category B) — two of them, on different cadences
#
# They do not ride in every payload, nor on a generic 30-day pulse: they are anchored to the crop
# calendar.
#   * Soil analysis: every ~2 YEARS (Cenicafé AVT 214), with the sample taken 3-4 months AFTER the
#     last fertilisation or amendment, on soil neither very wet nor very dry (AVT 497, Sadeghian
#     2018). Depth ~15 cm (active root, INIA-MIDAGRI Peru) or 20 cm (Colombia), zig-zag, 10-20+
#     sub-samples from the plate under the canopy projection.
#   * Liming: 1-2 years (AVT 466, Sadeghian 2016), with an effect lasting 2-4. Lime needs MOISTURE
#     to react, so the window is the onset of the rains, offset from fertilisation (~2 months
#     either side per Cenicafé; Ca sources 20-30 d before fertilising, and INIA Peru puts Mg
#     sources 2-3 months before). Do not mix lime with nitrogen or phosphate fertilisers.
# LIMITATION: without a fertilisation log the "3-4 months after" cannot be anchored; today it is
# emitted on cadence and the exact anchoring is explained in the message (see the phase 2 plan).
# The cadence these reminders are RESENT on does not live here: `main.py` sets it
# (`COFFEETECH_SOIL_ANALYSIS_DAYS` / `COFFEETECH_LIMING_DAYS`, 730 and 365 days by default),
# because it depends on state held on disk rather than on the reading. The engine only decides what
# the reminder says.

def lab_reminders() -> List[Alert]:
    return [
        _A(rule_id="LAB_SOIL_ANALYSIS", category=CAT_B, severity="warning",
            farmer_message=("El sensor no mide la acidez del suelo, que es clave en esta zona. Hazte un "
                            "análisis de suelo cada dos años. Toma la muestra unos 3 o 4 meses después de "
                            "la última abonada, con el suelo ni muy mojado ni muy seco."),
            agronomist_message=("Sensor no capta pH/Al/Ca/Mg/B/Zn/Fe/S/MO. Análisis cada ~2 años; muestreo 3-4 "
                                "meses post-fertilización, ~15-20 cm en el plato, zig-zag con 10-20 submuestras."),
            refer=True,
            referral_note="Pide a tu técnico de APROCASSI el muestreo de suelo y su interpretación."),
        _A(rule_id="LAB_LIMING", category=CAT_B, severity="info",
            farmer_message=("Si tu análisis mostró suelo muy ácido, el encalado se hace al empezar las lluvias: "
                            "la cal necesita humedad para actuar. Aplícala unos 20 a 30 días antes de abonar y "
                            "nunca mezclada con el abono."),
            agronomist_message=("Encalado cada 1-2 años (efecto 2-4 años). Si saturación de Al > 30%: cal "
                                "dolomítica/calcítica en bandas, ventana de inicio de lluvias, desfasada de la "
                                "fertilización (Ca 20-30 d antes; Mg 2-3 meses). pH objetivo 5.0-5.5. No mezclar "
                                "con N ni P."),
            refer=True,
            referral_note="Consulta a tu técnico la dosis de cal según la saturación de aluminio del análisis."),
    ]

# --------------------------------------------------------------------------- #
# ANTICIPATED RULES — they read the FORECAST, not the window that already happened
#
# The rules above warn about what already happened, and for several decisions that arrives too
# late. Warning that the rain washed the fertiliser away is useless; warning that rain is coming
# before fertilising saves the fertiliser and the day's labour. Rust is treated with copper BEFORE
# the infection window. The borer flies with the first rain after a dry spell, and the traps have
# to be out already.
#
# WHAT IS READ FROM THE FORECAST AND WHAT IS NOT. Measured against the three pilot campaigns (see
# `coffeetech_weather`): the difference between sensor and regional model is not a fixed bias. With
# the altitude corrected to 1823 m it changes by campaign -- +1.9 °C in nov 2025, +6.1 °C in feb
# 2026, −0.2 °C in may 2026 -- and varies 5.7-7.1 °C within the same day. Hence the line between
# the two:
#
#   * Timing and change: yes. When rain arrives and how much, wind, cloud cover. Either the sensor
#     does not measure it (wind), or it is a question about the future that no local measurement
#     can answer. There the API is the only source, and that is what it is for.
#   * Absolute temperature level: not directly. Comparing the API's temperature against a threshold
#     can be several degrees off on this plot. When a rule needs the level, it is corrected with
#     the sensor − API offset OBSERVED over the last hours (`_recent_offset`), and the threshold is
#     kept wide to absorb what remains.
#
# These rules do not use the forest from `coffeetech_forecast` beyond 6 h. The own forecast was
# measured at 24 and 72 h, did not beat persistence, and was not published (see
# `LONG_HORIZONS_NOT_PUBLISHED`). At those ranges the forecast comes from Open-Meteo, which is a
# valid weather product there.

#: Residual error when estimating the sensor's temperature from the API corrected by the last 24 h
#: offset: 1.3-2.1 °C MAE at 24-48 h ahead.
#:
#: MEASURED OVER nov + feb, before the may campaign existed. It has not been redone with all three,
#: and that is worth knowing because may is the campaign with the smallest offset (−0.2 °C against
#: +6.1 in feb): including it would probably lower this residual, so ±2 °C is if anything
#: conservative. It does not change without re-measuring -- a threshold is not loosened on an
#: expectation.
#:
#: The rolling window is used rather than the campaign's mean offset even though the latter
#: measured slightly better (1.7 against 2.0 °C), because that number is a rear-view mirror:
#: computing it needs the whole campaign, including the future being predicted. In production it
#: does not exist. Preferring the prettier number would have slipped a leak into the validation.
FORECAST_TEMP_UNCERTAINTY_C = 2.0

#: Window used to estimate the sensor − API offset. 24 h because the offset has a strong diurnal
#: component and a shorter window would sample it biased.
OFFSET_WINDOW_H = 24

# ── Where the MILLIMETRE thresholds come from ───────────────────────────────────────────
#
# The nitrate leaching literature (Frontiers 2025 on N losses in Coffea arabica; Cannavo et al. on
# coffee Andosols) establishes the MECHANISM -- losses correlate with rain, are minimal in dry
# months and maximal in wet ones, and splitting the N is the mitigation -- but publishes NO
# universal threshold in mm. It cannot: that depends on the soil's retention capacity, on the N
# applied and on percolation, not only on how much fell.
#
# So these numbers are declared for what they are: "heavy rain FOR THIS SITE", derived from the
# local regime. Percentiles measured over three years of reanalysis at the pilot's coordinates
# (2023-2025). The product is ECMWF IFS at 9 km, pinned with `models=ecmwf_ifs`, and it is not
# ERA5. Measured at 1823 m the IFS gives 861 mm/year against ERA5's 2442 -- nearly triple, so the
# product is not a provenance detail but a factor of three over these numbers.
#
#     window       p75    p90    p95    p98
#      24 h        2.5    6.4   10.0   16.3
#      48 h        5.7   11.7   18.0   25.7
#
# THE ELEVATION IS PART OF THE MEASUREMENT. Altitude moves the model's rain as well as its
# temperature -- the same series gives 1286 mm/year asked at 1450 m and 861 at the pilot's real
# 1823 -- so a percentile taken at the wrong elevation describes another site. These come from
# 1823 m.
#
# Round numbers picked by eye do not work here either: 5 mm over 72 h, which sounds like a real
# rain event, is exceeded on 46,0 % [medido:clima.horas_con_lluvia_72h_sobre_5mm_pct] of the year's
# hours and filters almost nothing.
#
# To recalculate for other coordinates: `scripts/frecuencia_reglas.py` fetches the same history.
FORECAST = {
    # R7: rain that washes the fertiliser away. p98 of 24 h. Splitting the N is the answer, and it
    # has to be decided BEFORE fertilising, not after losing it.
    "leaching_mm_24h": 16.0,
    # R8: waterlogging. p98 of 48 h, crossed with soil already near saturation.
    "waterlog_mm_24h": 26.0,
    "waterlog_soil_margin": 5.0,      # points below saturation that already count as risk
    # R3: the borer flies with the first rain after the dry spell. p90 of 24 h, which is a real
    # rain event and not drizzle. What limits this rule is the preceding dry run; the rain
    # threshold only has to tell wetting from raining.
    "borer_dry_hours": 48,            # minimum preceding dry spell, observed by the sensor
    "borer_rain_mm": 6.0,
    # R12: VPD during fruit filling.
    "vpd_fruit_fill_kpa": 0.82,
    "vpd_min_hours": 4,               # minimum hours with data for the mean to mean anything
    # R13: spray window. Both limits come from extension guidance -- drift above ~16, thermal
    # inversion below ~5 -- and are kept as they are, but measured against this farm's real wind
    # they behave very differently from each other. See the rule's note.
    "spray_wind_min_kmh": 5.0,
    "spray_wind_max_kmh": 16.0,
    "spray_dry_hours_after": 4,       # drying time without rain after spraying
    #: MEASURED hit rate and false alarm of the wind forecast when classifying an hour as suitable
    #: (5-16 km/h), over a year against what the previous day's run said. Reproducible with
    #: `python scripts/acierto_pronostico.py`; both go into the agronomist's message.
    #:
    #: Publishing the hit rate alone would keep the flattering half: of every ten hours the
    #: forecast announces as suitable, more than four are not. That is what makes R13 a suggestion
    #: with a field check rather than an instruction.
    "spray_window_hit_rate_pct": 79,
    "spray_window_false_alarm_pct": 43,
    # R14: rain expected. In rain-fed country -- nobody irrigates here, they depend on the rain --
    # knowing water is coming is the information that moves the most decisions: when to fertilise,
    # when to plant, when to lay the coffee out to dry.
    "rain_expected_mm": 1.0,          # below this it is dew or drizzle, not an event
    "rain_dry_hours": 24,             # observed dry spell that turns forecast rain into news
    "rain_expected_horizon_h": 72,
    #: MEASURED hit rate and false alarm of the rain forecast (>1 mm/day) at these coordinates,
    #: over a full year against what the run 1 and 3 days earlier said. Reproducible with
    #: `python scripts/acierto_pronostico.py`. Both go into the message: a rain warning without its
    #: rate invites more trust than the data supports.
    #:
    #: A FULL YEAR, and it has to be: the hit rate runs from 55 % to 86 % by quarter and RISES with
    #: rain frequency, because in the middle of the wet season getting "it will rain" right is
    #: easy. Any seasonal window publishes the season it was taken in.
    #:
    #: Which is also why the JSON publishes the ETS, discounting the hits chance would give anyway:
    #: that stays between 0.27 and 0.36 all year, so the forecast's quality is steadier than the
    #: headline rate makes it look.
    "rain_hit_rate_24h_pct": 79,
    "rain_hit_rate_72h_pct": 67,
    "rain_false_alarm_24h_pct": 29,
    "rain_false_alarm_72h_pct": 29,
    #: Days the above was verified over. It goes into the message, so it is published data rather
    #: than a note: saying "79 % over 365 days" is a different claim from saying it over 93.
    "hit_rate_dias": 365,
    # R11: flowering expected. Arabica anthesis is triggered by rain following a dry period, and
    # the flower opens 8-15 days later. Systematic review in Frontiers in Sustainable Food Systems
    # 2025 over Boreux et al. 2016 and Lara-Estrada et al. 2024: "anthesis promotion can occur
    # after a precipitation event greater than 10 mm in a single day, preceded by a dry period".
    # Physiological basis in Drinnan & Menzel 1994.
    "flowering_rain_mm": 10.0,
    # GAP WITHOUT INDUCTIVE RAIN, in days. Replaces `flowering_dry_hours = 24`.
    #
    # The august 2026 agronomic review rejected the 24 h -- "they break no latency because no
    # dormancy was ever induced" -- and asked for a 15-20 day dry period. Measured here, that
    # criterion leaves the rule DEAD: under the published definition (≥11 days below 0.6 mm/day)
    # the longest run in three years is 12 days, and requiring 11, 15 or 20 produces no event at
    # all.
    #
    # The cause is the site's and comes out of the data: between may and september not one day
    # above 10 mm falls, and the dry season does not end abruptly but on a ramp (20 mm in august,
    # 35 in september, 72 in october). Demanding a dry run IMMEDIATELY followed by a downpour asks
    # for a sharp transition that does not exist here.
    #
    # So the MECHANISM the review describes is kept -- dormancy accumulated over a period without
    # inductive rain, broken by the first water shock -- and calibrated to the site: the first event
    # of ≥10 mm/24 h after 90 days without one. That fires ONCE a year, all three times inside the
    # september-november window the review states for San Ignacio (26-oct-2023, 28-sep-2024,
    # 16-oct-2025), with no false positives.
    #
    # The reanalysis reproducing on its own the window the agronomist named is the cross validation
    # of both: `scripts/calibracion_floracion.py` and `docs/calibracion_floracion.json`.
    "flowering_gap_days": 90,
    # THERE IS NO SOIL GATE, and that is a declared scope decision rather than an oversight.
    #
    # The second agronomic review requires the dry period to be measured in SOIL humidity -- 15-20
    # days below field capacity -- rather than in rain, and it is right: 29.5 % of dry-season days
    # here drizzle between 0.6 and 2 mm, a depth that evaporates off the litter without reaching
    # the 20-30 cm of absorbing root. The rain gauge records rain while the plant goes thirsty.
    #
    # It is not implemented because the two requirements that would make it valid are OUT OF SCOPE:
    #   1. "Below field capacity" needs the capacitive probe calibrated against gravimetry, and
    #      there will be no soil sampling.
    #   2. Checking whether it is right needs flowering observed in the field, and there will be no
    #      further data collection.
    # Without the first the threshold is a relative number; without the second there is no way to
    # know whether it works. Coding it anyway would leave a branch that can neither run nor be
    # validated, which is exactly the defect this engine already retired from potassium's "severe"
    # band.
    #
    # Declared as future work with its literature basis in `docs/DECISIONES_DESCARTADAS.md`.
    "flowering_anthesis_days_min": 8,
    "flowering_anthesis_days_max": 15,
    # R1: anticipated rust. See the rule's long note: the TEMPERATURE comes from literature (De
    # Jong 1987; Diniz 2012) and the DURATION is a site calibration, not a published value.
    "rust_wet_hours": 12,
    "rust_temp_min": 15.0,
    "rust_temp_max": 28.0,
}


#: Peru runs at UTC−5 all year with no daylight saving, so the conversion is a subtraction and no
#: timezone database is needed. The whole system travels in UTC and local time is computed ONLY
#: here, when writing a window for the grower -- a conversion applied anywhere else leaves two
#: clocks in the same pipeline and neither of them labelled. Nobody on the farm thinks in UTC.
PERU_UTC_OFFSET_H = -5


def _a_hora_local(t: datetime) -> datetime:
    return t + timedelta(hours=PERU_UTC_OFFSET_H)


def _fc_hours(forecast, hours_ahead: int, span_hours: int, now=None) -> List[dict]:
    """Trims the forecast to a window and returns it as workable rows.

    Accepts any object with `.times` and `.values` (`coffeetech_weather`'s `HourlyForecast`) so this
    module does not depend on the HTTP client: the tests pass a double with no network.
    """
    if forecast is None:
        return []
    w = forecast.window(hours_ahead=hours_ahead, span_hours=span_hours, now=now)
    filas = []
    for i, t in enumerate(w.times):
        fila = {"t": t}
        for k, v in w.values.items():
            fila[k] = v[i] if i < len(v) else None
        filas.append(fila)
    return filas


def _sum(filas: List[dict], var: str) -> float:
    return sum(f.get(var) or 0.0 for f in filas)


def _recent_offset(win: List[Reading], forecast, hours: int = OFFSET_WINDOW_H) -> Optional[float]:
    """Mean sensor − API offset over the last `hours`, with the forecast as reference.

    Returns `None` when there is not enough overlap. That `None` is NOT replaced by zero: a zero
    would assert that the plot coincides with the regional model, and here that has been worth 4 °C.
    With no measurable offset, the rule that needs it is not emitted.
    """
    if forecast is None or not win:
        return None
    por_hora = {t.replace(minute=0, second=0, microsecond=0): v
                for t, v in zip(forecast.times, forecast.values.get("temperature_2m", []))
                if v is not None}
    if not por_hora:
        return None
    ultimo = max(r.ts for r in win)
    diffs = []
    for r in win:
        if r.air_temp is None:
            continue
        if (ultimo - r.ts).total_seconds() > hours * 3600:
            continue
        api = por_hora.get(r.ts.replace(minute=0, second=0, microsecond=0))
        if api is not None:
            diffs.append(r.air_temp - api)
    if len(diffs) < 6:      # fewer than a handful of hours is not an estimate, it is an anecdote
        return None
    return sum(diffs) / len(diffs)


def forecast_rules(win: List[Reading], forecast, stage: str = "vegetativo",
                   altitude: Optional[float] = None, now=None,
                   pending_application: bool = False) -> List[Alert]:
    """Anticipated alerts from the regional forecast plus the sensor's current state.

    Every rule crosses both sources on purpose: the sensor says how the plot is NOW (saturated
    soil, dry run) and the forecast says what is coming. Neither alone would do.

    With no forecast -- farm without a coordinate, or the API down -- it returns an empty list. Not
    a failure: the diagnosis over measured data still goes out by its own path.
    """
    if forecast is None or len(forecast) == 0:
        return []

    out: List[Alert] = []
    H = _to_hourly(win)
    # No altitude means no band, and no band means no modulation: `_weight` returns
    # `WEIGHT_UNKNOWN` and `_apply_weight` leaves the severity as it came, saying so in the
    # technician's voice.
    peso = {r: _weight(altitude, r) for r in ("rust", "borer", "leaf_spot", "cold_phoma")}
    # FRUIT LOAD here too. The MEASURED rule (`R1_RUST`) raises rust's weight one step in
    # fructificacion and maduracion -- López-Bravo, Virginio-Filho & Avelino 2012 measure +28.9 %
    # incidence and +129.2 % severity at 500 fruiting nodes against a plant with no fruit -- and
    # the ANTICIPATED one did not. Same disease, same weakened plant, and the agronomic review
    # validated the factor expressly. If either of the two had to carry it, it is this one, because
    # copper works BEFORE the infection window, not after.
    peso_rust_altitud = peso["rust"]
    if stage in ("fructificacion", "maduracion"):
        peso["rust"] = _bump_weight(peso_rust_altitud)

    prox24 = _fc_hours(forecast, 0, 24, now)
    prox48 = _fc_hours(forecast, 0, 48, now)
    prox72 = _fc_hours(forecast, 0, 72, now)
    lluvia24 = _sum(prox24, "precipitation")

    # ── R7 · ANTICIPATED nitrogen leaching ─────────────────────────────────────────────
    # The retrospective rule warns once the fertiliser has already washed away, which is late. This
    # one gets ahead of the decision: do not fertilise today.
    if lluvia24 >= FORECAST["leaching_mm_24h"]:
        out.append(_A(rule_id="R7_N_LEACHING_FORECAST", category=CAT_A, severity="warning",
            forecast=True, horizon_hours=24,
            farmer_message=(f"Se esperan unos {lluvia24:.0f} mm de lluvia en las próximas 24 horas. "
                            "No abones hoy: el abono se lavaría y perderías el jornal. Espera a que "
                            "pase y reparte el abono en 3 o 4 aplicaciones pequeñas."),
            agronomist_message=(f"Precipitación prevista {lluvia24:.1f} mm/24 h ≥ "
                                f"{FORECAST['leaching_mm_24h']:g} mm: riesgo de lixiviación de NO3- en "
                                "Andosoles. Diferir la fertilización nitrogenada y fraccionar."),
            refer=False))

    # ── R8 · ANTICIPATED waterlogging ──────────────────────────────────────────────────
    # The cross is necessary: the same rain on drained soil does not waterlog. The sensor supplies
    # the soil state, which the forecast cannot know.
    suelo = [hr["soil"] for hr in H if hr["soil"] is not None]
    suelo_actual = suelo[-1] if suelo else None
    lluvia48 = _sum(prox48, "precipitation")
    if (suelo_actual is not None
            and suelo_actual >= ENV["soil_saturation"] - FORECAST["waterlog_soil_margin"]
            and lluvia48 >= FORECAST["waterlog_mm_24h"]):
        out.append(_A(rule_id="R8_WATERLOGGING_FORECAST", category=CAT_A, severity="alert",
            forecast=True, horizon_hours=48,
            farmer_message=(f"El suelo ya está muy mojado ({suelo_actual:.0f}%) y se esperan "
                            f"{lluvia48:.0f} mm más en dos días. Abre las zanjas de drenaje ahora, "
                            "sobre todo en las partes bajas, antes de que se encharque."),
            agronomist_message=(f"Suelo {suelo_actual:.1f}% (saturación {ENV['soil_saturation']:g}%) + "
                                f"{lluvia48:.1f} mm previstos/48 h: riesgo de hipoxia radicular. "
                                "Adelantar drenaje."),
            refer=True,
            referral_note="Si el drenaje de la parcela no da abasto, pide a tu técnico evaluarlo."))

    # ── R3 · ANTICIPATED borer flight ──────────────────────────────────────────────────
    # Same stage gating as the retrospective rule: without susceptible grain there is no possible
    # colonisation, only sanitation, which runs all year in `borer_sanitation_note`.
    if stage in ("fructificacion", "maduracion", "cosecha"):
        seco_previo = _max_run([not hr["rain"] for hr in H]) >= FORECAST["borer_dry_hours"]
        lluvia72 = _sum(prox72, "precipitation")
        if seco_previo and lluvia72 >= FORECAST["borer_rain_mm"]:
            # When does that rain arrive? That is the actionable part: how many days there are
            # to set the traps.
            horas = next((int((f["t"] - prox72[0]["t"]).total_seconds() // 3600)
                          for f in prox72 if (f.get("precipitation") or 0) > 0.2), 0)
            a = _apply_weight(_A(rule_id="R3_BERRY_BORER_FORECAST", category=CAT_A, severity="alert",
                forecast=True, horizon_hours=max(horas, 1),
                farmer_message=(f"Tras el tiempo seco se espera lluvia en unas {max(horas,1)} horas: "
                                "es cuando la broca vuela y ataca el grano. Pon las trampas ahora, "
                                "antes de que llueva, y prepárate para revisar granos perforados."),
                agronomist_message=(f"Racha seca ≥{FORECAST['borer_dry_hours']} h observada + "
                                    f"{lluvia72:.1f} mm previstos a {max(horas,1)} h: ventana de vuelo de "
                                    f"colonización de H. hampei. Umbral de acción "
                                    f"{ENV['broca_action_pct']:g}% (orgánico estricto)."),
                refer=True,
                referral_note=(f"Si más del {ENV['broca_action_pct']:g}% de granos están brocados, coordina "
                               "con tu técnico la aplicación de Beauveria bassiana.")), peso["borer"])
            if a:
                out.append(a)

    # ── R14 · Rain expected ────────────────────────────────────────────────────────────
    # The simplest rule in the module and the one resting most on measurements taken here. In
    # San Ignacio nobody irrigates: they depend on the rain. Knowing water is coming moves the
    # decisions a rain-fed grower can actually make -- fertilise or wait, lay the coffee out to dry
    # or bring it in, transplant -- and none of them needs a model. They need the fact in time.
    #
    # Verified over a FULL YEAR at these coordinates, comparing what earlier days' runs said with
    # what finally fell (`python scripts/acierto_pronostico.py`):
    #
    #                 hits when it rains   false alarm   ETS
    #   1 day ahead         78.6 %            28.6 %     0.34
    #   2 days ahead        73.8 %            30.7 %     0.30
    #   3 days ahead        67.3 %            28.9 %     0.28
    #
    # The year is not a formality: the hit rate runs 55 % to 86 % by quarter and rises with rain
    # frequency, so a wet-season window alone reports something closer to 89 %. The ETS, which
    # discounts chance hits, stays between 0.27 and 0.36 across all four quarters.
    #
    # Of everything the external forecast contributes, this is the best verified. Which is why
    # the rule says rain is expected and never that it will rain, and carries both its hit rate AND
    # its false alarm: publishing only the first would keep the flattering half.
    #
    # ONLY ON THE CHANGE OF STATE. Announcing rain whenever any is forecast fires on
    # 87,5 % [medido:cf.lluvia_sin_puerta_seca] of days against
    # 23,8 % [medido:cf.lluvia_sin_puerta_seca.servido] with the gate in place: 861 mm fall here
    # each year and almost every 72 h window has water in it. A true fact and a useless warning.
    # news is water arriving AFTER a dry period, which is when decisions change, so the sensor is
    # required not to have seen rain in the last hours. The plot supplies the dry part; the API,
    # the rain that is coming.
    seco_reciente = _max_run([not hr["rain"] for hr in H[-FORECAST["rain_dry_hours"]:]]) >= \
        FORECAST["rain_dry_hours"] if len(H) >= FORECAST["rain_dry_hours"] else False
    primera_lluvia = next(
        (f for f in prox72 if (f.get("precipitation") or 0) >= 0.2), None)
    lluvia72 = _sum(prox72, "precipitation")
    if seco_reciente and primera_lluvia is not None and lluvia72 >= FORECAST["rain_expected_mm"]:
        horas = max(1, int((primera_lluvia["t"] - prox72[0]["t"]).total_seconds() // 3600))
        cuando = _a_hora_local(primera_lluvia["t"])
        corto = horas <= 24
        acierto = FORECAST["rain_hit_rate_24h_pct" if corto else "rain_hit_rate_72h_pct"]
        falsas = FORECAST["rain_false_alarm_24h_pct" if corto else "rain_false_alarm_72h_pct"]
        # The probability the API publishes for that hour, when present. It belongs to the
        # forecast rather than the archive, so it does not exist in historical reports and the
        # message adapts.
        prob = primera_lluvia.get("precipitation_probability")
        detalle = f" (probabilidad {prob:.0f}%)" if prob is not None else ""
        out.append(_A(rule_id="R14_RAIN_EXPECTED", category=CAT_A, severity="info",
            forecast=True, horizon_hours=horas,
            farmer_message=(f"Se espera lluvia a partir de las {cuando.hour:02d}:00 "
                            f"{'de hoy' if horas <= 18 else 'del ' + format(cuando, '%d/%m')}, "
                            f"unos {lluvia72:.0f} mm en tres días. Aprovecha para lo que necesita "
                            "agua y deja para después lo que la lluvia estropea: abonar justo "
                            "antes de un aguacero es perder el abono."),
            agronomist_message=(f"Precipitación prevista {lluvia72:.1f} mm/72 h, inicio a {horas} h"
                                f"{detalle}. Verificado en estas coordenadas sobre "
                                f"{FORECAST['hit_rate_dias']} días: acierta el {acierto}% de las "
                                f"veces que llueve, con {falsas}% de falsas alarmas. Ventana para "
                                "programar fertilización, trasplante y secado."),
            refer=False))

    # ── R11 · Flowering expected ───────────────────────────────────────────────────────
    # It cannot live in `water_rules`: over measured data all it can ask for is a dry hour followed
    # at some point by a rainy one, which is true on nearly every day. The published thresholds
    # need depth -- anthesis is triggered by more than 10 mm in one day after a dry period, and the
    # flower opens 8-15 days later (Frontiers in Sustainable Food Systems 2025, over Boreux et al.
    # 2016 and Lara-Estrada et al. 2024; physiology in Drinnan & Menzel 1994).
    #
    # THE DRY PERIOD IS MEASURED IN THE API, NOT IN THE SENSOR. The sensor only gives rain yes/no,
    # and the question here is in millimetres: has there been any 10 mm downpour in the last 90
    # days? That is why `PAST_DAYS` went from 2 to 91 -- without that past the question cannot be
    # answered.
    #
    # It is measured on a ROLLING 24 h accumulation, not by calendar day. A downpour spread between
    # 22:00 and 02:00 is an inductive event just the same, and splitting it at midnight would hide
    # it -- the same defect of form this engine already corrected in the persistence rules.
    if stage in ("vegetativo", "floracion"):
        lluvia_dia = max((_sum(_fc_hours(forecast, h, 24, now), "precipitation")
                          for h in range(0, 48, 6)), default=0.0)
        gap_h = 24 * FORECAST["flowering_gap_days"]
        pasado = _fc_hours(forecast, -gap_h, gap_h, now)
        # Largest rolling 24 h accumulation inside the gap. If any reached the threshold, the
        # plant already had its water shock and there is no dormancy left to break.
        mayor_pasado, suma, cola = 0.0, 0.0, []
        for f in pasado:
            cola.append(f.get("precipitation") or 0.0)
            suma += cola[-1]
            if len(cola) > 24:
                suma -= cola.pop(0)
            mayor_pasado = max(mayor_pasado, suma)
        # Without enough past it does NOT fire. Claiming "it has not rained in 90 days" on 3 days
        # of data is claiming something unknown, and this rule exists to announce a rare event: a
        # false positive here sends someone to prepare boron and calcium for a flowering that is
        # not coming.
        hay_pasado = len(pasado) >= gap_h * 0.9
        seco_previo = hay_pasado and mayor_pasado < FORECAST["flowering_rain_mm"]

        if seco_previo and lluvia_dia >= FORECAST["flowering_rain_mm"]:
            out.append(_A(rule_id="R11_FLOWERING_EXPECTED", category=CAT_C, severity="info",
                forecast=True, horizon_hours=24 * FORECAST["flowering_anthesis_days_min"],
                farmer_message=(f"Tras la seca se esperan unos {lluvia_dia:.0f} mm de lluvia, que es lo que "
                                "despierta la floración. Si se cumple, el café debería florear en una o dos "
                                "semanas: ve preparando el abono con boro y calcio para que la flor cuaje."),
                agronomist_message=(f"Lluvia prevista {lluvia_dia:.1f} mm/24 h "
                                    f"(umbral {FORECAST['flowering_rain_mm']:g} mm) tras "
                                    f"{FORECAST['flowering_gap_days']} días sin ningún evento de esa "
                                    f"magnitud: rompe latencia de yemas. Antesis esperada en "
                                    f"{FORECAST['flowering_anthesis_days_min']}-"
                                    f"{FORECAST['flowering_anthesis_days_max']} días. "
                                    "LÍMITE DECLARADO: el periodo seco se mide en LLUVIA, no en la "
                                    "humedad del suelo, que es lo que pide la fisiología — la planta "
                                    "responde a la tensión hídrica de la raíz, no al pluviómetro. "
                                    f"Los {FORECAST['flowering_gap_days']} días son un artificio "
                                    "operativo para sortear la llovizna andina, calibrado sobre "
                                    "reanálisis y NO validado contra floración observada. Úsese como "
                                    "aviso de ventana fenológica, no como predicción de fecha."),
                refer=False))

    # ── R12 · VPD during fruit filling ─────────────────────────────────────────────────
    # Threshold: VPD above 0.82 kPa during fruit development (Nature Food 2022,
    # s43016-022-00614-8: yield falls sharply above that value). DaMatta 2025
    # (10.1002/ael2.70050): what limits coffee's photosynthesis is VPD rather than temperature
    # itself, through stomatal closure.
    #
    # It does not duplicate R9_HEAT_QUALITY, which looks at temperature: two days at the same
    # temperature and different humidities give very different VPD, and it is the VPD that closes
    # the stomata.
    #
    # WHAT KIND OF QUANTITY 0.82 kPa IS, because two plausible readings of it are both wrong.
    #
    # It is NOT an hourly PEAK. "At least 4 hours above 0.82" fires almost daily:
    # 19,5 % [medido:clima.vpd.horas_feb_jul_sobre_umbral_pct] of feb-jul hours exceed the threshold
    # raw, so asking for four of them is barely asking for anything.
    #
    # And it is NOT a DAYTIME mean, however reasonable that sounds -- before dawn there is no
    # photosynthesis to limit. Kath et al. average over ALL hours of the season, and a daytime mean
    # runs systematically higher. "Daytime" also has no standard definition, so a figure computed
    # that way cannot be reproduced without the window it used.
    #
    # So the comparison is made against something that depends on no unwritten choice. Measured
    # over a year
    # at these coordinates and at 1823 m (`scripts/contrafactuales.py`), the 24 h mean is
    # 0,515 kPa [medido:clima.vpd.media_24h_anual_kpa] and the daytime mean's offset runs from
    # +0,198 [medido:clima.vpd.desfase_min_kpa] to +0,461 [medido:clima.vpd.desfase_max_kpa] kPa
    # depending on where the day is cut, across eight windows from 06-20 h to 10-15 h local.
    #
    # The conclusion holds across that range: the review itself puts at 0.10-0.20 kPa the offset
    # that would force a recalibration, and here it is exceeded even by the widest window -- 06-20 h
    # gives +0.198 and grazes the limit -- and doubled or tripled by any narrow definition of
    # daytime. A quantity was being compared against a threshold derived from a different one, and
    # the bias is real however it is defined.
    #
    # WHAT IS STILL NOT EXACT, declared: Kath et al. average over the flowering and filling SEASON,
    # months long, from NATIONAL yields via a GAM, not from plot physiology. This averages a 48 h
    # window on one farm. So this is an early-warning heuristic, not a plot-validated threshold.
    if stage in ("fructificacion", "maduracion"):
        horas_vpd = [f for f in prox48 if f.get("vapour_pressure_deficit") is not None]
        medio = (sum(f["vapour_pressure_deficit"] for f in horas_vpd) / len(horas_vpd)
                 if horas_vpd else 0.0)
        if len(horas_vpd) >= FORECAST["vpd_min_hours"] and medio > FORECAST["vpd_fruit_fill_kpa"]:
            out.append(_A(rule_id="R12_VPD_FRUIT_FILL", category=CAT_A, severity="warning",
                forecast=True, horizon_hours=48,
                # Nobody irrigates in this area: they depend on the rain. So what is actionable
                # comes first -- shade and mulching, which are within reach -- and irrigation is
                # left to the end and conditional, as R10 already does. Leading with "irrigate"
                # would be an instruction most cannot follow.
                farmer_message=("Vienen días de aire muy seco justo cuando el grano está llenando. "
                                "La planta cierra sus poros y deja de alimentar el grano. Revisa que "
                                "la sombra esté cubriendo bien y pon acolchado con los restos de tu poda o "
                                "deshierbe "
                                "para que el suelo aguante la humedad que le queda. No podes la sombra "
                                "estos días. Si tienes riego, esta parcela es la prioridad."),
                agronomist_message=(f"VPD medio previsto {medio:.2f} kPa sobre {len(horas_vpd)} h "
                                    f"(media de todas las horas, como Kath et al. 2022; umbral "
                                    f"{FORECAST['vpd_fruit_fill_kpa']:g} kPa) en etapa {stage}: cierre "
                                    "estomático y caída de llenado. Alerta temprana, no umbral validado "
                                    "en parcela. Manejo de sombra; riego prioritario donde exista."),
                refer=False))

    # ── R1 · ANTICIPATED rust ──────────────────────────────────────────────────────────
    # Thresholds: a forecast leaf wetness window of at least 12 h with temperature between 15 and
    # 28 °C.
    #
    # TEMPERATURE: 15-28 °C, with citations. De Jong, Eskes, Hoogstraten et al. 1987 (Neth. J.
    # Plant Pathol. 93:61-71, doi 10.1007/BF01998091) puts the germination limits at 13 and ~30 °C,
    # fastest between 22 and 28. Diniz et al. 2012 gives minimum 15.5, optimum 22.0 and maximum
    # 28.5 °C. Nutman & Roberts 1963 (Trans. Br. Mycol. Soc.) fixes the classic optimum at ~22 °C.
    #
    # The floor is 15 and not 18 °C, which the citations allow and this site requires: at 1823 m
    # the 15-18 °C range covers the small hours and the mornings, and a year of reanalysis puts
    # 1763 hours a year in it WITH WET LEAVES. An 18 °C floor drops all of them.
    #
    # CAVEAT: the optima vary by H. vastatrix race and may be shifting -- Rozo et al. 2012 reports
    # 24 °C against the historical 22 -- so any fixed window is an approximation. The temperature
    # also comes from the regional forecast and carries ±2 °C even after correcting for the recent
    # offset (`FORECAST_TEMP_UNCERTAINTY_C`).
    #
    # DURATION: 12 h, and NOT a literature value. The literature spans a wide range depending on
    # conditions:
    #   * De Jong 1987 / Avelino et al. 2004: infection in as little as 4-6 h when high temperatures
    #     (22-28 °C, favouring germination) ALTERNATE with cool ones (13-16 °C, favouring appressorium
    #     formation).
    #   * Gichuru et al. 2021 (Agronomy 11(12):2590): 24-48 h of CONTINUOUS free moisture for
    #     uredinial germination and infection.
    # The alternation of temperatures that enables the fast case cannot be verified here, and the
    # leaf wetness proxy -- RH ≥ 90 % in a 25 km cell -- is generous: it is not free water on the
    # leaf. At 6 h the rule fires on 33,0 % [medido:cf.roya_anticipada_6h] of the year's days; at
    # the 12 h served, on 3,9 % [medido:cf.roya_anticipada_6h.servido].
    #
    # So 12 h is declared for what it is: a site-calibrated compensation for a coarse proxy, sitting
    # between the fast case and continuous wetness. It is not presented as a published threshold,
    # because it is not one.
    #
    # WIND IS NOT USED HERE even though the API provides it: the literature attributes two opposite
    # effects to it -- dry wind SHORTENS leaf wetness duration (less infection) but RELEASES and
    # disperses urediniospores (more spread) -- and without local data settling which dominates,
    # adding it with a sign would be invention. Documented so it is not retried blind.
    desfase = _recent_offset(win, forecast)
    if desfase is not None:
        mojado = []
        for f in prox72:
            rh = f.get("relative_humidity_2m") or 0
            llueve = (f.get("precipitation") or 0) > 0.1
            t_api = f.get("temperature_2m")
            if t_api is None:
                mojado.append(False)
                continue
            t = t_api + desfase       # brought onto the sensor's scale, which is where the threshold lives
            mojado.append((rh >= ENV["rh_leafwet"] or llueve)
                          and FORECAST["rust_temp_min"] <= t <= FORECAST["rust_temp_max"])
        racha = _max_run(mojado)
        if racha >= FORECAST["rust_wet_hours"]:
            horas = next((int((f["t"] - prox72[0]["t"]).total_seconds() // 3600)
                          for f, m in zip(prox72, mojado) if m), 0)
            a = _apply_weight(_A(rule_id="R1_RUST_FORECAST", category=CAT_A, severity="alert",
                forecast=True, horizon_hours=max(horas, 1),
                farmer_message=(f"En unas {max(horas,1)} horas se juntan hoja mojada y temperatura buena "
                                "para la roya. El cobre sirve ANTES, no después: si no ha llovido, aplícalo "
                                "hoy aprovechando la hora de buen viento."),
                agronomist_message=(f"Ventana de hoja mojada prevista {racha} h "
                                    f"(≥{FORECAST['rust_wet_hours']} h) con T "
                                    f"{FORECAST['rust_temp_min']:g}-{FORECAST['rust_temp_max']:g} °C "
                                    f"corregida por desfase reciente {desfase:+.1f} °C "
                                    f"(±{FORECAST_TEMP_UNCERTAINTY_C:g} °C): germinación de H. vastatrix. "
                                    "Cobre preventivo antes de la ventana; respetar CU_CAP."),
                refer=False), peso["rust"], peso_rust_altitud)
            if a:
                out.append(a)

    # ── R13 · Spray window ─────────────────────────────────────────────────────────────
    # Wind between 5 and 16 km/h. Above ~16 drift grows fast; below ~5 the wind is erratic and
    # thermal inversion becomes a risk, which concentrates the droplet and carries it further (SDSU
    # Extension "How to stop drift"; the 3-10 mph consensus also comes from Purdue, Rutgers, Montana
    # State and U. Minnesota Extension).
    #
    # It answers a question nobody else answers -- can I spray now? -- and completes the rust
    # warning, which says what to apply but not when. Wind is also the variable where the API is
    # indisputable: the sensor does not measure it.
    #
    # HOW THESE TWO LIMITS BEHAVE HERE, over a verified year. This farm's wind is light: median
    # 3.1 km/h, p95 8.2, p99 9.9, maximum 14.7. An hour falls inside the suitable band only 27.7 %
    # of the time. Measured over a year with `python scripts/acierto_pronostico.py`:
    #
    #   * The UPPER limit (16 km/h) NEVER binds: 0.00 % of the year's hours exceed it. It is kept
    #     because the drift citation is valid and another farm may have more wind, but nobody
    #     should read it as an active condition in San Ignacio.
    #   * The LOWER limit (5 km/h) is what governs, and more than expected: 72.3 % of hours fall
    #     BELOW it. What limits the spray window here is not too much wind but too little. The
    #     floor also sits above the median, in the thick of the distribution, and there the
    #     forecast's error (MAE 1.99 km/h at one day) flips the classification easily: it catches
    #     79 % of the genuinely suitable hours but announces as suitable 43 % THAT ARE NOT.
    #
    # That second figure is the one that decides how the message is worded, and it is also the one
    # a short window flatters: measured over a single season the false alarm comes out below 20 %,
    # against the 43 % the full year gives.
    #
    # Which is why the message does NOT present the band as a certainty and the agronomist's
    # carries BOTH rates. Promising an exact hour with that margin would sell precision that is not
    # there.
    hay_que_aplicar = pending_application or any(
        a.rule_id in ("R1_RUST_FORECAST", "R3_BERRY_BORER_FORECAST") for a in out)
    if stage != "plantula" and hay_que_aplicar:
        franjas = []
        for i, f in enumerate(prox24):
            v = f.get("wind_speed_10m")
            if v is None or not (FORECAST["spray_wind_min_kmh"] <= v <= FORECAST["spray_wind_max_kmh"]):
                continue
            # It has to stay dry as long as the spray takes to dry, or the application washes
            # off.
            secado = prox24[i:i + 1 + FORECAST["spray_dry_hours_after"]]
            if _sum(secado, "precipitation") > 0.2:
                continue
            franjas.append(f)
        # The band has to be CONTINUOUS. Taking the first and last suitable hours would announce
        # "from 8:00 to 23:00" when what exists is two good stretches with a bad afternoon between
        # them, and the grower would go out to spray in exactly the hours the rule discarded.
        mejor: List[dict] = []
        actual: List[dict] = []
        for f in franjas:
            if actual and (f["t"] - actual[-1]["t"]).total_seconds() > 3700:
                actual = []
            actual.append(f)
            if len(actual) > len(mejor):
                mejor = list(actual)
        franjas = mejor
        if franjas:
            desde, hasta = _a_hora_local(franjas[0]["t"]), _a_hora_local(franjas[-1]["t"])
            out.append(_A(rule_id="R13_SPRAY_WINDOW", category=CAT_A, severity="info",
                forecast=True, horizon_hours=24,
                farmer_message=(f"Según el pronóstico, la mejor hora para aplicar sería entre las "
                                f"{desde.hour:02d}:00 y las {hasta.hour:02d}:00: el viento acompaña y no "
                                "se espera lluvia después. Es una previsión, así que asómate antes de "
                                "preparar la mezcla; si notas el aire quieto o muy movido, déjalo."),
                agronomist_message=(f"Ventana de aplicación {desde:%d/%m %H:%M}-{hasta:%H:%M} hora local "
                                    f"({len(franjas)} h continuas; el pronóstico de viento acierta el "
                                    f"{FORECAST['spray_window_hit_rate_pct']}% de las horas aptas pero "
                                    f"anuncia como aptas un {FORECAST['spray_window_false_alarm_pct']}% "
                                    f"que no lo son): viento "
                                    f"{FORECAST['spray_wind_min_kmh']:g}-{FORECAST['spray_wind_max_kmh']:g} km/h "
                                    f"y sin precipitación en las {FORECAST['spray_dry_hours_after']} h de "
                                    "secado. Fuera del rango: deriva (>16) o inversión térmica (<5)."),
                refer=False))
        else:
            # Saying so is as useful as the window itself: without it the grower cannot tell
            # whether there is no band or the rule simply did not run.
            out.append(_A(rule_id="R13_SPRAY_WINDOW_NONE", category=CAT_A, severity="info",
                forecast=True, horizon_hours=24,
                farmer_message=("Hoy no hay una buena hora para aplicar: o el viento está muy fuerte o "
                                "muy calmo, o llueve poco después. Si puedes, deja la aplicación para "
                                "mañana."),
                agronomist_message=(f"Sin ventana en 24 h: ninguna hora cumple viento "
                                    f"{FORECAST['spray_wind_min_kmh']:g}-{FORECAST['spray_wind_max_kmh']:g} km/h "
                                    f"con {FORECAST['spray_dry_hours_after']} h secas posteriores."),
                refer=False))

    return out

# --------------------------------------------------------------------------- #
# PUBLISHABLE REFERENCE — what the engine considers "adequate"
#
# It exists so the frontend's Reports chart stops keeping its own range table: a second table falls
# out of sync and ends up painting red what the diagnosis calls adequate. Everything here is
# DERIVED from SENSOR_DOMAIN / NPK_BANDS / ENV / WATER; no number is rewritten. Each band's
# severity is read from the real Alert the engine emits, not from a parallel map.
def _npk_reference(nut: str, stage: str) -> Dict[str, Any]:
    dmin, dmax = SENSOR_DOMAIN[nut]
    raw = NPK_BANDS[nut]
    bands: List[Dict[str, Any]] = []
    lower = float("-inf")
    for upper, _label in raw:
        hi = float("inf") if upper is None else float(upper)
        # Outside the calibrated domain the engine does not classify: it cuts earlier
        # (NPK_*_OUT_OF_DOMAIN). So the bands are trimmed to the domain and the ones left empty are
        # not published.
        lo_c, hi_c = max(lower, dmin), min(hi, dmax)
        lower = hi
        if hi_c <= lo_c:
            continue
        probe = (lo_c + hi_c) / 2.0
        band = _stage_adjust_band(nut, probe, _classify(probe, raw), stage)
        bands.append({
            "from": lo_c,
            "to": hi_c,
            "label": band,
            "label_es": _BAND_ES.get(band, band),
            "severity": _npk_alert(nut, probe, band, stage).severity,
        })
    adequate = [b for b in bands if b["label"] == "adequate"]
    ref: Dict[str, Any] = {
        "kind": "band",
        "unit": "mg/kg",
        "domain": [dmin, dmax],
        "optimal": [adequate[0]["from"], adequate[-1]["to"]] if adequate else None,
        "bands": bands,
    }
    if nut == "N":
        # The engine marks all N as provisional: it is a proxy for mineral N, not traceable to
        # Kjeldahl or foliar analysis. The frontend must draw it as indicative, not as truth.
        ref["provisional"] = True
        ref["note"] = ("Lectura provisional: el sensor estima N mineral, que varía mucho con la "
                       "lluvia. La referencia real es un análisis foliar.")
    if not adequate:
        ref["note"] = (f"En etapa '{stage}' el motor no reconoce una banda adecuada para {nut} "
                       f"dentro del dominio del sensor.")
    return ref

def _temperature_bands() -> List[Dict[str, Any]]:
    """Thermal ranges, obtained by exercising `env_alerts` at the midpoint of each.

    That way the label and the severity are decided by the rule that actually runs in the
    diagnosis: move `temp_hot` tomorrow and this moves with it."""
    cold, warn, hot = ENV["temp_cold"], ENV["temp_warn"], ENV["temp_hot"]
    # Ends clamped to what a field sensor can read, so no infinities are published.
    edges = [(0.0, cold), (cold, warn), (warn, hot), (hot, 50.0)]
    labels = {"R9_COLD": ("cold", "frío"), "R9_HEAT_QUALITY": ("warm", "calor"),
              "R9_ACUTE_HEAT": ("acute_heat", "calor agudo")}
    out: List[Dict[str, Any]] = []
    for lo, hi in edges:
        probe = (lo + hi) / 2.0
        alerts = env_alerts(Reading(ts=datetime(2026, 1, 1), air_temp=probe))
        if alerts:
            label, label_es = labels[alerts[0].rule_id]
            severity = alerts[0].severity
        else:
            label, label_es, severity = "adequate", "adecuado", "info"
        out.append({"from": lo, "to": hi, "label": label, "label_es": label_es,
                    "severity": severity})
    return out


def reference_ranges(stage: Optional[str] = None) -> Dict[str, Any]:
    """The reference ranges the engine applies, for publishing to the interface.

    `stage` adjusts what the diagnosis really adjusts: P in plantula, K in fructificacion and
    maduracion. With no stage the base reading is returned, unadjusted."""
    stage = stage if stage in VALID_STAGES else None
    effective = stage or ""
    return {
        "engine": "coffeetech_rules_v5",
        "stage": stage,
        "metrics": {
            **{nut: _npk_reference(nut, effective) for nut in ("N", "P", "K")},
            "temperature": {
                "kind": "band",
                "unit": "°C",
                "optimal": [ENV["temp_cold"], ENV["temp_warn"]],
                # Without `bands` the interface cannot issue a verdict; this was the only metric
                # with a band and no chip. They are derived by exercising `env_alerts`, so the
                # label and severity come from the real rule rather than a parallel map.
                "bands": _temperature_bands(),
                # `short_es` labels the chart, where space is what it is; `label_es` is the
                # explanation for a tooltip or legend.
                "thresholds": [
                    {"below": ENV["temp_cold"], "severity": "warning", "short_es": "Frío",
                     "label_es": "frío: la planta casi no absorbe agua ni abono"},
                    {"above": ENV["temp_warn"], "severity": "warning", "short_es": "Calor",
                     "label_es": "calor: estrés y riesgo de maduración prematura"},
                    {"above": ENV["temp_hot"], "severity": "alert", "short_es": "Calor agudo",
                     "label_es": "calor agudo: cierre estomático"},
                ],
            },
            "air_humidity": {
                # The engine has no "optimal band" for air humidity: these are fungal RISK
                # thresholds going upward. Publishing a band would be inventing one.
                "kind": "threshold",
                "unit": "%",
                "optimal": None,
                "thresholds": [
                    {"above": ENV["rh_fungal"], "severity": "warning", "short_es": "Riesgo fúngico",
                     "label_es": "humedad alta sostenida: riesgo fúngico"},
                    {"above": ENV["rh_leafwet"], "severity": "alert", "short_es": "Hoja mojada",
                     "label_es": "hoja mojada: infección de roya"},
                ],
            },
            "soil_humidity": {
                # Deliberately without a band: the capacitive probe is not calibrated in volume,
                # so an absolute % means nothing. The engine derives the irrigation threshold from
                # the wet-dry envelope observed at each deployment (see deficit_threshold).
                "kind": "relative",
                "unit": "%",
                "optimal": None,
                "depletion_frac": WATER["depletion_frac"],
                "saturation": ENV["soil_saturation"],
                "note": ("El riego no se decide con un porcentaje fijo: se compara cada lectura "
                         "con el recorrido húmedo-seco propio de esta parcela."),
            },
        },
    }

# --------------------------------------------------------------------------- #
def evaluate_reading(r: Reading) -> List[Alert]:
    return diagnose_npk(r) + env_alerts(r) + stage_notes(r.stage)

def _declarar_etapa_desconocida(alerts: List[Alert], stage: Optional[str]) -> List[Alert]:
    """Records in the technician's voice that the stage was not registered, as is done for altitude.

    Without this note the technician reads a recommendation that ALREADY degraded to its neutral
    branch with no way of knowing why, and would believe the system evaluated the stage and decided
    that. Staying quiet turns a prudent degradation into a false claim.
    """
    if stage in VALID_STAGES:
        return alerts
    for a in alerts:
        a.agronomist_message += (
            " [Etapa no registrada para esta sección: no se aplicó ninguna regla condicionada por "
            "etapa y el manejo hídrico degradó a su recomendación neutra. Registrar la etapa "
            "fenológica de la sección.]")
    return alerts


def evaluate_window(win: List[Reading]) -> List[Alert]:
    if not win:
        return []
    last = win[-1]
    return _declarar_etapa_desconocida(evaluate_reading(last)
            + borer_sanitation_note(last.stage)
            + inferential_rules(win, altitude=last.altitude, stage=last.stage)
            + water_rules(win, stage=last.stage, altitude=last.altitude)
            + lab_reminders(), last.stage)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import csv, sys
    path = sys.argv[1] if len(sys.argv) > 1 else "feb.csv"
    stage = sys.argv[2] if len(sys.argv) > 2 else "fructificacion"
    win: List[Reading] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ts = datetime.fromisoformat(row["createdAt"].replace("Z", "+00:00"))
            except Exception:
                continue
            g = lambda k: float(row[k]) if row.get(k) not in (None, "") else None
            win.append(Reading(ts=ts, N=g("nitrogen_mg_kg"), P=g("phosphorus_mg_kg"), K=g("potassium_mg_kg"),
                air_temp=g("celcius_grade_temperature"), air_rh=g("air_humidity_percent"),
                soil_moist=g("soil_humidity_percent"),
                rain=str(row.get("precipitation_detected", "0")).strip() in ("1", "1.0", "true", "True"),
                stage=stage))
    # With no altitude on the command line, none is invented: `None` means unknown and the engine
    # does not modulate. A fallback value would insert a band nobody measured.
    alt = float(sys.argv[3]) if len(sys.argv) > 3 else None
    win.sort(key=lambda x: x.ts)
    for r in win:
        r.stage = stage; r.altitude = alt
    def mean(a):
        v = [getattr(x, a) for x in win if getattr(x, a) not in (None, 0.0)]
        return sum(v) / len(v) if v else None
    rep = Reading(ts=win[-1].ts, N=mean("N"), P=mean("P"), K=mean("K"),
                  air_temp=mean("air_temp"), stage=stage, altitude=alt)
    print(f"Ventana: {len(win)} lecturas | etapa = {stage} | altitud = {alt:g} msnm ({altitude_band(alt)})\n")
    print("── Diagnóstico puntual (lectura media) ──")
    for a in diagnose_npk(rep) + stage_notes(stage):
        print(a); print()
    print("── Reglas inferenciales (moduladas por altitud) ──")
    for a in inferential_rules(win, altitude=alt, stage=stage):
        print(a); print()
    print("── Reglas de manejo hídrico ──")
    wr = water_rules(win, stage=stage, altitude=alt)
    for a in (wr or []):
        print(a); print()
    if not wr:
        print("  (ninguna regla hídrica se disparó)\n")
    print("── Recordatorio de laboratorio ──")
    for a in lab_reminders():
        print(a)
