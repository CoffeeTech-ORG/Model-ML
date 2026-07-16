# -*- coding: utf-8 -*-
"""
CoffeeTech — adapter between the rule engine and the backend.

Turns the validated engine's alerts (`coffeetech_rules_v5`) into `Recommendation` objects,
enriches them with the organic dose layer (`coffeetech_doses`) and maps them to the payload the
backend expects. It is the single source of truth for the operational diagnosis: no random forest
and no circular labelling take part.

Standard library only, so it can be exercised offline against the telemetry CSV.

Flow:
    Reading window  --evaluate_window-->  List[Alert]
                    --_alert_to_rec-->    List[Recommendation]  (ordered by severity)
                    --map_to_azure_payload-->  {recommendationDescription, deviceHubId}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Iterable

from coffeetech_rules_v5 import (
    Reading,
    Alert,
    STAGE_UNKNOWN,
    evaluate_window,
    env_alerts,
    _classify,
    NPK_BANDS,
    SENSOR_DOMAIN,
    CONVENTIONAL_BLOCKED,
    VALID_STAGES,
)
from coffeetech_doses import build_prescription, Prescription


# --------------------------------------------------------------------------- #
# The recommendation object: what the adapter produces for each detected condition.
@dataclass
class Recommendation:
    rule_id: str
    category: str                         # "A" real time | "B" laboratory | "C" management
    rec_type: str                         # nutrition|irrigation|disease|pest|thermal|lab|management
    severity: str                         # info | warning | alert | critical
    farmer_message: str
    agronomist_message: str
    action: Optional[str] = None
    product: Optional[str] = None         # ORGANIC product (or None)
    dose: Optional[str] = None            # dose with its unit (a starting point)
    method: Optional[str] = None          # soil | foliar | cultural | None
    timing: Optional[str] = None
    verification: Optional[str] = None    # verification step (scouting/trap/laboratory)
    refer: bool = False
    referral_note: Optional[str] = None
    provisional: bool = False             # true for N (and for readings outside the domain)
    dose_disclaimer: Optional[str] = None
    # Anticipated alert: it comes from the microclimate forecast, not from observed telemetry.
    forecast: bool = False
    horizon_h: Optional[float] = None     # forecast horizon, in hours
    forecast_detail: Optional[str] = None  # forecast detail for the text

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# SUPPORTING TABLES
SEV_RANK = {"critical": 0, "alert": 1, "warning": 2, "info": 3}

# rule_id (or its prefix) -> recommendation type
def _type_for(rule_id: str) -> str:
    if rule_id.endswith("_OUT_OF_DOMAIN"):
        return "lab"
    # An anticipated rule is about the same problem as its measured version: forecast rust is
    # still disease and forecast borer is still a pest. Without this they would all land in
    # "manejo" for not matching the identifiers below exactly.
    if rule_id.endswith("_FORECAST"):
        return _type_for(rule_id[:-len("_FORECAST")])
    if rule_id.startswith("R12_VPD"):
        return "thermal"          # stress from atmospheric demand, same family as the thermal signal
    if rule_id.startswith("R13_SPRAY_WINDOW"):
        return "disease"          # exists to say WHEN to apply what another rule called for
    if rule_id.startswith("NPK_"):
        return "nutrition"
    if rule_id.startswith("R9_"):
        return "thermal"
    if rule_id in ("R1_RUST", "R2_AMERICAN_LEAF_SPOT", "R4_CERCOSPORA", "R6_COLD_PHOMA"):
        return "disease"
    if rule_id in ("R3_BERRY_BORER", "R3_BORER_SANITATION", "R5_LEAF_MINER"):
        return "pest"
    if rule_id.startswith("R10_"):
        return "irrigation"
    if rule_id in ("R7_N_LEACHING", "R8_WATERLOGGING", "R11_FLOWERING_EXPECTED"):
        return "management"
    if rule_id.startswith("STAGE_"):
        return "management"
    if rule_id.startswith("LAB_"):
        return "lab"
    if rule_id.startswith("CU_"):
        return "disease"
    return "management"


# rule_id -> field verification step. Never a definitive diagnosis.
#
# Every FORECAST alert has to have one. Without it `_actionability_for` leaves the alert as
# "direct" -- you can apply this now -- and the system ends up more assertive about what it merely
# predicts than about what it measured: `R1_RUST` said "check before applying copper" while
# `R1_RUST_FORECAST`, the less certain version of the same thing, said "apply". That inverts the
# project's honesty rule. `_verification_for` prevents it and a test pins it down.
_VERIFICATION = {
    "R1_RUST": "Revisar el envés de las hojas buscando polvillo anaranjado; confirmar incidencia antes de aplicar cobre.",
    "R2_AMERICAN_LEAF_SPOT": "Revisar hojas y hojarasca por lesiones de ojo de gallo; retirar material enfermo y evaluar sombra.",
    "R3_BERRY_BORER": "Instalar trampas etanol–metanol y hacer conteo físico de granos brocados (umbral de acción 2%).",
    "R4_CERCOSPORA": "Revisar manchas foliares y el estado nutricional; monitorear humedad y sombra.",
    "R5_LEAF_MINER": "Revisar minas en hojas del tercio superior; conservar parasitoides (avispitas).",
    "R6_COLD_PHOMA": "Revisar muerte descendente en ramas; evitar heridas y colocar cortavientos.",
    "R8_WATERLOGGING": "Inspeccionar raíces finas en el 10% de plantas buscando pudrición.",
    # Anticipated rules with no measured version to inherit from. All three are checked by eye
    # in a minute, and that check is exactly what turns a forecast into something actionable.
    "R7_N_LEACHING_FORECAST": "Antes de abonar, mirar el cielo y el parte: si la lluvia ya entró o se adelantó, fraccionar el abono en vez de aplicarlo entero.",
    "R12_VPD_FRUIT_FILL": "Revisar al mediodía si la sombra cubre bien y si las hojas nuevas se ven marchitas en las horas de más sol.",
    "R13_SPRAY_WINDOW": "Antes de preparar la mezcla, salir a comprobar el viento: si las hojas no se mueven o se mueven las ramas, no es la hora.",
    "R14_RAIN_EXPECTED": "Confirmar mirando el cielo y el parte del día antes de decidir el abonado, el secado o el trasplante.",
}


def _verification_for(rule_id: str) -> Optional[str]:
    """A rule's verification step, falling back to the base identifier.

    An anticipated rule is about the same problem as its measured version -- forecast rust is
    checked by looking at the underside of the same leaf -- so it inherits the verification instead
    of duplicating it. Without this fallback, `R1_RUST_FORECAST` came out with no check at all.
    """
    paso = _VERIFICATION.get(rule_id)
    if paso is None and rule_id.endswith("_FORECAST"):
        paso = _VERIFICATION.get(rule_id[: -len("_FORECAST")])
    return paso


def _nutrient_from_rule(rule_id: str) -> Optional[str]:
    """'NPK_N' -> 'N'; 'NPK_K_OUT_OF_DOMAIN' -> 'K'."""
    parts = rule_id.split("_")
    if len(parts) >= 2 and parts[0] == "NPK" and parts[1] in ("N", "P", "K"):
        return parts[1]
    return None


def current_band(reading: Reading, nut: str) -> Optional[str]:
    """Replicates `diagnose_npk`'s band classification for the current reading, including the
    stage exceptions (P in plantula, K in fructificacion and maduracion)."""
    val = getattr(reading, nut, None)
    if val is None or val == 0.0:
        return None
    dmin, dmax = SENSOR_DOMAIN[nut]
    if val < dmin or val > dmax:
        return "out_of_domain"
    band = _classify(val, NPK_BANDS[nut])
    if nut == "P" and reading.stage == "plantula" and val < 20.0 and band == "adequate":
        band = "low"
    if nut == "K" and reading.stage in ("fructificacion", "maduracion") and band == "moderate":
        band = "low_ripening"
    return band


def _action_from(prescription: Prescription) -> Optional[str]:
    if not prescription.product:
        return None
    action = f"Aplicar {prescription.product}"
    if prescription.timing:
        action += f" ({prescription.timing})"
    return action


# --------------------------------------------------------------------------- #
def _alert_to_rec(alert: Alert, current: Reading, copper_ledger=None, *,
                  forecast: bool = False, horizon_h: Optional[float] = None,
                  forecast_detail: Optional[str] = None) -> Recommendation:
    """Turns a v5 engine alert into a Recommendation enriched with a dose. When it comes from a
    forecast it is flagged as anticipated, with its horizon and detail."""
    nut = _nutrient_from_rule(alert.rule_id)
    band = current_band(current, nut) if nut else None

    presc = build_prescription(alert.rule_id, band, current.stage, copper_ledger=copper_ledger)

    refer = bool(alert.refer or presc.refer)
    provisional = bool(alert.provisional or presc.provisional)
    referral_note = alert.referral_note or None

    return Recommendation(
        rule_id=alert.rule_id,
        category=alert.category,
        rec_type=_type_for(alert.rule_id),
        severity=alert.severity,
        farmer_message=alert.farmer_message,
        agronomist_message=alert.agronomist_message,
        action=_action_from(presc),
        product=presc.product,
        dose=presc.dose,
        method=presc.method,
        timing=presc.timing,
        verification=_verification_for(alert.rule_id),
        refer=refer,
        referral_note=referral_note if refer else None,
        provisional=provisional,
        dose_disclaimer=presc.dose_disclaimer,
        forecast=forecast,
        horizon_h=horizon_h,
        forecast_detail=forecast_detail,
    )


def build_recommendations(
    window: List[Reading],
    *,
    stage: Optional[str] = None,
    altitude: Optional[float] = None,
    copper_ledger=None,
) -> List[Recommendation]:
    """
    Runs the v5 engine over the window and returns the recommendations ordered by severity
    (critical -> alert -> warning -> info). Passing `stage` or `altitude` pins them on every
    reading, which is what the Azure context map needs.
    """
    if not window:
        return []
    win = list(window)
    if stage is not None:
        # An unrecognised stage is NOT replaced by another: it becomes the sentinel, and the
        # engine degrades to its neutral branch and says so. Substituting "vegetativo" made a
        # section in grain filling receive "no need to irrigate", the opposite advice.
        stage = stage if stage in VALID_STAGES else STAGE_UNKNOWN
        for r in win:
            r.stage = stage
    if altitude is not None:
        for r in win:
            r.altitude = altitude

    current = win[-1]
    alerts = evaluate_window(win)
    recs = [_alert_to_rec(a, current, copper_ledger=copper_ledger) for a in alerts]
    # Stable order by severity; within one severity the engine's emission order is kept
    # (NPK -> environmental -> stage -> inferential -> water -> lab).
    recs.sort(key=lambda r: SEV_RANK.get(r.severity, 99))
    return recs


# --------------------------------------------------------------------------- #
# ANTICIPATED RECOMMENDATIONS (from the microclimate forecast)
# Coarse trigger for the anticipated fungal warning: high air humidity plus mild temperature.
_FUNGAL_FORECAST_RH = 85.0
_FUNGAL_FORECAST_TEMP = (18.0, 28.0)


def build_anticipated_recommendations(
    current: Reading,
    forecasts: List[Dict[str, float]],
) -> List[Recommendation]:
    """Raises anticipated alerts from the microclimate forecast at one or more horizons: thermal
    ones (R9 evaluated on a future reading, the most reliable forecast there is) and a coarse,
    probabilistic fungal warning. Only T and RH move to their forecast values; NPK, soil and rain
    keep their current ones.

    `forecasts` is a list of dicts {"horizon_h", "air_temp", "air_rh"}."""
    out: List[Recommendation] = []
    for fc in forecasts:
        h = fc.get("horizon_h")
        temp = fc.get("air_temp")
        rh = fc.get("air_rh")
        if temp is None or rh is None:
            continue
        # The forecast is rounded: more precision than 0.1 is not meaningful and clutters the text.
        temp = round(float(temp), 1)
        rh = round(float(rh), 1)
        future = replace(current, air_temp=temp, air_rh=rh)

        # Thermal (R9): anticipates heat and cold stress, and the cup quality lost to heat.
        for alert in env_alerts(future):
            out.append(_alert_to_rec(
                alert, future, forecast=True, horizon_h=h,
                forecast_detail=f"temperatura de aire ≈ {temp:.1f} °C prevista",
            ))

        # Fungal: a coarse early warning. Air humidity skill is modest, so it stays explicitly
        # probabilistic and carries a verification step -- it is not a diagnosis.
        low, high = _FUNGAL_FORECAST_TEMP
        if rh >= _FUNGAL_FORECAST_RH and low <= temp <= high:
            out.append(Recommendation(
                rule_id="FORECAST_FUNGAL_RISK",
                category="A",
                rec_type="disease",
                severity="warning",
                farmer_message=("En las próximas horas el clima podría volverse húmedo y templado, "
                                "condiciones que favorecen hongos como la roya o el ojo de gallo. "
                                "Conviene que empieces a revisar el cafetal."),
                agronomist_message=(f"Pronóstico a ~{h:g} h: HR≈{rh:.0f}% y T≈{temp:.1f} °C entran en el rango "
                                    "favorable a infección fúngica. Aviso probabilístico, no diagnóstico."),
                verification=("Revisar el envés de las hojas (roya) y buscar lesiones de ojo de gallo; "
                              "preparar el monitoreo antes de que se abra la ventana."),
                forecast=True,
                horizon_h=h,
                forecast_detail=f"HR≈{rh:.0f}% y T≈{temp:.1f} °C previstas",
            ))
    return out


def build_forecast_recommendations(alerts, current: Optional[Reading] = None,
                                   copper_ledger=None) -> List[Recommendation]:
    """Turns `forecast_rules` alerts into recommendations flagged as anticipated.

    Kept separate from `build_anticipated_recommendations` because the two anticipate different
    things: that one extends the sensor's own microclimate a few hours, this one reads the regional
    forecast, which is the only thing that reaches several days out. Both end up with
    `forecast=True` and their horizon, so the interface can say they are prediction, not
    measurement.
    """
    out: List[Recommendation] = []
    for a in alerts:
        ref = current or Reading(ts=datetime.now(timezone.utc))
        out.append(_alert_to_rec(
            a, ref, copper_ledger=copper_ledger,
            forecast=True,
            horizon_h=float(a.horizon_hours) if a.horizon_hours else None,
            forecast_detail=(f"Previsión a {a.horizon_hours} h" if a.horizon_hours else "Previsión"),
        ))
    out.sort(key=lambda r: SEV_RANK.get(r.severity, 99))
    return out


def combine_recommendations(
    immediate: List[Recommendation],
    anticipated: List[Recommendation],
) -> List[Recommendation]:
    """Merges immediate and anticipated recommendations and orders them by severity. Deduplicates:
    a condition already active now does not repeat as a forecast, and across horizons the earliest
    alert for each condition is the one kept."""
    active_now = {r.rule_id for r in immediate}
    earliest: Dict[str, Recommendation] = {}
    for r in anticipated:
        if r.rule_id in active_now:
            continue
        cur = earliest.get(r.rule_id)
        if cur is None or (r.horizon_h or float("inf")) < (cur.horizon_h or float("inf")):
            earliest[r.rule_id] = r
    merged = list(immediate) + list(earliest.values())
    merged.sort(key=lambda r: SEV_RANK.get(r.severity, 99))
    return merged


# --------------------------------------------------------------------------- #
# ORGANIC CERTIFICATION — safety net
def find_conventional(recommendations: Iterable[Recommendation]) -> List[str]:
    """Returns the blocked synthetic products found in any text field of the recommendations. It
    must always come back empty."""
    blocked_lower = {b.lower() for b in CONVENTIONAL_BLOCKED}
    hits: List[str] = []
    for rec in recommendations:
        blob = " ".join(
            str(v) for v in (
                rec.farmer_message, rec.agronomist_message, rec.action,
                rec.product, rec.dose, rec.method, rec.timing,
                rec.verification, rec.referral_note, rec.dose_disclaimer,
            ) if v
        ).lower()
        for term in blocked_lower:
            if term in blob:
                hits.append(term)
    return hits


def recommendations_signature(recommendations: Iterable[Recommendation]) -> str:
    """Qualitative signature of a set of recommendations, used to decide whether to resend to the
    backend. It ignores the numeric values in the text, which move every run because of the
    forecast, and rests on which alerts exist: it changes only when an alert appears or disappears,
    its severity changes, or a condition moves from immediate to anticipated. That way the grower
    gets a new recommendation only when the diagnosis actually changes."""
    parts = sorted(
        f"{r.rule_id}|{r.severity}|{'F' if r.forecast else 'N'}|"
        f"{r.horizon_h if r.horizon_h is not None else ''}"
        for r in recommendations
    )
    return ";".join(parts)


def is_lab_reminder(rec: Recommendation) -> bool:
    """A lab reminder (category B): static, emitted by the engine on every evaluation, but only
    sent on its own cadence (soil analysis ~24 months, liming ~12)."""
    return rec.rule_id.startswith("LAB_")


def select_for_sending(
    recommendations: List[Recommendation],
    *,
    last_signature: Optional[str],
    due_reminders: Optional[Iterable[str]] = None,
) -> tuple[Optional[List[Recommendation]], str]:
    """Decides what goes out this run, combining content deduplication with each reminder's
    cadence. `due_reminders` holds the `rule_id`s of reminders that are due today, each on its own
    cadence.

    The signature is computed over the dynamic diagnosis ONLY, with reminders excluded: since the
    engine emits them on every evaluation, including them would make entering or leaving a cadence
    count as a new diagnosis and cause spurious resends.

    Returns (to_send, signature). `to_send` is None when nothing should go out: the diagnosis has
    not changed and no reminder is due."""
    due = set(due_reminders or ())
    dynamic = [r for r in recommendations if not is_lab_reminder(r)]
    reminders_due = [r for r in recommendations if is_lab_reminder(r) and r.rule_id in due]
    signature = recommendations_signature(dynamic)
    if not recommendations:
        # Empty window: the "no readings" notice is sent only when it is new (ordinary
        # deduplication); the reminder cadence does not apply here.
        return ([] if signature != last_signature else None), signature
    diagnosis_changed = bool(dynamic) and signature != last_signature
    if not diagnosis_changed and not reminders_due:
        return None, signature
    return (dynamic + reminders_due), signature


# --------------------------------------------------------------------------- #
# BUILDING THE WINDOW FROM dict RECORDS — reused by Azure and by the tests
def _to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _parse_bool_rain(v: Any) -> bool:
    return str(v).strip() in ("1", "1.0", "true", "True")


def reading_from_record(
    record: Dict[str, Any],
    *,
    stage: str = "vegetativo",
    altitude: Optional[float] = None,
    ts: Optional[datetime] = None,
) -> Optional[Reading]:
    """
    Builds a Reading from a dict record. Accepts both the CSV/edge keys (`nitrogen_mg_kg`,
    `celcius_grade_temperature`, …) and the Azure backend ones (`nitrogen`,
    `celciusGradeTemperature`, …).
    """
    def g(*keys):
        for k in keys:
            if k in record and record[k] not in (None, ""):
                return record[k]
        return None

    if ts is None:
        # The backend exposes the data-records timestamp as `updatedAt`; several names are
        # accepted so the origin (CSV, edge or backend) does not matter.
        raw_ts = g("createdAt", "created_at", "timestamp", "recordDate", "updatedAt", "updatedDate")
        if raw_ts is not None:
            try:
                ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            except Exception:
                ts = None
    if ts is None:
        return None

    return Reading(
        ts=ts,
        N=_to_float(g("nitrogen_mg_kg", "nitrogen")),
        P=_to_float(g("phosphorus_mg_kg", "phosphorus")),
        K=_to_float(g("potassium_mg_kg", "potassium")),
        air_temp=_to_float(g("celcius_grade_temperature", "celciusGradeTemperature")),
        air_rh=_to_float(g("air_humidity_percent", "airHumidityPercent")),
        soil_moist=_to_float(g("soil_humidity_percent", "soilHumidityPercent")),
        rain=_parse_bool_rain(g("precipitation_detected", "precipitationDetected") or 0),
        stage=stage if stage in VALID_STAGES else STAGE_UNKNOWN,
        altitude=altitude,
    )


def build_window_from_records(
    records: Iterable[Dict[str, Any]],
    *,
    stage: str = "vegetativo",
    altitude: Optional[float] = None,
) -> List[Reading]:
    """Turns dict records into a window ordered by time."""
    win: List[Reading] = []
    for rec in records:
        r = reading_from_record(rec, stage=stage, altitude=altitude)
        if r is not None:
            win.append(r)
    win.sort(key=lambda x: x.ts)
    return win


# --------------------------------------------------------------------------- #
# Mapping to the backend payload.
#   The recommendations endpoint expects exactly:
#       { "recommendationDescription": <str>, "deviceHubId": <str> }
#   That contract is not changed: everything a recommendation carries (both voices, the action, the
#   verification and the forecast) is concatenated inside `recommendationDescription`.
_SEVERITY_ES = {"critical": "Crítico", "alert": "Alerta", "warning": "Aviso", "info": "Info"}
_TYPE_ES = {
    "nutrition": "nutrición", "irrigation": "riego", "disease": "enfermedad",
    "pest": "plaga", "thermal": "térmico", "lab": "laboratorio", "management": "manejo",
}
_METHOD_ES = {"soil": "suelo", "foliar": "foliar", "cultural": "cultural"}
_REFER_PREFIX = "⚠ Consultar técnico APROCASSI: "


def _render_one(rec: Recommendation) -> str:
    """Serialises one recommendation into the readable text block that goes inside the payload."""
    sev = _SEVERITY_ES.get(rec.severity, rec.severity.capitalize())
    tipo = _TYPE_ES.get(rec.rec_type, rec.rec_type)
    prefix = _REFER_PREFIX if rec.refer else ""
    lines = [f"[{sev} · {tipo}] {prefix}{rec.farmer_message}"]
    lines.append(f"Técnico: {rec.agronomist_message}")

    if rec.product:
        accion = rec.product
        if rec.dose:
            accion += f" — {rec.dose}"
        metodo = _METHOD_ES.get(rec.method, rec.method) if rec.method else None
        meta = [x for x in (metodo, rec.timing) if x]
        if meta:
            accion += f" ({', '.join(meta)})"
        lines.append(f"Acción: {accion}")

    if rec.verification:
        lines.append(f"Verificar: {rec.verification}")

    if rec.forecast:
        horizon = f"~{rec.horizon_h:g} h" if rec.horizon_h is not None else "las próximas horas"
        detalle = f" — {rec.forecast_detail}" if rec.forecast_detail else ""
        lines.append(f"Previsión: en {horizon}{detalle}")

    notes = []
    if rec.dose_disclaimer:
        notes.append(rec.dose_disclaimer)
    if rec.refer and rec.referral_note:
        notes.append(rec.referral_note)
    if notes:
        lines.append(f"Nota: {' '.join(notes)}")

    return "\n".join(lines)


def render_description(recommendations: List[Recommendation]) -> str:
    """Composes the `recommendationDescription` text by concatenating each recommendation (both
    voices, action, verification and forecast) separated by a blank line."""
    if not recommendations:
        # An empty window means there were no readings to evaluate. It is not an "all clear":
        # a real evaluation always emits at least the lab reminder.
        return ("No hay lecturas recientes del sensor para este dispositivo; "
                "no se pudo generar un diagnóstico. Verifique el envío de datos del sensor.")
    return "\n\n".join(_render_one(rec) for rec in recommendations)


# --------------------------------------------------------------------------- #
# JSON serialisation, so the frontend can render cards and filter by role. It travels inside the
# same `recommendationDescription` field (still a string), leaving the backend contract alone. The
# frontend does a JSON.parse and shows:
#   - Manager: everything (technical voice, product/dose/method, citations, verification).
#   - Farmer: only what is actionable, in the plain voice, with no jargon and no citations.
PAYLOAD_VERSION = 1
_EMPTY_MESSAGE = ("No hay lecturas recientes del sensor para este dispositivo; "
                  "no se pudo generar un diagnóstico. Verifique el envío de datos del sensor.")

_NUTRIENT_ES = {"N": "Nitrógeno", "P": "Fósforo", "K": "Potasio"}
_SUBJECT = {
    "R1_RUST": "Roya", "R2_AMERICAN_LEAF_SPOT": "Ojo de gallo", "R3_BERRY_BORER": "Broca",
    "R4_CERCOSPORA": "Cercospora", "R5_LEAF_MINER": "Minador", "R6_COLD_PHOMA": "Frío / Phoma",
    "R7_N_LEACHING": "Lixiviación de nitrógeno", "R8_WATERLOGGING": "Encharcamiento",
    "R9_COLD": "Frío", "R9_ACUTE_HEAT": "Calor", "R9_HEAT_QUALITY": "Calor",
    "R10_DEFICIT_CRITICAL": "Riego", "R10_DEFICIT_INDUCTION": "Riego", "R10_DEFICIT_MILD": "Riego",
    "R11_FLOWERING_EXPECTED": "Floración", "STAGE_FLOWERING_BCA": "Floración (boro y calcio)",
    "STAGE_HARVEST_MAINTENANCE": "Poscosecha",
    "R3_BORER_SANITATION": "Broca (saneamiento RE-RE)",
    "LAB_SOIL_ANALYSIS": "Análisis de suelo", "LAB_LIMING": "Encalado",
    "FORECAST_FUNGAL_RISK": "Riesgo fúngico", "CU_CAP": "Cobre", "CU_OK": "Cobre",
}


def _subject_for(rule_id: str) -> Optional[str]:
    nut = _nutrient_from_rule(rule_id)
    if nut:
        return _NUTRIENT_ES.get(nut)
    return _SUBJECT.get(rule_id)


# Static reminders (lab analysis, stage notes) do not change with the reading, so the frontend
# shows them in their own section rather than as an alert.
def _kind_for(rule_id: str) -> str:
    if rule_id.startswith("LAB_") or rule_id.startswith("STAGE_"):
        return "reminder"
    return "recommendation"


# Actionability: what should the user do with this card? The chip summarises the expected action
# so that "you can apply this now" never appears where there is nothing to apply (an adequate
# level) or where the only lever is long term (shade).
_ACTIONABILITY_ES = {
    "direct": "Puedes aplicarlo ahora",   # there is a concrete input to apply now
    "verify": "Verifica primero",         # check in the field before acting
    "consult": "Consulta al técnico",     # needs expert judgement or a laboratory
    "structural": "Medida de fondo",      # long-term lever (shade, agroforestry)
    "monitor": "Observa y espera",        # watch or wait; no immediate input
    "none": "Todo en orden",              # adequate level: nothing to apply
}

# Rules whose only lever is structural or long term, not resolvable the same day.
_STRUCTURAL_RULES = {"R9_ACUTE_HEAT"}
# Watch-and-wait rules: the instruction is to observe or hold, applying nothing now.
# `R13_SPRAY_WINDOW_NONE` says exactly that -- there is no good hour to spray today, leave it for
# tomorrow -- so it carries no verification step: there is nothing to check, there is something not
# to do.
_MONITOR_RULES = {"R9_COLD", "R9_HEAT_QUALITY", "R13_SPRAY_WINDOW_NONE"}


def _actionability_for(rec: Recommendation) -> str:
    if rec.refer:
        return "consult"
    if rec.verification:
        return "verify"
    if rec.rule_id in _STRUCTURAL_RULES:
        return "structural"
    if rec.rule_id in _MONITOR_RULES:
        return "monitor"
    if rec.product:
        return "direct"     # there is a concrete product and dose to apply now
    # No product does NOT mean nothing to do: many measures are cultural and are real actions
    # (collecting leftover berries, mulching, opening drainage, splitting the fertiliser). Only a
    # nutritional diagnosis at an adequate or high level is a genuine all clear.
    if rec.rule_id.startswith("NPK_") and rec.severity == "info":
        return "none"
    return "direct"


# "Saving" recommendations: the ones that say NOT to spend a resource (do not fertilise, do not
# irrigate).
_SAVING_KEYWORDS = (
    "no apliques", "no compres", "no aplicar", "no es necesario regar",
    "no regar", "no apliques nada", "suspender",
)


def _is_saving(rec: Recommendation) -> bool:
    blob = f"{rec.farmer_message} {rec.agronomist_message}".lower()
    return any(k in blob for k in _SAVING_KEYWORDS)


def recommendation_to_dict(rec: Recommendation) -> Dict[str, Any]:
    """Represents a recommendation as a structured dict for the payload's JSON."""
    actionability = _actionability_for(rec)
    item: Dict[str, Any] = {
        "rule_id": rec.rule_id,
        "severity": rec.severity,
        "severity_label": _SEVERITY_ES.get(rec.severity, rec.severity.capitalize()),
        "type": rec.rec_type,
        "type_label": _TYPE_ES.get(rec.rec_type, rec.rec_type),
        "subject": _subject_for(rec.rule_id),
        "kind": _kind_for(rec.rule_id),            # "reminder" | "recommendation"
        "category": rec.category,                  # A (real time) | B (lab) | C (management)
        "actionability": actionability,
        "actionability_label": _ACTIONABILITY_ES[actionability],
        "saving": _is_saving(rec),                 # true = ahorra recursos (no aplicar/regar)
        "farmer_message": rec.farmer_message,        # voz simple (Farmer)
        "agronomist_message": rec.agronomist_message,  # voz técnica (Manager)
        "provisional": rec.provisional,
        "refer": rec.refer,
        "referral_note": rec.referral_note if rec.refer else None,
        "verification": rec.verification,
        "forecast": None,
        "action": None,
    }
    if rec.forecast:
        item["forecast"] = {"horizon_h": rec.horizon_h, "detail": rec.forecast_detail}
    if rec.product:
        item["action"] = {
            "product": rec.product,
            "dose": rec.dose,
            "method": rec.method,
            "method_label": _METHOD_ES.get(rec.method, rec.method) if rec.method else None,
            "timing": rec.timing,
            "disclaimer": rec.dose_disclaimer,
        }
    return item


def build_payload_json(recommendations: List[Recommendation]) -> str:
    """Serialises the recommendations into the compact JSON the frontend consumes."""
    if not recommendations:
        payload = {"v": PAYLOAD_VERSION, "summary": {}, "items": [], "empty_message": _EMPTY_MESSAGE}
        return json.dumps(payload, ensure_ascii=False)
    summary: Dict[str, int] = {}
    for r in recommendations:
        summary[r.severity] = summary.get(r.severity, 0) + 1
    payload = {
        "v": PAYLOAD_VERSION,
        "summary": summary,
        "items": [recommendation_to_dict(r) for r in recommendations],
    }
    return json.dumps(payload, ensure_ascii=False)


def map_to_azure_payload(
    recommendations: List[Recommendation],
    device_hub_id: str,
) -> Dict[str, str]:
    """Maps the Recommendation list to the payload the backend already expects: the structured
    content (JSON) travels inside the `recommendationDescription` string."""
    return {
        "recommendationDescription": build_payload_json(recommendations),
        "deviceHubId": device_hub_id,
    }
