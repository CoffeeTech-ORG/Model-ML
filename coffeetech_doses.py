# -*- coding: utf-8 -*-
"""
CoffeeTech — dose / organic prescription layer.
El Milagro plot (APROCASSI · USDA-Organic + Fairtrade).

Turns each agronomic condition the engine detects into a concrete prescription, under this site's
constraints:

  * Organic-certified products only. Nothing in `coffeetech_rules_v5.CONVENTIONAL_BLOCKED` can
    appear here.
  * Every dose is a literature starting point and carries `DOSE_DISCLAIMER`.
  * Nitrogen is advisory: no absolute N dose is ever issued from the sensor. The only dosing tied
    to N is of the organic amendment itself (compost/bocashi in kg per plant), not of the nutrient.
  * Outside the sensor's calibrated domain there is no product and no dose; the case is referred to
    a laboratory.
  * Copper is subject to the `CopperLedger` accumulator (4 kg Cu/ha/year cap). Over the cap, the
    copper recommendation is replaced by cultural management.

Standard library only, so it can be tested without FastAPI or pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict

# Fixed disclaimer attached to every recommendation that carries a dose.
DOSE_DISCLAIMER = (
    "Dosis de referencia basada en literatura; ajústela al análisis de suelo/foliar "
    "y confírmela con su técnico de APROCASSI."
)


@dataclass
class DoseOption:
    """One organic prescription option (a literature starting point)."""
    product: str
    dose: Optional[str] = None        # None when the source sets no specific dose
    method: Optional[str] = None      # "soil" | "foliar" | "cultural"
    timing: Optional[str] = None      # when to apply it
    note: Optional[str] = None
    # N: organic amendment whose dose is of the amendment, NOT an absolute N dose.
    provisional: bool = False
    # Flags the copper options so the layer consults the CopperLedger.
    copper: bool = False


# --------------------------------------------------------------------------- #
# Organic prescription table. Each key is an already-resolved agronomic condition pointing at one
# or more product/dose options (literature starting points).
ORGANIC_PRESCRIPTIONS: Dict[str, List[DoseOption]] = {
    # --- NITROGEN (always ADVISORY; no absolute N dose) ---
    "N_low": [
        DoseOption(
            product="Compost o bocashi",
            dose="2–5 kg/planta/año",
            method="soil",
            timing="con las lluvias",
            note="Sin dosis puntual de N; dimensionar según el análisis foliar.",
            provisional=True,
        ),
        DoseOption(
            product="Guano de isla",
            dose="según análisis",
            method="soil",
            timing="con las lluvias",
            note="Aprovechar los eventos de lluvia para incorporarlo.",
            provisional=True,
        ),
    ],

    # --- PHOSPHORUS (low/severe) ---
    "P_low": [
        DoseOption(
            product="Roca fosfórica reactiva + compost",
            dose="según análisis (aplicación en hoyo/almácigo en establecimiento)",
            method="soil",
            timing="siembra / floración",
            note="Suelos ácidos con fijación de P; favorecer micorrizas.",
        ),
    ],

    # --- POTASSIUM (low / grain filling) ---
    "K_low": [
        DoseOption(
            product="Sulfato de potasio (SOP) o sul-po-mag",
            dose="según el análisis de suelo",
            method="soil",
            timing="antes/durante el llenado de grano",
            note="Pico de demanda de K en llenado/maduración.",
        ),
    ],

    # --- LIMING / Ca / Mg (category B, needs a soil analysis) ---
    "liming": [
        DoseOption(
            product="Cal dolomítica o calcítica",
            dose="según el análisis de suelo (arranque ~30 g/planta)",
            method="soil",
            timing="~30 días antes de abonar",
            note="Si saturación de Al > 30%; pH objetivo 5.0–5.5. Aplicar en banda.",
        ),
        DoseOption(
            product="Yeso agrícola",
            dose="según análisis",
            method="soil",
            timing=None,
            note="Aporta Ca + S sin cambiar el pH.",
        ),
    ],

    # --- COFFEE LEAF RUST / AMERICAN LEAF SPOT (copper control, subject to the cap) ---
    "copper": [
        DoseOption(
            product="Cobre autorizado (oxicloruro/hidróxido de cobre)",
            dose="dentro del tope 4 kg Cu/ha/año",
            method="foliar",
            timing="solo si el riesgo escapa al manejo cultural",
            note="Priorizar poda, sombra y variedad resistente antes del cobre.",
            copper=True,
        ),
    ],

    # --- COFFEE BERRY BORER (biological control) ---
    "borer": [
        DoseOption(
            product="Beauveria bassiana",
            dose="~2 kg/ha (≈1×10⁷ conidias/mL), en días húmedos/nublados",
            method="foliar",
            timing="tras confirmar infestación > 2%",
            note=("Trampas etanol–metanol 20–34/ha; conteo físico de granos "
                  "brocados obligatorio antes de aplicar."),
        ),
    ],

    # --- CERCOSPORA / iron spot: a disease of WEAKNESS, so nutrition and shade rather than a
    #     fungicide up front. Deliberately kept away from copper. ---
    "cercospora_management": [
        DoseOption(
            product="Corrección nutricional orgánica (N y/o K) + manejo de sombra",
            dose=None,
            method="cultural",
            timing="al detectar debilidad + humedad sostenida",
            note=("Cercospora coffeicola es enfermedad de debilidad; primero nutrir "
                  "(abono orgánico) y regular sombra 35–65%. El cobre no es la primera opción."),
        ),
    ],

    # --- FLOWERING: B + Ca. The sensor measures neither, so the dose goes to the technician ---
    "flowering_bca": [
        DoseOption(
            product="Fuente orgánica de boro + calcio",
            dose="según análisis foliar (dosis a definir por el técnico)",
            method="foliar",
            timing="en floración/cuajado",
            note="B para el tubo polínico y Ca para la pared celular; evita caída de flor.",
        ),
    ],

    # --- Replacement once the copper cap is exceeded (cultural management) ---
    "copper_cultural_management": [
        DoseOption(
            product="Manejo cultural (poda sanitaria, regulación de sombra, variedad resistente)",
            dose=None,
            method="cultural",
            timing="de inmediato",
            note="Tope de cobre alcanzado: no aplicar más cobre este año.",
        ),
    ],
}


# --------------------------------------------------------------------------- #
# RESOLVING the prescription key from (rule_id, band, crop stage).
def prescription_key(rule_id: str, band: Optional[str], stage: Optional[str]) -> Optional[str]:
    """Maps a v5 engine alert to an `ORGANIC_PRESCRIPTIONS` key, or None when the condition does
    not warrant dosing (adequate or high level, or an informational alert)."""
    if rule_id.endswith("_OUT_OF_DOMAIN"):
        return None  # outside the domain -> laboratory, no product and no dose
    if rule_id == "NPK_N":
        return "N_low" if band in ("severe", "low") else None
    if rule_id == "NPK_P":
        return "P_low" if band in ("severe", "low") else None
    if rule_id == "NPK_K":
        # 'moderate' (planning) and 'low_ripening' (zero tolerance) do dose. 'high', 'excess'
        # and 'adequate' do not: they recommend applying no potassium.
        #
        # 'severe' is absent on purpose: the engine cannot produce it for K. Its band asked for
        # <78 mg/kg and the sensor's domain starts at 90, so it was retired from `NPK_BANDS`.
        return "K_low" if band in ("moderate", "low_ripening") else None
    if rule_id in ("R1_RUST", "R2_AMERICAN_LEAF_SPOT"):
        return "copper"
    if rule_id == "R4_CERCOSPORA":
        return "cercospora_management"
    if rule_id == "R3_BERRY_BORER":
        return "borer"
    if rule_id == "LAB_LIMING":
        # Lime rides on the LIMING reminder; the soil-analysis one prescribes nothing, because
        # the analysis has to exist first.
        return "liming"
    if rule_id == "STAGE_FLOWERING_BCA":
        return "flowering_bca"
    return None


@dataclass
class Prescription:
    """What the dose layer produces for one recommendation."""
    product: Optional[str] = None
    dose: Optional[str] = None
    method: Optional[str] = None
    timing: Optional[str] = None
    dose_disclaimer: Optional[str] = None
    provisional: bool = False
    refer: bool = False
    extra_note: Optional[str] = None
    options: List[DoseOption] = field(default_factory=list)


def build_prescription(
    rule_id: str,
    band: Optional[str],
    stage: Optional[str],
    *,
    copper_ledger=None,
    planned_cu_kg_ha: float = 2.0,
) -> Prescription:
    """
    Fills in the prescription (product, dose, method, timing, disclaimer) for an alert.

      - N never gets an absolute dose; `provisional=True`.
      - Outside the domain there is no product and no dose; `refer=True`.
      - Copper consults `copper_ledger` and falls back to cultural management once the cap is
        exceeded. `planned_cu_kg_ha` is the hypothetical copper application being evaluated.
    """
    # Outside the calibrated domain: no dose, referred to a laboratory.
    if rule_id.endswith("_OUT_OF_DOMAIN"):
        return Prescription(refer=True)

    key = prescription_key(rule_id, band, stage)
    if key is None:
        return Prescription()

    # Copper: consult the accumulator before recommending.
    if key == "copper" and copper_ledger is not None:
        remaining = copper_ledger.remaining()
        # With no room left for the planned application, fall back to cultural management.
        if remaining <= 0 or planned_cu_kg_ha > remaining:
            key = "copper_cultural_management"

    options = list(ORGANIC_PRESCRIPTIONS.get(key, []))
    if not options:
        return Prescription()

    main = options[0]
    provisional = any(o.provisional for o in options)
    has_dose = main.dose is not None
    refer = key in ("liming", "flowering_bca", "copper_cultural_management")

    return Prescription(
        product=main.product,
        dose=main.dose,
        method=main.method,
        timing=main.timing,
        dose_disclaimer=DOSE_DISCLAIMER if has_dose else None,
        provisional=provisional,
        refer=refer,
        options=options,
    )
