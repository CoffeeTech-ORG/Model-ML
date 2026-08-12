# -*- coding: utf-8 -*-
"""
The net that makes translating the comments to English safe.

Why this exists
---------------
Rewriting 4400 lines of comment by hand has three ways of going wrong that the normal tests do not
see, because none of them changes measured behaviour:

  1. Translating a `farmer_message` believing it a comment. The farmer is from San Ignacio.
  2. Leaving a file half done, with half of it in each language.
  3. Moving the docstring summary onto the line of the quotes, which leaves twelve scripts without
     a `--help` and nothing goes red.

And two more that break in silence: turning `52,4 %` into `52.4 %` while touching the paragraph
around it —which severs the `[medido:]` mark defending that figure— and dropping a mark entirely.

The `TRADUCIDOS` list grows by one file per commit. That is what makes stopping halfway leave a
state you can name instead of a limbo.
"""
from __future__ import annotations

import ast
import io
import os
import re
import tokenize

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Files whose comments and docstrings are already in English. A translation commit adds its own
#: here, in that same commit, or this check says nothing about it.
TRADUCIDOS: tuple = (
    "coffeetech_rules_v5.py",
    "coffeetech_doses.py",
    "coffeetech_recommendations.py",
    "coffeetech_forecast.py",
    "coffeetech_weather.py",
    "coffeetech_state.py",
    "main.py",
    "tests/conftest.py",
    "tests/test_acceptance.py",
    "tests/test_forecast_rules.py",
    "tests/test_reminder_cadence.py",
    "tests/test_etapa.py",
    "tests/test_reference_ranges.py",
    "tests/test_altitud.py",
    "tests/test_agregacion.py",
    "tests/test_npk_bandas.py",
    "tests/test_weather.py",
    "tests/test_forecast.py",
    "tests/test_comment_policy.py",
)

#: Proper nouns and terms kept in Spanish inside an English comment: values that arrive from the
#: backend, place names, or the name of an institution.
PERMITIDOS = {
    "plantula", "vegetativo", "floracion", "fructificacion", "maduracion", "cosecha",
    "no_registrada", "san", "ignacio", "cajamarca", "milagro", "aprocassi", "cenicafe",
    "cenicafé", "peru", "perú", "colombia", "brasil", "inia", "midagri", "senamhi",
    "arabica", "coffea", "hemileia", "vastatrix", "phoma", "cercospora", "mycena",
    "citricolor", "hypothenemus", "hampei", "leucoptera", "bandolas", "broca", "roya",
    "almacigo", "almácigo",
}

#: Frequent Spanish words. One is enough to give away an untranslated comment; they are picked for
#: being impossible in technical English.
DELATORAS = re.compile(
    r"\b(que|para|porque|cuando|pero|este|esta|esto|desde|hasta|entre|sobre|donde|"
    r"según|aunque|además|también|sólo|solo|cada|todos|todas|más|menos|hace|"
    r"tiene|puede|debe|dice|sale|vive|queda|falta|sin|con|del|las|los|una|uno)\b",
    re.IGNORECASE)


def _fuente(rel: str) -> str:
    return io.open(os.path.join(RAIZ, rel), encoding="utf-8").read()


def _ficheros_py() -> list:
    fuera = {"node_modules", ".git", "__pycache__", ".venv", "venv"}
    out = []
    for base, dirs, files in os.walk(RAIZ):
        dirs[:] = [d for d in dirs if d not in fuera]
        for f in files:
            if f.endswith(".py"):
                out.append(os.path.relpath(os.path.join(base, f), RAIZ).replace("\\", "/"))
    return sorted(out)


def _texto_comentado(fuente: str) -> list:
    """Comments and docstrings, with their line number. Nothing else: the code is not looked at.

    Comments come out through `tokenize`, not by looking for lines that start with `#`. The
    difference matters in `scripts/informe_motor.py`: its strings carry the report's markdown
    headings —`## F · El modelo de predicción`— and those are PUBLISHED PROSE in Spanish, not
    comments. In the naive form, this check asked for the delivered document to be translated.
    """
    trozos = []
    for tok in tokenize.generate_tokens(io.StringIO(fuente).readline):
        if tok.type == tokenize.COMMENT:
            trozos.append((tok.start[0], tok.string.lstrip("#: ").strip()))
    arbol = ast.parse(fuente)
    for n in ast.walk(arbol):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d = ast.get_docstring(n, clean=False)
            if d:
                trozos.append((getattr(n, "lineno", 1), d))
    return trozos


def _limpiar(t: str) -> str:
    """Strips what may legitimately be in Spanish inside an English comment."""
    t = re.sub(r"`[^`]*`", " ", t)                      # identifiers and paths in backticks
    t = re.sub(r"«[^»]*»", " ", t)                      # text quoted from the system itself
    t = re.sub(r'"[^"]*"', " ", t)                      # quoted strings
    t = re.sub(r"\b\w+\.\w+[\w.]*", " ", t)             # dotted paths: ENV.temp_cold
    t = re.sub(r"--[\w-]+", " ", t)                     # CLI options: `--desde`, `--stage`
    t = re.sub(r"\[medido:[^\]]*\]", " ", t)            # the measurement marks
    return " ".join(p for p in t.split() if p.strip(".,;:()").lower() not in PERMITIDOS)


# --------------------------------------------------------------------------- #
# 1 · A file declared translated cannot carry Spanish.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rel", TRADUCIDOS or ["(ninguno todavía)"])
def test_los_ficheros_traducidos_no_llevan_espanol(rel):
    if not TRADUCIDOS:
        pytest.skip("Todavía no ha cruzado ningún fichero.")
    restos = []
    for ln, t in _texto_comentado(_fuente(rel)):
        m = DELATORAS.search(_limpiar(t))
        if m:
            restos.append(f"{rel}:{ln} «{m.group(0)}» en: {t[:70]}")
    assert not restos, (
        "Este fichero está en TRADUCIDOS pero le quedan comentarios en español:\n  "
        + "\n  ".join(restos[:12]))


# --------------------------------------------------------------------------- #
# 2 · The reverse guard: what a person reads stays in Spanish.
#
# The one that matters most. It fails BEFORE the 34 acceptance assertions and with a better message:
# if someone translates a `farmer_message`, here you read why it is wrong, and there you only see a
# string that does not match.
# --------------------------------------------------------------------------- #
CAMPOS_DE_PERSONA = ("farmer_message", "agronomist_message", "referral_note", "note")


def _mensajes_del_motor() -> list:
    arbol = ast.parse(_fuente("coffeetech_rules_v5.py"))
    out = []
    for n in ast.walk(arbol):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_A"):
            continue
        for k in n.keywords:
            if k.arg not in CAMPOS_DE_PERSONA:
                continue
            partes = []
            for s in ast.walk(k.value):
                if isinstance(s, ast.Constant) and isinstance(s.value, str):
                    partes.append(s.value)
            if partes:
                out.append((n.lineno, k.arg, " ".join(partes)))
    return out


#: English words that never turn up in a Spanish message. It looks for ENGLISH rather than for the
#: absence of Spanish: half a dozen of these messages are telegraphic —«K=120 mg/kg (adecuado
#: 115-156). Mantener plan base.»— and carry too little signal to say which language they are in.
#: Asking whether there is English does have an answer, and it is the one that matters: that nobody
#: has translated them.
ANGLOSAJONAS = re.compile(
    r"\b(the|and|with|your|you|for|this|that|these|those|from|when|where|there|their|"
    r"will|would|should|must|does|is|are|was|were|has|have|been|because|however|about|"
    r"apply|check|keep|watch|soil|leaf|rain|shade|crop|plant)\b", re.IGNORECASE)


def test_los_mensajes_a_personas_siguen_en_espanol():
    mensajes = _mensajes_del_motor()
    assert len(mensajes) > 40, "No se están encontrando los mensajes; el extractor está roto."
    sospechosos = [f"línea {ln} ({campo}): «{m.group(0)}» en {txt[:60]}"
                   for ln, campo, txt in mensajes
                   for m in [ANGLOSAJONAS.search(txt)] if m]
    assert not sospechosos, (
        "Estos mensajes llevan inglés, y los lee un agricultor de San Ignacio o el técnico que lo "
        "acompaña. Un comentario se traduce; esto no:\n  " + "\n  ".join(sospechosos[:10]))


# --------------------------------------------------------------------------- #
# 3 · The exact set of [medido:] marks is frozen.
#
# `test_contrafactuales.py` requires every mark to have its measurement, and at least ten of them.
# That does not catch one disappearing while eleven remain. Here the set is frozen: dropping a mark
# is a decision, not a side effect of rewriting the paragraph around it.
# --------------------------------------------------------------------------- #
MARCAS = re.compile(r"\[medido:([\w.]+)\]")

MARCAS_ESPERADAS = frozenset({
    "cf.lluvia_sin_puerta_seca", "cf.lluvia_sin_puerta_seca.servido",
    "cf.phoma_24h", "cf.phoma_24h.servido",
    "cf.roya_anticipada_6h", "cf.roya_anticipada_6h.servido",
    "clima.horas_con_lluvia_72h_sobre_5mm_pct",
    "clima.horas_en_banda_ojo_gallo_pct",
    "clima.horas_en_banda_vuelo_broca_pct",
    "clima.max_media_diaria_c",
    "clima.max_media_movil_96h_c",
    "clima.vpd.desfase_max_kpa", "clima.vpd.desfase_min_kpa",
    "clima.vpd.horas_feb_jul_sobre_umbral_pct", "clima.vpd.media_24h_anual_kpa",
})


def test_las_marcas_medido_no_desaparecen_al_reescribir():
    presentes = frozenset(MARCAS.findall(_fuente("coffeetech_rules_v5.py")))
    faltan = MARCAS_ESPERADAS - presentes
    assert not faltan, (
        f"Estas marcas de medición han desaparecido del motor: {sorted(faltan)}. Cada una defiende "
        f"una cifra publicada; si la cifra ya no está, hay que retirarla también de "
        f"docs/contrafactuales.json y de esta lista, a propósito.")


def test_las_cifras_medidas_conservan_la_coma_decimal():
    """`52,4 %` rewritten as `52.4 %` breaks the link with its measurement without going red."""
    fuente = _fuente("coffeetech_rules_v5.py")
    malas = []
    for linea in fuente.split("\n"):
        if "[medido:" not in linea:
            continue
        for cifra in re.findall(r"\d+\.\d+\s*(?:%|°C|kPa|mm|h)", linea):
            malas.append(f"{cifra} en: {linea.strip()[:80]}")
    assert not malas, (
        "Cifra con punto decimal en una línea que lleva marca de medición. El detector de "
        "`tests/test_contrafactuales.py` exige coma:\n  " + "\n  ".join(malas[:8]))


# --------------------------------------------------------------------------- #
# 5 · The tics that give away machine-written text.
# --------------------------------------------------------------------------- #
TICS = (
    "this function is responsible for", "note that", "in other words",
    "it's worth noting", "it is worth noting", "leverage", "robust", "comprehensive",
    "delve", "seamless", "furthermore", "moreover",
)


def test_los_comentarios_traducidos_no_suenan_a_maquina():
    if not TRADUCIDOS:
        pytest.skip("Todavía no ha cruzado ningún fichero.")
    encontrados = []
    for rel in TRADUCIDOS:
        for ln, t in _texto_comentado(_fuente(rel)):
            bajo = t.lower()
            for tic in TICS:
                if tic in bajo:
                    encontrados.append(f"{rel}:{ln} «{tic}»")
    assert not encontrados, (
        "Estas expresiones están prohibidas por docs/INFORME_CODIFICACION.md: un desarrollador "
        "explica qué hace el código, no lo resume con muletillas:\n  "
        + "\n  ".join(encontrados[:12]))


# --------------------------------------------------------------------------- #
# 6 · The comment explains the code, not how the project reached it.
#
# A threshold's comment has to survive being read by someone who never saw this repository change.
# The measurement stays, the sensitivity stays; the account of what the value was before, who
# spotted it and in which round does not -- that is `docs/DECISIONES_DESCARTADAS.md`.
#
# The list only holds formulas with no legitimate technical reading. `used to` alone is not one:
# "the window used to estimate the offset" is describing what the window is for.
# --------------------------------------------------------------------------- #
RETROSPECTIVA = (
    "used to be", "used to carry", "used to say", "used to fire", "used to live", "used to come",
    "used to publish", "used to keep", "used to call", "there used to be", "it used to",
    "the previous version", "an external review", "the git history", "in git history",
    "throwaway script", "throwaway run", "no longer exists", "for months", "(new rule)",
    "this project has", "was withdrawn", "had been marked",
)


def test_los_comentarios_no_cuentan_la_historia_del_proyecto():
    if not TRADUCIDOS:
        pytest.skip("Todavía no ha cruzado ningún fichero.")
    encontrados = []
    for rel in TRADUCIDOS:
        for ln, t in _texto_comentado(_fuente(rel)):
            bajo = " ".join(t.lower().split())
            for f in RETROSPECTIVA:
                if f in bajo:
                    encontrados.append(f"{rel}:{ln} «{f}» en: {t[:60]}")
    assert not encontrados, (
        "Esto es relato retrospectivo, no documentación del código. Decir qué hace y qué pasa si "
        "el número cambia; la crónica va en docs/DECISIONES_DESCARTADAS.md:\n  "
        + "\n  ".join(encontrados[:12]))
