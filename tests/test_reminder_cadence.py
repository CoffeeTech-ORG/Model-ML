# -*- coding: utf-8 -*-
"""
Tests for the laboratory reminder's cadence.

The engine emits the reminder (LAB_*) on every evaluation; these tests check the sending layer only
includes it when it is due -- a cadence with state persisted to disk -- and that content
deduplication does not break as the reminder enters and leaves the payload.

Run with:  python -m pytest tests/test_reminder_cadence.py -v
"""
from datetime import datetime, timedelta, timezone

from coffeetech_rules_v5 import Reading
from coffeetech_recommendations import (
    build_recommendations,
    is_lab_reminder,
    recommendations_signature,
    select_for_sending,
)
from coffeetech_state import ReminderState


def recs_with_reminder():
    window = [Reading(ts=datetime(2026, 2, 1, tzinfo=timezone.utc),
                      N=34.0, P=12.0, K=130.0, air_temp=20.0, air_rh=70.0,
                      soil_moist=60.0, rain=False, stage="vegetativo", altitude=1450.0)]
    recs = build_recommendations(window, stage="vegetativo", altitude=1450.0)
    assert any(is_lab_reminder(r) for r in recs)   # the engine always emits it
    return recs


# --------------------------------------------------------------------------- #
# select_for_sending: what travels on each run
def test_primer_envio_incluye_recordatorio_cuando_toca():
    recs = recs_with_reminder()
    selected, signature = select_for_sending(recs, last_signature=None, due_reminders={"LAB_SOIL_ANALYSIS", "LAB_LIMING"})
    assert selected is not None
    assert any(is_lab_reminder(r) for r in selected)
    assert signature  # signature of the dynamic diagnosis, not empty


def test_diagnostico_nuevo_sin_recordatorio_si_no_toca():
    # The diagnosis changed but the reminder went out recently: the diagnosis travels alone,
    # without the reminder.
    recs = recs_with_reminder()
    selected, _ = select_for_sending(recs, last_signature="otro-diagnostico", due_reminders=set())
    assert selected is not None
    assert not any(is_lab_reminder(r) for r in selected)
    assert "LAB_PH_AL_KAMPRATH" not in [r.rule_id for r in selected]
    # The rest of the diagnosis is untouched.
    assert len(selected) == len(recs) - 2   # both LAB_* reminders are filtered out


def test_sin_cambio_y_sin_cadencia_no_se_envia():
    recs = recs_with_reminder()
    _, signature = select_for_sending(recs, last_signature=None, due_reminders=set())
    selected, _ = select_for_sending(recs, last_signature=signature, due_reminders=set())
    assert selected is None


def test_cadencia_cumplida_fuerza_envio_aunque_no_cambie_el_diagnostico():
    # Diagnosis identical to the last one sent, but the reminder is due again: the full
    # diagnosis goes out with the reminder included.
    recs = recs_with_reminder()
    _, signature = select_for_sending(recs, last_signature=None, due_reminders=set())
    selected, sig2 = select_for_sending(recs, last_signature=signature, due_reminders={"LAB_SOIL_ANALYSIS", "LAB_LIMING"})
    assert selected is not None
    assert any(is_lab_reminder(r) for r in selected)
    assert sig2 == signature  # the signature (dynamic diagnosis) did not change


def test_firma_estable_con_o_sin_recordatorio():
    # Filtering the reminder must not count as a new diagnosis: the signature is computed over
    # the dynamic diagnosis, so it is the same in both cases.
    recs = recs_with_reminder()
    dynamic = [r for r in recs if not is_lab_reminder(r)]
    _, sig_full = select_for_sending(recs, last_signature=None, due_reminders={"LAB_SOIL_ANALYSIS", "LAB_LIMING"})
    _, sig_dyn = select_for_sending(recs, last_signature=None, due_reminders=set())
    assert sig_full == sig_dyn == recommendations_signature(dynamic)


def test_ventana_vacia_avisa_una_vez_y_no_repite_por_cadencia():
    # With no readings the "no data" notice is sent only when new; the reminder's cadence must
    # not cause it to be resent on every run.
    selected, signature = select_for_sending([], last_signature=None, due_reminders={"LAB_SOIL_ANALYSIS", "LAB_LIMING"})
    assert selected == []          # first time: new, so the notice goes out
    selected2, _ = select_for_sending([], last_signature=signature, due_reminders={"LAB_SOIL_ANALYSIS", "LAB_LIMING"})
    assert selected2 is None       # already notified: not repeated even with lab_due True


def test_solo_recordatorio_no_se_confunde_con_sin_datos():
    # A quiet device whose only item is the reminder: when it is not due, nothing is sent. An
    # empty list must not travel, because it would render the "no readings" message.
    recs = recs_with_reminder()
    only_reminder = [r for r in recs if is_lab_reminder(r)]
    selected, _ = select_for_sending(only_reminder, last_signature=None, due_reminders=set())
    assert selected is None
    selected2, _ = select_for_sending(only_reminder, last_signature=None, due_reminders={"LAB_SOIL_ANALYSIS", "LAB_LIMING"})
    assert selected2 == only_reminder


# --------------------------------------------------------------------------- #
# ReminderState: cadence persisted to disk
def test_estado_recuerda_envio_y_cadencia(tmp_path):
    state = ReminderState(path=str(tmp_path / "state.json"))
    dev = "1C:69:20:31:4B:78"
    assert state.is_due(dev, "LAB_SOIL_ANALYSIS", every_days=30)        # never sent -> due
    state.mark_sent(dev, "LAB_SOIL_ANALYSIS")
    assert not state.is_due(dev, "LAB_SOIL_ANALYSIS", every_days=30)    # just sent -> not due
    # After 31 days it is due again ("now" is injected so the test does not depend on the
    # clock).
    future = datetime.now(timezone.utc) + timedelta(days=31)
    assert state.is_due(dev, "LAB_SOIL_ANALYSIS", every_days=30, now=future)


def test_estado_sobrevive_reinicio(tmp_path):
    path = str(tmp_path / "state.json")
    ReminderState(path=path).mark_sent("dev-1", "LAB_SOIL_ANALYSIS")
    reloaded = ReminderState(path=path)            # as if the service restarted
    assert not reloaded.is_due("dev-1", "LAB_SOIL_ANALYSIS", every_days=30)
    assert reloaded.is_due("dev-2", "LAB_SOIL_ANALYSIS", every_days=30)  # another device is not contaminated


def test_estado_corrupto_no_revienta(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{esto no es json", encoding="utf-8")
    state = ReminderState(path=str(path))          # tolerant load: starts from scratch
    assert state.is_due("dev-1", "LAB_SOIL_ANALYSIS", every_days=30)
    state.mark_sent("dev-1", "LAB_SOIL_ANALYSIS")                       # and it can write over it again
    assert not ReminderState(path=str(path)).is_due("dev-1", "LAB_SOIL_ANALYSIS", every_days=30)
