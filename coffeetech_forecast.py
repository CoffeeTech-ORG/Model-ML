# -*- coding: utf-8 -*-
"""
CoffeeTech — machine learning component: microclimate forecast (random forest).
El Milagro pilot, San Ignacio (Cajamarca, Peru).

Forecasts air temperature and humidity several hours ahead from the sensor's own recent history
plus the hour of day. The label is the value actually observed H steps later, not a rule's output,
which is what keeps it non-circular -- unlike the retired classification forest. The forecast feeds
the rule engine so it can raise anticipated alerts.

What the method does and does not claim:

  * It learns the CHANGE against the current reading (`y − lag_0`), not the future value, and adds
    that reading back when predicting. A random forest cannot extrapolate, so with an absolute
    target its ceiling is a ceiling IN DEGREES. See `TARGET_FORMULATION`.
  * Persistence is the baseline (ŷ(t+H) = current value); the forest only counts if it beats it.
    Under this formulation persistence is exactly "change = 0".
  * Generalisation is estimated by leaving one campaign out (LOCO), not by an internal holdout: a
    split inside one campaign tests on a regime the model has already seen. The MINIMUM fold
    decides, never the mean. See `evaluar_loco` and `CRITERIO_SERVICIO`.
  * The served model trains on every available campaign. Which one sits out rotates for the
    ESTIMATE only, so no campaign is spent on being able to measure.
  * It is still ONE farm. Three campaigns from the same hub do not fix a single site, and that is
    declared rather than glossed over.

Three uses: validation (`validar`), which writes the figures the reports publish; training and
persistence of the operational models (`train`); and runtime prediction from the recent window
(Forecaster).

Retraining means running `train` and THEN `validar`, in that order. The report generator fails when
the validation JSON is older than the artifacts, so the table of a model other than the served one
cannot be published.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Iterable

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

# Targets worth forecasting. Soil humidity is excluded on purpose: it is such a slow signal that
# persistence beats it, and that is published as a result rather than hidden.
TARGETS = {
    "celcius_grade_temperature": "temperatura de aire (°C)",
    "air_humidity_percent": "humedad de aire (%)",
}
# Short key for naming artifacts by target.
TARGET_KEY = {"celcius_grade_temperature": "temp", "air_humidity_percent": "rh"}
SAMPLE_MIN = 2   # minutos por muestra
N_LAGS = 30

#: Horizons that are TRAINED and persisted. All four models still exist: they are the record of the
#: experiment and the report's performance table comes from them.
DEFAULT_HORIZONS_H = (2, 6)

#: Horizons that are SERVED. Not the same thing, and separating them is the point.
#:
#: Measured by `python coffeetech_forecast.py validar` (august 2026, 1823 m, ECMWF IFS pinned,
#: "change" target, three campaigns). Each column is a LOCO fold: trained on the OTHER campaigns
#: whole and tested on the one left out.
#:
#:                       out: feb    out: may     out: nov     MINIMUM      blocks
#:   temperature 2 h       +7.8 %      +5.6 %      −6.5 %      −6.5 %   +29.9 % ±19
#:   temperature 6 h      +13.5 %     +33.8 %     +48.5 %     +13.5 %   +49.5 % ±3
#:   humidity    2 h       +2.8 %     −23.0 %     −20.8 %     −23.0 %   +18.7 % ±31
#:   humidity    6 h      +24.6 %     +19.7 %     +19.7 %     +19.7 %   +31.4 % ±24
#:
#: The MINIMUM decides, not the mean. A horizon that works in two seasons and fails in the third
#: will fail in the field when the third arrives, and averaging hides it behind the two that went
#: well. The blocks sit alongside as diagnosis -- they say how much of the number depends on where
#: the cut fell -- but they test on regimes already seen, so they do not decide: at 2 h they read
#: +29.9 % and +18.7 %, which is the mirage LOCO removes.
#:
#: At 2 h humidity falls to −23.0 % leaving may out and −20.8 % leaving november: worse than
#: repeating the current value. Temperature holds in two folds and collapses in the third. One is
#: enough to retire it, and the verdict now rests on a cold, dry 24.5-day campaign inside the
#: rotation, so "it only needed a different season" has been tested and did not hold.
#:
#: 6 h passes, with a caveat that gets published rather than buried: the temperature minimum is
#: +13.5 %, the fold leaving february out, against +48.5 % for the one leaving november. It beats
#: persistence in all three, which is why it is served, but promising the blocks' +49.5 % would be
#: promising the best case. Humidity is steadier at +19.7 % in its worst fold.
#:
#: That temperature minimum was +3.1 % until the cloud-cover ablation was redone: the february fold
#: sank because the model was learning the OTHER campaigns' relationship between regional cloud
#: cover and local heating. Dropping it from the temperature target multiplies the minimum by four.
#: See `WEATHER_TARGETS`.
#:
#: Serving 2 h would mean raising anticipated alerts from a forecast that predicts WORSE than
#: repeating the current value. It is retired, not deleted: it keeps being trained, its measurement
#: is published and the reason it is not served is stated.
#:
#: The criterion lives in `horizontes_servibles()`, and `validar` FAILS when this constant stops
#: matching what was measured, so the two cannot diverge silently again.
SERVED_HORIZONS_H = (6,)

ARTIFACTS_DIR = "artifacts"
METADATA_FILE = os.path.join(ARTIFACTS_DIR, "forecast_metadata.json")

#: The three validations, measured and written to disk by `python coffeetech_forecast.py validar`.
#: The report reads them from here instead of carrying them typed out, which is how the 6 h
#: temperature holdout went on publishing +55.0 % while the retrained model gave +42.0 %.
VALIDATION_FILENAME = "forecast_validation.json"
VALIDATION_FILE = os.path.join(ARTIFACTS_DIR, VALIDATION_FILENAME)

# Sensor columns the forecast needs, with their aliases in the backend payload.
_COLUMN_ALIASES = {
    "celcius_grade_temperature": ("celcius_grade_temperature", "celciusGradeTemperature"),
    "air_humidity_percent": ("air_humidity_percent", "airHumidityPercent"),
    "soil_humidity_percent": ("soil_humidity_percent", "soilHumidityPercent"),
    "precipitation_detected": ("precipitation_detected", "precipitationDetected"),
}


def horizon_steps(hours: float) -> int:
    """Converts a horizon in hours into a number of sampling steps."""
    return int(round(hours * 60 / SAMPLE_MIN))


def load_session(path: str, session: str) -> pd.DataFrame:
    """Loads a telemetry CSV, sorts it by time and interpolates the ambient zeros (nulls the
    sensor reports as 0)."""
    d = pd.read_csv(path)
    d["t"] = pd.to_datetime(d["createdAt"])
    d = d.sort_values("t").reset_index(drop=True)
    d["session"] = session
    for c in ["soil_humidity_percent", "celcius_grade_temperature", "air_humidity_percent"]:
        d[c] = d[c].replace(0, np.nan).interpolate()
    return d


#: Horizon from which regional weather enters the model.
#:
#: Weather was first added as seven columns at once and measured on a single 70/30 split, which
#: reported a large gain (6 h temperature: +43 % without weather against +59.6 % with). It did not
#: survive blocked temporal validation, where the between-fold spread swallows the difference
#: (51.7 % ±16.9 against 48.7 % ±9.9). Isolating each column instead -- mean of the four cross
#: tests between campaigns, the hardest test the two campaigns then available allowed:
#:
#:   sensor only      33.6 %   ← reference
#:   + wind           33.8 %   no contribution
#:   + radiation      31.9 %   HURTS
#:   + VPD            29.5 %   HURTS
#:   + all seven      34.4 %   diluted: one column's signal drowns in the others' noise
#:   + CLOUD COVER    36.8 %   contributes
#:
#: Which is why `WEATHER_FEATURES` is a single variable. Confirmed with 6 seeds per cell: +2.6
#: points on average, positive in 4/4, seed noise ±0.2 to ±0.8, so the gain sits well outside the
#: random variation. It is also physically coherent: cloud cover modulates daytime heating, which
#: `hsin/hcos` can only assume, being a clock rather than a sky.
#:
#: Cloud cover was tested at both horizons (4 seeds, cross between campaigns):
#:
#:                    T nov→feb   T feb→nov   RH nov→feb   RH feb→nov    mean
#:   at 2 h             +1.6        +10.9        +2.3         −13.8     inconsistent
#:   at 6 h             +0.5         +3.9        +0.5          +6.7     +2.2 / +3.6
#:
#: At 6 h it wins in all four cells; at 2 h the sign flips by cell, so it does not enter.
#:
#: DECLARED LIMIT: these ablations date from the absolute target and two campaigns (nov+feb).
#: Redone under the current design -- LOCO over three campaigns, "change" target -- the "cloud
#: cover contributes" conclusion splits by target: it helps humidity and hurts temperature, with
#: the numbers and five seeds in `WEATHER_TARGETS`. The between-variable comparison above (wind,
#: radiation, VPD, the block of seven) has NOT been redone. It does not change what is served,
#: since none of them entered then or now, but reconsidering any of them means measuring it under
#: LOCO rather than recycling this table.
#:
#: This constant only decides whether cloud cover enters the 2 h horizon, and 2 h is not served
#: (see `SERVED_HORIZONS_H`).
MIN_HORIZON_H_FOR_WEATHER = 6.0

#: Long horizons: MEASURED AND NOT PUBLISHED.
#:
#: The design for these horizons is "regional forecast + learned correction": the API provides the
#: base and the forest learns how far this plot departs from it. Learning a correction needs far
#: less data than learning the dynamics, which is what makes it worth measuring at pilot scale.
#:
#: Measured with the three campaigns and the "change" target (`python
#: scripts/horizontes_largos.py`, results in `docs/horizontes_largos.json`):
#:
#:   AT 24 H — leaving one campaign out is possible, and it does NOT pass:
#:                    out feb     out may     MINIMUM     blocks
#:     temperature    −100.0 %     −1.2 %    −100.0 %   +1.5 % ±9.2
#:     humidity        +11.0 %    −10.0 %     −10.0 %   −8.7 % ±16.8
#:
#:   AT 72 H — LOCO IS NOT APPLICABLE. Only may leaves a long enough stretch (514 usable hours);
#:   feb leaves 826 rows spanning 27.9 h, barely more than one diurnal cycle, and nov leaves none.
#:   Leaving one campaign out requires two, so this horizon can be MEASURED but not validated
#:   across seasons. Blocked validation gives +9.3 % ±12.6 on temperature and +6.1 % ±7.5 on
#:   humidity: positive, within their own spread, and from a single season.
#:
#: The two targets fail for DIFFERENT reasons, which only shows when the error is decomposed:
#:
#:   * TEMPERATURE fails by OFFSET. Trained on may and tested on february the correlation between
#:     predicted and actual change is POSITIVE (+0.33), so it does capture signal, but it predicts
#:     a 3.81 °C mean drop where the real one is 0.35. Bias is 62 % of the squared error. May
#:     averages 16.5 °C and february 21.7: given a february reading the forest lands in leaves
#:     populated by may's warmest samples, which are exactly the ones followed by the sharpest
#:     cooling.
#:   * HUMIDITY fails by CONTRIBUTING NOTHING. Its bias is ~0 % of the error; it simply does not
#:     beat persistence outside the campaign it trained on.
#:
#: At 6 h this matters far less because the change is dominated by the diurnal cycle, which does
#: transfer. At 24 h the cycle cancels -- the target falls at the SAME hour of day -- and what
#: remains is regime drift, which is precisely what does not transfer between seasons.
#:
#: The API is no better as a base:
#:
#:                      persistence MAE   raw API MAE
#:   temperature 24 h        1.68            3.21
#:   humidity    24 h        4.91           11.41
#:
#: Read that carefully: the "raw API" is the ARCHIVE, a reanalysis that has already seen what
#: happened. It plays with an advantage a real forecast does not have, and its error still nearly
#: doubles that of repeating the current reading. Starting from it would be starting behind. That
#: says less about the weather product than about how much a 9 km cell resembles this coffee plot.
#:
#: Which is why `DEFAULT_HORIZONS_H` stays (2, 6). Anticipated alerts at 24-72 h are not built on
#: this model but by reading the Open-Meteo forecast directly, which is a valid weather product at
#: those ranges. See `coffeetech_weather`.
LONG_HORIZONS_NOT_PUBLISHED = (24, 72)

#: ONE variable from the regional model, the only one measured to contribute.
#:
#: `hsin/hcos` give the forest a clock: it knows the hour but not whether the sky is overcast, so
#: it assumes the same heating at two in the afternoon come rain or shine. Cloud cover is exactly
#: what that clock lacks, which is why it wins where radiation, VPD and wind hurt or do nothing.
#: The comparison table is in `MIN_HORIZON_H_FOR_WEATHER`.
#:
#: The discarded ones do not come back just in case: every column that enters without contributing
#: spreads useless splits through the forest and dilutes the one that does, which is what happened
#: with the block of seven. See `docs/DECISIONES_DESCARTADAS.md`.
WEATHER_FEATURES = ("cloud_cover",)

#: WHICH TARGETS that cloud cover enters. Not both, and this is a result rather than a preference.
#:
#: The original ablation said "cloud cover contributes" flat, measured on two campaigns with the
#: absolute target and a nov↔feb cross. Redone under the current design -- LOCO over three
#: campaigns, "change" target -- it splits in two, in opposite directions. Five seeds per cell, at
#: 6 h:
#:
#:                        with cloud     without cloud   difference   seeds in favour
#:   TEMPERATURE
#:     out: feb            +3.8 % ±0.2    +13.3 % ±0.1     −9.6          0/5
#:     out: may           +33.9 % ±0.2    +34.1 % ±0.2     −0.2          1/5
#:     out: nov           +50.2 % ±0.2    +49.1 % ±0.2     +1.1          5/5
#:     MINIMUM (decides)   +3.8 %         +13.3 %        ← without cloud is 3.5× better
#:   HUMIDITY
#:     out: feb           +24.5 % ±0.3    +24.8 % ±0.3     −0.3          0/5
#:     out: may           +19.8 % ±0.3    +16.4 % ±0.2     +3.3          5/5
#:     out: nov           +19.5 % ±0.4    +14.1 % ±0.3     +5.4          5/5
#:     MINIMUM (decides)  +19.5 %         +14.1 %        ← with cloud
#:
#: Seed noise runs ±0.1 to ±0.4 points and the deciding effects are 9.6 and 5.4, with 0/5 and 5/5
#: agreement. This is not one lucky forest.
#:
#: The likely mechanism, stated as far as the evidence goes: reanalysis cloud cover belongs to a
#: 9 km cell, not to the coffee plot. For HUMIDITY that is enough -- overcast and humid travel
#: together at regional scale -- so it transfers between campaigns. For TEMPERATURE the model
#: learns how much the sun heats under a given cloud cover IN THAT CAMPAIGN, and that relationship
#: does not travel: the sensor − regional model offset is already measured and changes by campaign
#: (−0.3 °C in nov, +4.0 °C in feb). The fold that sinks being february's is consistent with that.
#: Consistent is not demonstrated, and it is declared as such.
#:
#: Both targets still pass the criterion under either configuration, so this rescues and retires no
#: horizon. What changes is the number PROMISED, and promising +3.8 % when +13.3 % is available
#: would publish worse than necessary.
WEATHER_TARGETS = ("air_humidity_percent",)


def _feature_columns(
    g: pd.DataFrame,
    target: str,
    n_lags: int = N_LAGS,
    horizon_h: Optional[float] = None,
) -> pd.DataFrame:
    """Builds the features (all from the past, <= t) for a session already sorted by time.

    Same recipe for training and for predicting, so the vector cannot differ between them.
    `horizon_h` decides whether regional weather enters: see `MIN_HORIZON_H_FOR_WEATHER`.
    """
    g = g.sort_values("t").reset_index(drop=True)
    f = pd.DataFrame(index=g.index)
    for L in range(n_lags):
        f[f"lag_{L}"] = g[target].shift(L)
    f["roll_mean_30"] = g[target].shift(1).rolling(15, min_periods=5).mean()   # ~30 min
    f["roll_std_30"] = g[target].shift(1).rolling(15, min_periods=5).std()
    f["trend_30"] = g[target].shift(1) - g[target].shift(15)                    # recent slope
    f["temp"] = g["celcius_grade_temperature"]
    f["rh"] = g["air_humidity_percent"]
    f["soil"] = g["soil_humidity_percent"]
    f["rain"] = g["precipitation_detected"]
    h = g["t"].dt.hour + g["t"].dt.minute / 60
    f["hsin"] = np.sin(2 * np.pi * h / 24)
    f["hcos"] = np.cos(2 * np.pi * h / 24)

    # Regional weather enters when the session was joined with it (`join_weather`), the horizon is
    # long enough for it to contribute AND the target is one that benefits. When it is absent -- a
    # plot with no coordinate -- the column is dropped entirely rather than filled with a zero: a
    # zero in cloud cover means "clear sky", not "unknown", and the model must not confuse them.
    if horizon_h is not None and horizon_h < MIN_HORIZON_H_FOR_WEATHER:
        return f
    # Measured: regional cloud cover HURTS temperature outside its own campaign. See
    # `WEATHER_TARGETS` for the numbers and the reason.
    if target not in WEATHER_TARGETS:
        return f

    for var in WEATHER_FEATURES:
        col = f"api_{var}"
        if col in g.columns:
            f[col] = g[col]
    return f


#: Variables that are hourly TOTALS (rain fallen, water evaporated) rather than an instantaneous
#: state. Interpolating them would smear the signal: 2 mm in the 14:00 hour does not mean 1 mm at
#: 14:30. They are carried unchanged across that hour's minutes.
_ACUMULADAS = ("precipitation", "et0_fao_evapotranspiration")


def join_weather(d: pd.DataFrame, weather) -> pd.DataFrame:
    """Joins telemetry (every 2 min) with regional weather (hourly).

    A timezone offset is the silent failure this join can produce: five hours of misalignment
    displaces the diurnal cycle, the model simply gets worse, and nothing visibly breaks. Both
    sides are therefore required to carry an explicit timezone rather than trusted to agree.

    Returns the DataFrame with `api_<variable>` columns. With no weather it returns it untouched:
    without `api_` columns `_feature_columns` generates no such features and the model stays the
    sensor-only one.
    """
    if weather is None or len(weather) == 0:
        return d

    if d["t"].dt.tz is None:
        raise ValueError(
            "La telemetría llega sin zona horaria. Cruzarla con la meteorología así descuadraría "
            "el ciclo diurno cinco horas sin que nada falle a la vista."
        )

    w = pd.DataFrame({"t": weather.times, **{f"api_{k}": v for k, v in weather.values.items()}})
    w["t"] = pd.to_datetime(w["t"], utc=True)
    w = w.set_index("t").sort_index()

    idx = pd.DatetimeIndex(d["t"])
    unido = d.copy()
    for col in w.columns:
        var = col[len("api_"):]
        serie = w[col].astype(float)
        if var in _ACUMULADAS:
            # The hour's value is held across all of its minutes.
            unido[col] = serie.reindex(serie.index.union(idx)).ffill().reindex(idx).values
        else:
            # Continuous state: interpolated by time between the known hours.
            unido[col] = (
                serie.reindex(serie.index.union(idx))
                .interpolate(method="time")
                .reindex(idx)
                .values
            )
    return unido


def attach_archive_weather(d: pd.DataFrame, latitude: Optional[float] = None,
                           longitude: Optional[float] = None,
                           elevation: Optional[float] = None,
                           obligatorio: bool = False) -> pd.DataFrame:
    """Joins a telemetry session with the reanalysis for ITS dates, for training.

    It exists so training and serving use one recipe. In production cloud cover arrives from the
    forecast; here it comes from the same grid with the same elevation correction, but from the
    archive, because the campaigns already happened.

    With no network or no coordinate it returns the DataFrame untouched: training continues without
    the weather column, which is exactly what happens for a farm with no location.

    `obligatorio=True` changes that and FAILS instead. It is for the path that PERSISTS the
    production artifacts, and it exists because this already happened: a 502 from the archive left
    all four models trained without cloud cover, 6 h humidity dropping from +22.6 % to +15.6 %, and
    nothing to show for it but two warnings in the log. A served model with a different feature
    recipe from the one production will hand it is a silent failure. Degrading quietly is fine for
    evaluation, not for what gets deployed.
    """
    from coffeetech_weather import (PILOT_ELEVATION, PILOT_LAT, PILOT_LON, get_archive)

    lat = PILOT_LAT if latitude is None else latitude
    lon = PILOT_LON if longitude is None else longitude
    alt = PILOT_ELEVATION if elevation is None else elevation

    # A day of margin on each side: the hourly window has to cover the session's edges.
    desde = (d["t"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    hasta = (d["t"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    w = get_archive(lat, lon, desde, hasta, elevation=alt)
    if w is None:
        if obligatorio:
            raise RuntimeError(
                f"No se pudo obtener el reanálisis {desde}..{hasta} y se pidió obligatorio. "
                "NO se persisten modelos sin la nubosidad: producción la alimenta y entrenar sin "
                "ella deja un modelo con otra receta de rasgos. Reintenta cuando el archivo "
                "responda.")
        return d
    return join_weather(d, w)


def make_features(d: pd.DataFrame, target: str, horizon: int, n_lags: int = N_LAGS) -> pd.DataFrame:
    """Builds the training matrix: features from the past and the target at t+horizon. Computed per
    session so the months-long gap between campaigns is never crossed."""
    out = []
    for _, g in d.groupby("session"):
        g = g.sort_values("t").reset_index(drop=True)
        f = _feature_columns(g, target, n_lags, horizon_h=horizon * SAMPLE_MIN / 60)
        f["y"] = g[target].shift(-horizon)   # future target (the value actually observed)
        f["t"] = g["t"]
        f["session"] = g["session"]
        out.append(f)
    return pd.concat(out, ignore_index=True).dropna().reset_index(drop=True)


def _feature_names(sample_features: pd.DataFrame) -> List[str]:
    return [c for c in sample_features.columns if c not in ("y", "t", "session")]


def _new_rf(seed: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_leaf=4,
                                 n_jobs=-1, random_state=seed)


def _mae(a, b) -> float:
    return float(np.mean(np.abs(a - b)))


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def evaluate(nov: pd.DataFrame, feb: pd.DataFrame, target: str, horizon_steps_: int, verbose=True) -> tuple:
    """Trains on (all of nov + 60 % of feb) and tests on feb's final 40 %, comparing the forest
    against persistence. Returns (model, holdout metrics)."""
    data = pd.concat([nov, feb], ignore_index=True)
    F = make_features(data, target, horizon_steps_)
    feb_F = F[F["session"] == "feb"].sort_values("t")
    nov_F = F[F["session"] == "nov"]
    cut = int(len(feb_F) * 0.60)
    train = pd.concat([nov_F, feb_F.iloc[:cut]], ignore_index=True)
    test = feb_F.iloc[cut:].reset_index(drop=True)
    feat_cols = _feature_names(F)
    Xtr, ytr = train[feat_cols], train["y"]
    Xte, yte = test[feat_cols], test["y"]

    rf = _new_rf()
    rf.fit(Xtr, ytr)
    yhat = rf.predict(Xte)
    yhat_p = Xte["lag_0"].values  # persistence: the current value as the forecast

    res = {
        "horizon_h": horizon_steps_ * SAMPLE_MIN / 60,
        "n_train": len(train), "n_test": len(test),
        "mae_persist": _mae(yte, yhat_p), "mae_rf": _mae(yte, yhat),
        "rmse_persist": _rmse(yte, yhat_p), "rmse_rf": _rmse(yte, yhat),
    }
    res["skill_rmse"] = 1 - res["rmse_rf"] / res["rmse_persist"]
    res["importances"] = sorted(zip(feat_cols, rf.feature_importances_), key=lambda x: -x[1])[:6]
    if verbose:
        print(f"  H={res['horizon_h']:.0f}h | train={res['n_train']} test={res['n_test']} | "
              f"RMSE persist={res['rmse_persist']:.3f} RF={res['rmse_rf']:.3f} | "
              f"MAE persist={res['mae_persist']:.3f} RF={res['mae_rf']:.3f} | skill={res['skill_rmse']:+.1%}")
    return rf, res


# --------------------------------------------------------------------------- #
# The validations that accompany LOCO.
#
# They live here, and write to the validation JSON, so the report reads them instead of quoting
# them. A retraining moves these columns by more than ten points -- the 6 h temperature holdout
# spans +55.0 % to +42.0 % across retrainings -- and a figure typed into prose does not move with
# it.
#
# They share `make_features`, `_feature_names` and `_new_rf` with LOCO, so they compare the SAME
# model with the same feature recipe and differ only in how they split the data.

def evaluar_bloques_n(campanas: List[pd.DataFrame], target: str, horizon_steps_: int,
                      delta: bool = True, n_splits: int = 5,
                      semillas: Iterable[int] = (0, 1, 2, 3, 4)) -> dict:
    """The same blocked validation, over N campaigns and with whichever formulation applies."""
    from sklearn.model_selection import TimeSeriesSplit

    F = make_features(pd.concat(campanas, ignore_index=True), target, horizon_steps_)
    F = F.sort_values("t").reset_index(drop=True)
    feat_cols = _feature_names(F)
    X, y = F[feat_cols], F["y"]
    base = F["lag_0"].values

    mejoras: List[float] = []
    for semilla in semillas:
        for tr, te in TimeSeriesSplit(n_splits=n_splits).split(X):
            rf = _new_rf(seed=semilla)
            rf.fit(X.iloc[tr], _aplicar_formulacion(y.iloc[tr].values, base[tr], delta))
            pred = _deshacer_formulacion(rf.predict(X.iloc[te]), base[te], target, delta)
            rmse_rf = _rmse(y.iloc[te].values, pred)
            rmse_p = _rmse(y.iloc[te].values, base[te])
            mejoras.append(1 - rmse_rf / rmse_p)

    return {
        "horizon_h": horizon_steps_ * SAMPLE_MIN / 60,
        "n_pliegues": len(mejoras),
        "skill_rmse": float(np.mean(mejoras)),
        "skill_sd": float(np.std(mejoras)),
        "skill_min": float(np.min(mejoras)),
        "skill_max": float(np.max(mejoras)),
    }


# --------------------------------------------------------------------------- #
# HOW THE TARGET IS FORMULATED
#
# A random forest does NOT extrapolate: each leaf returns the mean of the targets that fell into
# it, and 300 trees are then averaged. An average of averages cannot leave the training range, and
# it shrinks toward the centre on top of that. Measured on the model of the time -- two campaigns,
# absolute target -- which is the one that had the problem:
#
#     training targets (nov+feb)      13.4 - 28.7 °C
#     what the model could ever say   15.0 - 26.4 °C
#
# With an ABSOLUTE target that ceiling is a ceiling IN DEGREES. In practice the model could not
# anticipate cold below 15 °C -- exactly where `R9_COLD` and the Phoma band at 1823 m live -- nor
# heat above 26. That is not a training defect, it is the geometry of the method.
#
# The fix is to predict the CHANGE against persistence, `y − lag_0`. The ceiling becomes a ceiling
# on the CHANGE, and the prediction comes free because `lag_0` is the sensor's real reading:
#
#     prediction = current_reading + predicted_change
#
# Measured with that single variable isolated (same data, same model, same features):
#
#                             absolute -> change
#     feb tail      T 2h       +10.0 %    +22.2 %      KNOWN regime: flat or slightly worse
#     feb tail      RH 6h      +24.7 %    +22.6 %
#     nov whole     T 6h       +51.2 %    +56.3 %
#     may tail      T 2h      −135.2 %     +4.5 %      NEW regime: up to 140 points
#     may tail      T 6h       +12.9 %    +34.8 %
#
# The correct reading is not "change predicts better". It is that change survives a change of
# regime: where the model already knows the range it loses 2-8 points, and where it does not it
# gains orders of magnitude. For a system running all year on one farm, meeting an unseen regime is
# not a hypothesis, it is a matter of time.
#
# The mechanism and the remedy are documented in the tree-based time series literature (difference
# or detrend before fitting), but the evidence behind this decision is the measurement above, made
# on this farm's data.
TARGET_FORMULATION = "cambio"

#: PHYSICAL bounds per target, applied to the final prediction.
#:
#: Releasing the ceiling downward releases it upward too: while measuring, the change formulation
#: predicted 103.5 % relative humidity, which does not exist -- relative humidity is by definition
#: a fraction of saturation. This clamp is physics, not judgement.
#:
#: Temperature is deliberately left UNCLAMPED: there is no physical limit to appeal to, and an
#: invented one would restore the very limitation this formulation removes. What bounds it in
#: practice is the largest change observed in the archive, which is a measured quantity.
LIMITES_FISICOS = {
    "air_humidity_percent": (0.0, 100.0),
}


def _aplicar_formulacion(y, base, delta: bool):
    """The target TRAINED on: the future value, or its change against the current reading."""
    return (y - base) if delta else y


def _deshacer_formulacion(crudo, base, target: str, delta: bool):
    """Model output -> value in real units, clamped to what physics allows."""
    valor = (base + crudo) if delta else crudo
    lim = LIMITES_FISICOS.get(target)
    if lim is not None:
        valor = np.clip(valor, lim[0], lim[1])
    return valor


def train_operational(campanas: List[pd.DataFrame], target: str, horizon_steps_: int,
                      delta: bool = True) -> tuple:
    """Trains the final model on EVERY available campaign, for production use.

    Training on everything and estimating generalisation separately (with `evaluar_loco`) means
    there is no need to choose between using a campaign and reserving it for testing: which one
    sits out rotates for the ESTIMATE, and the served model trains on all of them.
    """
    F = make_features(pd.concat(campanas, ignore_index=True), target, horizon_steps_)
    feat_cols = _feature_names(F)
    rf = _new_rf()
    rf.fit(F[feat_cols], _aplicar_formulacion(F["y"].values, F["lag_0"].values, delta))
    return rf, feat_cols


def evaluar_loco(campanas: Dict[str, pd.DataFrame], target: str, horizon_steps_: int,
                 delta: bool = True) -> dict:
    """Leave-one-campaign-out cross validation.

    It replaces the held-out set because the alternative was a false dilemma: either may enters
    training or it stays as the test. Standard practice dissolves that -- generalisation is
    ESTIMATED by rotating which group sits out, and the served model trains on everything. What is
    needed is an honest estimator, not sacrificed data.

    The group here is the CAMPAIGN, which is the natural unit: each is a different season, and rows
    within one campaign are correlated. Splitting by row, or at random, would give an inflated
    number, because two readings two minutes apart are nearly the same reading.

    Each fold trains on the other campaigns WHOLE and tests on the one left out, so each answers
    "does it work in a season it has not seen?". Folds from short campaigns (nov is 48 h, feb
    100 h) are noisy and get published as such rather than averaged in as if they weighed the same
    as the 24.5-day one.
    """
    pliegues = {}
    for fuera in campanas:
        entrena = [d for k, d in campanas.items() if k != fuera]
        prueba = campanas[fuera]
        if not entrena:
            continue
        Ftr = make_features(pd.concat(entrena, ignore_index=True), target, horizon_steps_)
        Fte = make_features(prueba, target, horizon_steps_)
        cols = [c for c in _feature_names(Ftr) if c in Fte.columns]
        if Ftr.empty or Fte.empty or not cols:
            pliegues[fuera] = None
            continue

        rf = _new_rf()
        rf.fit(Ftr[cols], _aplicar_formulacion(Ftr["y"].values, Ftr["lag_0"].values, delta))
        pred = _deshacer_formulacion(rf.predict(Fte[cols]), Fte["lag_0"].values, target, delta)
        yte = Fte["y"].values
        rmse_p = _rmse(yte, Fte["lag_0"].values)
        pliegues[fuera] = {
            "n_train": len(Ftr), "n_test": len(Fte),
            "rmse_persist": rmse_p, "rmse_rf": _rmse(yte, pred),
            "skill_rmse": 1 - _rmse(yte, pred) / rmse_p,
            "pred_min": float(np.min(pred)), "pred_max": float(np.max(pred)),
        }

    validos = [p["skill_rmse"] for p in pliegues.values() if p is not None]
    return {
        "horizon_h": horizon_steps_ * SAMPLE_MIN / 60,
        "pliegues": pliegues,
        # The MINIMUM outranks the mean: a horizon that fails in one season will fail in the field
        # when that season arrives, and averaging hides it.
        "skill_min": float(min(validos)) if validos else None,
        "skill_medio": float(np.mean(validos)) if validos else None,
    }


# --------------------------------------------------------------------------- #
# THE FULL VALIDATION, PERSISTED
CRITERIO_SERVICIO = (
    "Un horizonte se sirve si TODOS sus objetivos superan a la persistencia en TODOS los pliegues "
    "de la validación dejando una campaña fuera (LOCO). No la media: el MÍNIMO. Un horizonte que "
    "falla en una estación va a fallar en campo cuando llegue esa estación, y promediarlo lo "
    "escondería. Los bloques temporales se siguen publicando como diagnóstico —dicen cuánto del "
    "número depende de dónde cayó el corte— pero no deciden: prueban sobre régimenes que el "
    "modelo ya vio.")


def horizontes_servibles(filas: List[dict]) -> List[float]:
    """Which horizons pass the criterion, computed from the measurement rather than chosen by hand.

    The LOCO folds decide. A validation JSON without them falls back to the cross between
    campaigns, which is the criterion those files were written under. No verdict is invented: a
    horizon with no evidence is not served.
    """
    por_horizonte: Dict[float, List[dict]] = {}
    for f in filas:
        por_horizonte.setdefault(f["horizon_h"], []).append(f)
    servibles = []
    for h, grupo in sorted(por_horizonte.items()):
        # LOCO decides: each target's minimum across all of its folds.
        minimos = [f.get("loco", {}).get("skill_min") if f.get("loco") else None for f in grupo]
        if all(m is not None for m in minimos):
            if all(m > 0 for m in minimos):
                servibles.append(h)
            continue
        # With no LOCO, no verdict is invented: it falls back to the cross between campaigns, the
        # criterion an older validation JSON was written under. If that is missing too the list
        # stays empty, and a horizon with no evidence is not served.
        alt = [(f.get("entre_campanas") or {}).get("skill_min") for f in grupo]
        if all(m is not None and m > 0 for m in alt):
            servibles.append(h)
    return servibles


def validar_todo(campanas: Dict[str, pd.DataFrame],
                 horizons_h: Iterable[float] = DEFAULT_HORIZONS_H,
                 artifacts_dir: str = ARTIFACTS_DIR, verbose: bool = True) -> dict:
    """Runs every validation for each target and horizon, and writes them to disk.

    The report reads this JSON to draw its performance table instead of carrying the numbers typed
    out. Same discipline as the threshold inventory: what gets published comes from the code, or it
    does not get published.

    LOCO is what DECIDES. The others are kept as diagnosis -- they say how much of the number
    depends on where the cut fell, which is useful -- but they test on regimes the model has
    already seen.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    delta = TARGET_FORMULATION == "cambio"
    nombres = sorted(campanas)
    entrenamiento = [campanas[k] for k in nombres]
    filas = []
    for target in TARGETS:
        for hours in horizons_h:
            steps = horizon_steps(hours)
            if verbose:
                print(f"  {TARGETS[target]} a {hours} h…", flush=True)
            filas.append({
                "target": target,
                "etiqueta": TARGETS[target],
                "horizon_h": hours,
                "loco": evaluar_loco(campanas, target, steps, delta),
                "bloques": evaluar_bloques_n(entrenamiento, target, steps, delta),
            })

    servibles = horizontes_servibles(filas)
    validacion = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "servibles": servibles,
        "criterio": CRITERIO_SERVICIO,
        "campanas": nombres,
        "objetivo": TARGET_FORMULATION,
        "nota": ("LOCO —dejar una campaña fuera— es la validación que decide: cada pliegue "
                 "entrena con las demás campañas enteras y prueba con la que queda fuera, así que "
                 "responde «¿sirve en una estación que no ha visto?». Los bloques temporales se "
                 "publican como diagnóstico: dicen cuánto del número depende de dónde cayó el "
                 "corte, pero prueban sobre régimenes ya vistos."),
        "limite": ("Los pliegues de campañas cortas son ruidosos y NO valen lo mismo: nov son "
                   "48 h y feb 100 h, frente a los 24,5 días de mayo. Se publican por separado en "
                   "vez de promediarlos. Y sigue siendo UNA finca: ninguna cantidad de campañas "
                   "del mismo hub arregla eso."),
        "modelos": filas,
    }
    with open(os.path.join(artifacts_dir, VALIDATION_FILENAME), "w", encoding="utf-8") as fh:
        json.dump(validacion, fh, ensure_ascii=False, indent=2)
    return validacion


# --------------------------------------------------------------------------- #
# PERSISTENCIA
def _artifact_path(target: str, hours: float, artifacts_dir: str = ARTIFACTS_DIR) -> str:
    return os.path.join(artifacts_dir, f"forecast_{TARGET_KEY[target]}_h{int(hours)}.joblib")


def train_and_persist(campanas: Dict[str, pd.DataFrame],
                      horizons_h: Iterable[float] = DEFAULT_HORIZONS_H,
                      artifacts_dir: str = ARTIFACTS_DIR,
                      loco: Optional[Dict] = None) -> dict:
    """Trains and saves one operational model per target (temperature, air humidity) and
    horizon, alongside a forecast_metadata.json recording the features, the target formulation,
    which campaigns it trained on and the LOCO folds that estimate its performance. Returns the
    metadata dictionary."""
    os.makedirs(artifacts_dir, exist_ok=True)
    delta = TARGET_FORMULATION == "cambio"
    models_meta = []
    for target in TARGETS:
        for hours in horizons_h:
            steps = horizon_steps(hours)
            # The SERVED model trains on every campaign. Its performance is NOT estimated with
            # an internal holdout -- that would test on a regime already seen -- but with the LOCO
            # folds passed in separately.
            rf, feat_cols = train_operational(list(campanas.values()), target, steps, delta)
            path = _artifact_path(target, hours, artifacts_dir)
            joblib.dump({"model": rf, "feat_cols": feat_cols, "target": target,
                         "horizon_steps": steps, "horizon_h": hours,
                         # The formulation travels WITH the artifact: `Forecaster.predict` reads
                         # it from here rather than from the module constant, so an old model is
                         # never interpreted with the new recipe.
                         "objetivo": TARGET_FORMULATION}, path)
            pl = (loco or {}).get((target, hours))
            models_meta.append({
                "file": os.path.basename(path),
                "target": target,
                "horizon_h": hours,
                "horizon_steps": steps,
                "feat_cols": feat_cols,
                "objetivo": TARGET_FORMULATION,
                "entrenado_con": sorted(campanas),
                "loco": pl,
                # Trained and saved is not the same as served. A model that does not beat
                # persistence outside its own campaign is kept as the record of the experiment but
                # raises no alerts, and the metadata says which is which so nobody has to infer it
                # from the code.
                "servido": hours in SERVED_HORIZONS_H,
            })
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sample_min": SAMPLE_MIN,
        "n_lags": N_LAGS,
        "targets": list(TARGETS),
        "soil_note": ("La humedad de suelo NO se pronostica: en la evaluación la persistencia la "
                      "supera (señal demasiado lenta). El sensor de suelo se usa tal cual en las reglas."),
        "horizontes_entrenados_h": [float(h) for h in horizons_h],
        "horizontes_servidos_h": [float(h) for h in SERVED_HORIZONS_H],
        "criterio_servicio": CRITERIO_SERVICIO,
        "models": models_meta,
    }
    with open(os.path.join(artifacts_dir, "forecast_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    return metadata


# --------------------------------------------------------------------------- #
# RUNTIME
def window_from_records(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Builds the window (a DataFrame sorted by time) the forecast consumes from dict records,
    accepting both the CSV keys and the backend's camelCase ones. Interpolates the ambient zeros
    the same way load_session does."""
    rows = []
    for rec in records:
        t = (rec.get("createdAt") or rec.get("timestamp") or rec.get("created_at")
             or rec.get("updatedAt") or rec.get("updatedDate"))
        if t is None:
            continue
        row = {"t": t}
        for canonical, aliases in _COLUMN_ALIASES.items():
            value = None
            for alias in aliases:
                if rec.get(alias) not in (None, ""):
                    value = rec.get(alias)
                    break
            row[canonical] = value
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    for c in ("soil_humidity_percent", "celcius_grade_temperature", "air_humidity_percent"):
        df[c] = pd.to_numeric(df[c], errors="coerce").replace(0, np.nan).interpolate()
    df["precipitation_detected"] = pd.to_numeric(df["precipitation_detected"], errors="coerce").fillna(0)
    return df


def latest_feature_row(window_df: pd.DataFrame, target: str, feat_cols: List[str],
                       n_lags: int = N_LAGS) -> Optional[pd.DataFrame]:
    """Builds the feature vector for the window's last reading, the one used to predict the
    value at t+H. Returns None when there is not enough data or NaNs remain.

    The horizon does not need to be passed: `feat_cols` comes from the trained artifact and says
    exactly which columns that model expects, so the `reindex` below keeps its own. When a 6 h
    model asks for weather and that run has none -- plot without a coordinate, or the API down --
    NaNs remain and `None` is returned: without the inputs it learned on, that forecast is not
    issued. Same rule as everywhere else in the system: rather than give an invented number, give
    none.
    """
    if window_df is None or len(window_df) <= n_lags:
        return None
    f = _feature_columns(window_df, target, n_lags)
    row = f.iloc[[-1]].reindex(columns=feat_cols)
    if row.isnull().any(axis=1).iloc[0]:
        return None
    return row


class Forecaster:
    """Loads the persisted models and predicts a target's future value at a given horizon from a
    device's recent window."""

    def __init__(self, bundles: Dict[tuple, dict]):
        # bundles: {(target, horizon_h): {"model", "feat_cols", ...}}
        self._bundles = bundles

    @classmethod
    def load(cls, artifacts_dir: str = ARTIFACTS_DIR,
             horizons_h: Iterable[float] = SERVED_HORIZONS_H) -> "Forecaster":
        """Loads only the SERVABLE horizons, not every trained one.

        The 2 h artifact stays on disk -- it is the record of the experiment and backs the
        report's table -- but it is not loaded, so `horizons()` never announces it and no
        anticipated alert can come from it. See the `SERVED_HORIZONS_H` note for the measurement.
        """
        bundles: Dict[tuple, dict] = {}
        for target in TARGETS:
            for hours in horizons_h:
                path = os.path.join(artifacts_dir, f"forecast_{TARGET_KEY[target]}_h{int(hours)}.joblib")
                if os.path.exists(path):
                    try:
                        bundles[(target, hours)] = joblib.load(path)
                    except Exception:
                        pass
        return cls(bundles)

    def available(self) -> bool:
        return bool(self._bundles)

    def horizons(self) -> List[float]:
        return sorted({h for (_t, h) in self._bundles})

    def predict(self, window_df: pd.DataFrame, target: str, horizon_h: float) -> Optional[float]:
        bundle = self._bundles.get((target, horizon_h))
        if bundle is None:
            return None
        row = latest_feature_row(window_df, target, bundle["feat_cols"])
        if row is None:
            return None
        # The formulation is read FROM THE ARTIFACT, not from the module constant, so an old
        # model trained on the absolute target is still interpreted correctly after the module
        # moved to change. An artifact that does not say is from before, and was absolute.
        delta = bundle.get("objetivo") == "cambio"
        crudo = bundle["model"].predict(row)
        base = row["lag_0"].values
        return float(_deshacer_formulacion(crudo, base, target, delta)[0])

    def forecast_climate(self, window_df: pd.DataFrame, horizon_h: float) -> Optional[Dict[str, float]]:
        """Devuelve {'air_temp', 'air_rh'} pronosticados al horizonte, o None si no se pudo."""
        temp = self.predict(window_df, "celcius_grade_temperature", horizon_h)
        rh = self.predict(window_df, "air_humidity_percent", horizon_h)
        if temp is None or rh is None:
            return None
        return {"air_temp": temp, "air_rh": rh}


# --------------------------------------------------------------------------- #
# CLI de reproducción de resultados
def _resolve_session_path(session: str) -> str:
    """Locates a session's CSV: <session>.csv in the current directory first, otherwise the
    matching file inside data/."""
    candidates = {
        "nov": [f"{session}.csv", os.path.join("data", "data_sensors_san_ignacio_nov2025.csv")],
        "feb": [f"{session}.csv", os.path.join("data", "data_sensors_san_ignacio_feb2026.csv")],
        # May is optional: without the CSV, `validar` and `train` carry on with two campaigns and
        # say so. See the `__main__` block for what it adds and what doubt it carries.
        "mayo": [f"{session}.csv", os.path.join("data", "data_sensors_san_ignacio_may2026.csv")],
    }
    for path in candidates.get(session, [f"{session}.csv"]):
        if os.path.exists(path):
            return path
    return f"{session}.csv"


if __name__ == "__main__":
    import sys

    nov = load_session(_resolve_session_path("nov"), "nov")
    feb = load_session(_resolve_session_path("feb"), "feb")

    # May: one more campaign in the rotation, and the only cold regime the project has. 17 392
    # rows -- four times what nov and feb add up to -- over 24.5 continuous days averaging 16.5 °C
    # against 20.1 and 21.7.
    #
    # It enters training because LOCO estimates generalisation by rotating which campaign is left
    # out, so nothing has to be sacrificed to measure. The reservation it carries is its own: the
    # DATES could not be certified by any control (see `scripts/cargar_campana.py`), and it is 4×
    # everything else, so an error in it would dominate.
    #
    # What makes including it acceptable is a check the folds provide: the "out: nov" fold
    # trains mostly on may and predicts november, which IS authentic -- and it is the best fold of
    # the three. Were may noise, or telemetry from somewhere else, that fold would collapse. It
    # does not certify the dates; it does say the data carries this site's signal.
    ruta_mayo = _resolve_session_path("mayo")
    mayo = load_session(ruta_mayo, "mayo") if os.path.exists(ruta_mayo) else None
    CAMPANAS = {"nov": nov, "feb": feb}
    if mayo is not None:
        CAMPANAS["mayo"] = mayo

    # `train`   trains and persists the operational models, with their LOCO folds alongside.
    # `validar` runs LOCO plus the blocks and writes the JSON the report reads.
    # With no arguments it reproduces the honest evaluation (forest vs persistence) over nov+feb.
    modo = sys.argv[1] if len(sys.argv) > 1 else ""
    a_disco = modo in ("train", "validar")

    # Reanalysis cloud cover for those dates. Without it the models come out sensor-only and the
    # measured gain never reaches production: this is the step that was missing after adding the
    # feature. When the
    # resultado va A DISCO —artefactos o números publicables— su ausencia rompe en vez de
    # degradarse en silencio.
    for k in list(CAMPANAS):
        CAMPANAS[k] = attach_archive_weather(CAMPANAS[k], obligatorio=a_disco)
    nov, feb = CAMPANAS["nov"], CAMPANAS["feb"]

    if modo == "train":
        # The served model's performance is estimated with LOCO rather than an internal holdout,
        # so it has to be computed BEFORE persisting, to be stored alongside the artifact.
        print(f"Estimando generalización (LOCO sobre {len(CAMPANAS)} campañas)…")
        delta = TARGET_FORMULATION == "cambio"
        loco = {(t, h): evaluar_loco(CAMPANAS, t, horizon_steps(h), delta)
                for t in TARGETS for h in DEFAULT_HORIZONS_H}
        meta = train_and_persist(CAMPANAS, loco=loco)
        print(f"\nModelos en {ARTIFACTS_DIR}/ (objetivo «{TARGET_FORMULATION}», entrenados con "
              f"{', '.join(sorted(CAMPANAS))}):")
        for m in meta["models"]:
            pl = m.get("loco") or {}
            estado = "SERVIDO" if m["servido"] else "no servido"
            print(f"  {m['file']:26} {estado:11} LOCO mín={pl.get('skill_min', float('nan')):+.1%}")
        sys.exit(0)

    if modo == "validar":
        if mayo is None:
            print("Falta data/data_sensors_san_ignacio_may2026.csv.")
            print("Sin ella LOCO se queda en dos pliegues y ninguno cubre un régimen frío, que es "
                  "justo donde el modelo tiene que demostrar que generaliza. Se avisa y se sigue, "
                  "pero el veredicto vale menos.")
        print(f"Validando (objetivo «{TARGET_FORMULATION}», campañas: "
              f"{', '.join(sorted(CAMPANAS))})…")
        v = validar_todo(CAMPANAS)
        print(f"\nEscrito {VALIDATION_FILE}\n")

        # One column per LOCO fold: each says "trained on the others, tested on THIS one". The
        # row's minimum is what decides, so it is printed separately instead of hiding behind a
        # mean.
        nombres = sorted(CAMPANAS)
        cab = "".join(f"{'fuera: ' + n:>14}" for n in nombres)
        print(f"{'modelo':28}{cab}{'MÍNIMO':>10}{'bloques':>14}")
        print("-" * (28 + 14 * len(nombres) + 24))
        for m in v["modelos"]:
            pl, b = m["loco"], m["bloques"]
            celdas = ""
            for n in nombres:
                p = pl["pliegues"].get(n)
                celdas += f"{p['skill_rmse']:+13.1%} " if p else f"{'—':>14}"
            mn = pl["skill_min"]
            nombre = f"{m['etiqueta']} a {m['horizon_h']:.0f} h"
            print(f"{nombre:28}{celdas}{mn:+9.1%} " if mn is not None
                  else f"{nombre:28}{celdas}{'—':>10}", end="")
            print(f"{b['skill_rmse']:+9.1%} ±{b['skill_sd']:.0%}")

        # What the model EMITTED over each fold's real readings. Not the same as what it COULD
        # emit given any input -- that is measured by pushing synthetic trajectories, in
        # `scripts/demo/predicciones.py` -- and confusing the two would promise coverage that has
        # not been measured.
        print("\nRango EMITIDO sobre los datos reales de los pliegues:")
        for m in v["modelos"]:
            ps = [p for p in m["loco"]["pliegues"].values() if p]
            if not ps:
                continue
            print(f"  {m['etiqueta']} a {m['horizon_h']:.0f} h: "
                  f"{min(p['pred_min'] for p in ps):.1f} … {max(p['pred_max'] for p in ps):.1f}")

        # What was measured outranks what was declared. When `SERVED_HORIZONS_H` stops matching
        # what the criterion says, this FAILS: it is the only way to stop the constant drifting
        # silently from the measurement that justifies it.
        medidos, declarados = v["servibles"], sorted(SERVED_HORIZONS_H)
        print(f"\nCriterio: {v['criterio']}")
        print(f"Horizontes servibles según la medición: {medidos} h")
        if medidos != [float(h) for h in declarados]:
            print(f"\n  SERVED_HORIZONS_H dice {declarados} y la medición dice {medidos}.\n"
                  "  Actualiza la constante Y su nota con los números de arriba, o explica ahí\n"
                  "  por qué se sirve algo que el criterio rechaza. No se publica una cosa\n"
                  "  midiendo otra.", file=sys.stderr)
            sys.exit(1)
        print("SERVED_HORIZONS_H coincide con la medición.")
        sys.exit(0)

    print(f"Datos: nov={len(nov)} filas (~{len(nov)*SAMPLE_MIN/60:.0f} h) + "
          f"feb={len(feb)} filas (~{len(feb)*SAMPLE_MIN/60:.0f} h)\n")
    for target, label in TARGETS.items():
        print(f"=== {label} — RF vs persistencia (train: nov+feb60% / test: feb40% final) ===")
        for h in (30, 60, 180):  # 1 h, 2 h, 6 h
            evaluate(nov, feb, target, h)
        _, r6 = evaluate(nov, feb, target, 180, verbose=False)
        print("  Variables más informativas (H=6h):",
              ", ".join(f"{n}={v:.2f}" for n, v in r6["importances"]))
        print()
