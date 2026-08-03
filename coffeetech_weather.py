# -*- coding: utf-8 -*-
"""
CoffeeTech — external weather forecast (Open-Meteo).

The hub sensor measures the plot; this looks ahead. Current conditions keep coming from the hub,
because that is what is measured on site, and nothing here overrides them. Two consumers: the
microclimate model's forecast features (`coffeetech_forecast`), and the anticipated rules, which
warn of a rust infection window before it opens, of rain before it leaches the fertiliser, and of
whether there is a spray window today.

Requests are in UTC, not Peru time. Open-Meteo would return local stamps with
`timezone=America/Lima`, but they would have to be converted back to cross with telemetry, which
is UTC -- and a conversion that lands the wrong way shifts the whole series five hours with no
error to show for it. Local time is computed only when a window
is shown to a user.

Elevation is mandatory. The grid cell is about 9 km and the relief here is steep. Open-Meteo
statistically downscales with a 90 m elevation model when `elevation` is passed, and
`farms.altitude` has it. Asking for 1823 m returns `elevation: 1823.0`, and the lapse it applies
matches the standard ~0.65 °C/100 m: 0.8 °C over 121 m (0.66 °C/100 m), 0.10 °C over the 14 m
between the pin and the declared altitude.

Without `elevation` the API returns 1809 m for this coordinate. That is not the cell's mean
orography -- it is the 90 m elevation model AT THE POINT, confirmed by `/v1/elevation` giving the
same 1809 m. The four neighbours a kilometre out read 1645, 1634, 1581 and 1540 m, so the terrain
spans 1540-1809 m within one kilometre. At that relief downscaling is not a refinement; it decides
which hillside is being read.

A farm with no coordinate is not queried. There is nothing to point at, and an invented coordinate
looks exactly as convincing as a real one. Those farms keep the sensor's own short-horizon
forecast, which does not need to know where it is.

Measured against the three pilot campaigns at the corrected 1823 m, the difference between sensor
and regional model is NOT a fixed bias that can be subtracted:

    sensor − regional model (temp.)  mean    range        daily swing    min at     max at
      nov 2025  (n=50)               +1.9 °C  −2.6..+5.7   5.7 (max 7.1)  −0.9 (01h) +5.1 (18h)
      feb 2026  (n=103)              +6.1 °C  +0.2..+9.7   6.4 (max 8.8)  +4.7 (02h) +8.2 (19h)
      may 2026  (n=589)              −0.2 °C  −6.2..+8.2   7.1 (max  — )  −1.8 (01h) +2.2 (19h)

What the three share is the SHAPE: minimum around 01-02 h local, maximum around 18-19 h, and a
within-day swing of 5.7 to 7.1 °C. The LEVEL changes by campaign -- the means span 6.3 °C, from
−0.2 to +6.1. An altitude bias would be constant on both axes and produces neither. The likely
cause is sensor siting or shielding; the available data cannot settle it, so it is declared rather
than corrected.

The consequence for the rules:

  * ABSOLUTE API temperatures cannot be compared against a threshold. A rule asking "will it be
    between 22 and 24 °C?" can be several degrees off on this plot. Thresholds of that kind are
    evaluated against the SENSOR.
  * Timing and change do hold: when rain arrives and how much, wind, cloud cover, radiation.
    Either the sensor does not measure them, or they are questions about the future that no local
    measurement can answer. There the API is the only source, and that is what it is for.

Open-Meteo data, CC BY 4.0 -- attribution lives in the interface.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

API_URL = "https://api.open-meteo.com/v1/forecast"

#: Historical reanalysis, needed for TRAINING: the november and february campaigns are in the past,
#: so their weather is in the archive rather than the forecast. Same grid and same elevation
#: correction, so training and serving features come from one source instead of the model learning
#: one distribution and living in another.
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

#: The archive product, PINNED. Open-Meteo defaults to the ECMWF IFS operational reanalysis at 9 km
#: since 2017, not ERA5 -- requesting the same day with an explicit `models=` shows the default
#: matching `ecmwf_ifs` and differing ~3 °C from `era5`. Pinned so a change of default cannot move
#: the series the rain percentiles come from. Measured over 2023-2025 at 1823 m: ECMWF IFS gives
#: 861 mm/year, ERA5 2442, and ERA5-Land does not cover the period at all (returns 0). That is a
#: factor of three on the thresholds.
ARCHIVE_MODEL = "ecmwf_ifs"

#: First date with ECMWF IFS in the archive. Before it there is only ERA5-Land, and the series
#: stops being homogeneous with the one that calibrated the thresholds.
ARCHIVE_IFS_SINCE = "2017-01-01"

# What the sensor does not measure and the rules need, plus what it does measure so anomalies can
# be computed.
HOURLY_VARS = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "shortwave_radiation",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "wind_speed_10m",
    "cloud_cover",
)

#: The archive does not publish precipitation probability -- it is a forecast output, not a
#: reanalysis one -- so these are the features common to training and serving. Probability is used
#: in rules only, never as a model feature, so the vector cannot differ between the two.
HOURLY_VARS_ARCHIVE = tuple(v for v in HOURLY_VARS if v != "precipitation_probability")

# The forecast refreshes hourly at the source; asking more often spends quota for nothing. Five
# farms come to 120 calls a day against a 10 000 limit.
CACHE_SECONDS = 3600

# Longest horizon the rules consume is 72 h. Four days are requested so a window closing at the
# edge of the third still has room.
FORECAST_DAYS = 4

#: Pilot coordinate and altitude: the grower's own pin for the El Milagro farm, San Ignacio,
#: Cajamarca.
#:
#: PRODUCTION does not use these. Every farm brings its own from the backend, and a farm without a
#: coordinate is not queried. They live here for the two things that need a fixed, known point:
#: training on the pilot campaigns and generating the rule-frequency report.
#:
#: 1823 m is the hub's altitude, and it DISAGREES with the `altitude_masl` column of the nov 2025
#: and feb 2026 CSVs, which say 1450. The CSVs are not edited -- that column records what was
#: loaded, and rewriting raw data is what makes an audit impossible -- so the value is corrected
#: here, where it is consumed, and anyone reading both should know which one governs.
#:
#: The gap is not cosmetic. It moves the altitude band from `medium` to `high`, which reweights
#: four rules -- coffee leaf rust and coffee berry borer drop from alert to warning and lose their
#: referral, Phoma is reinforced -- and it moves the downscaling by ~1.9 °C.
#:
#: The 90 m DEM reads 1809 m at this pin against the 1823 declared, so coordinate and altitude
#: agree to 14 m. `docs/VERIFICACION_FASE11.md` §1 carries the measurement.
PILOT_LAT, PILOT_LON = -5.137820098212263, -79.06301379203798
PILOT_ELEVATION = 1823.0

#: Elevation API. Resolves a coordinate's altitude against THE SAME 90 m elevation model the
#: forecast downscaling uses, so an altitude derived here is consistent with the thermal correction
#: by construction.
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

#: Gap beyond which the declared altitude and the coordinate are worth flagging as describing
#: different points. On a hillside, 100 m of drop is a few hundred metres horizontally.
ELEVATION_MISMATCH_M = 100.0

#: PAST days requested alongside the forecast. Without them the series starts at the current hour
#: and never overlaps telemetry, which lives in the past. That overlap is the only way to estimate
#: the sensor − regional model offset, and without the offset the anticipated rust rule cannot fire
#: at all, because its threshold lives on the sensor's scale -- and it would fail silently, looking
#: like a threshold nothing reaches.
#:
#: 91 and not fewer: `R11_FLOWERING_EXPECTED` has to know whether an inductive rain event happened
#: in the previous 90 days -- see `FORECAST["flowering_gap_days"]` -- and a shorter past cannot
#: answer that.
#:
#: The cost, measured: the same request goes from 9 KB and 144 hours to 129 KB
#: and 2280 hours, same response time (~1.5 s), same endpoint and same tariff. Open-Meteo documents
#: a 92 day maximum and accepted 93 in the test, so 91 leaves room. The response is cached
#: (`CACHE_SECONDS`), so the size is paid once per farm per cache window, not per evaluation.
PAST_DAYS = 91

_cache: Dict[str, Dict[str, Any]] = {}


@dataclass
class HourlyForecast:
    """Normalised hourly forecast. Every stamp is UTC with an explicit timezone."""

    latitude: float
    longitude: float
    #: The altitude Open-Meteo actually used. If it differs from the one requested, downscaling did
    #: not happen and the temperature carries the cell's bias.
    elevation: float
    times: List[datetime]
    values: Dict[str, List[Optional[float]]]

    def __len__(self) -> int:
        return len(self.times)

    def series(self, name: str) -> List[Optional[float]]:
        return self.values.get(name, [])

    def window(self, hours_ahead: int, span_hours: int,
               now: Optional[datetime] = None) -> "HourlyForecast":
        """Slices the window starting `hours_ahead` from now and lasting `span_hours`.

        This is what the anticipated rules consume: "is there a 6 h leaf wetness window in the next
        48?". The cut is made on wall-clock time rather than indices, because the series can start
        at a different hour depending on when it was requested.

        `now` pins the reference instant. Production omits it and reads the clock, but the
        firing-frequency report replays past days and has to say "stand on 3 march". With the clock
        hardcoded that report -- which is the validation deciding whether a threshold applies in
        San Ignacio -- could not be produced.
        """
        now = now or datetime.now(timezone.utc)
        desde = now.timestamp() + hours_ahead * 3600
        hasta = desde + span_hours * 3600
        idx = [i for i, t in enumerate(self.times) if desde <= t.timestamp() < hasta]
        return HourlyForecast(
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=self.elevation,
            times=[self.times[i] for i in idx],
            values={k: [v[i] for i in idx] for k, v in self.values.items()},
        )


def _cache_key(lat: float, lon: float, elevation: Optional[float]) -> str:
    # Rounded to 3 decimals (~110 m), so two plots on the same farm share a forecast. That is the
    # right answer when the cell is kilometres across.
    return f"{lat:.3f},{lon:.3f},{elevation}"


def _parse(payload: Dict[str, Any], variables: tuple = HOURLY_VARS) -> HourlyForecast:
    hourly = payload.get("hourly") or {}
    crudas = hourly.get("time") or []

    # Open-Meteo returns ISO with no timezone suffix ("2026-08-03T00:00") even when UTC is
    # requested. Stamping it explicitly: naive datetimes are how silent offsets start.
    times = [
        datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        for t in crudas
    ]

    return HourlyForecast(
        latitude=float(payload.get("latitude", 0.0)),
        longitude=float(payload.get("longitude", 0.0)),
        elevation=float(payload.get("elevation", 0.0)),
        times=times,
        values={v: hourly.get(v, []) for v in variables},
    )


#: Environment variable enabling the SCRIPTED FEED. Points at a JSON with the shape Open-Meteo
#: returns.
#:
#: Nine of the engine's 34 rules are anticipated: they read the forecast. They cannot be triggered
#: by inserting readings into the database, because they depend on the weather of the day someone
#: looks. For a demonstration before an examining board that is not enough -- either all of them
#: can be shown or the system cannot be claimed to cover them.
#:
#: With the variable set, `get_forecast` serves the file and never calls the API. Without it this
#: branch never runs: no default value, no file by convention, and the first use shouts in the log
#: so a scripted run can never be mistaken for a real one. Same discipline as the rest of the
#: project: what is simulated gets declared.
DEMO_FORECAST_ENV = "COFFEETECH_DEMO_FORECAST"

_demo_avisado = False


def forecast_guionizado() -> Optional[HourlyForecast]:
    """The file's forecast when the scripted feed is on, `None` when it is not.

    Hours may be given as relative offsets (`+3` = three hours from now) as well as absolute
    stamps. Without that, a script written today expires tomorrow and the demonstration stops
    working exactly when it is needed.
    """
    global _demo_avisado
    ruta = os.environ.get(DEMO_FORECAST_ENV)
    if not ruta:
        return None
    try:
        with open(ruta, encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        # Fails on purpose. If someone asked for the scripted feed and the file is missing,
        # serving the real forecast is the worst outcome: the demonstration would appear to work
        # while measuring something else.
        raise RuntimeError(f"{DEMO_FORECAST_ENV}={ruta} no se pudo leer: {exc}") from exc

    horas = (payload.get("hourly") or {}).get("time") or []
    if horas and isinstance(horas[0], str) and horas[0].lstrip("+-").isdigit():
        ahora = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        payload["hourly"]["time"] = [
            (ahora + timedelta(hours=int(h))).strftime("%Y-%m-%dT%H:%M") for h in horas
        ]

    pronostico = _parse(payload)
    if not _demo_avisado:
        logger.warning(
            "PRONÓSTICO GUIONIZADO ACTIVO (%s=%s, %d horas). Las reglas anticipadas NO están "
            "leyendo el tiempo real. Quita la variable para volver a producción.",
            DEMO_FORECAST_ENV, ruta, len(pronostico))
        _demo_avisado = True
    return pronostico


async def get_forecast(
    latitude: Optional[float],
    longitude: Optional[float],
    elevation: Optional[float] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Optional[HourlyForecast]:
    """Hourly forecast for a farm. `None` when it cannot or should not be requested.

    `None` is a valid and expected answer: no coordinate means no forecast, and the caller has to
    keep working without one. No anticipated rule may assume this returns data.
    """
    guionizado = forecast_guionizado()
    if guionizado is not None:
        return guionizado

    if latitude is None or longitude is None:
        logger.debug("Finca sin coordenada: no se consulta el pronóstico.")
        return None

    clave = _cache_key(latitude, longitude, elevation)
    entrada = _cache.get(clave)
    if entrada and time.time() - entrada["timestamp"] < CACHE_SECONDS:
        return entrada["data"]

    params: Dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": FORECAST_DAYS,
        "past_days": PAST_DAYS,
        "timezone": "UTC",
    }
    if elevation is not None:
        params["elevation"] = elevation

    propio = client is None
    client = client or httpx.AsyncClient()
    try:
        respuesta = await client.get(API_URL, params=params, timeout=30.0)
        respuesta.raise_for_status()
        pronostico = _parse(respuesta.json())
    except Exception as exc:
        # The forecast is an extra: when it fails, the diagnosis on what was measured still runs.
        logger.warning(f"No se pudo obtener el pronóstico para ({latitude}, {longitude}): {exc}")
        return None
    finally:
        if propio:
            await client.aclose()

    if elevation is not None and abs(pronostico.elevation - elevation) > 1.0:
        # Not fatal, but worth a warning: without downscaling the temperature stays at whatever the
        # 90 m DEM gives for the point (1809 m here), and on this hillside that is several degrees.
        logger.warning(
            f"Open-Meteo usó {pronostico.elevation} m y se pidieron {elevation} m: "
            "el descenso de escala por altitud no se aplicó."
        )

    _cache[clave] = {"data": pronostico, "timestamp": time.time()}
    logger.info(
        f"Pronóstico obtenido para ({latitude}, {longitude}) a {pronostico.elevation} m: "
        f"{len(pronostico)} horas."
    )
    return pronostico


def get_archive(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    elevation: Optional[float] = None,
) -> Optional[HourlyForecast]:
    """Historical reanalysis for a date range (`YYYY-MM-DD`), synchronous.

    For TRAINING, not production: the telemetry campaigns already happened, so their weather is in
    the archive rather than the forecast. It goes through `urllib` with no cache because it is
    called a handful of times from the training script, never from the service.

    Same grid, same variables and same elevation correction as `get_forecast`. Training features
    have to come from where the served ones will come from; training on reanalysis and predicting
    on forecast would have the model learning one distribution and living in another.

    The Open-Meteo archive does NOT default to ERA5. It serves the ECMWF IFS operational reanalysis
    at 9 km from 2017 onward and falls back to ERA5-Land before that. Requesting the same day with
    an explicit `models=` shows the default matching `ecmwf_ifs` digit for digit and differing
    ~3 °C from `era5`.

    `models=ecmwf_ifs` is pinned so the product cannot change if Open-Meteo moves its default. The
    rain thresholds are percentiles of this series, and measured over 2023-2025 at 1823 m the IFS
    gives 861 mm/year against ERA5's 2442 -- nearly triple, so an unnoticed product change would
    triple the thresholds.

    Before 2017 there is no IFS: requesting it returns an empty series, so the default is left in
    place -- ERA5-Land -- with a warning, because the series is then NOT homogeneous with the one
    that calibrated the thresholds.
    """
    import json
    import urllib.parse
    import urllib.request

    params: Dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS_ARCHIVE),
        "timezone": "UTC",
    }
    if start_date >= ARCHIVE_IFS_SINCE:
        params["models"] = ARCHIVE_MODEL
    else:
        logger.warning(
            f"El histórico empieza en {start_date}, antes de {ARCHIVE_IFS_SINCE}: no hay "
            f"{ARCHIVE_MODEL} y Open-Meteo servirá ERA5-Land. La serie no es homogénea con la que "
            "calibró los umbrales de lluvia."
        )
    if elevation is not None:
        params["elevation"] = elevation

    url = f"{ARCHIVE_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = json.load(resp)
    except Exception as exc:
        logger.warning(f"No se pudo obtener el histórico {start_date}..{end_date}: {exc}")
        return None

    return _parse(payload, variables=HOURLY_VARS_ARCHIVE)


def elevation_for(latitude: Optional[float], longitude: Optional[float]) -> Optional[float]:
    """A coordinate's altitude from the 90 m elevation model. `None` when it cannot be resolved.

    It exists because of an expensive mistake: the pilot's altitude was typed as 1450 m when the
    hub sits at 1823, and that number propagated into the downscaling, into the altitude band --
    which sets the weight of four rules -- and into every threshold derived from the archive. A
    text field is the wrong place for a fact the map already holds.

    Uses THE SAME elevation model as the forecast downscaling, so an altitude derived here and the
    thermal correction refer to the same point.

    It does NOT replace the altitude the user declares: they know where the equipment went, and a
    90 m model smooths the relief. It is there to catch the mistyped one -- see `check_elevation`.
    """
    if latitude is None or longitude is None:
        return None
    clave = f"elev:{latitude:.4f},{longitude:.4f}"
    entrada = _cache.get(clave)
    if entrada:
        return entrada["data"]
    import json
    import urllib.parse
    import urllib.request
    url = f"{ELEVATION_URL}?{urllib.parse.urlencode({'latitude': latitude, 'longitude': longitude})}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            valores = json.load(resp).get("elevation") or []
        altitud = float(valores[0]) if valores else None
    except Exception as exc:
        # With no altitude from the map the declared one still stands: this is a check, not a
        # requirement.
        logger.warning(f"No se pudo resolver la altitud de ({latitude}, {longitude}): {exc}")
        return None
    _cache[clave] = {"data": altitud, "timestamp": time.time()}
    return altitud


def check_elevation(latitude: Optional[float], longitude: Optional[float],
                    declared: Optional[float]) -> Optional[float]:
    """Compares the declared altitude against the map's and warns on a mismatch. Returns the one to
    use.

    The declared one wins when it exists. A large gap does not mean the user is wrong: it means the
    altitude and the coordinate do not describe the same point, which matters on a hillside --
    around the pilot's pin the terrain spans 1540-1809 m within a kilometre. There the check passes,
    1809 m from the map against 1823 declared, 14 m apart.
    """
    del_mapa = elevation_for(latitude, longitude)
    if declared is None:
        return del_mapa
    if del_mapa is not None and abs(del_mapa - declared) > ELEVATION_MISMATCH_M:
        logger.warning(
            f"La altitud declarada ({declared:.0f} m) y la del modelo de elevación en esas "
            f"coordenadas ({del_mapa:.0f} m) difieren {abs(del_mapa - declared):.0f} m. "
            "Revisa que la coordenada apunte donde está el equipo."
        )
    return declared


def clear_cache() -> None:
    """Empties the cache. Used by the tests and by the endpoint's manual refresh."""
    _cache.clear()
