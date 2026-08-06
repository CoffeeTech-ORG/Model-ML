"""
CoffeeTech's FastAPI recommendation service.

Pulls telemetry and context (crop stage plus altitude) from the .NET/Azure backend, produces
recommendations with the agronomic rule engine (coffeetech_rules_v5) -- immediate ones over what
was observed and anticipated ones from the microclimate forecast (coffeetech_forecast) -- and
sends them back. The service only consumes the backend (GET in, POST out); it changes neither its
contracts nor its database.
"""
import asyncio
import os
import time
import math
import logging
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple

import httpx
import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Domain engine: the single source of truth for the diagnosis. Standard library only.
from coffeetech_rules_v5 import (
    COFFEE_ALTITUDE_MAX_M,
    COFFEE_ALTITUDE_MIN_M,
    CopperLedger,
    STAGE_UNKNOWN as V5_STAGE_UNKNOWN,
    VALID_STAGES as V5_VALID_STAGES,
    forecast_rules,
    reference_ranges,
)
from coffeetech_weather import elevation_for, get_forecast


def _coord(v):
    """A usable coordinate, or `None`.

    A 0.0 is not a valid coordinate here: it comes from an empty field, not from a farm in the
    Gulf of Guinea. Letting it through would request the forecast for a point at sea and return it
    looking exactly as convincing as a real one.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f == 0.0:          # NaN o campo vacío
        return None
    return f
from coffeetech_recommendations import (
    build_window_from_records,
    build_forecast_recommendations,
    build_recommendations,
    build_anticipated_recommendations,
    combine_recommendations,
    map_to_azure_payload,
    find_conventional,
    is_lab_reminder,
    select_for_sending,
)
from coffeetech_state import ReminderState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Loads the local .env (not versioned). Without python-dotenv installed it falls back to the
# system environment variables.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    logger.debug("python-dotenv no disponible; se usan las variables de entorno del sistema.")

# Optional internal scheduler: when enabled, the service triggers recommendation generation
# every N minutes, which suits an always-on host such as App Service with "Always On". Off by
# default; the endpoint can trigger a run just as well.
SCHEDULER_ENABLED = os.getenv("COFFEETECH_SCHEDULER_ENABLED", "0") == "1"
SCHEDULER_MINUTES = int(os.getenv("COFFEETECH_SCHEDULER_MINUTES", "30"))
_scheduler = None


async def _scheduled_run():
    """Automatic run: reuses the same logic as the endpoint."""
    try:
        logger.info("Scheduler: corrida automática de recomendaciones.")
        await trigger_recommendation_generation_and_send()
    except Exception as exc:
        logger.error(f"Scheduler: fallo en la corrida automática: {exc}", exc_info=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Starts (and stops) the internal scheduler when it is enabled."""
    global _scheduler
    if SCHEDULER_ENABLED:
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            _scheduler = AsyncIOScheduler()
            # max_instances=1 plus coalesce stop two runs overlapping or piling up.
            _scheduler.add_job(_scheduled_run, "interval", minutes=SCHEDULER_MINUTES,
                               max_instances=1, coalesce=True, id="recommendations")
            _scheduler.start()
            logger.info(f"Scheduler interno activo: cada {SCHEDULER_MINUTES} min.")
        except Exception as exc:
            logger.error(f"No se pudo iniciar el scheduler ({exc}); disparo solo por endpoint.")
    yield
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)


app = FastAPI(
    title="Coffee Agriculture Recommendation System API",
    description="Recomendaciones agrícolas para café (Perú): motor de reglas agronómicas v5 + pronóstico de microclima para alertas anticipadas.",
    version="5.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Configuration: backend endpoints and token ---
AZURE_RECOMMENDATION_ENDPOINT = os.getenv("AZURE_RECOMMENDATION_ENDPOINT", "https://coffeetech-netcoreappweb-f6hwc3fph9hndhhg.centralus-01.azurewebsites.net/api/v1/recommendations")

# The JWT is read from the environment only; a real token is never left as a default in the code,
# where it would be exposed in the repository. Missing, it stays None and the backend calls are
# skipped rather than failing abruptly.
JWT_TOKEN = os.getenv("AZURE_JWT_TOKEN")
if not JWT_TOKEN:
    logger.warning(
        "AZURE_JWT_TOKEN no configurado. Define la variable de entorno (o el archivo .env) "
        "antes de ejecutar; las llamadas al backend se omitirán hasta entonces."
    )

API_BASE_URL = os.getenv("COFFEETECH_API_BASE_URL", "https://coffeetech-netcoreappweb-f6hwc3fph9hndhhg.centralus-01.azurewebsites.net/api/v1")
DATA_RECORDS_URL = f"{API_BASE_URL}/data-records"; DEVICES_URL = f"{API_BASE_URL}/devices"
ASSIGNMENTS_URL = f"{API_BASE_URL}/assignments"; SECTIONS_URL = f"{API_BASE_URL}/sections"
FARMS_URL = f"{API_BASE_URL}/farms"

# Cache of the per-device context map (stage plus altitude), to avoid repeating GETs.
CACHE_DURATION_SECONDS = 300
device_stage_map_cache = {"data": None, "timestamp": 0}
# Qualitative signature of the last recommendation sent per device. It is resent only when the
# diagnosis changes, not every run, so the grower is not flooded with identical messages.
last_recommendation_signature: Dict[str, str] = {}
# Each reminder's cadence, anchored to the crop calendar rather than a generic pulse: soil
# analysis is recommended every ~2 years (Cenicafé AVT 214) and liming every 1-2 (AVT 466). They
# enter the payload only once that cadence has elapsed for that device, tracked in state persisted
# to disk so it survives restarts.
REMINDER_CADENCE_DAYS: Dict[str, float] = {
    "LAB_SOIL_ANALYSIS": float(os.getenv("COFFEETECH_SOIL_ANALYSIS_DAYS", "730")),
    "LAB_LIMING": float(os.getenv("COFFEETECH_LIMING_DAYS", "365")),
}
reminder_state = ReminderState()
# Days of telemetry per device handed to the engine: the inferential and water rules (R1-R11)
# need a time series, not just the latest reading.
WINDOW_DAYS = int(os.getenv("COFFEETECH_WINDOW_DAYS", "10"))

# Microclimate forecast, which feeds the anticipated alerts. Loaded tolerantly: without sklearn
# or the trained artifacts the service carries on with immediate recommendations only (the
# artifacts come from `python coffeetech_forecast.py train`).
try:
    from coffeetech_forecast import Forecaster, join_weather, window_from_records
    forecaster = Forecaster.load()
    if forecaster.available():
        logger.info(f"Pronóstico cargado (horizontes: {forecaster.horizons()} h).")
    else:
        logger.warning("Sin artefactos de pronóstico; solo recomendaciones inmediatas.")
except Exception as exc:
    logger.warning(f"Pronóstico no disponible ({exc}); solo recomendaciones inmediatas.")
    forecaster = None
    window_from_records = None
    join_weather = None

# The backend can report the stage with accents and different capitalisation; this normalises it
# to the internal key the engine understands.
PLANT_STAGE_MAPPING = {
    'plantula': 'plantula',
    'plántula': 'plantula',
    'Plantula': 'plantula',
    'Plántula': 'plantula',
    'vegetativo': 'vegetativo',
    'Vegetativo': 'vegetativo',
    'floracion': 'floracion',
    'floración': 'floracion',
    'Floracion': 'floracion',
    'Floración': 'floracion',
    'fructificacion': 'fructificacion',
    'fructificación': 'fructificacion',
    'Fructificacion': 'fructificacion',
    'Fructificación': 'fructificacion',
    'maduracion': 'maduracion',
    'maduración': 'maduracion',
    'Maduracion': 'maduracion',
    'Maduración': 'maduracion',
    'cosecha': 'cosecha',
    'Cosecha': 'cosecha',
}


def map_plant_stage(api_stage_name: Optional[str]) -> str:
    """Normalises the stage name arriving from the backend (accents, capitalisation) to one of the
    internal keys. Returns 'default' when it is not recognised."""
    if not api_stage_name or not isinstance(api_stage_name, str):
        return 'default'
    # Direct match
    internal_name = PLANT_STAGE_MAPPING.get(api_stage_name)
    if internal_name:
        return internal_name
    # Match ignoring case and accents
    def _norm(s: str) -> str:
        try:
            s = unicodedata.normalize('NFKD', s)
            s = ''.join(ch for ch in s if not unicodedata.combining(ch))
        except Exception:
            pass
        return s.lower().strip()
    norm_input = _norm(api_stage_name)
    for key, value in PLANT_STAGE_MAPPING.items():
        if _norm(key) == norm_input:
            return value
    logger.warning(f"Etapa no reconocida desde el backend: {api_stage_name}")
    return 'default'


async def fetch_api_data(client: httpx.AsyncClient, url: str) -> Optional[List[Dict]]:
    """Generic authorised (JWT) GET against the backend. Returns the parsed list, or None so the
    caller can stop when there is no data."""
    if not JWT_TOKEN or len(JWT_TOKEN) < 50:
        logging.error(f"JWT_TOKEN inválido/faltante. No se consulta {url}")
        return None
    headers = {"Authorization": f"Bearer {JWT_TOKEN}"}
    logger.debug(f"Consultando {url}")
    try:
        response = await client.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"OK {url}")
        return data if isinstance(data, list) else None
    except Exception as e:
        logging.error(f"Error consultando {url}: {e}")
        return None


async def get_device_stage_map() -> Dict[str, Dict[str, Optional[float]]]:
    """Builds (and caches) the deviceHubId -> context {stage, altitude, provenance} map by crossing
    the backend's devices, assignments, sections and farms."""
    global device_stage_map_cache
    current_time = time.time()
    if device_stage_map_cache["data"] and (current_time - device_stage_map_cache["timestamp"] < CACHE_DURATION_SECONDS):
        logging.info("Usando el mapa de contexto en caché.")
        return device_stage_map_cache["data"]

    logging.info("Refrescando el mapa de contexto desde el backend...")
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            fetch_api_data(client, DEVICES_URL),
            fetch_api_data(client, ASSIGNMENTS_URL),
            fetch_api_data(client, SECTIONS_URL),
            fetch_api_data(client, FARMS_URL),
            return_exceptions=True,
        )
    devices_data, assignments_data, sections_data, farms_data = results

    if isinstance(devices_data, Exception) or not devices_data:
        logging.error(f"Falló la consulta de dispositivos: {devices_data}")
        return {}
    if isinstance(assignments_data, Exception) or not assignments_data:
        logging.error(f"Falló la consulta de asignaciones: {assignments_data}")
        return {}
    if isinstance(sections_data, Exception) or not sections_data:
        logging.error(f"Falló la consulta de secciones: {sections_data}")
        return {}
    if isinstance(farms_data, Exception) or farms_data is None:
        logging.error(f"Falló la consulta de fincas: {farms_data}")
        farms_data = []

    try:
        dev_df = pd.DataFrame(devices_data)[['id', 'deviceHubId']].rename(columns={'id': 'deviceId'})
        asg_df = pd.DataFrame(assignments_data)[['deviceId', 'sectionId']]
        sections_df = pd.DataFrame(sections_data)
        # The authoritative crop stage lives on the section; several key names are tried.
        stage_candidates = ['type', 'stage', 'plantStage', 'phenologicalStage', 'stageName', 'phenologyStage']
        stage_col = next((col for col in stage_candidates if col in sections_df.columns), None)
        sec_cols = ['id']
        if stage_col:
            sec_cols.append(stage_col)
        if 'farmId' in sections_df.columns:
            sec_cols.append('farmId')
        sec_df = sections_df[sec_cols].rename(columns={'id': 'sectionId'})
        if stage_col:
            sec_df = sec_df.rename(columns={stage_col: 'section_stage_raw'})
        else:
            sec_df['section_stage_raw'] = None
        sec_df['api_plant_stage'] = sec_df['section_stage_raw']
        sec_df['stage_source'] = stage_col or None

        farms_df = pd.DataFrame(farms_data) if farms_data else pd.DataFrame(columns=['id', 'altitude'])
        altitude_candidates = ['altitude', 'altitudeMeters', 'altitude_m', 'altitudeMasl', 'altitude_masl', 'elevation']
        altitude_col = next((col for col in altitude_candidates if col in farms_df.columns), None)
        # Coordinates are needed for the forecast. A farm without them is not queried -- an
        # invented coordinate looks exactly as convincing as a real one -- and goes without
        # anticipated alerts, while keeping the full diagnosis over what was measured.
        lat_col = next((c for c in ('latitude', 'lat') if c in farms_df.columns), None)
        lon_col = next((c for c in ('longitude', 'lon', 'lng') if c in farms_df.columns), None)
        if altitude_col:
            cols = ['id', altitude_col] + [c for c in (lat_col, lon_col) if c]
            farms_df = farms_df[cols].rename(columns={'id': 'farmId', altitude_col: 'altitude',
                                                      **({lat_col: 'latitude'} if lat_col else {}),
                                                      **({lon_col: 'longitude'} if lon_col else {})})
            farms_df['altitude_source'] = altitude_col
            for c in ('latitude', 'longitude'):
                if c not in farms_df.columns:
                    farms_df[c] = None
        else:
            if not farms_df.empty:
                logging.warning("El payload de fincas no trae altitud; se omitirá ese campo.")
            farms_df = farms_df[['id']].rename(columns={'id': 'farmId'}) if 'id' in farms_df.columns else pd.DataFrame(columns=['farmId'])
            farms_df['altitude'] = None
            farms_df['altitude_source'] = None
            farms_df['latitude'] = None
            farms_df['longitude'] = None

        merged = pd.merge(pd.merge(dev_df, asg_df, on='deviceId', how='left'), sec_df, on='sectionId', how='left')
        if 'farmId' in merged.columns:
            merged = pd.merge(merged, farms_df, on='farmId', how='left')
        else:
            merged['altitude'] = None
            merged['altitude_source'] = None
            merged['latitude'] = None
            merged['longitude'] = None

        device_stage_map: Dict[str, Dict[str, Optional[float]]] = {}
        for _, row in merged.iterrows():
            hub_id = row.get('deviceHubId')
            if pd.isna(hub_id):
                continue
            raw_stage = row.get('section_stage_raw')
            stage_value = map_plant_stage(raw_stage)
            stage_source = row.get('stage_source')
            altitude_value = row.get('altitude') if 'altitude' in row else None
            altitude_source = row.get('altitude_source')
            if pd.isna(altitude_value):
                altitude_value = None
            try:
                altitude_int = int(round(float(altitude_value))) if altitude_value is not None else None
            except (TypeError, ValueError):
                altitude_int = None

            # One hub can appear in several rows; stage and altitude are filled in without an
            # empty value overwriting one already resolved.
            entry = device_stage_map.get(str(hub_id))
            if not entry:
                entry = {
                    'stage': stage_value if stage_value else 'default',
                    'raw_stage': raw_stage,
                    'stage_source': stage_source,
                    'section_id': row.get('sectionId'),
                    'farm_id': row.get('farmId'),
                    'altitude': altitude_int,
                    'altitude_raw': altitude_value,
                    'altitude_source': altitude_source,
                    'latitude': _coord(row.get('latitude')),
                    'longitude': _coord(row.get('longitude')),
                }
            else:
                if entry.get('stage') in (None, 'default') and stage_value and stage_value != 'default':
                    entry['stage'] = stage_value
                    entry['raw_stage'] = raw_stage
                    entry['stage_source'] = stage_source
                    entry['section_id'] = row.get('sectionId')
                if entry.get('altitude') is None and altitude_int is not None:
                    entry['altitude'] = altitude_int
                    entry['altitude_raw'] = altitude_value
                    entry['altitude_source'] = altitude_source
                    entry['farm_id'] = row.get('farmId')
                if entry.get('latitude') is None:
                    entry['latitude'] = _coord(row.get('latitude'))
                    entry['longitude'] = _coord(row.get('longitude'))
            device_stage_map[str(hub_id)] = entry

        if device_stage_map:
            device_stage_map_cache = {"data": device_stage_map, "timestamp": current_time}
            logging.info(f"Mapa de contexto cacheado para {len(device_stage_map)} dispositivos.")
        else:
            logging.warning("El mapa de contexto generado quedó vacío.")
        return device_stage_map
    except Exception as exc:
        logging.error(f"Error procesando los dataframes del mapa: {exc}", exc_info=True)
        return {}


async def get_sensor_records_by_device() -> Dict[str, List[Dict[str, Any]]]:
    """Fetches every recent record (the same GET as always) and groups them into a per-device window
    ordered by time. The engine needs the series of readings, not just the last one, to evaluate
    its inferential and water rules."""
    logging.info("Trayendo ventanas de telemetría desde el backend...")
    async with httpx.AsyncClient() as client:
        data_records = await fetch_api_data(client, DATA_RECORDS_URL)
    if not data_records:
        logging.warning("No se recibieron registros para armar ventanas.")
        return {}
    try:
        df = pd.DataFrame(data_records)
        if df.empty or 'deviceHubId' not in df.columns:
            logging.warning("Los registros no traen deviceHubId; no se pueden armar ventanas.")
            return {}
        ts_candidates = ['createdAt', 'timestamp', 'recordDate', 'created_at', 'updatedAt']
        ts_col = next((col for col in ts_candidates if col in df.columns), None)
        result: Dict[str, List[Dict[str, Any]]] = {}
        if ts_col:
            # Sorted by time and, per device, trimmed to the last WINDOW_DAYS.
            df['_ts'] = pd.to_datetime(df[ts_col], errors='coerce', utc=True)
            df = df.dropna(subset=['_ts']).sort_values('_ts')
            kept = []
            for _dev, group in df.groupby('deviceHubId'):
                cutoff = group['_ts'].max() - pd.Timedelta(days=WINDOW_DAYS)
                kept.append(group[group['_ts'] >= cutoff])
            if kept:
                df = pd.concat(kept)
            df = df.drop(columns=['_ts'])
        for dev, group in df.groupby('deviceHubId'):
            if pd.isna(dev):
                continue
            result[str(dev)] = group.to_dict('records')
        logging.info(f"Ventanas armadas para {len(result)} dispositivos.")
        return result
    except Exception as e:
        logging.error(f"Error armando las ventanas de telemetría: {e}", exc_info=True)
        return {}


# --- Endpoints ---
@app.post("/trigger_recommendation_generation/")
async def trigger_recommendation_generation_and_send(refresh_map: bool = False):
    """Orchestrator: fetches context and telemetry windows (the same GETs as always), runs the rule
    engine, assembles the recommendations and sends them to the backend under the same payload
    contract. The backend is not modified at any point."""
    logger.info("Generando recomendaciones (motor de reglas v5)...")
    try:
        if refresh_map:
            device_stage_map_cache["data"] = None
            device_stage_map_cache["timestamp"] = 0
        device_context_map, records_by_device = await asyncio.gather(
            get_device_stage_map(),
            get_sensor_records_by_device(),
        )
    except Exception as exc:
        logger.error(f"Falló la obtención de datos del backend: {exc}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Error obteniendo datos del backend: {exc}")

    if not records_by_device:
        return {"message": "No new sensor data found."}

    # (device_hub_id, payload, content signature, does it carry a lab reminder?) ready to send
    # once deduplication and the reminder cadence have had their say.
    to_send: List[Tuple[str, Dict[str, str], str, bool]] = []

    for device_hub_id, records in records_by_device.items():
        if not device_hub_id or not records:
            continue

        # Context (crop stage plus altitude) from the backend's map.
        stage_info = device_context_map.get(device_hub_id, {}) if device_context_map else {}
        if isinstance(stage_info, dict):
            stage_raw = stage_info.get('stage', 'default') or 'default'
            altitude_value = stage_info.get('altitude')
        else:
            stage_raw = stage_info or 'default'
            altitude_value = None
        # NO fallback stage here, and 'vegetativo' in particular would be the wrong one: it is the
        # flower induction stage, where the water rule says there is no need to irrigate because
        # the dry spell helps flowering set evenly. A section in grain filling whose stage name
        # fails to map would get that advice, which is the opposite of correct, with nothing on
        # screen to show a guess was made.
        #
        # The sentinel flows through instead: no stage-gated rule recognises it, water management
        # degrades to its neutral branch -- mulching, valid in any stage -- and the engine says so
        # in the technician's voice.
        stage_value = stage_raw if stage_raw in V5_VALID_STAGES else V5_STAGE_UNKNOWN
        if stage_value == V5_STAGE_UNKNOWN:
            logger.warning(
                f"{device_hub_id}: etapa no registrada o no reconocida ('{stage_raw}'). Se omiten "
                f"las reglas condicionadas por etapa y se declara en la recomendación.")
        # Altitude: declared, derived from the coordinate, or nothing at all.
        #
        # NO numeric fallback, because there is no neutral altitude to fall back to: altitude
        # decides the band, and the band sets the weight of four pest and disease rules. A
        # mid-range guess lands in `medium`, which raises American leaf spot to `high`, so a farm
        # with no altitude would receive an invented modulation wearing the face of data.
        #
        # The order is: the declared one wins; without it the altitude is derived from the
        # coordinate with the 90 m DEM -- the same one the downscaling uses, so it is consistent by
        # construction -- and without coordinates either, the run happens with NO altitude
        # modulation, which is the honest answer when the farm's location is unknown.
        #
        # An altitude that cannot belong to a coffee plot is treated as UNKNOWN rather than as a
        # measurement. The database held farms with `altitude = 0` because the column was NOT NULL
        # and the form did not really require it; that 0 lands in band `low`, the profile of
        # MAXIMUM weight for rust and borer. The backend can no longer send it -- the column is
        # nullable now -- but this guard closes the door on old rows and on other clients.
        try:
            altitude_value = float(altitude_value) if altitude_value is not None else None
        except (TypeError, ValueError):
            altitude_value = None
        if altitude_value is not None and not (COFFEE_ALTITUDE_MIN_M <= altitude_value
                                               <= COFFEE_ALTITUDE_MAX_M):
            logger.warning(f"Dispositivo {device_hub_id}: altitud declarada {altitude_value:.0f} m "
                           f"fuera del rango del café ({COFFEE_ALTITUDE_MIN_M:.0f}-"
                           f"{COFFEE_ALTITUDE_MAX_M:.0f} m). Se trata como desconocida.")
            altitude_value = None
        if altitude_value is None and isinstance(stage_info, dict):
            altitude_value = elevation_for(stage_info.get('latitude'), stage_info.get('longitude'))
            if altitude_value is not None:
                logger.info(f"Dispositivo {device_hub_id}: altitud {altitude_value:.0f} m derivada "
                            "de las coordenadas (DEM 90 m); el backend no la traía.")
        if altitude_value is None:
            logger.warning(f"Dispositivo {device_hub_id}: sin altitud declarada ni coordenadas. "
                           "Se diagnostica SIN modulación por altitud.")

        logger.info(f"Dispositivo {device_hub_id}: etapa={stage_value}, altitud={altitude_value}, lecturas={len(records)}")

        # Window -> immediate recommendations -> (forecast -> anticipated) -> payload.
        try:
            window = build_window_from_records(records, stage=stage_value, altitude=altitude_value)
            recommendations = build_recommendations(
                window, stage=stage_value, altitude=altitude_value, copper_ledger=CopperLedger(),
            )

            # The regional forecast is requested ONCE and both blocks below use it.
            #
            # Fetching it for the anticipated rules alone would leave the microclimate model MUTE:
            # the 6 h models carry `api_cloud_cover` among their features, and without that column
            # `latest_feature_row` returns `None` on purpose, so as not to predict with a vector
            # other than the one it learned on. `forecast_climate` would then return `None` on
            # every run and no model-based alert could ever be raised -- silently, since a model
            # that declines to predict looks the same as one with nothing to say.
            lat = _coord(stage_info.get('latitude')) if isinstance(stage_info, dict) else None
            lon = _coord(stage_info.get('longitude')) if isinstance(stage_info, dict) else None
            pronostico = None
            if lat is not None and lon is not None and window:
                try:
                    pronostico = await get_forecast(lat, lon, altitude_value)
                except Exception as wexc:
                    logger.warning(f"No se pudo obtener el pronóstico para {device_hub_id}: {wexc}")

            # Anticipated alerts: T and RH are forecast at each horizon and the climate rules
            # are evaluated over those future readings. With no forecast available, only the
            # immediate recommendations remain.
            if forecaster is not None and forecaster.available() and window:
                try:
                    # Cloud cover is joined with THE SAME function training uses, so the feature
                    # vector at prediction is the one it learned on.
                    forecast_window = join_weather(window_from_records(records), pronostico)
                    forecasts = []
                    for horizon in forecaster.horizons():
                        climate = forecaster.forecast_climate(forecast_window, horizon)
                        if climate:
                            climate["horizon_h"] = horizon
                            forecasts.append(climate)
                    if forecasts:
                        anticipated = build_anticipated_recommendations(window[-1], forecasts)
                        recommendations = combine_recommendations(recommendations, anticipated)
                except Exception as fexc:
                    logger.warning(f"No se pudo pronosticar para {device_hub_id}: {fexc}")

            # ANTICIPATED alerts over the regional forecast (24-72 h). A different thing from
            # the block above: those extend the sensor's own microclimate a few hours, these read
            # Open-Meteo, which is the only thing reaching several days out. Reuses the
            # `pronostico` already fetched -- without a coordinate it is `None` and this block does
            # not run.
            if pronostico is not None and window:
                try:
                    anticipadas = forecast_rules(
                        window, pronostico, stage=stage_value, altitude=altitude_value,
                        # The spray window only appears when there is already something to
                        # apply. Otherwise it is daily advice nobody asked for -- measured: there
                        # is a window on 82 % of days.
                        pending_application=bool(recommendations),
                    )
                    if anticipadas:
                        recommendations = combine_recommendations(
                            recommendations, build_forecast_recommendations(anticipadas))
                except Exception as wexc:
                    # The forecast is an extra: when it fails, the diagnosis over what was measured still goes out.
                    logger.warning(f"Sin alertas anticipadas para {device_hub_id}: {wexc}")

            # Organic certification safeguard: a conventional synthetic must never slip through.
            conventional = find_conventional(recommendations)
            if conventional:
                logger.error(
                    f"Bloqueo orgánico: términos convencionales {conventional} en {device_hub_id}; "
                    f"revisar catálogo. No se envía."
                )
                continue

            # Content deduplication plus the lab reminder's cadence: a resend happens when the
            # diagnosis changes (an alert appears or disappears, its severity moves, or it becomes
            # anticipated) or when the reminder is due again. That reminder travels only every
            # LAB_REMINDER_DAYS days, not in every payload.
            due_reminders = {
                rid for rid, days in REMINDER_CADENCE_DAYS.items()
                if reminder_state.is_due(device_hub_id, rid, every_days=days)
            }
            selected, signature = select_for_sending(
                recommendations,
                last_signature=last_recommendation_signature.get(device_hub_id),
                due_reminders=due_reminders,
            )
            if selected is None:
                logger.info(f"Se omite {device_hub_id}: el diagnóstico no cambió desde el último envío.")
                continue

            sent_reminders = {r.rule_id for r in selected if is_lab_reminder(r)}
            payload = map_to_azure_payload(selected, device_hub_id)
            to_send.append((device_hub_id, payload, signature, sent_reminders))
        except Exception as exc:
            logger.error(f"Error generando recomendaciones para {device_hub_id}: {exc}", exc_info=True)
            continue

    sent_count = 0
    error_count = 0
    results_summary = []

    if not to_send:
        return {"message": "No new sensor data found."}
    if not JWT_TOKEN or len(JWT_TOKEN) < 50:
        return {
            "message": "Recomendaciones generadas pero NO ENVIADAS (JWT inválido).",
            "status": "send_config_error",
        }

    headers = {"Authorization": f"Bearer {JWT_TOKEN}", "Content-Type": "application/json"}
    logger.info(f"Enviando {len(to_send)} recomendaciones a {AZURE_RECOMMENDATION_ENDPOINT}")

    for dev_id, payload, signature, sent_reminders in to_send:
        # payload = {"recommendationDescription": <texto v5>, "deviceHubId": <id>}
        try:
            response = requests.post(AZURE_RECOMMENDATION_ENDPOINT, json=payload, headers=headers, timeout=15)
            if 200 <= response.status_code < 300:
                logger.info(f"Envío OK para {dev_id}.")
                sent_count += 1
                results_summary.append({"device": dev_id, "status": "sent"})
                # The signature is stored only once the backend confirms receipt, so a change of
                # diagnosis is not lost to a failed send. Same for the reminder: if the send fails
                # it stays due and is retried later.
                last_recommendation_signature[dev_id] = signature
                for rid in sent_reminders:
                    reminder_state.mark_sent(dev_id, rid)
            else:
                logger.error(f"Error de envío para {dev_id}. Estado: {response.status_code}, Resp: {response.text[:300]}")
                error_count += 1
                results_summary.append({"device": dev_id, "status": "error", "code": response.status_code})
        except Exception as exc:
            logger.error(f"Excepción de envío para {dev_id}: {exc}")
            error_count += 1
            results_summary.append({"device": dev_id, "status": "exception"})

    final_message = (
        f"Proceso completo (motor de reglas v5). Generadas: {len(to_send)}. Enviadas: {sent_count}. Errores: {error_count}."
    )
    logger.info(final_message)
    status = "success" if error_count == 0 else ("partial_success" if sent_count > 0 else "send_failed")
    return {"message": final_message, "status": status, "details": results_summary}


@app.get("/health")
def health_check():
    """Quick check: token configured, diagnosis engine, and forecast status."""
    token_ok = bool(JWT_TOKEN and len(JWT_TOKEN) > 50)
    forecast_ok = bool(forecaster is not None and forecaster.available())
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "token_configured": token_ok,
        "diagnosis_engine": "coffeetech_rules_v5",
        "forecast_enabled": forecast_ok,
        "forecast_horizons_h": forecaster.horizons() if forecast_ok else [],
    }


@app.get("/reference-ranges")
def get_reference_ranges(stage: Optional[str] = None):
    """The reference ranges the diagnosis engine applies.

    Consumed by the frontend's Reports chart, so it does not need a table of its own. A second copy
    ends up contradicting the diagnosis on screen -- painting as out of range a potassium the
    engine calls adequate -- and the user has no way to tell which view is right.

    `stage` is optional; when it is not a valid stage it is ignored and the base reading returned.
    """
    return {
        "version": app.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **reference_ranges(stage),
    }


@app.get("/debug/device-stage-map")
async def debug_device_stage_map(refresh: bool = False):
    """Exposes the cached context map so the stage and altitude, with their provenance, can be
    checked straight from the API."""
    try:
        if refresh:
            device_stage_map_cache["data"] = None
            device_stage_map_cache["timestamp"] = 0
        mapping = await get_device_stage_map()
        # DataFrames leave numpy/pandas types and NaNs that are not valid JSON; they are cleaned.
        def _json_safe(value):
            if isinstance(value, dict):
                return {k: _json_safe(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_json_safe(v) for v in value]
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    return None
            if isinstance(value, (np.floating, np.integer)):
                value = value.item()
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    return None
                return value
            if hasattr(value, "isoformat"):
                try:
                    return value.isoformat()
                except Exception:
                    return str(value)
            if value is pd.NA:
                return None
            return value
        return {"count": len(mapping), "mapping": _json_safe(mapping)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error construyendo el mapa de contexto: {exc}")
