"""FastAPI service combining Random Forest outputs with agronomic expert rules."""
import asyncio
import json
import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timedelta, timezone
import requests
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Union, Tuple
import logging
import time
import joblib
import math
import numpy as np
import unicodedata

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Coffee Agriculture Recommendation System API",
    description="API para recomendaciones agrícolas en café (Perú) usando Sensores, Conocimiento Experto Detallado y Random Forest (v3.2 - Experto Integrado).",
    version="3.2.0"
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Pydantic Models ---
class SensorDataInput(BaseModel):
    device_hub_id: str; air_humidity_percent: float; soil_humidity_percent: float
    celcius_grade_temperature: float; precipitation_detected: int
    nitrogen_mg_kg: Optional[float] = None; phosphorus_mg_kg: Optional[float] = None
    potassium_mg_kg: Optional[float] = None; plant_stage: Optional[str] = None
    altitude_masl: Optional[int] = None
    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))

# Modelo Pydantic ACTUALIZADO para producto/acción
class RecommendationProduct(BaseModel):
    name: str
    composition: Optional[str] = None
    dose: Optional[str] = None
    method: Optional[str] = None
    timing_frequency: Optional[str] = None
    stage_applicability: Optional[List[str]] = Field(default_factory=list) # Lista de etapas donde aplica
    notes: Optional[str] = None
    distributor: Optional[str] = None
    senasa_registro: Optional[str] = None

class RecommendationOutput(BaseModel):
    device_hub_id: str; recommendation_description: str; recommendation_type: str
    condition_detected: str; urgency_level: int
    condition_code: Optional[str] = None  # Valor listo para UI (alias del detectado)
    condition_internal_code: Optional[str] = None  # Etiqueta técnica para integraciones
    specific_products: List[RecommendationProduct] = [] # Lista estructurada con detalles
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    based_on_data_at: Optional[datetime] = None

# --- Constantes y Configuración ---
# Endpoints y Token
AZURE_RECOMMENDATION_ENDPOINT = os.getenv("AZURE_RECOMMENDATION_ENDPOINT", "https://coffeetech-netcoreappweb-f6hwc3fph9hndhhg.centralus-01.azurewebsites.net/api/v1/recommendations")
JWT_TOKEN = os.getenv("AZURE_JWT_TOKEN", "eyJhbGciOiJodHRwOi8vd3d3LnczLm9yZy8yMDAxLzA0L3htbGRzaWctbW9yZSNobWFjLXNoYTI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NjMzMjk1NTksImh0dHA6Ly9zY2hlbWFzLnhtbHNvYXAub3JnL3dzLzIwMDUvMDUvaWRlbnRpdHkvY2xhaW1zL3NpZCI6IjEzIiwiaHR0cDovL3NjaGVtYXMueG1sc29hcC5vcmcvd3MvMjAwNS8wNS9pZGVudGl0eS9jbGFpbXMvbmFtZSI6Ik1vZGVsb1JGIiwiaWF0IjoxNzYyNzI0NzU5LCJuYmYiOjE3NjI3MjQ3NTl9.9AVRfM-u1ktoF0nowwSoZBQM2jMHvKJI9uH4pn6Qj2s") # ¡¡¡ IMPORTANTE: USAR TOKEN VALIDO !!!
API_BASE_URL = os.getenv("COFFEETECH_API_BASE_URL", "https://coffeetech-netcoreappweb-f6hwc3fph9hndhhg.centralus-01.azurewebsites.net/api/v1")
DATA_RECORDS_URL = f"{API_BASE_URL}/data-records"; DEVICES_URL = f"{API_BASE_URL}/devices"
ASSIGNMENTS_URL = f"{API_BASE_URL}/assignments"; SECTIONS_URL = f"{API_BASE_URL}/sections"
FARMS_URL = f"{API_BASE_URL}/farms"

MODEL_ARTIFACT_PATH = os.getenv("COFFEETECH_MODEL_PATH", "artifacts/coffee_rf_pipeline.joblib")
CACHE_DURATION_SECONDS = 300; device_stage_map_cache = {"data": None, "timestamp": 0}
last_sensor_signature_cache: Dict[str, Optional[str]] = {}

# --- ML Functions ---

PLANT_STAGE_MAPPING = {  # Mapeo de nombres API a claves internas
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

VALID_STAGES = list(dict.fromkeys(PLANT_STAGE_MAPPING.values())) + ['default'] # Lista de etapas válidas internas

STAGE_NAME_ES = {
    'plantula': 'Plántula',
    'vegetativo': 'Vegetativo',
    'floracion': 'Floración',
    'fructificacion': 'Fructificación',
    'maduracion': 'Maduración',
    'cosecha': 'Cosecha',
    'default': 'Sin etapa'
}

NUTRIENT_NAME_ES = {'N': 'nitrógeno', 'P': 'fósforo', 'K': 'potasio'}
NUTRIENT_VALUE_KEYS = {
    'N': ('nitrogen_mg_kg', 'nitrogen'),
    'P': ('phosphorus_mg_kg', 'phosphorus'),
    'K': ('potassium_mg_kg', 'potassium'),
}

SEVERITY_NAME_ES = {'severe': 'severa', 'moderate': 'moderada'}
FUNGAL_AGENT_ES = {
    'botrytis': 'Botrytis',
    'ojo_gallo': 'Ojo de gallo',
    'roya': 'Roya'
}
PEST_AGENT_ES = {'broca': 'Broca'}

# --- Umbrales y Niveles Críticos ---
# Estos parámetros consolidan las guías agronómicas internas de CoffeeTech
# (basadas en INIA 2019 y Rainforest Alliance 2020). Si el equipo técnico
# actualiza los rangos óptimos o umbrales críticos, deben modificarse aquí
# y no en otras secciones del código para mantener una sola fuente de verdad.
STAGE_OPTIMAL_PARAMS = { # Rangos óptimos por etapa
    'plantula': {'N': (30, 40), 'P': (20, 30), 'K': (25, 35), 'soil_hum': (70, 80), 'temp': (18, 25)},
    'vegetativo': {'N': (40, 60), 'P': (25, 35), 'K': (35, 50), 'soil_hum': (65, 75), 'temp': (18, 25)},
    'floracion': {'N': (40, 50), 'P': (35, 45), 'K': (50, 60), 'soil_hum': (65, 70), 'temp': (18, 25)},
    'fructificacion': {'N': (40, 50), 'P': (30, 40), 'K': (60, 80), 'soil_hum': (70, 75), 'temp': (17, 23)},
    'maduracion': {'N': (30, 40), 'P': (25, 35), 'K': (70, 90), 'soil_hum': (60, 70), 'temp': (17, 23)},
    'cosecha': {'N': (30, 40), 'P': (20, 30), 'K': (50, 70), 'soil_hum': (60, 70), 'temp': (18, 22)},
    'default': {'N': (35, 55), 'P': (20, 40), 'K': (40, 70), 'soil_hum': (60, 75), 'temp': (17, 26)},
}
NPK_CRITICAL_LEVELS = { # Niveles NPK generales
    'N': {'severe_low': 25, 'moderate_low': 40, 'optimal_low': 40, 'optimal_high': 60, 'excess': 80},
    'P': {'severe_low': 15, 'moderate_low': 25, 'optimal_low': 25, 'optimal_high': 45, 'excess': 60},
    'K': {'severe_low': 30, 'moderate_low': 50, 'optimal_low': 50, 'optimal_high': 80, 'excess': 100},
}
HUMIDITY_ALERT_THRESHOLDS = { # Umbrales Humedad Suelo por etapa 
    'plantula': {'deficit': 60, 'excess': 85}, 'vegetativo': {'deficit': 55, 'excess': 85},
    'floracion': {'deficit': 55, 'excess': 80}, 'fructificacion': {'deficit': 60, 'excess': 85},
    'maduracion': {'deficit': 50, 'excess': 80}, 'cosecha': {'deficit': 50, 'excess': 80},
    'default': {'deficit': 55, 'excess': 85},
}
ENV_ALERT_THRESHOLDS = {
    'temp_low_alert': 15,
    'temp_high_alert': 32,
    'air_hum_fungal_risk': 85,
    'air_hum_broca_risk': 70,
    'temp_broca_risk': 24,
}


class ThresholdedPipeline:
    """
    Wrapper used during training to allow label-specific probability thresholds.
    Must be present at import time to unpickle artifacts saved from training.
    """
    def __init__(self, base_pipeline, thresholds):
        self.base_pipeline = base_pipeline
        self.thresholds = thresholds or {}
        self.classes_ = getattr(base_pipeline, "classes_", None)

    def predict(self, X):
        probabilities = self.base_pipeline.predict_proba(X)
        base_labels = self.base_pipeline.classes_
        base_predictions = base_labels.take(np.argmax(probabilities, axis=1))
        if not self.thresholds:
            return base_predictions
        adjusted = base_predictions.astype(object).copy()
        for label, threshold in self.thresholds.items():
            if self.classes_ is None:
                continue
            try:
                idx = list(base_labels).index(label)
            except ValueError:
                continue
            mask = probabilities[:, idx] >= threshold
            if mask.any():
                adjusted[mask] = label
        return np.asarray(adjusted)

    def predict_proba(self, X):
        return self.base_pipeline.predict_proba(X)

    def __getattr__(self, item):
        base = object.__getattribute__(self, "base_pipeline")
        return getattr(base, item)

    def __getstate__(self):
        return {"base_pipeline": self.base_pipeline, "thresholds": self.thresholds}

    def __setstate__(self, state):
        self.base_pipeline = state["base_pipeline"]
        self.thresholds = state.get("thresholds", {})
        self.classes_ = getattr(self.base_pipeline, "classes_", None)


PRODUCTS_CATALOG = {
    'n_severe': [
        {
            'name': 'Guía de combinación de nitrógeno',
            'notes': 'Seleccionar una opción al suelo y, si es posible, complementar con una foliar. Evitar aplicar todos los productos en simultáneo.'
        },
        {
            'name': 'Nitrato de amonio (Nitrax/Yara)',
            'composition': '33.5% N',
            'dose': '40 g/planta',
            'method': 'Aplicar al suelo alrededor de la línea de gotero',
            'notes': 'Regar después de aplicar para favorecer la absorción y evitar quemaduras.',
            'distributor': 'Yara Perú',
        },
        {
            'name': 'Nitrato de potasio (Ultrasol foliar)',
            'composition': '13-0-46',
            'dose': '4 g/L de agua',
            'method': 'Aplicación foliar',
            'timing_frequency': 'Cada 10-14 días mientras dure la deficiencia severa',
            'notes': 'Complementar con aporte de materia orgánica para mejorar disponibilidad.',
            'distributor': 'SQM/Ultrasol',
        },
    ],
    'n_moderate': [
        {
            'name': 'Urea granulada',
            'composition': '46% N',
            'dose': '25 g/planta',
            'method': 'Incorporar superficialmente y cubrir con mulch',
            'timing_frequency': 'Cada 30 días hasta normalizar niveles',
            'distributor': 'Molinos & CIA',
        },
        {
            'name': 'Nutricafé Crecimiento',
            'composition': '19-9-19 + micro',
            'dose': '50 g/planta',
            'method': 'Aplicación al suelo',
            'notes': 'Fraccionar la dosis en dos aplicaciones para mejorar eficiencia.',
            'distributor': 'Molinos & CIA',
        },
    ],
    'p_severe': [
        {
            'name': 'Superfosfato triple',
            'composition': '46% P2O5',
            'dose': '45 g/planta',
            'method': 'Incorporar al suelo en media luna',
            'notes': 'Aplicar sobre suelo húmedo y cubrir ligeramente.',
            'distributor': 'Corporación Misti',
        },
        {
            'name': 'Fosfito potásico (Fitopron)',
            'composition': '0-28-26',
            'dose': '3 ml/L de agua',
            'method': 'Aplicación foliar',
            'timing_frequency': 'Cada 15 días mientras se corrija la deficiencia',
            'distributor': 'Farmex',
        },
    ],
    'p_moderate': [
        {
            'name': 'Guía de ajuste de fósforo moderado',
            'notes': 'El nivel está cerca del umbral crítico; repetir muestreo en 7-10 días y complementar con una fuente ligera si se mantiene <25 mg/kg.'
        },
        {
            'name': 'Fosfato diamónico (DAP)',
            'composition': '18-46-0',
            'dose': '25 g/planta',
            'method': 'Aplicación al suelo con ligera incorporación',
            'notes': 'Evitar contacto directo con raíces expuestas.',
            'distributor': 'Gavilon Perú',
        },
        {
            'name': 'Guano de isla compostado',
            'composition': '12% P2O5 promedio',
            'dose': '80 g/planta',
            'method': 'Aplicación en corona',
            'notes': 'Aporta fósforo y materia orgánica para mejorar la microbiología.',
            'distributor': 'Agrorural',
        },
    ],
    'k_severe': [
        {
            'name': 'Guía de combinación de potasio',
            'notes': 'Elegir una estrategia al suelo y una foliar según disponibilidad; evaluar mezcla con riego para optimizar absorción.'
        },
        {
            'name': 'Sulfato de potasio granular',
            'composition': '50% K2O',
            'dose': '45 g/planta',
            'method': 'Aplicación al suelo; incorporar ligeramente',
            'notes': 'Aportar humedad posterior para evitar salinizar la rizósfera.',
            'distributor': 'SQM Perú',
        },
        {
            'name': 'Nitrato de potasio foliar',
            'composition': '13-0-46',
            'dose': '5 g/L de agua',
            'method': 'Nebulización foliar fina',
            'timing_frequency': 'Cada 10 días hasta normalizar la lectura',
            'distributor': 'Ultrasol',
        },
    ],
    'k_moderate': [
        {
            'name': 'Sulfato de potasio',
            'composition': '50% K2O',
            'dose': '30 g/planta',
            'method': 'Aplicación al suelo',
            'notes': 'Fraccionar la dosis en dos aplicaciones con 15 días de separación.',
            'distributor': 'SQM Perú',
        },
        {
            'name': 'Cafexport Producción',
            'composition': '15-15-23 + Mg + S',
            'dose': '70 g/planta',
            'method': 'Aplicación al suelo alrededor de la planta',
            'distributor': 'Yara Perú',
        },
    ],
    'water_deficit_general': [
        {
            'name': 'Riego de recuperación por goteo',
            'method': 'Aplicar riego profundo de 25-30 L por planta en 2 pulsos separados por 12 h',
            'timing_frequency': 'Repetir a las 48 h si la humedad sigue por debajo del umbral',
            'notes': 'Verificar uniformidad de humedad con tensiómetros y evitar encharcamiento.'
        },
        {
            'name': 'Cobertura con acolchado orgánico',
            'method': 'Instalar capa de 5 cm de material orgánico (hojarasca o paja) sobre la gotera',
            'notes': 'Reduce evaporación y protege microbiota; renovar cada 30 días en época seca.'
        },
    ],
    'water_deficit_maduracion': [
        {
            'name': 'Guía de combinación para déficit hídrico en maduración',
            'notes': 'Priorizar una acción de riego principal y una medida complementaria (sombra, acolchado o bioestimulante). Ajustar según disponibilidad de agua.'
        },
        {
            'name': 'Riego suplementario en maduración',
            'method': 'Aplicar 35 L/planta por goteo o microaspersión al atardecer',
            'timing_frequency': 'Cada 3-4 días hasta recuperar humedad objetivo',
            'notes': 'Añadir 2 L/planta con bioestimulante de algas (0.3%) para reducir estrés.'
        },
        {
            'name': 'Sombras temporales con malla 35%',
            'method': 'Instalar malla en las horas de mayor radiación para reducir evapotranspiración',
            'notes': 'Retirar al final de la ola de calor para evitar reducción de fotosíntesis.'
        },
    ],
    'water_deficit_general': [
        {
            'name': 'Guía de combinación para déficit hídrico',
            'notes': 'Seleccionar un esquema de riego (profundo o suplementario) y complementarlo con manejo de cobertura o sombra según los recursos disponibles.'
        },
        {
            'name': 'Riego de recuperación por goteo',
            'method': 'Aplicar riego profundo de 25-30 L por planta en 2 pulsos separados por 12 h',
            'timing_frequency': 'Repetir a las 48 h si la humedad sigue por debajo del umbral',
            'notes': 'Verificar uniformidad de humedad con tensiómetros y evitar encharcamiento.'
        },
        {
            'name': 'Cobertura con acolchado orgánico',
            'method': 'Instalar capa de 5 cm de material orgánico (hojarasca o paja) sobre la gotera',
            'notes': 'Reduce evaporación y protege microbiota; renovar cada 30 días en época seca.'
        },
        {
            'name': 'Sombras temporales con malla 35%',
            'method': 'Instalar malla en horas de máxima radiación para reducir evapotranspiración',
            'notes': 'Retirar al final de la ola de calor para evitar reducción de fotosíntesis.'
        },
    ],
    'n_excess_moderate': [
        {
            'name': 'Lavado ligero de nitrógeno',
            'method': 'Aplicar 20 L/planta por goteo en dos pulsos de 10 L separados por 6 h',
            'notes': 'Favorece lixiviación controlada sin arrastrar otros nutrientes.'
        },
        {
            'name': 'Mulch alto en carbono',
            'method': 'Incorporar 3 kg/planta de rastrojo seco o cascarilla',
            'notes': 'Captura el exceso de N y mejora la microbiología.'
        },
    ],
    'n_excess_severe': [
        {
            'name': 'Lavado intensivo escalonado',
            'method': 'Aplicar 40 L/planta por goteo divididos en 4 pulsos de 10 L',
            'notes': 'Supervisar conductividad del drenaje para detener cuando baje <1.5 dS/m.'
        },
        {
            'name': 'Aplicación de yeso agrícola',
            'dose': '200 g/planta',
            'method': 'Distribuir alrededor del gotero y regar',
            'notes': 'El calcio favorece el intercambio catiónico y reduce el N amoniacal disponible.'
        },
        {
            'name': 'Biochar activado',
            'dose': '1 kg/planta',
            'method': 'Incorporar superficialmente',
            'notes': 'Fija amonio y mejora estructura del suelo.'
        },
    ],
    'p_excess_moderate': [
        {
            'name': 'Materia orgánica fresca sin P',
            'method': 'Aplicar 2 kg/planta de compost lignificado',
            'notes': 'Secuestra P disponible y alimenta microbios inmovilizadores.'
        },
        {
            'name': 'Quelato de zinc preventivo',
            'dose': '2 cc/L foliar',
            'method': 'Aplicar cada 15 días',
            'notes': 'Evita bloqueos de Zn y Fe causados por exceso de P.'
        },
    ],
    'p_excess_severe': [
        {
            'name': 'Yeso agrícola granular',
            'dose': '250 g/planta',
            'method': 'Aplicar en corona y regar',
            'notes': 'El Ca favorece la fijación de fosfatos al complejo arcillo-húmico.'
        },
        {
            'name': 'Riego de lixiviación dirigido',
            'method': '30 L/planta en 3 pulsos de 10 L',
            'notes': 'Aplicar solo en suelos con buen drenaje para evitar anoxia.'
        },
    ],
    'k_excess_moderate': [
        {
            'name': 'Riego de arrastre con aporte Ca/Mg',
            'method': '20 L/planta con 200 ppm de Ca y Mg',
            'notes': 'Rebalancea la relación K-Ca-Mg en el complejo de intercambio.'
        },
        {
            'name': 'Aplicación de sulfato de magnesio',
            'dose': '15 g/planta',
            'method': 'Disolver y aplicar al suelo',
            'notes': 'Compensa antagonismo de K sobre Mg.'
        },
    ],
    'k_excess_severe': [
        {
            'name': 'Yeso agrícola + lavado',
            'dose': '250 g/planta de yeso y 30 L de agua',
            'method': 'Aplicar yeso, regar lentamente para lixiviar K',
            'notes': 'Monitorear conductividad eléctrica para evitar pérdida excesiva de Ca.'
        },
        {
            'name': 'Enmienda con zeolita',
            'dose': '1 kg/planta',
            'method': 'Incorporar superficialmente',
            'notes': 'La zeolita intercambia K y reduce su actividad inmediata.'
        },
    ],
    'temp_heat_general': [
        {
            'name': 'Sombras móviles 40-50%',
            'method': 'Instalar mallas temporales en horas de máxima radiación',
            'notes': 'Bajar temperatura foliar 2-3 °C y reducir estrés fotooxidativo; retirar al estabilizarse el clima.'
        },
        {
            'name': 'Nebulización fina o riego evaporativo',
            'method': 'Aplicar microaspersión 5-7 min cada 90 min entre 11:00-15:00',
            'notes': 'Usar agua limpia para evitar enfermedades; suspender si la humedad supera 90%.'
        },
        {
            'name': 'Bioestimulante antiestrés (silicio + algas)',
            'composition': 'Si 3% + extractos de algas',
            'dose': '2-3 cc/L foliar',
            'timing_frequency': 'Cada 7 días mientras dure la ola de calor',
            'notes': 'Fortalece paredes celulares y mejora la regulación estomática.'
        },
    ],
    'temp_heat_maduracion': [
        {
            'name': 'Riego fraccionado en maduración',
            'method': 'Aplicar 15 L/planta al amanecer y 15 L al atardecer',
            'notes': 'Evita golpes osmóticos y mantiene la turgencia del fruto.'
        },
        {
            'name': 'Cobertura ligera de frutos',
            'method': 'Colocar malla 35% sobre racimos expuestos en la cara oeste de la planta',
            'notes': 'Disminuye quemado de fruto y caída prematura.'
        },
    ],
    'temp_cold_general': [
        {
            'name': 'Cobertura con mulch + cal agrícola',
            'method': 'Aplicar capa de 5 cm de mulch y espolvorear 150 g de cal agrícola alrededor del tallo',
            'notes': 'Aísla la raíz de descensos bruscos y mejora disponibilidad de Ca.'
        },
        {
            'name': 'Bioestimulante de resistencia (aminoácidos + K)',
            'dose': '3 cc/L foliar',
            'timing_frequency': 'Aplicar 24-48 h antes de la helada prevista',
            'notes': 'Favorece la síntesis de proteínas anticongelantes.'
        },
        {
            'name': 'Cortinas rompeviento temporales',
            'method': 'Instalar plástico microperforado o costales en la cara sur del lote',
            'notes': 'Reduce la velocidad del viento frío en 30-40%.'
        },
    ],
    'temp_cold_maduracion': [
        {
            'name': 'Cobertura nocturna con manta térmica',
            'method': 'Cubrir hileras con manta agrícola 17 g/m² entre 19:00 y 7:00',
            'notes': 'Retener calor residual del suelo y proteger racimos en maduración.'
        },
        {
            'name': 'Foliar potásico-calcio',
            'composition': '6% K2O + 4% CaO + aminoácidos',
            'dose': '3 cc/L',
            'timing_frequency': 'Cada 7 días durante la ola fría',
            'notes': 'Mejora firmeza de fruto y reduce microfisuras por frío.'
        },
    ],
    'k_fructification_boost': [
        {
            'name': 'Sulfato de potasio (frutificación)',
            'composition': '50% K2O',
            'dose': '35-40 g/planta',
            'method': 'Aplicación al suelo con incorporación ligera',
            'notes': 'Apoyar con riego para mejorar la absorción.',
            'distributor': 'SQM Perú',
        },
        {
            'name': 'Nitrato de potasio foliar',
            'composition': '13-0-46',
            'dose': '3 g/L de agua',
            'method': 'Aplicación foliar (atomizador fino)',
            'distributor': 'SQM/Ultrasol',
        },
    ],
    'ca_deficiency': [
        {
            'name': 'Nitrato de calcio (YaraLiva)',
            'composition': '15.5% N / 26.5% CaO',
            'dose': '30 g/planta',
            'method': 'Aplicación al suelo',
            'notes': 'Fortalece paredes celulares y mejora el llenado del grano.',
            'distributor': 'Yara Perú',
        },
    ],
    'roya_ojo_gallo_control': [
        {
            'name': 'Cyproconazole (Alto®)',
            'dose': '1 ml/L de agua',
            'method': 'Aplicación foliar dirigida',
            'timing_frequency': 'Repetir cada 21 días según monitoreo',
            'notes': 'Rotar modos de acción para evitar resistencia.',
            'distributor': 'Syngenta',
        },
        {
            'name': 'Azoxystrobin (Amistar®)',
            'dose': '0.5 g/L de agua',
            'method': 'Nebulización foliar homogénea',
            'distributor': 'Syngenta',
        },
    ],
    'botrytis_control': [
        {
            'name': 'Azoxystrobin + Difenoconazole',
            'dose': '0.6 g/L de agua',
            'method': 'Aplicación foliar dirigida a floración',
            'notes': 'Realizar en horas frescas para mejor absorción.',
        },
        {
            'name': 'Caldo bordelés',
            'composition': 'Sulfato de cobre + cal',
            'dose': 'Seguir etiqueta comercial',
            'method': 'Aplicación foliar',
            'notes': 'Mantener intervalo de seguridad previo a cosecha.',
        },
    ],
    'bio_control_broca': [
        {
            'name': 'Guía de control integrado de broca',
            'notes': 'Combinar control biológico con trampas; priorizar una alternativa principal y complementar según infestación.'
        },
        {
            'name': 'Beauveria bassiana (formulación comercial)',
            'dose': 'Seguir etiqueta; referencia 200 g/200 L',
            'method': 'Aplicación foliar al atardecer',
            'timing_frequency': 'Repetir cada 15 días en picos poblacionales',
            'notes': 'Mantener cobertura de follaje y humedad para favorecer infección del hongo.'
        },
    ],
    'entomopathogenic_baits': [
        {
            'name': 'Trampas Brocap® con atrayente',
            'dose': '20 trampas/ha',
            'method': 'Distribuir uniformemente en el lote',
            'notes': 'Monitorear capturas semanalmente para ajustar densidad y plan de aspersión complementaria.'
        },
        {
            'name': 'Cebo Etológico con alcohol/melon',
            'method': 'Preparar mezcla artesanal e instalar en envases con tapa perforada',
            'notes': 'Renovar cada 7 días; útil en zonas donde no hay trampa comercial disponible.'
        },
    ],
    'broca_control': [
        {
            'name': 'Beauveria bassiana (Beauvesol)',
            'dose': '200 g/200 L de agua',
            'method': 'Aplicación foliar y al fruto',
            'timing_frequency': 'Repetir cada 15 días en picos poblacionales',
            'distributor': 'Solagro',
        },
        {
            'name': 'Trampas Brocap® con atrayente',
            'dose': '20 trampas/ha',
            'method': 'Distribuir uniformemente en el lote',
            'notes': 'Relevar semanalmente capturas para ajustar densidad.',
            'distributor': 'Procafé',
        },
    ],
    'general_pest_preventive': [
        {
            'name': 'Extracto botánico (Bioplag)',
            'dose': '5 ml/L de agua',
            'method': 'Pulverización foliar',
            'notes': 'Acción repelente; reforzar cada 10 días en viveros y plántulas.',
        },
    ],
    'excess_humidity_soil_actions': [
        {
            'name': 'Suspender riego temporalmente',
            'method': 'Pausar cualquier riego programado durante 48 h y reevaluar humedad',
            'notes': 'Evita saturación y asfixia radicular cuando hay precipitación activa.'
        },
        {
            'name': 'Apertura y limpieza de drenajes',
            'method': 'Revisar y despejar cunetas, salidas y filtros',
            'notes': 'Facilita la evacuación del exceso de agua y reduce charcos persistentes.'
        },
        {
            'name': 'Monitoreo de raíces',
            'method': 'Inspeccionar raíces finas en 10% de plantas',
            'notes': 'Detectar signos de pudrición para decidir aplicaciones correctivas.'
        },
    ],
    'fungal_preventive_excess_humidity': [
        {
            'name': 'Hidróxido de cobre (Kocide®)',
            'dose': '3 g/L de agua',
            'method': 'Aplicación foliar preventiva',
            'notes': 'Aplicar después de lluvias intensas para proteger tejidos jóvenes.',
        },
        {
            'name': 'Fungicida biológico a base de Bacillus subtilis',
            'dose': 'Seguir etiqueta comercial',
            'method': 'Pulverización foliar',
            'notes': 'Compatibilizar con planes de control químico para reducir residuos.',
        },
    ],
    'plantula_base': [
        {
            'name': 'Abono orgánico enriquecido (Mallki®)',
            'dose': '50 g/planta',
            'method': 'Aplicación al sustrato',
            'notes': 'Mejora estructura y retención de humedad en viveros.',
        },
        {
            'name': 'Guano de isla compostado',
            'dose': '80 g/planta',
            'method': 'Mezclar con el sustrato en trasplante',
            'notes': 'Aporta microelementos clave para raíces jóvenes.',
        },
    ],
    'vegetativo_integral': [
        {
            'name': 'Fertilizante granular 20-10-10',
            'dose': '60 g/planta cada 3 meses',
            'method': 'Aplicación al suelo dividida en dos medias lunas',
            'notes': 'Complementar con cobertura orgánica para reducir lixiviación.',
        },
    ],
    'floracion_foliar': [
        {
            'name': 'Fertilizante foliar alto en fósforo (12-48-8)',
            'dose': '2.5 ml/L de agua',
            'method': 'Pulverización fina al inicio de floración',
            'notes': 'Añadir boro si el análisis foliar lo requiere.',
        },
    ],
    'floracion_base_k': [
        {
            'name': 'Sulfato de potasio',
            'dose': '35 g/planta',
            'method': 'Aplicación al suelo alrededor de la planta',
            'notes': 'Favorece cuajado y desarrollo de frutos.',
        },
    ],
    'fructificacion_base': [
        {
            'name': 'Fertilizante 15-15-23',
            'dose': '90 g/planta fraccionado',
            'method': 'Aplicar en dos medias lunas opuestas',
            'notes': 'Acompañar con control de malezas para reducir competencia.',
        },
    ],
    'maduracion_integral': [
        {
            'name': 'Fertilizante 5-5-30 + Ca/Mg',
            'dose': '60 g/planta',
            'method': 'Aplicación al suelo',
            'notes': 'Favorece llenado final y firmeza del grano.',
        },
    ],
    'base_fertilizer_plantula': [
        {'name': 'Plan base plántula', 'notes': 'Mantener aporte orgánico ligero y monitorear riego.'},
    ],
    'base_fertilizer_vegetativo': [
        {'name': 'Plan base vegetativo', 'notes': 'Aporte balanceado NPK y manejo de sombra al 40-50%.'},
    ],
    'base_fertilizer_floracion': [
        {'name': 'Plan base floración', 'notes': 'Mantener aporte de fósforo y potasio, vigilar humedad.'},
    ],
    'base_fertilizer_fructificacion': [
        {'name': 'Plan base fructificación', 'notes': 'Potenciar potasio y monitorear carga de frutos.'},
    ],
    'base_fertilizer_maduracion': [
        {'name': 'Plan base maduración', 'notes': 'Refuerzo de potasio y calcio para uniformidad en cosecha.'},
    ],
    'base_fertilizer_cosecha': [
        {'name': 'Plan base cosecha', 'notes': 'Priorizar sanidad del fruto y suspender fertilizaciones químicas 30 días antes.'},
    ],
    'base_fertilizer_default': [
        {'name': 'Plan base general', 'notes': 'Mantener programa NPK equilibrado según análisis de suelo.'},
    ],
    'altitude_guidance_baja': [
        {'name': 'Manejo plano bajo', 'notes': 'Incrementar sombra, riego y control de plagas térmicas.'},
    ],
    'altitude_guidance_media': [
        {'name': 'Manejo plano medio', 'notes': 'Conservar prácticas integrales y monitoreo semanal.'},
    ],
    'altitude_guidance_alta': [
        {'name': 'Manejo plano alto', 'notes': 'Reforzar protección contra heladas ligeras y monitorear humedad de suelo.'},
    ],
    'altitude_guidance_muy_alta': [
        {'name': 'Manejo plano muy alto', 'notes': 'Implementar barreras contra heladas y ajustar carga productiva.'},
    ],
    'error_model_not_loaded': [
        {'name': 'Alerta de sistema', 'notes': 'Modelo ML no disponible. Verificar servicios antes de ejecutar labores.'},
    ],
    'error_prediction_failed': [
        {'name': 'Alerta de predicción', 'notes': 'Se produjo un fallo al generar la recomendación. Revisar registros y volver a intentar.'},
    ],
    'error_missing_data': [
        {'name': 'Error de datos', 'notes': 'Faltan datos esenciales del sensor. Validar dispositivo y repetir medición.'},
    ],
    'error_formatting': [
        {'name': 'Error de formato', 'notes': 'No se pudo formatear la recomendación. Revisar logs del servicio.'},
    ],
    'unknown_condition': [
        {'name': 'Condición no reconocida', 'notes': 'Validar con un agrónomo para definir plan de acción.'},
    ],
    'optimal': [
        {'name': 'Condiciones óptimas', 'notes': 'Mantener monitoreo y buenas prácticas culturales.'},
    ],

}

CONDITION_ACTION_MAP = {
    'n_deficiency_severe': ['n_severe'],
    'n_deficiency_moderate': ['n_moderate'],
    'p_deficiency_severe': ['p_severe'],
    'p_deficiency_moderate': ['p_moderate'],
    'k_deficiency_severe': ['k_severe'],
    'k_deficiency_moderate': ['k_moderate'],
    'k_deficiency_fructificacion': ['k_fructification_boost'],
    'water_deficit_maduracion': ['water_deficit_maduracion', 'water_deficit_general'],
    'water_deficit_floracion': ['water_deficit_general'],
    'water_deficit_fructificacion': ['water_deficit_general'],
    'water_deficit_plantula': ['water_deficit_general'],
    'water_deficit_vegetativo': ['water_deficit_general'],
    'water_deficit_cosecha': ['water_deficit_general'],
    'excess_humidity_soil': ['excess_humidity_soil_actions', 'fungal_preventive_excess_humidity'],
    'cold_stress_high_altitude': ['altitude_guidance_alta'],
    'heat_stress_low_altitude': ['altitude_guidance_baja'],
    'fungal_risk_botrytis_floracion': ['fungal_preventive_excess_humidity', 'floracion_foliar'],
    'fungal_risk_roya_vegetativo': ['vegetativo_integral'],
    'fungal_risk_ojo_gallo_maduracion': ['maduracion_integral'],
    'pest_risk_broca_fructificacion': ['bio_control_broca', 'entomopathogenic_baits'],
    'n_excess_moderate': ['n_excess_moderate'],
    'n_excess_severe': ['n_excess_severe'],
    'p_excess_moderate': ['p_excess_moderate'],
    'p_excess_severe': ['p_excess_severe'],
    'k_excess_moderate': ['k_excess_moderate'],
    'k_excess_severe': ['k_excess_severe'],
}

def _resolve_catalog_actions(label: str, stage: str) -> List[Dict[str, Any]]:
    keys = list(CONDITION_ACTION_MAP.get(label, []))
    actions: List[Dict[str, Any]] = []
    if not keys:
        if label.startswith('water_deficit_'):
            stage_key = label.split('water_deficit_', 1)[-1]
            keys.extend([f'water_deficit_{stage_key}', 'water_deficit_general'])
        elif label.startswith('optimal_'):
            stage_key = label.split('optimal_', 1)[-1]
            keys.extend([f'base_fertilizer_{stage_key}', 'base_fertilizer_default'])
        elif label == 'excess_humidity_soil':
            keys.extend(['fungal_preventive_excess_humidity'])
        elif label in {'cold_stress_high_altitude', 'heat_stress_low_altitude'}:
            keys.extend(CONDITION_ACTION_MAP.get(label, []))
        elif label.startswith('temp_high_stage_'):
            stage_key = label.split('temp_high_stage_', 1)[-1]
            keys.extend([f'temp_heat_{stage_key}', 'temp_heat_general'])
        elif label.startswith('temp_low_stage_'):
            stage_key = label.split('temp_low_stage_', 1)[-1]
            keys.extend([f'temp_cold_{stage_key}', 'temp_cold_general'])
        elif label.startswith('precip_saturated_'):
            keys.extend(['excess_humidity_soil_actions'])
    for key in keys:
        actions.extend(PRODUCTS_CATALOG.get(key, []) or [])
    # deduplicar por nombre o id
    unique: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for action in actions:
        name = action.get('name') or action.get('id') or repr(action)
        if name in seen:
            continue
        seen.add(name)
        unique.append(action)
    return unique


ALTITUDE_THRESHOLDS = {
    'bajo': {'min': 800, 'max': 1199, 'label': 'Plano bajo', 'message': 'Zona baja: ciclo rápido, más plagas y granos menos densos; reforzar sombra y riego.', 'catalog_key': 'altitude_guidance_baja'},
    'medio': {'min': 1200, 'max': 1600, 'label': 'Plano medio', 'message': 'Zona media: óptimo rendimiento y calidad; mantener prácticas integrales y monitoreo constante.', 'catalog_key': 'altitude_guidance_media'},
    'alto': {'min': 1601, 'max': 1900, 'label': 'Plano alto', 'message': 'Zona alta: maduración lenta, grano denso y calidad superior; vigilar humedad y heladas ligeras.', 'catalog_key': 'altitude_guidance_alta'},
    'muy_alto': {'min': 1901, 'label': 'Plano muy alto', 'message': 'Zona muy alta: producción limitada, riesgo de heladas y tazas exóticas; proteger brotes y ajustar carga.', 'catalog_key': 'altitude_guidance_muy_alta'}
}

STAGE_ALTITUDE_NOTES = {
    'plantula': {
        'bajo': 'Sombra â‰¥ 60%, riegos frecuentes y control intensivo de broca y roya.',
        'medio': 'Sombra 40-50%, riego moderado y fertilización inicial estándar.',
        'alto': 'Proteger contra frío, sombra ligera 30-40% y asegurar buen drenaje.',
        'muy_alto': 'Viveros protegidos contra heladas y fertilización de liberación lenta.'
    },
    'vegetativo': {
        'bajo': 'Podas tempranas para controlar vigor y aporte alto de nitrógeno.',
        'medio': 'Podas formativas regulares y fertilización equilibrada NPK.',
        'alto': 'Aumentar materia orgánica y cobertura; moderar nitrógeno.',
        'muy_alto': 'Reducir nitrógeno, priorizar fósforo/potasio y evitar estrés por frío.'
    },
    'floracion': {
        'bajo': 'Gestionar sombra para evitar estrés térmico y reforzar control preventivo de roya.',
        'medio': 'Mantener sombra estable y aplicar micronutrientes foliares.',
        'alto': 'Minimizar exceso de humedad y asegurar buena polinización.',
        'muy_alto': 'Proteger de heladas y neblina; micronutrientes vía foliar.'
    },
    'fructificacion': {
        'bajo': 'Riego suplementario para uniformidad y control intensivo de broca.',
        'medio': 'Ajustar fertilización potásica y control moderado de plagas.',
        'alto': 'Combinar potasio con materia orgánica y asegurar drenaje.',
        'muy_alto': 'Priorizar fertilización orgánica, monitorear heladas y reducir carga si es necesario.'
    },
    'maduracion': {
        'bajo': 'Cosecha escalonada y refuerzo en control de plagas.',
        'medio': 'Cosecha más uniforme y monitoreo de roya.',
        'alto': 'Maduración lenta, realizar cosecha selectiva para calidad.',
        'muy_alto': 'Recolección cuidadosa y seguimiento del clima para evitar pérdidas.'
    },
    'cosecha': {
        'bajo': 'Procesamiento rápido para evitar fermentaciones por calor y transporte en sombra.',
        'medio': 'Procesamiento estándar con secado controlado.',
        'alto': 'Fermentaciones controladas y secado lento para especialidad.',
        'muy_alto': 'Procesamiento ultra selectivo y secado protegido de bajas temperaturas.'
    }
}



def map_plant_stage(api_stage_name: Optional[str]) -> str:
    """Normalize raw stage names from Azure (including accents/case) so downstream
    logic can rely on a small set of canonical keys."""
    if not api_stage_name or not isinstance(api_stage_name, str):
        return 'default'
    # Direct mapping
    internal_name = PLANT_STAGE_MAPPING.get(api_stage_name)
    if internal_name:
        return internal_name
    # Case-insensitive + accent-insensitive match
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
    logger.warning(f"Unrecognized plant stage from API: {api_stage_name}")
    return 'default'

async def fetch_api_data(client: httpx.AsyncClient, url: str) -> Optional[List[Dict]]:
    """Generic authorized GET against the CoffeeTech backend. Returns a parsed
    list or None so callers can short-circuit on missing data."""
    if not JWT_TOKEN or len(JWT_TOKEN) < 50: logging.error(f"JWT_TOKEN invalido/faltante. No fetch {url}"); return None
    headers = {"Authorization": f"Bearer {JWT_TOKEN}"}; logger.debug(f"Fetching {url}")
    try:
        response = await client.get(url, headers=headers, timeout=30.0)
        response.raise_for_status(); data = response.json()
        logger.debug(f"Success fetching {url}"); return data if isinstance(data, list) else None
    except Exception as e: logging.error(f"Error fetching {url}: {e}"); return None



async def get_device_stage_map() -> Dict[str, Dict[str, Optional[float]]]:
    """Build a cached map deviceHubId -> context (stage, altitude, provenance)
    by joining devices, assignments, sections and farms from Azure."""
    global device_stage_map_cache
    current_time = time.time()
    if device_stage_map_cache["data"] and (current_time - device_stage_map_cache["timestamp"] < CACHE_DURATION_SECONDS):
        logging.info("Using cached device-stage map.")
        return device_stage_map_cache["data"]

    logging.info("Fetching fresh device-stage map from API...")
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
        logging.error(f"Failed devices fetch: {devices_data}")
        return {}
    if isinstance(assignments_data, Exception) or not assignments_data:
        logging.error(f"Failed assignments fetch: {assignments_data}")
        return {}
    if isinstance(sections_data, Exception) or not sections_data:
        logging.error(f"Failed sections fetch: {sections_data}")
        return {}
    if isinstance(farms_data, Exception) or farms_data is None:
        logging.error(f"Failed farms fetch: {farms_data}")
        farms_data = []

    try:
        dev_df = pd.DataFrame(devices_data)[['id', 'deviceHubId']].rename(columns={'id': 'deviceId'})
        asg_df = pd.DataFrame(assignments_data)[['deviceId', 'sectionId']]
        sections_df = pd.DataFrame(sections_data)
        # Resolve stage from sections payload (the authoritative phenological stage)
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
        if altitude_col:
            farms_df = farms_df[['id', altitude_col]].rename(columns={'id': 'farmId', altitude_col: 'altitude'})
            farms_df['altitude_source'] = altitude_col
        else:
            if not farms_df.empty:
                logging.warning("Farms payload did not include an altitude field; altitude will be omitido.")
            farms_df = farms_df[['id']].rename(columns={'id': 'farmId'}) if 'id' in farms_df.columns else pd.DataFrame(columns=['farmId'])
            farms_df['altitude'] = None
            farms_df['altitude_source'] = None

        merged = pd.merge(pd.merge(dev_df, asg_df, on='deviceId', how='left'), sec_df, on='sectionId', how='left')
        if 'farmId' in merged.columns:
            merged = pd.merge(merged, farms_df, on='farmId', how='left')
        else:
            merged['altitude'] = None
            merged['altitude_source'] = None

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
            device_stage_map[str(hub_id)] = entry

        if device_stage_map:
            device_stage_map_cache = {"data": device_stage_map, "timestamp": current_time}
            logging.info(f"Cached map for {len(device_stage_map)} devices.")
        else:
            logging.warning("Generated device-stage map is empty.")
        return device_stage_map
    except Exception as exc:
        logging.error(f"Error processing map dataframes: {exc}", exc_info=True)
        return {}



async def get_latest_sensor_data() -> List[Dict[str, Any]]:
    """Fetch the most recent record per device (by timestamp/id) from Azure."""
    logging.info("Fetching latest sensor data from API...")
    async with httpx.AsyncClient() as client: data_records = await fetch_api_data(client, DATA_RECORDS_URL)
    if not data_records: logging.warning("No data records received."); return []
    try:
        df = pd.DataFrame(data_records); latest_data_list = []
        if df.empty: logging.info("Data records DataFrame empty."); return []
        potential_ts_cols = ['createdAt', 'timestamp', 'recordDate', 'created_at', 'updatedAt']; timestamp_col = next((col for col in potential_ts_cols if col in df.columns), None)
        if timestamp_col:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce', utc=True); df = df.dropna(subset=[timestamp_col])
            if not df.empty: latest_idx = df.loc[df.groupby('deviceHubId')[timestamp_col].idxmax()]; latest_data_list = latest_idx.to_dict('records')
            else: logging.warning("No valid timestamp data.")
        else:
            logging.warning("No timestamp column. Using 'id' fallback.");
            if 'id' in df.columns:
                df['id'] = pd.to_numeric(df['id'], errors='coerce'); df = df.dropna(subset=['id'])
                if not df.empty: latest_idx = df.loc[df.groupby('deviceHubId')['id'].idxmax()]; latest_data_list = latest_idx.to_dict('records')
                else: logging.warning("No valid 'id' data.")
            else: logging.error("Critical: Cannot get latest record."); return []
        logging.info(f"Found latest data for {len(latest_data_list)} unique devices."); return latest_data_list
    except Exception as e: logging.error(f"Error processing data records DF: {e}", exc_info=True); return []


def _get_sensor_value(payload: Union[pd.Series, Dict[str, Any]], *keys):
    for key in keys:
        if payload is None or key is None:
            continue
        if isinstance(payload, pd.Series):
            if key not in payload:
                continue
            value = payload[key]
        else:
            if key not in payload:
                continue
            value = payload[key]
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        return value
    return None


def _safe_float_value(value: Optional[Any]) -> Optional[float]:
    """Convert inputs from sensors/catalog (which may be strings, None or NaN)
    into safe floats so threshold logic is consistent."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _compute_sensor_signature(sensor_record: Dict[str, Any], iso_timestamp: Optional[str]) -> str:
    """Create a signature that lets us detect whether a device record is new."""
    if iso_timestamp:
        return f"ts:{iso_timestamp}"
    record_id = sensor_record.get("id")
    if record_id is not None:
        return f"id:{record_id}"
    try:
        return json.dumps(sensor_record, sort_keys=True, default=str)
    except Exception:
        return str(sensor_record)

def _classify_condition_meta(condition_label: str, stage: str) -> Tuple[str, int]:
    """Assign human-friendly metadata (recommendation type + urgency) to any
    condition label, consolidating logic so format_detailed_recommendation stays lean."""
    rec_type = 'unknown'
    urgency = 1
    if '_deficiency_severe' in condition_label:
        urgency = 5
        rec_type = 'fertilizacion'
    elif '_deficiency_moderate' in condition_label:
        urgency = 3
        rec_type = 'fertilizacion'
    elif '_excess_severe' in condition_label:
        urgency = 4
        rec_type = 'fertilizacion'
    elif '_excess_moderate' in condition_label:
        urgency = 2
        rec_type = 'fertilizacion'
    elif condition_label.endswith('_excess') or condition_label in {'n_excess', 'p_excess', 'k_excess'}:
        urgency = 2
        rec_type = 'fertilizacion'
    elif 'water_deficit' in condition_label:
        urgency = 5 if stage in ['floracion', 'fructificacion'] else 4
        rec_type = 'riego'
    elif condition_label.startswith('precip_saturated_'):
        urgency = 4
        rec_type = 'riego'
    elif 'excess_humidity_soil' in condition_label:
        urgency = 4
        rec_type = 'manejo_cultural'
    elif 'cold_stress' in condition_label or 'heat_stress' in condition_label:
        urgency = 4
        rec_type = 'manejo_cultural'
    elif condition_label.startswith('temp_low_stage_') or condition_label.startswith('temp_high_stage_'):
        urgency = 4
        rec_type = 'manejo_cultural'
    elif 'fungal_risk' in condition_label:
        urgency = 4
        rec_type = 'fitosanitario'
    elif 'pest_risk' in condition_label:
        urgency = 3
        rec_type = 'fitosanitario'
    elif condition_label.startswith('optimal'):
        urgency = 1
        rec_type = 'optimo'
    elif condition_label.startswith('error'):
        urgency = 5
        rec_type = 'error'
    return rec_type, urgency



def _build_condition_context_lines(condition_label: str, stage: str, sensor_data: Dict[str, Any]) -> List[str]:
    """Generate bullet points with the sensor evidence that justifies each
    detected condition (levels, ranges, alerts)."""
    lines: List[str] = []
    def safe_float(value: Optional[Any]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    opt_params = STAGE_OPTIMAL_PARAMS.get(stage, STAGE_OPTIMAL_PARAMS['default'])
    humidity_alerts = HUMIDITY_ALERT_THRESHOLDS.get(stage, HUMIDITY_ALERT_THRESHOLDS['default'])
    altitude = safe_float(_get_sensor_value(sensor_data, 'altitude_masl', 'altitudeMasl', 'altitude'))
    temp = safe_float(_get_sensor_value(sensor_data, 'celcius_grade_temperature', 'celciusGradeTemperature'))
    soil_hum = safe_float(_get_sensor_value(sensor_data, 'soil_humidity_percent', 'soilHumidityPercent'))
    air_hum = safe_float(_get_sensor_value(sensor_data, 'air_humidity_percent', 'airHumidityPercent'))
    for nutrient_key, aliases in NUTRIENT_VALUE_KEYS.items():
        if condition_label.startswith(f"{nutrient_key.lower()}_") or condition_label == f"{nutrient_key.lower()}_excess":
            raw_value = _get_sensor_value(sensor_data, *aliases)
            actual_val = safe_float(raw_value)
            if actual_val is not None:
                low, high = opt_params.get(nutrient_key, (None, None))
                lines.append(
                    f"Nivel {nutrient_key}: {actual_val:.1f} mg/kg (óptimo {low}-{high} mg/kg)"
                )
            break
    if condition_label.startswith('water_deficit') and soil_hum is not None:
        lines.append(
            f"Humedad del suelo: {soil_hum:.1f}% (alerta <{humidity_alerts['deficit']}%, óptimo {opt_params['soil_hum'][0]}-{opt_params['soil_hum'][1]}%)"
        )
    if condition_label == 'excess_humidity_soil' and soil_hum is not None:
        lines.append(
            f"Humedad del suelo: {soil_hum:.1f}% (alerta >{humidity_alerts['excess']}%, óptimo {opt_params['soil_hum'][0]}-{opt_params['soil_hum'][1]}%)"
        )
    if condition_label.startswith('precip_saturated_') and soil_hum is not None:
        lines.append(
            f"Humedad del suelo: {soil_hum:.1f}% con precipitación activa (óptimo {opt_params['soil_hum'][0]}-{opt_params['soil_hum'][1]}%)"
        )
    if 'fungal_risk' in condition_label or 'pest_risk' in condition_label:
        if air_hum is not None:
            lines.append(f"Humedad del aire: {air_hum:.1f}%")
        if temp is not None:
            lines.append(f"Temperatura: {temp:.1f} °C")
    if condition_label == 'cold_stress_high_altitude' and altitude is not None and temp is not None:
        lines.append(f"Altitud: {altitude:.0f} msnm con temperatura {temp:.1f} °C (riesgo de frío)")
    if condition_label == 'heat_stress_low_altitude' and altitude is not None and temp is not None:
        lines.append(f"Altitud: {altitude:.0f} msnm con temperatura {temp:.1f} °C (riesgo de calor)")
    if (condition_label.startswith('temp_low_stage_') or condition_label.startswith('temp_high_stage_')) and temp is not None:
        temp_range = opt_params.get('temp', (None, None))
        lines.append(
            f"Temperatura: {temp:.1f} °C (óptimo {temp_range[0]}-{temp_range[1]} °C)"
        )
    if condition_label == 'p_deficiency_moderate':
        lines.append('El fósforo está cerca del umbral crítico; repetir muestreo en 7-10 días y planificar refuerzo ligero si continúa <25 mg/kg.')
    return lines



def _collect_rule_conditions(row: pd.Series) -> List[str]:
    """Run the expert-rule detector on a single sensor row, returning the list
    of condition labels that apply before any ML prediction is considered."""
    stage_value = _get_sensor_value(row, 'plant_stage', 'plantStage')
    stage = stage_value if isinstance(stage_value, str) else 'default'
    if stage not in VALID_STAGES:
        stage = 'default'
    n = _safe_float_value(_get_sensor_value(row, 'nitrogen_mg_kg', 'nitrogen'))
    p = _safe_float_value(_get_sensor_value(row, 'phosphorus_mg_kg', 'phosphorus'))
    k = _safe_float_value(_get_sensor_value(row, 'potassium_mg_kg', 'potassium'))
    soil_hum = _safe_float_value(_get_sensor_value(row, 'soil_humidity_percent', 'soilHumidityPercent'))
    air_hum = _safe_float_value(_get_sensor_value(row, 'air_humidity_percent', 'airHumidityPercent'))
    temp = _safe_float_value(_get_sensor_value(row, 'celcius_grade_temperature', 'celciusGradeTemperature'))
    prec_raw = _get_sensor_value(row, 'precipitation_detected', 'precipitationDetected')
    try:
        prec = bool(int(prec_raw))
    except (TypeError, ValueError):
        prec = False
    altitude = _safe_float_value(_get_sensor_value(row, 'altitude_masl', 'altitudeMasl', 'altitude'))
    conditions: List[str] = []
    if altitude is not None and temp is not None:
        if altitude >= ALTITUDE_THRESHOLDS['muy_alto']['min'] and temp < ENV_ALERT_THRESHOLDS['temp_low_alert']:
            conditions.append('cold_stress_high_altitude')
        if altitude <= ALTITUDE_THRESHOLDS['bajo']['max'] and temp > ENV_ALERT_THRESHOLDS['temp_high_alert']:
            conditions.append('heat_stress_low_altitude')
    stage_opt = STAGE_OPTIMAL_PARAMS.get(stage, STAGE_OPTIMAL_PARAMS['default'])
    temp_range = stage_opt.get('temp') if isinstance(stage_opt, dict) else None
    if temp is not None and temp_range:
        temp_low_stage, temp_high_stage = temp_range
        if temp_low_stage is not None and temp < temp_low_stage:
            conditions.append(f'temp_low_stage_{stage}')
        if temp_high_stage is not None and temp > temp_high_stage:
            conditions.append(f'temp_high_stage_{stage}')

    def _nutrient_thresholds(key: str) -> Tuple[float, float]:
        opt_range = stage_opt.get(key, (None, None)) if isinstance(stage_opt, dict) else (None, None)
        opt_low = opt_range[0] if opt_range and opt_range[0] is not None else NPK_CRITICAL_LEVELS[key]['moderate_low']
        severe_default = NPK_CRITICAL_LEVELS[key]['severe_low']
        severe_thr = min(opt_low, severe_default)
        moderate_thr = max(opt_low, severe_thr + 1e-6)
        return severe_thr, moderate_thr

    nutrient_thresholds = {
        'n': _nutrient_thresholds('N'),
        'p': _nutrient_thresholds('P'),
        'k': _nutrient_thresholds('K'),
    }

    def _apply_nutrient_threshold(value: Optional[float], nutrient_key: str):
        if value is None:
            return
        severe_thr, moderate_thr = nutrient_thresholds[nutrient_key]
        severe_label = f"{nutrient_key}_deficiency_severe"
        moderate_label = f"{nutrient_key}_deficiency_moderate"
        if value < severe_thr:
            conditions.append(severe_label)
        elif value < moderate_thr:
            if severe_label not in conditions:
                conditions.append(moderate_label)

    _apply_nutrient_threshold(n, 'n')
    _apply_nutrient_threshold(p, 'p')
    _apply_nutrient_threshold(k, 'k')

    def _apply_excess_threshold(value: Optional[float], nutrient_key: str, upper_key: str):
        if value is None:
            return
        stage_range = stage_opt.get(upper_key) if isinstance(stage_opt, dict) else None
        stage_high = stage_range[1] if stage_range else None
        if stage_high is None:
            stage_high = NPK_CRITICAL_LEVELS[upper_key]['optimal_high']
        severe_thr = NPK_CRITICAL_LEVELS[upper_key]['excess']
        severe_label = f"{nutrient_key}_excess_severe"
        moderate_label = f"{nutrient_key}_excess_moderate"
        trigger_severe = severe_thr is not None and value > severe_thr
        trigger_moderate = stage_high is not None and value > stage_high
        if trigger_severe:
            conditions.append(severe_label)
        elif trigger_moderate:
            conditions.append(moderate_label)
    soil_range = stage_opt.get('soil_hum') if isinstance(stage_opt, dict) else None
    soil_opt_low = soil_range[0] if soil_range else None
    soil_opt_high = soil_range[1] if soil_range else None
    deficit_threshold = HUMIDITY_ALERT_THRESHOLDS.get(stage, HUMIDITY_ALERT_THRESHOLDS['default'])['deficit']
    excess_threshold = HUMIDITY_ALERT_THRESHOLDS.get(stage, HUMIDITY_ALERT_THRESHOLDS['default'])['excess']
    if soil_hum is not None:
        if soil_opt_low is not None and soil_hum < soil_opt_low and not prec:
            conditions.append(f'water_deficit_{stage}')
        elif soil_opt_high is not None and prec and soil_hum >= soil_opt_high:
            conditions.append(f'precip_saturated_{stage}')
        elif soil_hum > excess_threshold:
            conditions.append('excess_humidity_soil')
    if air_hum is not None and air_hum > ENV_ALERT_THRESHOLDS['air_hum_fungal_risk']:
        if stage == 'maduracion':
            conditions.append('fungal_risk_ojo_gallo_maduracion')
        if stage == 'floracion':
            conditions.append('fungal_risk_botrytis_floracion')
        if stage == 'vegetativo' and temp is not None and 18 <= temp <= 25:
            conditions.append('fungal_risk_roya_vegetativo')
    if stage in ['fructificacion', 'maduracion'] and air_hum is not None and air_hum > ENV_ALERT_THRESHOLDS['air_hum_broca_risk'] and temp is not None and temp > ENV_ALERT_THRESHOLDS['temp_broca_risk']:
        conditions.append('pest_risk_broca_fructificacion')
    _apply_excess_threshold(n, 'n', 'N')
    _apply_excess_threshold(p, 'p', 'P')
    _apply_excess_threshold(k, 'k', 'K')
    if not conditions:
        conditions.append(f'optimal_{stage}')
    return conditions



def _get_condition_label_from_rules(row: pd.Series) -> str:
    """Convenience helper used when the RF artifact is missing/failing."""
    conditions = _collect_rule_conditions(row)
    if not conditions:
        return 'optimal_default'
    return conditions[0]



def load_model_artifact(path: str):
    """Load the serialized Random Forest artifact while ensuring the custom
    ThresholdedPipeline class is available for unpickling."""
    if not path:
        logger.error('Model artifact path not configured.')
        return None
    artifact_path = Path(path)
    if not artifact_path.exists():
        logger.error(f'Model artifact not found at {artifact_path}')
        return None
    try:
        # Ensure pickled reference '__main__.ThresholdedPipeline' resolves when running under uvicorn.
        import sys, types
        main_mod = sys.modules.get('__main__')
        if main_mod is None or not hasattr(main_mod, '__dict__'):
            main_mod = types.ModuleType('__main__')
            sys.modules['__main__'] = main_mod
        if not hasattr(main_mod, 'ThresholdedPipeline'):
            setattr(main_mod, 'ThresholdedPipeline', ThresholdedPipeline)
        artifact = joblib.load(artifact_path)
    except Exception as exc:
        logger.error(f'Error loading model artifact from {artifact_path}: {exc}', exc_info=True)
        return None
    if not isinstance(artifact, dict):
        logger.error('Loaded artifact has unexpected format (expected dict).')
        return None
    if 'pipeline' not in artifact:
        logger.error("Loaded artifact missing 'pipeline' key.")
        return None
    return artifact


def predict_condition(sensor_data: Dict[str, Any], model_artifact) -> str:
    """Run inference if the RF pipeline is available, otherwise fall back to
    rules. Always logs errors so the API can keep serving recommendations."""
    if not model_artifact:
        logger.warning('Model artifact not available; falling back to expert rules.')
        return _get_condition_label_from_rules(pd.Series(sensor_data))

    pipeline = model_artifact.get('pipeline')
    feature_columns = model_artifact.get('feature_columns')
    if pipeline is None or not feature_columns:
        logger.error('Model artifact is incomplete; using rules fallback.')
        return _get_condition_label_from_rules(pd.Series(sensor_data))

    input_row = {column: sensor_data.get(column) for column in feature_columns}
    input_df = pd.DataFrame([input_row], columns=feature_columns)

    try:
        prediction = pipeline.predict(input_df)[0]
        logger.info(f'Predicted condition label: {prediction}')
        return str(prediction)
    except Exception as exc:
        logger.error(f'Prediction error: {exc}', exc_info=True)
        return _get_condition_label_from_rules(pd.Series(sensor_data))






def get_altitude_band_info(altitude: Optional[float]):
    """Clasifica la altitud y devuelve (clave, etiqueta, mensaje, catálogo, valor)."""
    if altitude is None:
        return None
    try:
        altitude_val = int(float(altitude))
    except (TypeError, ValueError):
        return None

    if altitude_val < ALTITUDE_THRESHOLDS['bajo']['min']:
        band_key = 'bajo'
    elif altitude_val <= ALTITUDE_THRESHOLDS['bajo']['max']:
        band_key = 'bajo'
    elif ALTITUDE_THRESHOLDS['medio']['min'] <= altitude_val <= ALTITUDE_THRESHOLDS['medio']['max']:
        band_key = 'medio'
    elif ALTITUDE_THRESHOLDS['alto']['min'] <= altitude_val <= ALTITUDE_THRESHOLDS['alto']['max']:
        band_key = 'alto'
    else:
        band_key = 'muy_alto'

    band = ALTITUDE_THRESHOLDS[band_key]
    return (band_key, band['label'], band['message'], band['catalog_key'], altitude_val)




def translate_condition_label(condition_label: str, stage: str) -> str:
    """Convert internal labels (snake_case) into Spanish phrases for the API."""
    stage_es = STAGE_NAME_ES.get(stage, stage.capitalize())

    if condition_label.startswith('optimal'):
        suffix = condition_label.split('optimal_', 1)[-1] if '_' in condition_label else ''
        if suffix and suffix == stage:
            return f'Óptimo ({stage_es})'
        if suffix:
            stage_text = STAGE_NAME_ES.get(suffix, suffix.capitalize())
            return f'Óptimo ({stage_text})'
        return 'Óptimo'

    if '_deficiency_' in condition_label:
        parts = condition_label.split('_')
        nutrient = NUTRIENT_NAME_ES.get(parts[0].upper(), parts[0])
        severity = SEVERITY_NAME_ES.get(parts[-1], parts[-1])
        return f'Deficiencia {severity} de {nutrient}'

    if '_excess_' in condition_label:
        parts = condition_label.split('_')
        nutrient = NUTRIENT_NAME_ES.get(parts[0].upper(), parts[0])
        severity = SEVERITY_NAME_ES.get(parts[-1], parts[-1])
        return f'Exceso {severity} de {nutrient}'

    if condition_label.endswith('_excess') or condition_label in {'n_excess', 'p_excess', 'k_excess'}:
        nutrient_key = condition_label.split('_')[0].upper()
        nutrient = NUTRIENT_NAME_ES.get(nutrient_key, nutrient_key)
        return f'Exceso de {nutrient}'

    if condition_label.startswith('water_deficit_'):
        stage_key = condition_label.split('water_deficit_', 1)[-1]
        stage_text = STAGE_NAME_ES.get(stage_key, stage_key.capitalize())
        return f'Déficit hídrico ({stage_text})'

    if condition_label == 'excess_humidity_soil':
        return 'Exceso de humedad en suelo'

    if condition_label == 'cold_stress_high_altitude':
        return 'Estrés por frío en zona alta'

    if condition_label == 'heat_stress_low_altitude':
        return 'Estrés por calor en zona baja'

    if condition_label.startswith('fungal_risk_'):
        remainder = condition_label.split('fungal_risk_', 1)[-1]
        stage_key = None
        for key in STAGE_NAME_ES:
            if remainder.endswith(f'_{key}'):
                stage_key = key
                agent_key = remainder[:-(len(key) + 1)]
                break
        else:
            agent_key = remainder
        agent = FUNGAL_AGENT_ES.get(agent_key, agent_key.replace('_', ' ').title())
        stage_text = STAGE_NAME_ES.get(stage_key, stage_es) if stage_key else stage_es
        return f'Riesgo fúngico: {agent} ({stage_text})'

    if condition_label.startswith('pest_risk_'):
        remainder = condition_label.split('pest_risk_', 1)[-1]
        stage_key = None
        for key in STAGE_NAME_ES:
            if remainder.endswith(f'_{key}'):
                stage_key = key
                agent_key = remainder[:-(len(key) + 1)]
                break
        else:
            agent_key = remainder
        agent = PEST_AGENT_ES.get(agent_key, agent_key.replace('_', ' ').title())
        stage_text = STAGE_NAME_ES.get(stage_key, stage_es) if stage_key else stage_es
        return f'Riesgo de plaga: {agent} ({stage_text})'

    if condition_label.startswith('temp_low_stage_'):
        stage_key = condition_label.split('temp_low_stage_', 1)[-1]
        stage_text = STAGE_NAME_ES.get(stage_key, stage_key.capitalize())
        return f'Temperatura baja ({stage_text})'

    if condition_label.startswith('temp_high_stage_'):
        stage_key = condition_label.split('temp_high_stage_', 1)[-1]
        stage_text = STAGE_NAME_ES.get(stage_key, stage_key.capitalize())
        return f'Temperatura alta ({stage_text})'

    if condition_label.startswith('precip_saturated_'):
        stage_key = condition_label.split('precip_saturated_', 1)[-1]
        stage_text = STAGE_NAME_ES.get(stage_key, stage_key.capitalize())
        return f'Suelo saturado por lluvia ({stage_text})'

    if condition_label.startswith('error'):
        return 'Error interno'

    if condition_label == 'unknown_condition':
        return 'Condición no reconocida'

    return condition_label.replace('_', ' ').capitalize()



def format_detailed_recommendation(condition_label: str, sensor_data: Dict[str, Any]) -> RecommendationOutput:
    """Build the human-readable recommendation (text + products) by merging
    ML prediction, rule-based alerts, catalog actions and contextual metadata."""
    device_id = _get_sensor_value(sensor_data, 'device_hub_id', 'deviceHubId') or sensor_data.get('device_hub_id') or 'Unknown_Device'
    stage = _get_sensor_value(sensor_data, 'plant_stage', 'plantStage') or sensor_data.get('plant_stage') or 'default'
    if stage not in VALID_STAGES:
        stage = 'default'

    base_time_iso = sensor_data.get('base_time_iso')
    base_time_dt = None
    if base_time_iso:
        try:
            base_time_dt = datetime.fromisoformat(base_time_iso).astimezone(timezone.utc)
        except Exception:
            logger.warning(f"Could not re-parse base_time_iso: {base_time_iso}")

    def safe_float(value: Optional[Any]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    description_lines: List[str] = [
        f"Etapa: {STAGE_NAME_ES.get(stage, stage.capitalize())}",
    ]

    altitude_value = safe_float(_get_sensor_value(sensor_data, 'altitude_masl', 'altitudeMasl', 'altitude'))
    altitude_info = get_altitude_band_info(altitude_value)
    altitude_actions: List[Dict[str, Any]] = []
    if altitude_info:
        band_key, zone_label, zone_message, altitude_catalog_key, altitude_val = altitude_info
        description_lines.append(f"Altitud: {altitude_val:.0f} msnm ({zone_label})")
        if zone_message:
            description_lines.append(f"Nota por altitud: {zone_message}")
        stage_notes = STAGE_ALTITUDE_NOTES.get(stage, {})
        stage_alt_note = stage_notes.get(band_key) if isinstance(stage_notes, dict) else None
        if stage_alt_note:
            description_lines.append(f"Recomendación etapa-altitud: {stage_alt_note}")
        if altitude_catalog_key:
            altitude_actions = PRODUCTS_CATALOG.get(altitude_catalog_key, []) or []

    original_prediction = condition_label
    rule_conditions = _collect_rule_conditions(pd.Series(sensor_data))
    if condition_label not in rule_conditions:
        rule_conditions.insert(0, condition_label)

    def _filter_false_nutrient_alerts(labels: List[str]) -> List[str]:
        stage_params = STAGE_OPTIMAL_PARAMS.get(stage, STAGE_OPTIMAL_PARAMS['default'])
        value_cache: Dict[str, Optional[float]] = {}

        def nutrient_value(letter_upper: str) -> Optional[float]:
            if letter_upper not in value_cache:
                keys = NUTRIENT_VALUE_KEYS.get(letter_upper)
                if not keys:
                    value_cache[letter_upper] = None
                else:
                    raw = _get_sensor_value(sensor_data, *keys)
                    value_cache[letter_upper] = _safe_float_value(raw)
            return value_cache[letter_upper]

        def lower_bound(letter_upper: str) -> Optional[float]:
            stage_range = stage_params.get(letter_upper)
            low = stage_range[0] if stage_range else None
            if low is None:
                low = NPK_CRITICAL_LEVELS[letter_upper]['moderate_low']
            return low

        def upper_bound(letter_upper: str) -> Optional[float]:
            stage_range = stage_params.get(letter_upper)
            high = stage_range[1] if stage_range else None
            if high is None:
                high = NPK_CRITICAL_LEVELS[letter_upper]['optimal_high']
            return high

        filtered: List[str] = []
        nutrient_pairs = [('n', 'N'), ('p', 'P'), ('k', 'K')]
        for label in labels:
            skip = False
            for lower, upper in nutrient_pairs:
                if label.startswith(f"{lower}_deficiency"):
                    value = nutrient_value(upper)
                    if value is not None:
                        bound = lower_bound(upper)
                        if bound is not None and value >= bound:
                            logger.debug(
                                "Dropping %s because %s=%.2f within optimal (>= %.2f).",
                                label,
                                upper,
                                value,
                                bound,
                            )
                            skip = True
                    break
                if label.startswith(f"{lower}_excess"):
                    value = nutrient_value(upper)
                    if value is not None:
                        if label.endswith('_severe'):
                            bound = NPK_CRITICAL_LEVELS[upper]['excess']
                            compare = value <= bound if bound is not None else False
                        else:
                            bound = upper_bound(upper)
                            compare = value <= bound if bound is not None else False
                        if bound is not None and compare:
                            logger.debug(
                                "Dropping %s because %s=%.2f within safe limit (<= %.2f).",
                                label,
                                upper,
                                value,
                                bound,
                            )
                            skip = True
                    break
            if not skip:
                filtered.append(label)
        return filtered

    rule_conditions = _filter_false_nutrient_alerts(rule_conditions)

    # Avoid duplicated severities for the same nutrient (e.g., N moderate and severe).
    # If both exist, keep the model-predicted one; otherwise keep the most severe.
    def _dedupe_npk_severity(labels: List[str], predicted: str) -> List[str]:
        families = {
            'n_def': ('n_deficiency_severe', 'n_deficiency_moderate'),
            'p_def': ('p_deficiency_severe', 'p_deficiency_moderate'),
            'k_def': ('k_deficiency_severe', 'k_deficiency_moderate'),
            'n_exc': ('n_excess_severe', 'n_excess_moderate'),
            'p_exc': ('p_excess_severe', 'p_excess_moderate'),
            'k_exc': ('k_excess_severe', 'k_excess_moderate'),
        }
        label_set = set(labels)
        to_remove: Set[str] = set()
        for sev, mod in families.values():
            if sev in label_set and mod in label_set:
                # Prefer the model-predicted label when present among the pair.
                if predicted == sev:
                    to_remove.add(mod)
                elif predicted == mod:
                    to_remove.add(sev)
                else:
                    # Default: keep the most severe, drop moderate
                    to_remove.add(mod)
        if not to_remove:
            return labels
        return [lbl for lbl in labels if lbl not in to_remove]

    rule_conditions = _dedupe_npk_severity(rule_conditions, original_prediction)

    if not rule_conditions:
        fallback_label = f"optimal_{stage}" if stage in VALID_STAGES else 'optimal_default'
        rule_conditions = [fallback_label]

    if original_prediction in rule_conditions:
        condition_label = original_prediction
    else:
        condition_label = rule_conditions[0]

    condition_entries: List[Tuple[str, str, int, List[str], List[Dict[str, Any]]]] = []
    seen_labels: Set[str] = set()
    for label in rule_conditions:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        rec_type, urgency = _classify_condition_meta(label, stage)
        context_lines = _build_condition_context_lines(label, stage, sensor_data)
        actions = _resolve_catalog_actions(label, stage)
        condition_entries.append((label, rec_type, urgency, context_lines, actions))

    # Ordenar por urgencia (mayor a menor), manteniendo orden original en empates.
    condition_entries.sort(key=lambda item: (-item[2], rule_conditions.index(item[0])))

    resolved_condition_label = condition_label

    if condition_entries:
        primary_label, primary_rec_type, primary_urgency, primary_context, _ = condition_entries[0]
        resolved_condition_label = primary_label
        description_lines.append(f"Condición detectada: {translate_condition_label(primary_label, stage)}")
        description_lines.append(f"Tipo: {primary_rec_type} | Urgencia: {primary_urgency}/5")
        if primary_context:
            description_lines.append('Datos relevantes:')
            description_lines.extend(f"- {item}" for item in primary_context)
    else:
        primary_rec_type, primary_urgency = 'optimo', 1

    all_rec_types: Set[str] = {primary_rec_type}
    max_urgency = primary_urgency

    additional_entries = condition_entries[1:]
    for label, rec_type, urgency, context_lines, _ in additional_entries:
        all_rec_types.add(rec_type)
        max_urgency = max(max_urgency, urgency)
        description_lines.append('')
        description_lines.append(f"Alerta adicional: {translate_condition_label(label, stage)}")
        description_lines.append(f"Tipo: {rec_type} | Urgencia: {urgency}/5")
        for item in context_lines:
            description_lines.append(f"- {item}")
        if PRODUCTS_CATALOG.get(label):
            description_lines.append('Acciones específicas disponibles para esta alerta: ver sección Acciones recomendadas.')

    # Construir listado completo de acciones por condición.
    description_lines.append('')
    description_lines.append('Acciones recomendadas:')
    action_counter = 1
    products_struct_list: List[RecommendationProduct] = []
    seen_product_keys: Set[str] = set()

    def _emit_actions(actions: List[Dict[str, Any]], heading: Optional[str] = None) -> None:
        nonlocal action_counter
        if not actions:
            return
        if heading:
            if description_lines and description_lines[-1]:
                description_lines.append('')
            description_lines.append(heading)
        local_index = 1
        for action in actions:
            name = action.get('name') or action.get('id') or 'Acción'
            key = name or repr(action)
            if key in seen_product_keys:
                continue
            seen_product_keys.add(key)
            if heading:
                if local_index == 1:
                    prefix = '	- '
                else:
                    prefix = f"	{local_index - 1}) "
            else:
                prefix = f"{action_counter}) "
            description_lines.append(f"{prefix}{name}")
            detail_indent = '		- ' if heading else '	- '
            if action.get('composition'):
                description_lines.append(f"{detail_indent}Composición: {action['composition']}")
            if action.get('dose'):
                description_lines.append(f"{detail_indent}Dosis: {action['dose']}")
            if action.get('method'):
                description_lines.append(f"{detail_indent}Método: {action['method']}")
            if action.get('timing_frequency'):
                description_lines.append(f"{detail_indent}Frecuencia: {action['timing_frequency']}")
            if action.get('stage_applicability'):
                description_lines.append(f"{detail_indent}Etapa aplicable: {', '.join(action['stage_applicability'])}")
            if action.get('notes'):
                description_lines.append(f"{detail_indent}Notas: {action['notes']}")
            if action.get('distributor'):
                description_lines.append(f"{detail_indent}Distribuidor: {action['distributor']}")
            if action.get('senasa_registro'):
                description_lines.append(f"{detail_indent}SENASA: {action['senasa_registro']}")
            products_struct_list.append(RecommendationProduct(**action))
            action_counter += 1
            if heading:
                local_index += 1


    for label, _, _, _, actions in condition_entries:
        if actions:
            heading = f"Acciones para {translate_condition_label(label, stage)}:" if len(condition_entries) > 1 else None
            _emit_actions(actions, heading)

    if altitude_actions:
        _emit_actions(altitude_actions, 'Acciones por altitud:')

    if action_counter == 1:
        description_lines.append('No se encontraron acciones específicas, consulte a un agrónomo para un plan detallado.')

    recommendation_type = primary_rec_type if len(all_rec_types) == 1 else 'mixto'

    condition_display = translate_condition_label(resolved_condition_label, stage)

    return RecommendationOutput(
        device_hub_id=device_id,
        recommendation_description="\n".join(description_lines),
        recommendation_type=recommendation_type,
        condition_detected=condition_display,
        condition_code=condition_display,
        condition_internal_code=resolved_condition_label,
        urgency_level=max_urgency,
        specific_products=products_struct_list,
        based_on_data_at=base_time_dt
    )
def get_altitude_band_info(altitude: Optional[float]):
    """Clasifica la altitud y devuelve (clave, etiqueta, mensaje, catálogo, valor)."""
    if altitude is None:
        return None
    try:
        altitude_val = int(float(altitude))
    except (TypeError, ValueError):
        return None

    if altitude_val < ALTITUDE_THRESHOLDS['bajo']['min']:
        band_key = 'bajo'
    elif altitude_val <= ALTITUDE_THRESHOLDS['bajo']['max']:
        band_key = 'bajo'
    elif ALTITUDE_THRESHOLDS['medio']['min'] <= altitude_val <= ALTITUDE_THRESHOLDS['medio']['max']:
        band_key = 'medio'
    elif ALTITUDE_THRESHOLDS['alto']['min'] <= altitude_val <= ALTITUDE_THRESHOLDS['alto']['max']:
        band_key = 'alto'
    else:
        band_key = 'muy_alto'

    band = ALTITUDE_THRESHOLDS[band_key]
    return (band_key, band['label'], band['message'], band['catalog_key'], altitude_val)




def translate_condition_label(condition_label: str, stage: str) -> str:
    stage_es = STAGE_NAME_ES.get(stage, stage.capitalize())

    if condition_label.startswith('optimal'):
        suffix = condition_label.split('optimal_', 1)[-1] if '_' in condition_label else ''
        if suffix and suffix == stage:
            return f'Óptimo ({stage_es})'
        if suffix:
            stage_text = STAGE_NAME_ES.get(suffix, suffix.capitalize())
            return f'Óptimo ({stage_text})'
        return 'Óptimo'

    if '_deficiency_' in condition_label:
        parts = condition_label.split('_')
        nutrient = NUTRIENT_NAME_ES.get(parts[0].upper(), parts[0])
        severity = SEVERITY_NAME_ES.get(parts[-1], parts[-1])
        return f'Deficiencia {severity} de {nutrient}'

    if condition_label.endswith('_excess') or condition_label in {'n_excess', 'p_excess', 'k_excess'}:
        nutrient_key = condition_label.split('_')[0].upper()
        nutrient = NUTRIENT_NAME_ES.get(nutrient_key, nutrient_key)
        return f'Exceso de {nutrient}'

    if condition_label.startswith('water_deficit_'):
        stage_key = condition_label.split('water_deficit_', 1)[-1]
        stage_text = STAGE_NAME_ES.get(stage_key, stage_key.capitalize())
        return f'Déficit hídrico ({stage_text})'

    if condition_label == 'excess_humidity_soil':
        return 'Exceso de humedad en suelo'

    if condition_label == 'cold_stress_high_altitude':
        return 'Estrés por frío en zona alta'

    if condition_label == 'heat_stress_low_altitude':
        return 'Estrés por calor en zona baja'

    if condition_label.startswith('fungal_risk_'):
        remainder = condition_label.split('fungal_risk_', 1)[-1]
        stage_key = None
        for key in STAGE_NAME_ES:
            if remainder.endswith(f'_{key}'):
                stage_key = key
                agent_key = remainder[:-(len(key) + 1)]
                break
        else:
            agent_key = remainder
        agent = FUNGAL_AGENT_ES.get(agent_key, agent_key.replace('_', ' ').title())
        stage_text = STAGE_NAME_ES.get(stage_key, stage_es) if stage_key else stage_es
        return f'Riesgo fúngico: {agent} ({stage_text})'

    if condition_label.startswith('pest_risk_'):
        remainder = condition_label.split('pest_risk_', 1)[-1]
        stage_key = None
        for key in STAGE_NAME_ES:
            if remainder.endswith(f'_{key}'):
                stage_key = key
                agent_key = remainder[:-(len(key) + 1)]
                break
        else:
            agent_key = remainder
        agent = PEST_AGENT_ES.get(agent_key, agent_key.replace('_', ' ').title())
        stage_text = STAGE_NAME_ES.get(stage_key, stage_es) if stage_key else stage_es
        return f'Riesgo de plaga: {agent} ({stage_text})'

    if condition_label.startswith('error'):
        return 'Error interno'

    if condition_label == 'unknown_condition':
        return 'Condición no reconocida'

    return condition_label.replace('_', ' ').capitalize()

# --- Carga del Modelo al Iniciar ---
model_data_global = None
try:
    model_data_global = load_model_artifact(MODEL_ARTIFACT_PATH)
    if model_data_global:
        logger.info(f'Model artifact loaded from {MODEL_ARTIFACT_PATH}')
    else:
        logger.warning('Modelo ML no disponible (artifact missing o inválido).')
except Exception as e: logger.critical(f'Error crítico al cargar el modelo: {e}', exc_info=True)

# --- Endpoints API ---
@app.post("/trigger_recommendation_generation/")
async def trigger_recommendation_generation_and_send(refresh_map: bool = False):
    """Fetch context + latest sensor data, run inference/rules, assemble
    recommendations and push them to the .NET backend."""
    logger.info("Triggering recommendation generation process (v3.2 - Expert Integrated)...")
    try:
        if refresh_map:
            device_stage_map_cache["data"] = None
            device_stage_map_cache["timestamp"] = 0
        device_context_map, latest_sensor_data_list = await asyncio.gather(
            get_device_stage_map(),
            get_latest_sensor_data(),
        )
    except Exception as exc:
        logger.error(f"Failed API fetching: {exc}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Error fetching backend data: {exc}")

    if not latest_sensor_data_list:
        return {"message": "No new sensor data found."}

    recommendations_to_send: List[RecommendationOutput] = []
    processed_devices: Set[str] = set()

    for sensor_record in latest_sensor_data_list:
        device_hub_id = sensor_record.get("deviceHubId")
        if not device_hub_id or device_hub_id in processed_devices:
            continue
        processed_devices.add(device_hub_id)

        stage_info = device_context_map.get(device_hub_id, {}) if device_context_map else {}
        if isinstance(stage_info, dict):
            stage_value = stage_info.get('stage', 'default') or 'default'
            altitude_value = stage_info.get('altitude')
        else:
            stage_value = stage_info or 'default'
            altitude_value = None
        if altitude_value is not None:
            try:
                altitude_value = int(float(altitude_value))
            except (TypeError, ValueError):
                altitude_value = None

        logger.info(f"Device {device_hub_id}: stage={stage_value}, altitude={altitude_value}")

        prepared_data = {
            "device_hub_id": device_hub_id,
            "deviceHubId": device_hub_id,
            "plant_stage": stage_value,
            "plantStage": stage_value,
            "airHumidityPercent": sensor_record.get("airHumidityPercent"),
            "soilHumidityPercent": sensor_record.get("soilHumidityPercent"),
            "celciusGradeTemperature": sensor_record.get("celciusGradeTemperature"),
            "precipitationDetected": int(sensor_record.get("precipitationDetected") or 0),
            "nitrogen": sensor_record.get("nitrogen"),
            "phosphorus": sensor_record.get("phosphorus"),
            "potassium": sensor_record.get("potassium"),
            "altitude": altitude_value,
            "altitudeMasl": altitude_value,
            "base_time_iso": None,
        }

        ts_str = sensor_record.get("createdAt") or sensor_record.get("timestamp")
        iso_timestamp = None
        if ts_str:
            try:
                iso_timestamp = pd.to_datetime(ts_str, errors='raise', utc=True).isoformat()
                prepared_data["base_time_iso"] = iso_timestamp
            except Exception:
                logger.warning(f"Could not parse timestamp {ts_str} for {device_hub_id}")
        record_signature = _compute_sensor_signature(sensor_record, iso_timestamp)
        last_signature = last_sensor_signature_cache.get(device_hub_id)
        if record_signature and last_signature == record_signature:
            logger.info(f"Skipping device {device_hub_id}: no new sensor data since last run.")
            continue

        predicted_condition = predict_condition(prepared_data, model_data_global)
        try:
            detailed_recommendation = format_detailed_recommendation(predicted_condition, prepared_data)
            recommendations_to_send.append(detailed_recommendation)
            if record_signature:
                last_sensor_signature_cache[device_hub_id] = record_signature
        except Exception as exc:
            logger.error(f"Error formatting rec for {device_hub_id} (Cond: {predicted_condition}): {exc}", exc_info=True)
            base_time_dt = None
            if prepared_data["base_time_iso"]:
                try:
                    base_time_dt = datetime.fromisoformat(prepared_data["base_time_iso"]).astimezone(timezone.utc)
                except Exception:
                    pass
            stage_for_display = stage_value if stage_value in VALID_STAGES else 'default'
            fallback_condition_text = translate_condition_label(predicted_condition, stage_for_display)
            recommendations_to_send.append(
                RecommendationOutput(
                    device_hub_id=device_hub_id,
                    recommendation_description="Error interno formateando recomendación.",
                    recommendation_type="error",
                    condition_detected=fallback_condition_text,
                    condition_code=fallback_condition_text,
                    condition_internal_code=predicted_condition,
                    urgency_level=5,
                    based_on_data_at=base_time_dt,
                )
            )
            if record_signature:
                last_sensor_signature_cache[device_hub_id] = record_signature

    sent_count = 0
    error_count = 0
    results_summary = []

    if not recommendations_to_send:
        return {"message": "No new sensor data found."}
    if not JWT_TOKEN or len(JWT_TOKEN) < 50:
        return {
            "message": "Recomendaciones generadas pero NO ENVIADAS (JWT inválido).",
            "status": "send_config_error",
        }

    headers = {"Authorization": f"Bearer {JWT_TOKEN}", "Content-Type": "application/json"}
    logger.info(f"Attempting to send {len(recommendations_to_send)} detailed recommendations to {AZURE_RECOMMENDATION_ENDPOINT}")

    for rec in recommendations_to_send:
        payload = {
            "recommendationDescription": rec.recommendation_description,
            "deviceHubId": rec.device_hub_id,
        }
        dev_id = rec.device_hub_id
        try:
            response = requests.post(AZURE_RECOMMENDATION_ENDPOINT, json=payload, headers=headers, timeout=15)
            if 200 <= response.status_code < 300:
                logger.info(f"Sent OK for {dev_id}...")
                sent_count += 1
                results_summary.append({"device": dev_id, "status": "sent"})
            else:
                logger.error(f"Send Error for {dev_id}. Status: {response.status_code}, Resp: {response.text[:300]}")
                error_count += 1
                results_summary.append({"device": dev_id, "status": "error", "code": response.status_code})
        except Exception as exc:
            logger.error(f"Send Exception for {dev_id}: {exc}")
            error_count += 1
            results_summary.append({"device": dev_id, "status": "exception"})

    final_message = (
        f"Processing complete (v3.2 Expert RF). Generated: {len(recommendations_to_send)}. Sent: {sent_count}. Errors: {error_count}."
    )
    logger.info(final_message)
    status = "success" if error_count == 0 else ("partial_success" if sent_count > 0 else "send_failed")
    return {"message": final_message, "status": status, "details": results_summary}


@app.get("/health")
def health_check():
    token_ok = bool(JWT_TOKEN and len(JWT_TOKEN) > 50)
    model_ok = model_data_global is not None
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat(), "token_configured": token_ok, "model_loaded": model_ok}

# Debug endpoint to inspect the current device -> {stage, altitude} mapping
@app.get("/debug/device-stage-map")
async def debug_device_stage_map(refresh: bool = False):
    """Expose the cached device context map so operators can verify stage +
    altitude provenance straight from the API."""
    try:
        if refresh:
            device_stage_map_cache["data"] = None
            device_stage_map_cache["timestamp"] = 0
        mapping = await get_device_stage_map()
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
        raise HTTPException(status_code=500, detail=f"Error building device-stage map: {exc}")


