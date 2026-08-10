# CoffeeTech — Servicio de recomendaciones (Model-ML)

Servicio FastAPI que genera recomendaciones agronómicas para café a partir de la telemetría
de sensores de la finca El Milagro (San Ignacio, Cajamarca), certificada USDA-Organic +
Fairtrade con la cooperativa APROCASSI.

El servicio **consume** el backend .NET/Azure (lee telemetría y contexto por GET, envía
recomendaciones por POST); no modifica sus endpoints, contratos ni base de datos. Toda la
lógica de diagnóstico vive del lado del modelo.

## Cómo funciona

Por cada corrida, y para cada dispositivo:

1. **Ingesta (GET).** Se obtiene la telemetría (`data-records`) y el contexto —etapa
   fenológica por sección y altitud de la finca— (`devices`, `assignments`, `sections`,
   `farms`).
2. **Recomendaciones inmediatas.** El motor de reglas agronómicas (`coffeetech_rules_v5`)
   evalúa la ventana reciente de lecturas: diagnóstico NPK (con dominio calibrado del sensor),
   riego, semáforo térmico y reglas inferenciales de plagas/enfermedades. Cada alerta habla en
   dos voces (agricultor y agrónomo) y marca cuándo derivar a un técnico.
3. **Recomendaciones anticipadas.** El pronóstico de microclima (`coffeetech_forecast`)
   predice la temperatura y la humedad de aire a 2 h y 6 h; con esas lecturas "futuras" se
   evalúan las reglas climáticas para avisar antes de que se abra una ventana de riesgo
   (estrés térmico o condiciones favorables a hongos).
4. **Composición y envío (POST).** Inmediatas y anticipadas se combinan, se ordenan por
   severidad y se serializan en un único texto que se envía al backend con el mismo payload
   de siempre: `{ "recommendationDescription": <texto>, "deviceHubId": <id> }`. Se reenvía
   solo cuando el **diagnóstico cambia** (aparece/desaparece una alerta o cambia su severidad),
   no en cada corrida, para no saturar al agricultor con mensajes iguales.

El diagnóstico proviene **siempre** del motor de reglas; el pronóstico solo lo anticipa.
No hay clasificación por Random Forest: un RF entrenado para reproducir las etiquetas que generan
las mismas reglas con las mismas variables es circular, y su exactitud no tiene valor agronómico.

## Módulos

| Archivo | Rol |
|---|---|
| `main.py` | Servicio FastAPI: ingesta, orquestación y envío. |
| `coffeetech_rules_v5.py` | Motor de reglas agronómicas validado (diagnóstico). Solo stdlib. |
| `coffeetech_doses.py` | Capa de prescripción orgánica (producto/dosis, cobre con tope, N consultivo). Solo stdlib. |
| `coffeetech_recommendations.py` | Adaptador: alertas → recomendaciones, anticipadas y formato del payload. Solo stdlib. |
| `coffeetech_forecast.py` | Pronóstico de microclima (Random Forest no circular) + persistencia. |
| `coffeetech_weather.py` | Cliente de Open-Meteo: pronóstico horario y archivo del reanálisis, con caché. |
| `coffeetech_state.py` | Estado mínimo en disco: cuándo se envió por última vez cada recordatorio. Solo stdlib. |
| `tests/` | Pruebas de aceptación del motor y del pronóstico. |

## Ejecutar

Requiere Python 3.12. Con Python instalado y las dependencias en él:

```bash
# Preparar (una sola vez)
python -m pip install -r requirements.txt
copy .env.example .env      # y definir AZURE_JWT_TOKEN

# Arrancar el servicio
uvicorn main:app --reload

# Pruebas
python -m pytest tests/ -q
```

> En Windows usar `python` (no `python3`). Si no hay Python instalado, se puede instalar por
> usuario con `winget install Python.Python.3.12 --scope user` (abrir una terminal nueva
> después). Alternativa sin instalar nada global: `uv run --with-requirements requirements.txt
> uvicorn main:app --reload`.

Endpoints: `POST /trigger_recommendation_generation/` (orquestador), `GET /health`
(estado del token, motor y pronóstico) y `GET /debug/device-stage-map` (soporte).

Si falta `AZURE_JWT_TOKEN`, el servicio arranca pero omite las llamadas al backend de forma
controlada. Si faltan `scikit-learn` o los artefactos del pronóstico, funciona igual con solo
las recomendaciones inmediatas.

### ¿Cuándo se generan las recomendaciones?

El servicio no se dispara solo: hay que llamar a `POST /trigger_recommendation_generation/`.
Dos formas:

- **Scheduler interno (opcional).** Con `COFFEETECH_SCHEDULER_ENABLED=1` el propio servicio se
  dispara cada `COFFEETECH_SCHEDULER_MINUTES` (30 por defecto). Requiere que el proceso corra
  siempre encendido (p. ej. App Service con *Always On*). Está apagado por defecto.
- **Disparo externo.** Un cron o un Azure Timer/Logic App que haga el POST al endpoint. Útil
  si se prefiere separar el "cuándo" del servicio o escalar a varias instancias.

En ambos casos la deduplicación por contenido evita reenviar recomendaciones repetidas, así que
es seguro dispararlo con frecuencia.

## Pronóstico de microclima

```bash
# Reproducir la validación (dejando una campaña fuera, contra la persistencia)
python coffeetech_forecast.py validar

# Entrenar y persistir los modelos operativos en artifacts/
python coffeetech_forecast.py train
```

El bosque aprende el **cambio** respecto a la lectura actual (`y − lag_0`), no el valor futuro; al
predecir se le vuelve a sumar esa lectura. La generalización se estima **dejando una campaña
fuera** por turnos, no con un holdout interno —que probaría sobre un régimen ya visto—, y decide el
**mínimo** de los pliegues, nunca la media. La línea base es la persistencia: el bosque sólo cuenta
si la supera, y sólo se sirve el horizonte de **6 h**.

`validar` escribe `artifacts/forecast_validation.json`, que es donde viven las cifras vigentes, y
**falla** si `SERVED_HORIZONS_H` deja de coincidir con lo medido. El límite declarado: sigue siendo
**una sola finca**, y tres campañas del mismo hub no arreglan que sea un único emplazamiento.

## Pruebas

```bash
python -m pytest tests/ -q            # toda la suite
python -m pytest tests/test_acceptance.py -v   # solo motor de reglas (sin scikit-learn)
python -m pytest tests/test_forecast.py -v     # solo pronóstico (requiere scikit-learn)
```

## Seguridad

El token JWT se lee únicamente de la variable de entorno `AZURE_JWT_TOKEN` (sin valor por
defecto en el código). El archivo `.env` está fuera del control de versiones. El token que
estuvo hardcodeado en el historial del repositorio debe considerarse comprometido: rotarlo en
el backend y purgarlo del historial de git.
