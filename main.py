from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from datetime import datetime, timedelta
import requests

app = FastAPI(
    title="Agriculture Recommendation System API",
    description="API for agriculture recommendations about coffee production.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de datos con Pydantic
class Recommendation(BaseModel):
    device_hub_id: str
    recommendation_description: str

# Ruta del archivo CSV
file_path = './new_training_sensor_data_validated.csv'  # Ajusta la ruta al archivo CSV

# Función para procesar los datos y generar recomendaciones por device_hub_id
def analyze_data_and_generate_recommendations():
    df = pd.read_csv(file_path, parse_dates=['created_at'])

    if df.empty:
        raise ValueError("El archivo CSV está vacío o no contiene datos suficientes.")

    time_threshold = datetime.now() - timedelta(minutes=5)
    df_time_filtered = df[df['created_at'] >= time_threshold]

    if df_time_filtered.empty:
        raise HTTPException(status_code=400, detail="No hay datos en el rango de la última media hora.")

    recommendations = []

    for device_id in df_time_filtered['device_hub_id'].unique():
        df_device = df_time_filtered[df_time_filtered['device_hub_id'] == device_id]

        if df_device.empty:
            continue

        def generate_detailed_recommendation(row):
            # Reglas de negocio complejas para recomendaciones
            if row['soil_humidity_percent'] < 20 and row['precipitation_detected'] == 0 and row['celcius_grade_temperature'] > 28:
                return ('Urgente: La humedad del suelo está muy baja y la temperatura es alta. Se recomienda un riego profundo '
                        'de inmediato para evitar que las plantas sufran estrés. Riegue al final de la tarde para evitar la evaporación y '
                        'permitir que el agua se absorba bien durante la noche.')
            elif 20 <= row['soil_humidity_percent'] < 25 and row['precipitation_detected'] == 0:
                return ('Advertencia de riego: El suelo está seco y no ha habido lluvias recientes. Se recomienda un riego moderado '
                        'para mantener la planta en un estado saludable. Prefiera regar temprano en la mañana.')

            # Manejo de precipitaciones y humedad alta
            elif row['precipitation_detected'] == 1 and row['soil_humidity_percent'] > 40:
                return ('Alerta de lluvia y suelo húmedo: Se ha detectado precipitación y el suelo ya está bastante húmedo. Revise si hay '
                        'acumulación de agua cerca de las plantas para evitar enfermedades en las raíces y mejorar el drenaje si es necesario.')
            elif row['precipitation_detected'] == 1 and row['air_humidity_percent'] > 85:
                return ('Lluvias con alta humedad: Estas condiciones favorecen la aparición de hongos como la roya. Inspeccione las hojas '
                        'en busca de manchas y considere aplicar tratamientos preventivos para evitar que la enfermedad se propague.')

            # Control de enfermedades debido a condiciones ambientales
            elif row['celcius_grade_temperature'] > 30 and row['air_humidity_percent'] > 80:
                return ('Alerta de enfermedades: Las altas temperaturas y la humedad pueden propiciar la aparición de enfermedades como la '
                        'antracnosis y la roya. Podar las ramas para mejorar la circulación del aire y aplicar fungicidas naturales es una medida clave.')
            elif row['celcius_grade_temperature'] < 20 and row['soil_humidity_percent'] > 50:
                return ('Cuidado: La combinación de baja temperatura y alta humedad puede causar la pudrición de raíces. Verifique el drenaje y '
                        'retire el agua estancada que pueda estar cerca de las plantas.')

            # Control de plagas basado en condiciones específicas
            elif row['celcius_grade_temperature'] > 25 and row['air_humidity_percent'] < 60 and row['soil_humidity_percent'] < 30:
                return ('Alerta de plagas: Las condiciones actuales pueden favorecer la broca del café. Revise los frutos y utilice métodos de control '
                        'biológico, como hongos naturales, para proteger el cultivo.')
            elif row['air_humidity_percent'] > 70 and row['celcius_grade_temperature'] < 22 and row['soil_humidity_percent'] > 40:
                return ('Presencia de insectos chupadores: Las altas condiciones de humedad y temperaturas suaves pueden atraer pulgones. Se recomienda '
                        'usar soluciones naturales como jabón potásico para reducir la población de plagas.')

            # Recomendaciones para el manejo preventivo y estabilidad
            elif 25 <= row['soil_humidity_percent'] <= 35 and 18 <= row['celcius_grade_temperature'] <= 26 and 50 <= row['air_humidity_percent'] <= 70:
                return ('Condiciones óptimas: El cultivo se encuentra en un buen estado. Mantenga el monitoreo constante y asegúrese de que las condiciones '
                        'no cambien bruscamente. No es necesario realizar intervenciones, pero esté atento a cambios climáticos inesperados.')
            
            # Estrategias para controlar sequías prolongadas
            elif row['soil_humidity_percent'] < 15 and row['precipitation_detected'] == 0 and row['celcius_grade_temperature'] > 30:
                return ('Sequía prolongada: Las plantas están bajo mucho estrés por la falta de agua y el calor. Considere instalar un sistema de riego '
                        'por goteo para mantener la humedad del suelo constante y evitar daños en las plantas.')
            
            # Estrategias de sombra y protección
            elif row['celcius_grade_temperature'] > 32 and row['air_humidity_percent'] < 50:
                return ('Protección contra calor extremo: Las temperaturas altas y la baja humedad pueden deshidratar las plantas rápidamente. Considere '
                        'colocar mallas de sombra y regar ligeramente las hojas para ayudar a bajar la temperatura de las plantas.')

            # Manejo de suelos compactados y mejora del drenaje
            elif row['soil_humidity_percent'] > 50 and row['precipitation_detected'] == 0 and row['air_humidity_percent'] < 70:
                return ('Humedad elevada sin lluvia: El suelo está reteniendo demasiada agua. Esto podría ser un signo de compactación. Se recomienda '
                        'airear el suelo para mejorar el drenaje y evitar la acumulación de agua en las raíces.')

            # Planificación a largo plazo
            elif 18 <= row['celcius_grade_temperature'] <= 25 and 55 <= row['air_humidity_percent'] <= 75 and row['soil_humidity_percent'] > 30:
                return ('Planificación de fertilización: Las condiciones actuales son ideales para aplicar fertilizantes naturales. Asegúrese de usar '
                        'fertilizantes que enriquezcan el suelo sin dañar los microorganismos beneficiosos.')

            # Control de plagas en temperaturas bajas
            elif row['celcius_grade_temperature'] < 18 and row['air_humidity_percent'] > 80:
                return ('Cuidado con plagas de clima fresco: Las bajas temperaturas y la alta humedad pueden atraer a los minadores de hojas. Inspeccione '
                        'las hojas en busca de líneas blancas o marrones y use extractos de ajo o aceites esenciales como tratamiento natural.')

            else:
                return 'No se necesita intervención: Las condiciones actuales son óptimas para el crecimiento del café. Continúe monitoreando de manera regular.'

        df_device['recommendation_description'] = df_device.apply(generate_detailed_recommendation, axis=1)
        recommendation = {
            "device_hub_id": device_id,
            "recommendation_description": df_device['recommendation_description'].mode()[0]
        }
        recommendations.append(recommendation)

    return recommendations

# Endpoint para generar y enviar recomendaciones a Azure
@app.post("/recommendations/")
async def post_generate_recommendations():
    try:
        recommendations = analyze_data_and_generate_recommendations()

        if not recommendations:
            raise HTTPException(status_code=404, detail="No se generaron recomendaciones.")

        azure_endpoint = "https://coffeetech-api-netcore.azurewebsites.net/api/v1/recommendations"

        for rec in recommendations:
            payload = {
                "deviceHubId": rec['device_hub_id'],
                "recommendationDescription": rec['recommendation_description']
            }
            try:
                response = requests.post(azure_endpoint, json=payload)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=500, detail=f"Error al enviar la recomendación a Azure: {e}")

        return {"message": "Recomendaciones generadas y enviadas a Azure.", "data": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
