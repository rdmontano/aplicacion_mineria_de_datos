# Aplicación demostrativa — Práctica Experimental 1 de Minería de Datos

## Caso de estudio
Predicción de abandono de clientes de una empresa de telecomunicaciones mediante datos simulados.

La aplicación reproduce los componentes del informe:

- definición del problema;
- exploración del conjunto de datos;
- preprocesamiento;
- comparación de modelos;
- evaluación con Exactitud, Precisión, Recall, F1 y AUC;
- predicción interactiva con regresión logística;
- segmentación K-Means;
- exportación de clientes priorizados por riesgo.

## Ejecutar en Windows

1. Descomprima la carpeta.
2. Tenga instalado **Python 3.10 o superior**.
3. Haga doble clic en `iniciar_app.bat`.
4. En la primera ejecución se instalarán automáticamente las dependencias.
5. La aplicación se abrirá en el navegador.

También puede ejecutar manualmente:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Ejecutar en Linux/macOS

```bash
chmod +x iniciar_app.sh
./iniciar_app.sh
```

## Estructura

- `app.py`: aplicación principal.
- `data/`: conjuntos de datos y resultados del informe.
- `docs/`: informe completo de la práctica.
- `output/`: archivos exportados por la aplicación.
- `requirements.txt`: dependencias.

## Datos y modelo

El conjunto inicial contiene 1.520 registros, con 20 duplicados intencionales. Después de la limpieza quedan 1.500 clientes únicos. El modelo principal es una regresión logística con preprocesamiento mediante pipeline, división estratificada 80/20 y semilla 42.

Resultados reproducidos del modelo principal:

- Accuracy: 0.693
- Precision: 0.419
- Recall: 0.671
- F1: 0.516
- AUC: 0.742

La aplicación es exclusivamente académica; los datos son simulados.
