# Modelo predictivo de riesgo de insolvencia — Pymes manufactureras y comerciales (Colombia)

Proyecto de grado — Maestría en Analítica de Datos para Contabilidad y Auditoría, Universidad Externado de Colombia.
Autora: Leidy Briggitte Muete Garzón · Tutora: Gabriela María Saucedo Meza, Ph.D.

Random Forest + SMOTE, contrastado con regresión logística, sobre los estados financieros
individuales (Grupo 2 NIIF) reportados a la Superintendencia de Sociedades, 2016–2025.

## Estructura

```
modelo_insolvencia/
├── config.py            <- TODAS las decisiones metodológicas (rutas, CIIU, etiqueta, horizonte, 70/30)
├── main.py              <- ejecuta el proceso completo
├── Modelo_insolvencia_Colab.ipynb  <- cuaderno para Google Colab
├── requirements.txt
├── data/
│   ├── macro_colombia.csv   <- tasa BanRep, IPC, PIB, TRM (Banco de la República)
│   └── panel_consolidado.parquet  <- base depurada 2016–2025 (82.108 estados financieros)
├── src/
│   ├── carga.py         <- SRS-01 lectura y depuración de los 4 reportes SIIS
│   ├── variables.py     <- SRS-02 indicadores (Tabla 1) + etiqueta t+1..t+3
│   ├── modelado.py      <- SRS-03/04/05 SMOTE, RF, Logit, métricas, bootstrap, ROC
│   └── reporte.py       <- SRS-06/07 calificación por empresa, factores (SHAP), Excel
└── resultados/          <- se llena al ejecutar
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
# 1. Ajuste RUTA_BD en config.py a la carpeta "bd"
python main.py            # proceso completo (~5-15 min según equipo)
python main.py --desde 3  # solo re-entrena, reutilizando la base ya construida
```

## Ejecutar en Google Colab (sin instalar nada)

Abra `Modelo_insolvencia_Colab.ipynb` en [Google Colab](https://colab.research.google.com) (*Archivo → Abrir cuaderno → GitHub*
y pegue la URL de este repositorio) y elija **Entorno de ejecución → Ejecutar todas**.
En la celda 1 seleccione `FUENTE = GitHub` y escriba la URL del repositorio. La base consolidada
(`data/panel_consolidado.parquet`, 11 MB) reproduce la depuración de los 40 archivos del SIIS, así que no se
necesita descargar los Excel. Modo rápido: unos 5–10 minutos; modo completo: 30–45 minutos.

## Fuentes de datos (carpeta `bd/pymes_individuales-AAAA/`)

| Código | Reporte |
|---|---|
| 10000 | Carátula (NIT, CIIU, estado actual — reorganización Ley 1116) |
| 210030 | Estado de situación financiera, corriente/no corriente |
| 310030 | Estado de resultado integral, por función del gasto |
| 520000 | Estado de flujos de efectivo, método indirecto (depreciación para EBITDA) |

Cifras en miles de pesos. Los datos no se suben al repositorio (son pesados); se descargan del portal SIIS.

## Resultados que genera

`01_bitacora_depuracion.csv`, `02_prevalencia_criterios.csv`, `02_distribucion_etiqueta.csv`,
`03_metricas_prueba.csv`, `03_comparacion_modelos.json` (IC 95 % bootstrap del AUC),
`03_curva_roc.png`, `03_importancia_variables.csv/.png`, `03_metricas_por_sector.csv`,
`04_validacion_fuera_de_tiempo.csv`, `05_reporte_riesgo_insolvencia.xlsx`, `modelo_rf_smote.joblib`.
