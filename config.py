"""
config.py — Parámetros del modelo predictivo de riesgo de insolvencia.

Todas las decisiones metodológicas del documento V3 (Marco Metodológico)
están centralizadas aquí. Cambie los valores en este archivo, no en el código.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. RUTAS
# ---------------------------------------------------------------------------
# Carpeta que contiene las subcarpetas pymes_individuales-2016 ... -2025
# (ajuste la ruta a su equipo; en Windows use r"C:\...").
RUTA_BD = Path(r"C:\Users\Usuario\OneDrive\Documentos\MACA2\Semestre 3\2. Proyecto II\bd")
RUTA_PROYECTO = Path(__file__).resolve().parent
RUTA_MACRO = RUTA_PROYECTO / "data" / "macro_colombia.csv"
RUTA_INTERMEDIOS = RUTA_PROYECTO / "data"           # bases consolidadas (.parquet / .pkl)
RUTA_RESULTADOS = RUTA_PROYECTO / "resultados"       # métricas, gráficos, reportes

ANIOS = list(range(2016, 2026))                      # 10 años de estados financieros

# Archivos SIIS (Supersociedades) por año
ARCHIVOS = {
    "caratula": "10000_Carátula.xlsx",
    "esf":      "210030_Estado de situación financiera, corriente_no corriente.xlsx",
    "eri":      "310030_Estado de resultado integral, resultado del periodo, por funcion de gasto.xlsx",
    "efe":      "520000_Estado de flujos de efectivo - método indirecto.xlsx",
}

# Indicadores macroeconómicos descargados de los informes estadísticos del
# Banco de la República (guardados en la misma carpeta bd)
ARCHIVOS_MACRO = {
    "tasa":      "tasa de intervención.xlsx",
    "inflacion": "Inflación y meta.xlsx",
    "trm":       "4_Serie_historica_TRM_iqy.xlsx",
    "pib":       "5_PIB_2015_Producto_Interno_Bruto_ramas_de_actividades_economicas_trimestral_iqy.xlsx",
}

# ---------------------------------------------------------------------------
# 2. POBLACIÓN (CIIU Rev. 4 A.C.)
# ---------------------------------------------------------------------------
# C = Industrias manufactureras (divisiones 10-33)
# G = Comercio al por mayor y al por menor; reparación de vehículos (45-47)
SECCIONES_CIIU = {"C": "Manufacturero", "G": "Comercial"}

# Solo cierres anuales (31 de diciembre) para que los años sean comparables
SOLO_CIERRE_ANUAL = True

# Sociedades con regímenes especiales que se excluyen (no son actividad real
# manufacturera/comercial o tienen naturaleza financiera)
EXCLUIR_TIPO_SOCIEDAD = ["FACTORING", "LIBRANZA", "AUTOFINANCIAMIENTO",
                         "FONDO GANADERO", "MULTINIVEL", "FUTBOL"]

# ---------------------------------------------------------------------------
# 3. VARIABLE DEPENDIENTE (definición operacional de insolvencia, V3)
# ---------------------------------------------------------------------------
# Columna "Estado actual" de la carátula (columna M). Estos cuatro estados
# identifican a las empresas en proceso de insolvencia (Ley 1116 de 2006,
# Ley 550 de 1999 y concordatos). Se comparan sin tildes ni mayúsculas.
ESTADOS_INSOLVENCIA_FORMAL = [
    "ACUERDO DE REORGANIZACION",
    "ACUERDO DE REESTRUCTURACION",
    "CONCORDATO EN EJECUCION",
    "CONCORDATO EN TRAMITE",
]

# Horizonte predictivo: variables en t, evento en t+1 ... t+HORIZONTE
HORIZONTE = 3

# Regla de etiquetado:
#   "documento" -> riesgo = 1 si cumple AL MENOS UNO de los 5 criterios
#                  (exactamente como está redactado en V3)
#   "estricta"  -> riesgo = 1 si: proceso formal Ley 1116, o patrimonio
#                  negativo, o al menos DOS de los criterios financieros
#                  (EBITDA<0, razón corriente<1, cobertura<1) en el mismo año
#   "formal"    -> riesgo = 1 solo si entra a proceso formal (Ley 1116)
REGLA_ETIQUETA = "documento"        # definida por la tutora (V3)

# Para que el modelo sea de ALERTA TEMPRANA se excluyen las observaciones de
# empresas que en el año t ya están en proceso formal de insolvencia
# (si no, el modelo "predice" algo que ya ocurrió).
EXCLUIR_YA_EN_PROCESO_EN_T = True

# Si True, también se excluyen las empresas que en t ya cumplen la regla
# de etiqueta (modelo de "nuevos casos"). Recomendado como análisis de
# robustez.
EXCLUIR_YA_EN_RIESGO_EN_T = False

# Mínimo de años observados en t+1..t+H para poder etiquetar la observación
MIN_ANIOS_FUTUROS_OBSERVADOS = 1

# ---------------------------------------------------------------------------
# 4. VARIABLES INDEPENDIENTES (Tabla 1 de V3)
# ---------------------------------------------------------------------------
VARIABLES_TABLA1 = [
    "razon_corriente",        # AC / PC                    (Beaver, 1966)
    "endeudamiento",          # PT / AT                    (Altman, 1968)
    "roa",                    # Utilidad neta / AT         (Beaver, 1966)
    "cobertura_intereses",    # EBIT / Gastos financieros  (Budhidharma et al., 2023)
    "capital_trabajo_at",     # (AC - PC) / AT             (Altman, 1968)
    "ebitda_ventas",          # EBITDA / Ventas            (Gabrielli et al., 2026)
    "rotacion_activos",       # Ventas / AT                (Altman, 1968)
    "tasa_banrep",            # Tasa de política monetaria (BanRep)
    "inflacion",              # Variación anual IPC (DANE)
    "crec_pib",               # Variación anual PIB (DANE)
    "trm_promedio",           # TRM promedio anual (BanRep)
]

# Variables complementarias (Anexo 1). Útiles para el objetivo específico 1
# (identificar indicadores con mayor capacidad explicativa). Póngalo en
# False si el comité exige usar solo la Tabla 1.
USAR_VARIABLES_ADICIONALES = True
VARIABLES_ADICIONALES = [
    "log_activos",            # tamaño
    "flujo_operativo_pt",     # Flujo de caja operativo / Pasivo total
    "ganancias_acum_at",      # Ganancias acumuladas / AT (Altman X2)
    "deuda_financiera_at",    # Préstamos / AT
    "crec_ventas",            # Variación anual de ventas
    "edad_empresa",           # años desde constitución
    "sector_manufacturero",   # 1 = C, 0 = G
    "opinion_rf_modificada",  # 1 = salvedad / negativa / abstención
    "crec_pib_sector",        # crecimiento real del valor agregado de su rama (DANE)
]

# ---------------------------------------------------------------------------
# 5. MODELADO
# ---------------------------------------------------------------------------
SEMILLA = 42
PROPORCION_PRUEBA = 0.30          # 70 % entrenamiento / 30 % prueba (V3)
DIVIDIR_POR_EMPRESA = True        # una misma empresa no queda en train y test
N_FOLDS_CV = 5
N_ITER_BUSQUEDA = 20              # combinaciones de hiperparámetros a probar
WINSORIZAR = (0.01, 0.99)         # recorte de valores extremos (percentiles)

# Validación fuera de tiempo (robustez): entrena con años t <= ANIO_CORTE_OOT
# y prueba con años posteriores
VALIDACION_FUERA_DE_TIEMPO = True
ANIO_CORTE_OOT = 2020

# Niveles de riesgo para el reporte (probabilidad estimada)
UMBRALES_RIESGO = {"Bajo": (0.0, 0.30), "Medio": (0.30, 0.60), "Alto": (0.60, 1.01)}

N_BOOTSTRAP = 1000                # IC del AUC y prueba RF vs. regresión logística
