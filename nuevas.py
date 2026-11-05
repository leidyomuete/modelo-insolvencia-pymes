"""
Calificación de empresas que NO están en la base de Supersociedades.

El usuario (auditor, analista, revisor fiscal) entrega las cifras de los
estados financieros de una o varias empresas —escritas a mano o en la
plantilla de Excel data/plantilla_empresas_nuevas.xlsx— y el modelo devuelve
la probabilidad de presentar señales de insolvencia en los tres años
siguientes, el nivel de riesgo y los indicadores que más pesan.
"""
import numpy as np
import pandas as pd

import config as cfg
from src import variables, reporte

# columna de la plantilla -> (nombre interno, descripción, obligatoria)
CAMPOS = {
    "NIT":                              ("nit", "NIT sin dígito de verificación", True),
    "Razón social":                     ("razon_social", "Nombre de la empresa", True),
    "Sector (Manufacturero/Comercial)": ("sector", "Manufacturero (CIIU sección C) o Comercial (sección G)", True),
    "Año de los estados financieros":   ("anio", "Año de cierre, p. ej. 2025", True),
    "Fecha de constitución":            ("fecha_constitucion", "AAAA-MM-DD", False),
    "Opinión del revisor fiscal":       ("opinion_rf", "Limpio, Con salvedad, Negativa o Abstención", False),
    "Activo corriente":                 ("activo_corriente", "Miles de pesos", True),
    "Activo total":                     ("activo_total", "Miles de pesos", True),
    "Pasivo corriente":                 ("pasivo_corriente", "Miles de pesos", True),
    "Pasivo total":                     ("pasivo_total", "Miles de pesos", True),
    "Patrimonio":                       ("patrimonio", "Miles de pesos", True),
    "Ganancias acumuladas":             ("ganancias_acumuladas", "Miles de pesos", False),
    "Préstamos corto plazo":            ("prestamos_cp", "Miles de pesos (obligaciones financieras corrientes)", False),
    "Préstamos largo plazo":            ("prestamos_lp", "Miles de pesos (obligaciones financieras no corrientes)", False),
    "Ingresos operacionales":           ("ingresos", "Miles de pesos", True),
    "Ingresos del año anterior":        ("ingresos_anterior", "Miles de pesos (para el crecimiento de ventas)", False),
    "Utilidad operacional":             ("utilidad_operacional", "Miles de pesos (ganancia por actividades de operación)", True),
    "Gastos financieros":               ("gastos_financieros", "Miles de pesos", True),
    "Utilidad neta":                    ("utilidad_neta", "Miles de pesos", True),
    "Depreciación y amortización":      ("depreciacion_amortizacion", "Miles de pesos (del flujo de efectivo o notas)", False),
    "Flujo de efectivo de operación":   ("flujo_operativo", "Miles de pesos", False),
}

EJEMPLO = {
    "NIT": 900123456, "Razón social": "EMPRESA EJEMPLO SAS", "Sector (Manufacturero/Comercial)": "Manufacturero",
    "Año de los estados financieros": 2025, "Fecha de constitución": "2012-03-15", "Opinión del revisor fiscal": "Limpio",
    "Activo corriente": 4_500_000, "Activo total": 9_800_000, "Pasivo corriente": 3_900_000, "Pasivo total": 6_200_000,
    "Patrimonio": 3_600_000, "Ganancias acumuladas": 1_100_000, "Préstamos corto plazo": 1_200_000,
    "Préstamos largo plazo": 1_500_000, "Ingresos operacionales": 12_400_000, "Ingresos del año anterior": 13_100_000,
    "Utilidad operacional": 310_000, "Gastos financieros": 420_000, "Utilidad neta": -95_000,
    "Depreciación y amortización": 280_000, "Flujo de efectivo de operación": 150_000,
}


def crear_plantilla(ruta=None):
    """Genera la plantilla de Excel con una fila de ejemplo y una hoja de instrucciones."""
    ruta = ruta or (cfg.RUTA_INTERMEDIOS / "plantilla_empresas_nuevas.xlsx")
    datos = pd.DataFrame([EJEMPLO], columns=list(CAMPOS))
    instr = pd.DataFrame([(k, v[1], "Sí" if v[2] else "No") for k, v in CAMPOS.items()],
                         columns=["Columna", "Qué escribir", "Obligatoria"])
    with pd.ExcelWriter(ruta, engine="openpyxl") as w:
        datos.to_excel(w, sheet_name="Empresas", index=False)
        instr.to_excel(w, sheet_name="Instrucciones", index=False)
        for hoja in w.sheets.values():
            for col in hoja.columns:
                hoja.column_dimensions[col[0].column_letter].width = 26
    return ruta


def _preparar(df):
    faltan = [k for k, v in CAMPOS.items() if v[2] and k not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias: {faltan}")
    d = df.rename(columns={k: v[0] for k, v in CAMPOS.items()}).copy()
    for v in CAMPOS.values():
        if v[0] not in d.columns:
            d[v[0]] = np.nan
    num = [v[0] for k, v in CAMPOS.items() if k not in
           ("NIT", "Razón social", "Sector (Manufacturero/Comercial)", "Fecha de constitución", "Opinión del revisor fiscal")]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce")
    d["anio"] = d["anio"].astype(int)
    d["nit"] = pd.to_numeric(d["nit"], errors="coerce")
    d["sector"] = d["sector"].astype(str).str.strip().str.capitalize()
    d["seccion_ciiu"] = np.where(d.sector.str.startswith("Manuf"), "C", "G")
    d["opinion_rf"] = d["opinion_rf"].fillna("Limpio").astype(str)
    d["prestamos_lp_porcion_cp"] = 0.0
    return d


def calificar_nuevas(df, modelo_rf, cols):
    """df: DataFrame con las columnas de la plantilla. Devuelve la calificación por empresa."""
    d = _preparar(df)
    avisos = []
    sectores_no = d[~d.sector.isin(["Manufacturero", "Comercial"])]
    if len(sectores_no):
        avisos.append("El modelo se entrenó con Pymes manufactureras y comerciales; otros sectores son solo referenciales.")

    d = variables.calcular_indicadores(d)
    # crecimiento de ventas con el dato del año anterior escrito por el usuario
    d["crec_ventas"] = ((d.ingresos - d.ingresos_anterior) / d.ingresos_anterior.abs()).clip(-1, 10)

    # variables macro del año de los estados financieros (si no existe, el último disponible)
    macro = pd.read_csv(cfg.RUTA_MACRO)
    ultimo = int(macro.anio.max())
    if (d.anio > ultimo).any():
        avisos.append(f"Para años posteriores a {ultimo} se usaron las variables macroeconómicas de {ultimo}.")
    d["anio_macro"] = d.anio.clip(upper=ultimo)
    d = d.merge(macro.rename(columns={"anio": "anio_macro"}), on="anio_macro", how="left")
    d["crec_pib_sector"] = np.where(d.seccion_ciiu == "C", d.crec_pib_manufactura, d.crec_pib_comercio)

    d["probabilidad_riesgo"] = modelo_rf.predict_proba(d[cols])[:, 1]
    d["nivel_riesgo"] = d.probabilidad_riesgo.apply(reporte.nivel_riesgo)
    d["principales_factores"] = reporte.factores_por_empresa(modelo_rf, d[cols], cols)
    d["horizonte"] = d.anio.map(lambda a: f"{a + 1}–{a + cfg.HORIZONTE}")

    # señales de alerta actuales (criterios de la definición operacional)
    alertas = []
    for _, r in d.iterrows():
        s = []
        if r.patrimonio < 0: s.append("patrimonio negativo")
        if r.ebitda < 0: s.append("EBITDA negativo")
        if r.razon_corriente < 1: s.append("razón corriente < 1")
        if r.gastos_financieros > 0 and r.cobertura_intereses < 1: s.append("cobertura de intereses < 1")
        alertas.append(", ".join(s) if s else "ninguna")
    d["alertas_actuales"] = alertas

    salida = d[["nit", "razon_social", "sector", "anio", "horizonte", "probabilidad_riesgo", "nivel_riesgo",
                "principales_factores", "alertas_actuales", "razon_corriente", "endeudamiento", "roa",
                "cobertura_intereses", "ebitda_ventas", "capital_trabajo_at"]].rename(columns={
        "anio": "año_estados_financieros", "horizonte": "riesgo_para_los_años"})
    for a in avisos:
        print("  Aviso:", a)
    return salida
