"""
Módulo 1 — Carga y depuración de datos (SRS-01).

Lee los 4 reportes SIIS de cada año, estandariza los nombres de columna
usando la etiqueta XBRL que viene entre paréntesis (p. ej. "(CurrentAssets)"),
filtra las secciones CIIU C y G, conserva cierres anuales y consolida un
panel empresa-año.
"""
import re
import pandas as pd
import numpy as np

import config as cfg


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _leer_excel(ruta):
    """Lee un .xlsx; usa 'calamine' (mucho más rápido) si está instalado."""
    try:
        return pd.read_excel(ruta, engine="calamine")
    except Exception:
        return pd.read_excel(ruta)


def _etiqueta_xbrl(nombre_columna):
    """'Activos corrientes totales (CurrentAssets)' -> 'CurrentAssets'."""
    m = re.findall(r"\(([A-Za-z_][A-Za-z0-9_]*)\)\s*$", str(nombre_columna))
    return m[0] if m else str(nombre_columna).strip()


def _estandarizar_columnas(df):
    nuevas, vistas = [], {}
    for c in df.columns:
        e = _etiqueta_xbrl(c)
        # el CIIU cambió de nombre entre años
        if e in ("CIIU",) or c.startswith("Clasificación Industrial Internacional"):
            e = "CIIU"
        if e in vistas:                       # etiquetas repetidas (EFE)
            vistas[e] += 1
            e = f"{e}_{vistas[e]}"
        else:
            vistas[e] = 0
        nuevas.append(e)
    df.columns = nuevas
    return df


def _limpiar_nit(s):
    return pd.to_numeric(s.astype(str).str.replace(r"\D", "", regex=True),
                         errors="coerce").astype("Int64")


# ---------------------------------------------------------------------------
# Lectura por año
# ---------------------------------------------------------------------------
COLS_CARATULA = {
    "NIT": "nit", "Fecha de Corte": "fecha_corte",
    "Razón social de la sociedad": "razon_social", "CIIU": "ciiu",
    "Estado actual": "estado", "Fecha de constitución (Aaaa-Mm-Dd)": "fecha_constitucion",
    "Tipo societario": "tipo_societario", "La sociedad es": "tipo_sociedad",
    "Concepto del Revisor fiscal en su informe": "opinion_rf",
    "Departamento de la dirección del domicilio": "departamento",
}

COLS_ESF = {
    "CurrentAssets": "activo_corriente", "CurrentLiabilities": "pasivo_corriente",
    "Assets": "activo_total", "Liabilities": "pasivo_total", "Equity": "patrimonio",
    "CashAndCashEquivalents": "efectivo", "Inventories": "inventarios",
    "TradeAndOtherCurrentReceivables": "cxc", "RetainedEarnings": "ganancias_acumuladas",
    "ShorttermBorrowings": "prestamos_cp", "CurrentPortionOfLongtermBorrowings": "prestamos_lp_porcion_cp",
    "LongtermBorrowings": "prestamos_lp",
}

COLS_ERI = {
    "Revenue": "ingresos", "CostOfSales": "costo_ventas", "GrossProfit": "utilidad_bruta",
    "GananciaPerdidaPorActividadesDeOperacion": "utilidad_operacional",
    "FinanceIncome": "ingresos_financieros", "FinanceCosts": "gastos_financieros",
    "ProfitLossBeforeTax": "utilidad_antes_impuestos", "ProfitLoss": "utilidad_neta",
}

COLS_EFE = {
    "AdjustmentsForDepreciationAndAmortisationExpense": "depreciacion_amortizacion",
    "CashFlowsFromUsedInOperatingActivities": "flujo_operativo",
    "CashFlowsFromUsedInInvestingActivities": "flujo_inversion",
    "CashFlowsFromUsedInFinancingActivities": "flujo_financiacion",
}


def _seleccionar(df, mapa):
    presentes = {k: v for k, v in mapa.items() if k in df.columns}
    out = df[["NIT", "Fecha de Corte"] + list(presentes)].rename(columns=presentes)
    for v in mapa.values():                    # columnas ausentes -> NaN
        if v not in out.columns:
            out[v] = np.nan
    return out.rename(columns={"NIT": "nit", "Fecha de Corte": "fecha_corte"})


def _periodo_actual(df):
    if "Periodo" not in df.columns:
        return df
    per = df["Periodo"].astype(str)
    if per.str.contains("Actual").any():
        return df[per.str.contains("Actual")]
    return df[pd.to_datetime(per, errors="coerce") == pd.to_datetime(df["Fecha de Corte"], errors="coerce")]


def leer_anio(anio):
    carpeta = cfg.RUTA_BD / f"pymes_individuales-{anio}"
    car = _estandarizar_columnas(_leer_excel(carpeta / cfg.ARCHIVOS["caratula"]))
    esf = _estandarizar_columnas(_leer_excel(carpeta / cfg.ARCHIVOS["esf"]))
    eri = _estandarizar_columnas(_leer_excel(carpeta / cfg.ARCHIVOS["eri"]))
    efe = _estandarizar_columnas(_leer_excel(carpeta / cfg.ARCHIVOS["efe"]))

    # ESF y ERI traen "Periodo Actual" y "Periodo Anterior": se usa el actual.
    # En 2016 (año de transición NIIF) la columna trae fechas en lugar de texto:
    # en ese caso se conserva la fila cuya fecha coincide con la fecha de corte.
    esf, eri = _periodo_actual(esf), _periodo_actual(eri)

    car = car.rename(columns={k: v for k, v in COLS_CARATULA.items() if k in car.columns})
    car = car[[c for c in COLS_CARATULA.values() if c in car.columns]]
    esf, eri, efe = _seleccionar(esf, COLS_ESF), _seleccionar(eri, COLS_ERI), _seleccionar(efe, COLS_EFE)

    for d in (car, esf, eri, efe):
        d["nit"] = _limpiar_nit(d["nit"])
        d["fecha_corte"] = pd.to_datetime(d["fecha_corte"], errors="coerce")
        d.drop_duplicates(["nit", "fecha_corte"], keep="last", inplace=True)

    llave = ["nit", "fecha_corte"]
    df = (car.merge(esf, on=llave, how="inner")
             .merge(eri, on=llave, how="inner")
             .merge(efe, on=llave, how="left"))
    df["anio"] = anio
    return df


# ---------------------------------------------------------------------------
# Panel consolidado
# ---------------------------------------------------------------------------
TEXTO = ["razon_social", "ciiu", "estado", "fecha_constitucion", "tipo_societario", "tipo_sociedad",
         "opinion_rf", "departamento", "seccion_ciiu", "division_ciiu", "sector"]


def exportar_parquet(panel, ruta):
    """Versión liviana (~11 MB) del panel para compartir y ejecutar en Google Colab."""
    try:
        p = panel.copy()
        for c in TEXTO:
            if c in p.columns:
                p[c] = p[c].astype("string")
        p["nit"] = p["nit"].astype("int64")
        p.to_parquet(ruta, compression="zstd", index=False)
    except Exception as e:                       # pyarrow no instalado, etc.
        print(f"  (no se generó {ruta.name}: {e})")


def construir_panel(guardar=True):
    partes = []
    for anio in cfg.ANIOS:
        print(f"  Leyendo {anio} ...", end=" ", flush=True)
        d = leer_anio(anio)
        print(f"{len(d):,} empresas")
        partes.append(d)
    panel = pd.concat(partes, ignore_index=True)
    bitacora = {"registros_iniciales": len(panel)}

    # 1) Cierre anual (31-dic) y año del corte = año de la carpeta
    if cfg.SOLO_CIERRE_ANUAL:
        panel = panel[(panel.fecha_corte.dt.month == 12) & (panel.fecha_corte.dt.day == 31)
                      & (panel.fecha_corte.dt.year == panel.anio)]
    bitacora["cierre_anual"] = len(panel)

    # 2) Sección CIIU C (manufactura) y G (comercio)
    panel["seccion_ciiu"] = panel["ciiu"].astype(str).str.strip().str[0].str.upper()
    panel["division_ciiu"] = panel["ciiu"].astype(str).str.extract(r"^[A-Z](\d{2})")[0]
    panel = panel[panel.seccion_ciiu.isin(cfg.SECCIONES_CIIU)]
    panel["sector"] = panel.seccion_ciiu.map(cfg.SECCIONES_CIIU)
    bitacora["sectores_C_G"] = len(panel)

    # 3) Excluir regímenes especiales / naturaleza financiera
    patron = "|".join(cfg.EXCLUIR_TIPO_SOCIEDAD)
    panel = panel[~panel["tipo_sociedad"].astype(str).str.upper().str.contains(patron, na=False)]
    bitacora["sin_regimen_especial"] = len(panel)

    # 4) Calidad mínima: activos > 0 y ecuación patrimonial consistente (±1 %)
    num = list(COLS_ESF.values()) + list(COLS_ERI.values()) + list(COLS_EFE.values())
    panel[num] = panel[num].apply(pd.to_numeric, errors="coerce")
    panel = panel[panel.activo_total > 0]
    dif = (panel.activo_total - panel.pasivo_total.fillna(0) - panel.patrimonio.fillna(0)).abs()
    panel = panel[dif <= 0.01 * panel.activo_total]
    bitacora["calidad_contable"] = len(panel)

    # 5) Una observación por empresa-año
    panel = panel.sort_values(["nit", "anio"]).drop_duplicates(["nit", "anio"], keep="last")
    bitacora["final_empresa_anio"] = len(panel)
    bitacora["empresas_unicas"] = panel.nit.nunique()

    if guardar:
        cfg.RUTA_INTERMEDIOS.mkdir(parents=True, exist_ok=True)
        panel.to_pickle(cfg.RUTA_INTERMEDIOS / "panel_consolidado.pkl")
        exportar_parquet(panel, cfg.RUTA_INTERMEDIOS / "panel_consolidado.parquet")
        pd.Series(bitacora).to_csv(cfg.RUTA_RESULTADOS / "01_bitacora_depuracion.csv",
                                   header=["registros"])
    print("\n  Bitácora de depuración:")
    for k, v in bitacora.items():
        print(f"    {k:<25}{v:>10,}")
    return panel.reset_index(drop=True)
