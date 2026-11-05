"""
Módulo 2 — Construcción de variables (SRS-02) y de la variable dependiente
con horizonte t+1..t+3 (definición operacional de insolvencia de V3).
"""
import numpy as np
import pandas as pd

import config as cfg


def _div(a, b):
    """División segura: denominador 0 o nulo -> NaN."""
    a, b = a.astype(float), b.astype(float)
    return np.where((b == 0) | b.isna(), np.nan, a / b)


# ---------------------------------------------------------------------------
# Indicadores financieros (Tabla 1 + Anexo 1)
# ---------------------------------------------------------------------------
def calcular_indicadores(panel):
    d = panel.copy()
    d["ebit"] = d["utilidad_operacional"]                       # aproximación de EBIT
    d["ebitda"] = d["ebit"] + d["depreciacion_amortizacion"].fillna(0)

    # --- Tabla 1 ---
    d["razon_corriente"] = _div(d.activo_corriente, d.pasivo_corriente)
    d["endeudamiento"] = _div(d.pasivo_total, d.activo_total)
    d["roa"] = _div(d.utilidad_neta, d.activo_total)
    gf = d.gastos_financieros.fillna(0)
    # Sin gastos financieros => la empresa "cubre" sus intereses: valor alto (tope 100)
    d["cobertura_intereses"] = np.where(gf <= 0, 100.0, d.ebit / gf.replace(0, np.nan))
    d["cobertura_intereses"] = d["cobertura_intereses"].clip(-100, 100)
    d["capital_trabajo_at"] = _div(d.activo_corriente - d.pasivo_corriente, d.activo_total)
    d["ebitda_ventas"] = _div(d.ebitda, d.ingresos)
    d["rotacion_activos"] = _div(d.ingresos, d.activo_total)

    # --- Variables adicionales (Anexo 1) ---
    d["log_activos"] = np.log1p(d.activo_total.clip(lower=0))
    d["flujo_operativo_pt"] = _div(d.flujo_operativo, d.pasivo_total)
    d["ganancias_acum_at"] = _div(d.ganancias_acumuladas, d.activo_total)
    deuda = d[["prestamos_cp", "prestamos_lp_porcion_cp", "prestamos_lp"]].fillna(0).sum(axis=1)
    d["deuda_financiera_at"] = _div(deuda, d.activo_total)
    d = d.sort_values(["nit", "anio"])
    ing_prev = d.groupby("nit")["ingresos"].shift(1)
    anio_prev = d.groupby("nit")["anio"].shift(1)
    d["crec_ventas"] = np.where(anio_prev == d.anio - 1, _div(d.ingresos - ing_prev, ing_prev.abs()), np.nan)
    d["crec_ventas"] = pd.Series(d["crec_ventas"], index=d.index).clip(-1, 10)
    fc = pd.to_datetime(d.fecha_constitucion, errors="coerce")
    d["edad_empresa"] = (d.anio - fc.dt.year).clip(lower=0)
    d["sector_manufacturero"] = (d.seccion_ciiu == "C").astype(int)
    d["opinion_rf_modificada"] = d.opinion_rf.astype(str).str.contains(
        "SALVEDAD|NEGATIV|ABSTEN", case=False, na=False).astype(int)

    # Reemplazar infinitos
    d = d.replace([np.inf, -np.inf], np.nan)
    return d


def agregar_macro(d):
    macro = pd.read_csv(cfg.RUTA_MACRO)
    d = d.merge(macro, on="anio", how="left")
    # crecimiento de la rama económica propia de cada empresa
    d["crec_pib_sector"] = np.where(d.seccion_ciiu == "C", d.crec_pib_manufactura, d.crec_pib_comercio)
    return d


# ---------------------------------------------------------------------------
# Criterios de insolvencia (V3) y etiqueta con horizonte
# ---------------------------------------------------------------------------
def criterios_insolvencia(d):
    import unicodedata
    def norm(x):
        x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
        return " ".join(x.upper().split())
    estados = {norm(e) for e in cfg.ESTADOS_INSOLVENCIA_FORMAL}
    d["crit_ley1116"] = d.estado.map(norm).isin(estados)
    d["crit_patrimonio_neg"] = d.patrimonio < 0
    d["crit_ebitda_neg"] = d.ebitda < 0
    d["crit_rc_menor1"] = d.razon_corriente < 1
    d["crit_cobertura_menor1"] = (d.gastos_financieros.fillna(0) > 0) & (d.cobertura_intereses < 1)

    fin = d[["crit_ebitda_neg", "crit_rc_menor1", "crit_cobertura_menor1"]].sum(axis=1)
    if cfg.REGLA_ETIQUETA == "documento":
        d["evento"] = d[["crit_ley1116", "crit_patrimonio_neg", "crit_ebitda_neg",
                         "crit_rc_menor1", "crit_cobertura_menor1"]].any(axis=1)
    elif cfg.REGLA_ETIQUETA == "estricta":
        d["evento"] = d.crit_ley1116 | d.crit_patrimonio_neg | (fin >= 2)
    elif cfg.REGLA_ETIQUETA == "formal":
        d["evento"] = d.crit_ley1116
    else:
        raise ValueError("REGLA_ETIQUETA debe ser 'documento', 'estricta' o 'formal'")
    d["evento"] = d["evento"].astype(int)
    return d


def construir_etiqueta(d):
    """
    y(t) = 1 si la empresa presenta el evento en ALGUNO de los años t+1..t+H.
    Se usan las variables del año t para predecirlo (modelo de alerta temprana).
    """
    H = cfg.HORIZONTE
    ev = d.pivot_table(index="nit", columns="anio", values="evento", aggfunc="max")
    filas = []
    for anio in sorted(d.anio.unique()):
        futuros = [a for a in range(anio + 1, anio + H + 1) if a in ev.columns]
        if len(futuros) < H:            # sin ventana futura completa: no se etiqueta
            continue
        sub = ev[futuros]
        n_obs = sub.notna().sum(axis=1)
        y = sub.max(axis=1)
        filas.append(pd.DataFrame({"nit": sub.index, "anio": anio,
                                   "y": y.values, "anios_futuros_obs": n_obs.values}))
    etq = pd.concat(filas, ignore_index=True)
    etq = etq[etq.anios_futuros_obs >= cfg.MIN_ANIOS_FUTUROS_OBSERVADOS]

    base = d.merge(etq, on=["nit", "anio"], how="inner")
    n0 = len(base)
    if cfg.EXCLUIR_YA_EN_PROCESO_EN_T:
        base = base[~base.crit_ley1116]
    if cfg.EXCLUIR_YA_EN_RIESGO_EN_T:
        base = base[base.evento == 0]
    base["y"] = base["y"].astype(int)
    print(f"  Observaciones etiquetadas: {n0:,} -> {len(base):,} tras exclusiones")
    return base


def columnas_modelo():
    cols = list(cfg.VARIABLES_TABLA1)
    if cfg.USAR_VARIABLES_ADICIONALES:
        cols += cfg.VARIABLES_ADICIONALES
    return cols


def construir_base_modelado(panel, guardar=True):
    d = calcular_indicadores(panel)
    d = agregar_macro(d)
    d = criterios_insolvencia(d)
    base = construir_etiqueta(d)

    # Resumen de prevalencia por regla (para discutir con la tutora)
    resumen = (d.groupby(["anio", "sector"])[[c for c in d.columns if c.startswith("crit_")] + ["evento"]]
                 .mean().round(4))
    if guardar:
        d.to_pickle(cfg.RUTA_INTERMEDIOS / "panel_indicadores.pkl")
        base.to_pickle(cfg.RUTA_INTERMEDIOS / "base_modelado.pkl")
        resumen.to_csv(cfg.RUTA_RESULTADOS / "02_prevalencia_criterios.csv")
        (base.groupby(["anio", "sector"]).y.agg(["count", "mean"]).round(4)
             .to_csv(cfg.RUTA_RESULTADOS / "02_distribucion_etiqueta.csv"))
    print(f"  Proporción de empresas en riesgo (y=1): {base.y.mean():.1%}")
    return d, base
