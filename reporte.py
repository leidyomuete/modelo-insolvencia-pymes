"""
Módulos 6 y 7 — Clasificación e interpretación por empresa (SRS-06) y
generación del reporte para usuarios no técnicos (SRS-07, StRS-02, StRS-05).
"""
import numpy as np
import pandas as pd
import joblib

import config as cfg

NOMBRES = {
    "razon_corriente": "Razón corriente", "endeudamiento": "Endeudamiento total",
    "roa": "Rentabilidad del activo (ROA)", "cobertura_intereses": "Cobertura de intereses",
    "capital_trabajo_at": "Capital de trabajo / Activo", "ebitda_ventas": "Margen EBITDA",
    "rotacion_activos": "Rotación de activos", "tasa_banrep": "Tasa BanRep",
    "inflacion": "Inflación", "crec_pib": "Crecimiento PIB", "trm_promedio": "TRM promedio",
    "log_activos": "Tamaño (log activos)", "flujo_operativo_pt": "Flujo operativo / Pasivo",
    "ganancias_acum_at": "Ganancias acumuladas / Activo", "deuda_financiera_at": "Deuda financiera / Activo",
    "crec_ventas": "Crecimiento de ventas", "edad_empresa": "Edad de la empresa",
    "sector_manufacturero": "Sector manufacturero", "opinion_rf_modificada": "Opinión RF modificada", "crec_pib_sector": "Crecimiento PIB de la rama",
}


def nivel_riesgo(p):
    for nivel, (a, b) in cfg.UMBRALES_RIESGO.items():
        if a <= p < b:
            return nivel
    return "Alto"


def factores_por_empresa(modelo_rf, X, cols, k=3):
    """Top-k variables que más empujan el riesgo hacia arriba (valores SHAP)."""
    try:
        import shap
        prep = modelo_rf[:-2]                      # winsor + imputación (sin SMOTE ni modelo)
        Xt = prep.transform(X)
        expl = shap.TreeExplainer(modelo_rf.named_steps["modelo"])
        sv = expl.shap_values(Xt, approximate=True, check_additivity=False)
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        orden = np.argsort(-sv, axis=1)[:, :k]
        return [", ".join(NOMBRES.get(cols[j], cols[j]) for j in fila) for fila in orden]
    except ImportError:
        print("  (instale 'shap' para obtener factores por empresa)")
        return [""] * len(X)


def calificar_empresas(panel_ind, modelo_rf, cols, anio=None):
    """Califica a todas las empresas del último año disponible (riesgo en t+1..t+3)."""
    anio = anio or int(panel_ind.anio.max())
    d = panel_ind[panel_ind.anio == anio].copy()
    d["probabilidad_riesgo"] = modelo_rf.predict_proba(d[cols])[:, 1]
    d["nivel_riesgo"] = d.probabilidad_riesgo.apply(nivel_riesgo)
    d["principales_factores"] = factores_por_empresa(modelo_rf, d[cols], cols)
    d["ranking_en_sector"] = d.groupby("sector").probabilidad_riesgo.rank(ascending=False, method="min").astype(int)
    d["en_proceso_formal_hoy"] = np.where(d.crit_ley1116, "Sí", "No")
    salida = d[["nit", "razon_social", "sector", "ciiu", "departamento", "anio", "estado", "en_proceso_formal_hoy",
                "probabilidad_riesgo", "nivel_riesgo", "ranking_en_sector", "principales_factores",
                "razon_corriente", "endeudamiento", "roa", "cobertura_intereses", "ebitda_ventas"]]
    salida = salida.rename(columns={"anio": "anio_estados_financieros"})
    return salida.sort_values("probabilidad_riesgo", ascending=False)


def exportar(calif, metricas, importancia, bitacora=None):
    ruta = cfg.RUTA_RESULTADOS / "05_reporte_riesgo_insolvencia.xlsx"
    with pd.ExcelWriter(ruta, engine="openpyxl") as w:
        calif.to_excel(w, sheet_name="Calificación empresas", index=False)
        (calif.groupby(["sector", "nivel_riesgo"]).size().unstack(fill_value=0)
              .to_excel(w, sheet_name="Resumen por sector"))
        metricas.round(4).to_excel(w, sheet_name="Métricas modelos")
        importancia.assign(variable=importancia.variable.map(lambda c: NOMBRES.get(c, c))) \
                   .round(5).to_excel(w, sheet_name="Importancia variables", index=False)
        if bitacora is not None:
            bitacora.to_excel(w, sheet_name="Depuración")
    print(f"  Reporte guardado en: {ruta}")
    return ruta


def guardar_modelo(modelo, cols, nombre="modelo_rf_smote.joblib"):
    joblib.dump({"modelo": modelo, "variables": cols}, cfg.RUTA_RESULTADOS / nombre)
