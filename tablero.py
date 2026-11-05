"""
Tablero web (index.html) con los resultados del modelo y un simulador.

Genera un único archivo HTML autocontenido (datos comprimidos dentro del
archivo, sin servidores ni librerías externas) que puede publicarse en
GitHub Pages: https://<usuario>.github.io/<repositorio>/

    from src import tablero
    tablero.generar_tablero(panel_ind, base, cols, rf, lr, "index.html")

El simulador usa una versión liviana del Random Forest (100 árboles) que se
exporta al navegador; su AUC y su correlación con el modelo de la tesis se
muestran en el propio tablero.
"""
import base64
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import config as cfg
from src import modelado, reporte

PLANTILLA = Path(__file__).with_name("tablero_plantilla.html")

FINANCIERAS = ["razon_corriente", "endeudamiento", "roa", "cobertura_intereses", "capital_trabajo_at",
               "ebitda_ventas", "rotacion_activos", "ganancias_acum_at", "flujo_operativo_pt",
               "deuda_financiera_at", "crec_ventas"]
MACRO = ["tasa_banrep", "inflacion", "crec_pib", "crec_pib_sector", "trm_promedio"]


def _r(x, n=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), n)


def _modelo_web(pipe, cols):
    """Exporta el pipeline (winsorización + imputación + bosque) a listas simples."""
    w, imp, rf = pipe.named_steps["winsor"], pipe.named_steps["imputar"], pipe.named_steps["modelo"]
    arboles = []
    for est in rf.estimators_:
        t = est.tree_
        v = t.value[:, 0, :]
        p1 = v[:, 1] / v.sum(axis=1)
        arboles.append({
            "f": t.feature.tolist(), "t": [float(x) for x in t.threshold],
            "l": t.children_left.tolist(), "r": t.children_right.tolist(), "p": [_r(x, 6) for x in p1]})
    return {"variables": cols, "inf": [float(x) for x in w.inf_], "sup": [float(x) for x in w.sup_],
            "mediana": [float(x) for x in imp.statistics_], "arboles": arboles}


def datos_tablero(panel_ind, base, cols, rf, lr, anio=None):
    anio = anio or int(panel_ind.anio.max())
    tr, te = modelado.dividir(base)
    p_rf, p_lr = rf.predict_proba(te[cols])[:, 1], lr.predict_proba(te[cols])[:, 1]
    m_rf, m_lr = modelado.metricas(te.y, p_rf), modelado.metricas(te.y, p_lr)
    boot = modelado.bootstrap_auc(te.y, p_rf, p_lr, n=300)

    roc = {}
    for nombre, p in [("rf", p_rf), ("lr", p_lr)]:
        f, t, _ = roc_curve(te.y, p)
        idx = np.unique(np.linspace(0, len(f) - 1, 120).astype(int))
        roc[nombre] = [[_r(f[i]), _r(t[i])] for i in idx]

    # modelo liviano para el simulador
    web = modelado.pipeline_rf()
    web.set_params(smote__k_neighbors=7, modelo__n_estimators=100, modelo__min_samples_leaf=150,
                   modelo__max_features="sqrt", modelo__max_depth=12)
    web.fit(tr[cols], tr.y)
    p_web = web.predict_proba(te[cols])[:, 1]

    imp_csv = cfg.RUTA_RESULTADOS / "03_importancia_variables.csv"
    if imp_csv.exists():
        imp = pd.read_csv(imp_csv)[["variable", "importancia_permutacion_rf"]]
        imp.columns = ["variable", "valor"]
    else:
        imp = pd.DataFrame({"variable": cols, "valor": rf.named_steps["modelo"].feature_importances_})
    imp = imp.sort_values("valor", ascending=False).head(12)

    oot_csv = cfg.RUTA_RESULTADOS / "04_validacion_fuera_de_tiempo.csv"
    oot = pd.read_csv(oot_csv, index_col=0) if oot_csv.exists() else None

    crit = {"crit_ley1116": "Proceso formal (Ley 1116)", "crit_patrimonio_neg": "Patrimonio negativo",
            "crit_ebitda_neg": "EBITDA negativo", "crit_rc_menor1": "Razón corriente < 1",
            "crit_cobertura_menor1": "Cobertura de intereses < 1"}
    prev = panel_ind.groupby("sector")[list(crit)].mean()
    anual = base.groupby(["anio", "sector"]).y.mean().unstack()

    # empresas del último año, con sus variables (para el simulador) y su trayectoria
    ult = panel_ind[panel_ind.anio == anio].copy()
    ult["prob"] = rf.predict_proba(ult[cols])[:, 1]
    ult["nivel"] = ult.prob.apply(reporte.nivel_riesgo)
    ult["factores"] = reporte.factores_por_empresa(rf, ult[cols], cols)
    hist = panel_ind[panel_ind.nit.isin(ult.nit)][["nit", "anio"] + cols].copy()
    hist["prob"] = rf.predict_proba(hist[cols])[:, 1]
    tray = {str(n): [[int(a), _r(p, 3)] for a, p in zip(g.anio, g.prob)] for n, g in hist.sort_values("anio").groupby("nit")}

    empresas = []
    for _, r in ult.sort_values("prob", ascending=False).iterrows():
        empresas.append({
            "nit": int(r.nit), "rs": str(r.razon_social), "sector": r.sector,
            "ciiu": str(r.ciiu)[:5], "act": str(r.ciiu)[8:90], "depto": str(r.departamento),
            "estado": str(r.estado), "prob": _r(r.prob, 4), "nivel": r.nivel, "factores": r.factores,
            "x": [_r(r[c], 5) for c in cols]})

    medianas = {s: [_r(v, 5) for v in g[cols].median()] for s, g in ult.groupby("sector")}
    nombres = {c: reporte.NOMBRES.get(c, c) for c in cols}
    nombres["crec_pib_sector"] = "Crecimiento PIB de la rama"

    return {
        "titulo": "Modelo predictivo de riesgo de insolvencia — Pymes manufactureras y comerciales",
        "anio": anio, "horizonte": cfg.HORIZONTE,
        "umbrales": {k: list(v) for k, v in cfg.UMBRALES_RIESGO.items()},
        "resumen": {"empresas": int(panel_ind.nit.nunique()), "estados": int(len(panel_ind)),
                    "observaciones": int(len(base)), "tasa_riesgo": _r(base.y.mean()),
                    "anios": [int(panel_ind.anio.min()), int(panel_ind.anio.max())],
                    "calificadas": len(empresas), "n_prueba": int(len(te))},
        "metricas": {"rf": {k: _r(v) for k, v in m_rf.items()}, "lr": {k: _r(v) for k, v in m_lr.items()},
                     "ic_rf": [_r(x) for x in boot["IC95_auc_rf"]], "ic_lr": [_r(x) for x in boot["IC95_auc_logit"]],
                     "ic_dif": [_r(x) for x in boot["IC95_diferencia"]]},
        "oot": None if oot is None else {"rf": _r(oot.loc["auc_roc"].iloc[0]), "lr": _r(oot.loc["auc_roc"].iloc[1]),
                                         "corte": cfg.ANIO_CORTE_OOT},
        "roc": roc,
        "importancia": [{"variable": nombres.get(v, v), "valor": _r(x, 5)} for v, x in zip(imp.variable, imp.valor)],
        "criterios": [{"criterio": crit[c], "Comercial": _r(prev.loc["Comercial", c]),
                       "Manufacturero": _r(prev.loc["Manufacturero", c])} for c in crit],
        "riesgo_anual": [{"anio": int(a), "Comercial": _r(r.get("Comercial")), "Manufacturero": _r(r.get("Manufacturero"))}
                         for a, r in anual.iterrows()],
        "variables": cols, "nombres": nombres, "financieras": FINANCIERAS, "macro": MACRO,
        "medianas": medianas, "empresas": empresas, "trayectorias": tray,
        "web": {"auc": _r(roc_auc_score(te.y, p_web)), "corr": _r(np.corrcoef(p_web, p_rf)[0, 1]),
                "modelo": _modelo_web(web, cols)},
    }


REPO = "https://github.com/leidyomuete/modelo-insolvencia-pymes"
COLAB = "https://colab.research.google.com/github/leidyomuete/modelo-insolvencia-pymes/blob/main/Modelo_insolvencia_Colab.ipynb"


def generar_tablero(panel_ind, base, cols, rf, lr, ruta=None, repo=REPO, colab=COLAB):
    ruta = Path(ruta or (cfg.RUTA_RESULTADOS / "index.html"))
    datos = datos_tablero(panel_ind, base, cols, rf, lr)
    datos.update({"repo": repo, "colab": colab})
    crudo = json.dumps(datos, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    b64 = base64.b64encode(gzip.compress(crudo, 9)).decode("ascii")
    html = PLANTILLA.read_text(encoding="utf-8").replace("__DATOS_GZIP_BASE64__", b64)
    ruta.write_text(html, encoding="utf-8")
    print(f"  Tablero guardado en: {ruta} ({ruta.stat().st_size / 1e6:.1f} MB)")
    return ruta
