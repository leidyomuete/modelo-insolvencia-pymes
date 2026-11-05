"""
Módulos 3, 4 y 5 — Balanceo SMOTE (SRS-03), entrenamiento Random Forest y
regresión logística (SRS-04) y evaluación sobre el 30 % de prueba (SRS-05).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, roc_curve)
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import config as cfg


class Winsorizador(BaseEstimator, TransformerMixin):
    """Recorta cada variable a los percentiles definidos, aprendidos SOLO en entrenamiento."""
    def __init__(self, limites=(0.01, 0.99)):
        self.limites = limites

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.inf_ = X.quantile(self.limites[0]).values
        self.sup_ = X.quantile(self.limites[1]).values
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.inf_, self.sup_)


# ---------------------------------------------------------------------------
# Pipelines (SMOTE se aplica únicamente dentro del entrenamiento de cada fold)
# ---------------------------------------------------------------------------
def pipeline_rf():
    return Pipeline([
        ("winsor", Winsorizador(cfg.WINSORIZAR)),
        ("imputar", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=cfg.SEMILLA)),
        ("modelo", RandomForestClassifier(random_state=cfg.SEMILLA, n_jobs=-1)),
    ])


def pipeline_logit():
    return Pipeline([
        ("winsor", Winsorizador(cfg.WINSORIZAR)),
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
        ("smote", SMOTE(random_state=cfg.SEMILLA)),
        ("modelo", LogisticRegression(max_iter=5000)),
    ])


GRID_RF = {
    "smote__k_neighbors": [3, 5, 7],
    "modelo__n_estimators": [300, 500],
    "modelo__max_depth": [6, 10, 15, None],
    "modelo__min_samples_leaf": [5, 20, 50, 100],
    "modelo__max_features": ["sqrt", 0.3, 0.5],
}
GRID_LOGIT = {
    "smote__k_neighbors": [3, 5, 7],
    "modelo__C": np.logspace(-3, 2, 12),
}


# ---------------------------------------------------------------------------
# División 70/30
# ---------------------------------------------------------------------------
def dividir(base):
    if cfg.DIVIDIR_POR_EMPRESA:
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.PROPORCION_PRUEBA, random_state=cfg.SEMILLA)
        i_tr, i_te = next(gss.split(base, base.y, groups=base.nit))
    else:
        from sklearn.model_selection import train_test_split
        i_tr, i_te = train_test_split(np.arange(len(base)), test_size=cfg.PROPORCION_PRUEBA,
                                      stratify=base.y, random_state=cfg.SEMILLA)
    return base.iloc[i_tr].copy(), base.iloc[i_te].copy()


def entrenar(nombre, pipe, grid, X, y, grupos):
    cv = StratifiedGroupKFold(n_splits=cfg.N_FOLDS_CV, shuffle=True, random_state=cfg.SEMILLA)
    n_iter = min(cfg.N_ITER_BUSQUEDA, int(np.prod([len(v) for v in grid.values()])))
    busq = RandomizedSearchCV(pipe, grid, n_iter=n_iter, scoring="roc_auc", cv=cv,
                              random_state=cfg.SEMILLA, n_jobs=-1, refit=True, verbose=0)
    busq.fit(X, y, groups=grupos)
    print(f"  {nombre}: AUC validación cruzada = {busq.best_score_:.4f}")
    print(f"    mejores hiperparámetros: {busq.best_params_}")
    return busq


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def metricas(y, p, umbral=0.5):
    yhat = (p >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "exactitud": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "sensibilidad_recall": recall_score(y, yhat),
        "especificidad": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y, yhat),
        "auc_roc": roc_auc_score(y, p),
        "auc_pr": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "VP": int(tp), "FP": int(fp), "FN": int(fn), "VN": int(tn),
    }


def bootstrap_auc(y, p_rf, p_lr, n=cfg.N_BOOTSTRAP, semilla=cfg.SEMILLA):
    """IC 95 % del AUC de cada modelo y de la diferencia RF - Logit (BRS-03 / SyRS-06)."""
    rng = np.random.default_rng(semilla)
    y, p_rf, p_lr = map(np.asarray, (y, p_rf, p_lr))
    a_rf, a_lr, dif = [], [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        r, l = roc_auc_score(y[i], p_rf[i]), roc_auc_score(y[i], p_lr[i])
        a_rf.append(r); a_lr.append(l); dif.append(r - l)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {"IC95_auc_rf": q(a_rf), "IC95_auc_logit": q(a_lr), "IC95_diferencia": q(dif),
            "p_valor_rf_no_supera_logit": float(np.mean(np.array(dif) <= 0))}


def graficar_roc(y, probs, ruta):
    plt.figure(figsize=(6, 6))
    for nombre, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{nombre} (AUC = {roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("Tasa de falsos positivos (1 - especificidad)")
    plt.ylabel("Tasa de verdaderos positivos (sensibilidad)")
    plt.title("Curva ROC — conjunto de prueba (30 %)")
    plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(ruta, dpi=200); plt.close()


def importancias(modelo_rf, modelo_lr, X_te, y_te, cols):
    rf = modelo_rf.named_steps["modelo"]
    imp = pd.DataFrame({"variable": cols, "importancia_gini_rf": rf.feature_importances_})
    muestra = X_te.sample(min(len(X_te), 8000), random_state=cfg.SEMILLA)
    perm = permutation_importance(modelo_rf, muestra, y_te.loc[muestra.index], scoring="roc_auc",
                                  n_repeats=5, random_state=cfg.SEMILLA, n_jobs=-1)
    imp["importancia_permutacion_rf"] = perm.importances_mean
    imp["coef_logit_estandarizado"] = modelo_lr.named_steps["modelo"].coef_[0]
    imp["odds_ratio_logit"] = np.exp(imp.coef_logit_estandarizado)
    imp = imp.sort_values("importancia_permutacion_rf", ascending=False)

    top = imp.sort_values("importancia_permutacion_rf")
    plt.figure(figsize=(7, max(4, 0.35 * len(cols))))
    plt.barh(top.variable, top.importancia_permutacion_rf, color="#2b6cb0")
    plt.xlabel("Disminución media del AUC al permutar la variable")
    plt.title("Importancia de variables — Random Forest")
    plt.tight_layout(); plt.savefig(cfg.RUTA_RESULTADOS / "03_importancia_variables.png", dpi=200); plt.close()
    return imp


def evaluar_y_comparar(base, cols):
    tr, te = dividir(base)
    print(f"  Entrenamiento: {len(tr):,} obs. ({tr.y.mean():.1%} en riesgo) | "
          f"Prueba: {len(te):,} obs. ({te.y.mean():.1%} en riesgo)")
    X_tr, y_tr, X_te, y_te = tr[cols], tr.y, te[cols], te.y

    b_rf = entrenar("Random Forest + SMOTE", pipeline_rf(), GRID_RF, X_tr, y_tr, tr.nit)
    b_lr = entrenar("Regresión logística + SMOTE", pipeline_logit(), GRID_LOGIT, X_tr, y_tr, tr.nit)
    p_rf = b_rf.predict_proba(X_te)[:, 1]
    p_lr = b_lr.predict_proba(X_te)[:, 1]

    res = pd.DataFrame({"Random Forest + SMOTE": metricas(y_te, p_rf),
                        "Regresión logística + SMOTE": metricas(y_te, p_lr)})
    boot = bootstrap_auc(y_te, p_rf, p_lr)
    print("\n  Métricas en prueba (umbral 0,5):\n", res.round(4).to_string())
    print("\n  Bootstrap AUC:", json.dumps(boot, indent=2))

    res.to_csv(cfg.RUTA_RESULTADOS / "03_metricas_prueba.csv")
    json.dump({**boot, "cv_auc_rf": b_rf.best_score_, "cv_auc_logit": b_lr.best_score_,
               "hiperparametros_rf": {k: str(v) for k, v in b_rf.best_params_.items()},
               "hiperparametros_logit": {k: str(v) for k, v in b_lr.best_params_.items()}},
              open(cfg.RUTA_RESULTADOS / "03_comparacion_modelos.json", "w"), indent=2)
    graficar_roc(y_te, {"Random Forest + SMOTE": p_rf, "Regresión logística + SMOTE": p_lr},
                 cfg.RUTA_RESULTADOS / "03_curva_roc.png")
    imp = importancias(b_rf.best_estimator_, b_lr.best_estimator_, X_te, y_te, cols)
    imp.to_csv(cfg.RUTA_RESULTADOS / "03_importancia_variables.csv", index=False)

    # Métricas por sector (StRS-03)
    te = te.assign(p_rf=p_rf, p_lr=p_lr)
    por_sector = te.groupby("sector").apply(
        lambda g: pd.Series({"n": len(g), "tasa_riesgo": g.y.mean(),
                             "auc_rf": roc_auc_score(g.y, g.p_rf), "auc_logit": roc_auc_score(g.y, g.p_lr)}))
    por_sector.to_csv(cfg.RUTA_RESULTADOS / "03_metricas_por_sector.csv")
    print("\n  Por sector:\n", por_sector.round(4).to_string())
    return b_rf.best_estimator_, b_lr.best_estimator_, res, imp


def validacion_fuera_de_tiempo(base, cols, modelo_rf, modelo_lr):
    """Robustez: reentrena con años <= corte y evalúa en años posteriores."""
    from sklearn.base import clone
    tr, te = base[base.anio <= cfg.ANIO_CORTE_OOT], base[base.anio > cfg.ANIO_CORTE_OOT]
    if te.empty or tr.empty:
        return None
    out = {}
    for nombre, m in {"Random Forest + SMOTE": modelo_rf, "Regresión logística + SMOTE": modelo_lr}.items():
        mm = clone(m).fit(tr[cols], tr.y)
        out[nombre] = metricas(te.y, mm.predict_proba(te[cols])[:, 1])
    res = pd.DataFrame(out)
    res.to_csv(cfg.RUTA_RESULTADOS / "04_validacion_fuera_de_tiempo.csv")
    print(f"\n  Validación fuera de tiempo (entrena <= {cfg.ANIO_CORTE_OOT}, prueba > {cfg.ANIO_CORTE_OOT}):\n",
          res.round(4).to_string())
    return res
