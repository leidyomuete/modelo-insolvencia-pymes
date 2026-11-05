"""
main.py — Ejecuta el pipeline completo del modelo predictivo de riesgo de
insolvencia (Random Forest + SMOTE vs. regresión logística).

Uso:
    python main.py                 # todo el proceso
    python main.py --desde 2       # reutiliza el panel ya consolidado (paso 1)
    python main.py --desde 3       # reutiliza la base de modelado (pasos 1-2)
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from src import carga, variables, modelado, reporte, macro


def main(desde=1):
    t0 = time.time()
    cfg.RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    cfg.RUTA_INTERMEDIOS.mkdir(parents=True, exist_ok=True)

    print("\n[0/5] Variables macroeconómicas (Banco de la República)")
    if desde <= 2:
        macro.construir_macro()

    print("\n[1/5] Carga y depuración de los estados financieros (SIIS)")
    if desde <= 1:
        panel = carga.construir_panel()
    else:
        panel = pd.read_pickle(cfg.RUTA_INTERMEDIOS / "panel_consolidado.pkl")

    print("\n[2/5] Indicadores financieros, variables macro y etiqueta (horizonte "
          f"{cfg.HORIZONTE} años, regla '{cfg.REGLA_ETIQUETA}')")
    if desde <= 2:
        panel_ind, base = variables.construir_base_modelado(panel)
    else:
        panel_ind = pd.read_pickle(cfg.RUTA_INTERMEDIOS / "panel_indicadores.pkl")
        base = pd.read_pickle(cfg.RUTA_INTERMEDIOS / "base_modelado.pkl")
    cols = variables.columnas_modelo()

    print("\n[3/5] Entrenamiento y evaluación (70 % / 30 %)")
    rf, lr, metr, imp = modelado.evaluar_y_comparar(base, cols)

    if cfg.VALIDACION_FUERA_DE_TIEMPO:
        print("\n[4/5] Validación fuera de tiempo")
        modelado.validacion_fuera_de_tiempo(base, cols, rf, lr)

    print("\n[5/5] Calificación de empresas y reporte")
    calif = reporte.calificar_empresas(panel_ind, rf, cols)
    bit = pd.read_csv(cfg.RUTA_RESULTADOS / "01_bitacora_depuracion.csv", index_col=0) \
        if (cfg.RUTA_RESULTADOS / "01_bitacora_depuracion.csv").exists() else None
    reporte.exportar(calif, metr, imp, bit)
    reporte.guardar_modelo(rf, cols)
    print(f"\nListo en {(time.time() - t0) / 60:.1f} minutos. Resultados en: {cfg.RUTA_RESULTADOS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=int, default=1, help="paso desde el cual ejecutar (1, 2 o 3)")
    main(ap.parse_args().desde)
