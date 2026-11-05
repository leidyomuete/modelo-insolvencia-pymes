"""
Construcción de las variables macroeconómicas a partir de los archivos
descargados de los informes estadísticos del Banco de la República
(carpeta bd). Genera data/macro_colombia.csv.

Variables (año calendario t):
  tasa_banrep        Tasa de política monetaria vigente al cierre del año (%)
  inflacion          Inflación total anual a diciembre (%)  — DANE vía BanRep
  crec_pib           Crecimiento real anual del PIB (%)     — DANE, base 2015, precios constantes
  trm_promedio       Promedio anual de la TRM diaria (COP/USD)
  crec_pib_manufactura / crec_pib_comercio
                     Crecimiento real anual del valor agregado de la rama
                     (se asigna a cada empresa según su sector -> crec_pib_sector)
"""
import pandas as pd
import numpy as np

import config as cfg


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(".", "", regex=False)
                         .str.replace(",", ".", regex=False).str.strip(), errors="coerce") \
        if s.dtype == object else pd.to_numeric(s, errors="coerce")


def tasa_intervencion():
    d = pd.read_excel(cfg.RUTA_BD / cfg.ARCHIVOS_MACRO["tasa"], header=None, skiprows=2)
    d = d.iloc[:, :2].dropna()
    d.columns = ["fecha", "tasa"]
    d["fecha"] = pd.to_datetime(d.fecha, format="%d/%m/%Y", errors="coerce")
    d["tasa"] = pd.to_numeric(d.tasa.astype(str).str.replace(",", "."), errors="coerce")
    d = d.dropna().sort_values("fecha")
    return d.groupby(d.fecha.dt.year).tasa.last().rename("tasa_banrep")


def inflacion():
    d = pd.read_excel(cfg.RUTA_BD / cfg.ARCHIVOS_MACRO["inflacion"], sheet_name="Datos",
                      header=None, skiprows=2)
    d = d.iloc[:, [0, 2]].dropna()
    d.columns = ["fecha", "inflacion"]
    d["fecha"] = pd.to_datetime(d.fecha, format="%Y/%m/%d", errors="coerce")
    d["inflacion"] = pd.to_numeric(d.inflacion, errors="coerce")
    d = d.dropna()
    dic = d[d.fecha.dt.month == 12]
    return dic.set_index(dic.fecha.dt.year).inflacion


def trm():
    d = pd.read_excel(cfg.RUTA_BD / cfg.ARCHIVOS_MACRO["trm"], header=None, skiprows=5)
    d = d.iloc[:, :2]
    d.columns = ["fecha", "trm"]
    d["fecha"] = pd.to_datetime(d.fecha, errors="coerce")
    d["trm"] = pd.to_numeric(d.trm, errors="coerce")
    d = d.dropna()
    return d.groupby(d.fecha.dt.year).trm.mean().round(2).rename("trm_promedio")


def pib():
    """Primer bloque del archivo: precios constantes base 2015 (variación real)."""
    d = pd.read_excel(cfg.RUTA_BD / cfg.ARCHIVOS_MACRO["pib"], header=None)
    fin = d.index[d[0].astype(str).str.strip() == "Producto interno bruto"][0]   # primer bloque
    fechas = pd.to_datetime(d.iloc[7], errors="coerce")
    col_valor = [c for c in d.columns if pd.notna(fechas[c])]
    bloque = d.iloc[:fin + 1]

    def serie(etiqueta, ocurrencia=0):
        fila = bloque[bloque[0].astype(str).str.strip() == etiqueta].iloc[ocurrencia]
        s = pd.Series(pd.to_numeric(fila[col_valor].values, errors="coerce"),
                      index=fechas[col_valor].values)
        anual = s.groupby(s.index.year).agg(["sum", "count"])
        anual = anual[anual["count"] == 4]["sum"]                 # solo años completos
        return (anual.pct_change() * 100).round(2)

    return pd.DataFrame({"crec_pib": serie("Producto interno bruto"),
                         "crec_pib_manufactura": serie("Industrias manufactureras"),
                         # La 1.a fila "Comercio al por mayor y al por menor" es la gran rama
                         # G-I (comercio + transporte + alojamiento); la 2.a es solo comercio (G)
                         "crec_pib_comercio": serie("Comercio al por mayor y al por menor", 1)})


def construir_macro(guardar=True):
    m = pd.concat([tasa_intervencion(), inflacion(), pib(), trm()], axis=1)
    m = m.loc[[a for a in cfg.ANIOS if a in m.index]]
    m.index.name = "anio"
    m = m.reset_index()
    if guardar:
        m.to_csv(cfg.RUTA_MACRO, index=False)
    print(m.to_string(index=False))
    faltan = m.columns[m.isna().any()].tolist()
    if faltan:
        print(f"  ATENCIÓN: datos faltantes en {faltan} (p. ej. año sin 4 trimestres de PIB)")
    return m
