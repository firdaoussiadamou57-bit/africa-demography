import pandas as pd
import numpy as np
from pathlib import Path
import os

# Chemin vers le fichier Excel 
BASE_DIR  = Path(os.path.abspath(__file__)).parent.parent
DATA_PATH = BASE_DIR / "data" / "africa_indicator_2016_2025.xlsx"

YEARS = [str(y) for y in range(2016, 2024)]

INDICATOR_LABELS = {
    "Population, total":                                                 "pop",
    "Population growth (annual %)":                                      "pop_growth",
    "Birth rate, crude (per 1,000 people)":                              "birth_rate",
    "Death rate, crude (per 1,000 people)":                              "death_rate",
    "Life expectancy at birth, total (years)":                           "life_exp",
    "Fertility rate, total (births per woman)":                          "fertility",
    "GDP (current US$)":                                                 "gdp",
    "GDP growth (annual %)":                                             "gdp_growth",
    "GDP per capita (current US$)":                                      "gdp_pc",
    "Inflation, consumer prices (annual %)":                             "inflation",
    "Unemployment, total (% of total labor force) (national estimate)":  "unemployment",
    "Government expenditure on education, total (% of GDP)":             "edu_exp",
    "Current health expenditure (% of GDP)":                             "health_exp",
    "Urban population (% of total population)":                          "urban",
    "Access to electricity (% of population)":                           "electricity",
    "School enrollment, secondary (% gross)":                            "school_enroll",
}

INDICATOR_FR = {
    "pop":           "Population totale",
    "pop_growth":    "Croissance démographique (%)",
    "birth_rate":    "Taux de natalité (‰)",
    "death_rate":    "Taux de mortalité (‰)",
    "life_exp":      "Espérance de vie (ans)",
    "fertility":     "Indice de fécondité",
    "gdp":           "PIB total (USD)",
    "gdp_growth":    "Croissance du PIB (%)",
    "gdp_pc":        "PIB par habitant (USD)",
    "inflation":     "Inflation (%)",
    "unemployment":  "Taux de chômage (%)",
    "edu_exp":       "Dépenses éducation (% PIB)",
    "health_exp":    "Dépenses santé (% PIB)",
    "urban":         "Population urbaine (%)",
    "electricity":   "Accès à l'électricité (%)",
    "school_enroll": "Scolarisation secondaire (%)",
}

ISO3_CODES = {
    "Algeria":"DZA","Angola":"AGO","Benin":"BEN","Botswana":"BWA",
    "Burkina Faso":"BFA","Burundi":"BDI","Cabo Verde":"CPV","Cameroon":"CMR",
    "Central African Republic":"CAF","Chad":"TCD","Comoros":"COM",
    "Congo, Dem. Rep.":"COD","Congo, Rep.":"COG","Cote d'Ivoire":"CIV",
    "Djibouti":"DJI","Egypt, Arab Rep.":"EGY","Equatorial Guinea":"GNQ",
    "Eritrea":"ERI","Eswatini":"SWZ","Ethiopia":"ETH","Gabon":"GAB",
    "Gambia, The":"GMB","Ghana":"GHA","Guinea":"GIN","Guinea-Bissau":"GNB",
    "Kenya":"KEN","Lesotho":"LSO","Liberia":"LBR","Libya":"LBY",
    "Madagascar":"MDG","Malawi":"MWI","Mali":"MLI","Mauritania":"MRT",
    "Mauritius":"MUS","Morocco":"MAR","Mozambique":"MOZ","Namibia":"NAM",
    "Niger":"NER","Nigeria":"NGA","Rwanda":"RWA","Sao Tome and Principe":"STP",
    "Senegal":"SEN","Seychelles":"SYC","Sierra Leone":"SLE",
    "Somalia, Fed. Rep.":"SOM","South Africa":"ZAF","South Sudan":"SSD",
    "Sudan":"SDN","Tanzania":"TZA","Togo":"TGO","Tunisia":"TUN",
    "Uganda":"UGA","Zambia":"ZMB","Zimbabwe":"ZWE",
}


def load_raw_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier Excel introuvable : {DATA_PATH}\n"
            f"BASE_DIR = {BASE_DIR}\n"
            f"Contenu de BASE_DIR : {list(BASE_DIR.iterdir())}"
        )
    return pd.read_excel(DATA_PATH, sheet_name="Data")


def clean_and_filter(df):
    valid_mask = (
        df["Country Name"].notna() &
        df["Country Name"].isin(ISO3_CODES.keys())
    )
    df = df[valid_mask].copy()
    year_col_map = {}
    for col in df.columns:
        for y in range(2010, 2026):
            if str(y) in str(col) and "YR" in str(col):
                year_col_map[col] = str(y)
    df = df.rename(columns=year_col_map)
    df = df.replace("..", np.nan)
    df = df[df["Series Name"].isin(INDICATOR_LABELS.keys())].copy()
    df["indicator"] = df["Series Name"].map(INDICATOR_LABELS)
    for y in YEARS:
        if y in df.columns:
            df[y] = pd.to_numeric(df[y], errors="coerce")
    return df


def build_panel(df):
    id_vars   = ["Country Name", "Country Code", "indicator"]
    year_cols = [y for y in YEARS if y in df.columns]
    panel = df[id_vars + year_cols].melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name="year",
        value_name="value"
    )
    panel = panel.rename(columns={"Country Name": "country", "Country Code": "iso3"})
    panel["year"]  = panel["year"].astype(int)
    panel["value"] = pd.to_numeric(panel["value"], errors="coerce")
    panel = panel.dropna(subset=["value"])
    return panel.reset_index(drop=True)


def build_wide(panel):
    wide = panel.pivot_table(
        index=["country", "iso3", "year"],
        columns="indicator",
        values="value",
        aggfunc="mean"
    ).reset_index()
    wide.columns.name = None
    return wide


def get_last_value(panel, country, indicator):
    sub = panel[(panel["country"] == country) & (panel["indicator"] == indicator)]
    if sub.empty:
        return None
    return sub.sort_values("year").iloc[-1]["value"]
