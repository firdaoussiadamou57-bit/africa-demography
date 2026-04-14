"""
=============================================================================
MODULE : data_loader.py
=============================================================================
Rôle    : Chargement, nettoyage et préparation des données démographiques
          et économiques africaines issues de la Banque Mondiale.
Auteur  : Projet académique – Démographie Africaine
Source  : World Bank – World Development Indicators (WDI)
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# CONSTANTES ET CONFIGURATION
# ---------------------------------------------------------------------------

# Chemin vers le fichier de données (relatif à la racine du projet)
import os
DATA_PATH = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "africa_indicator_2016_2025.xlsx"))

# Années couvertes par les données disponibles
YEARS = [str(y) for y in range(2016, 2024)]

# Correspondance : code interne → nom complet de l'indicateur
INDICATOR_LABELS = {
    "Population, total":                                                   "pop",
    "Population growth (annual %)":                                        "pop_growth",
    "Birth rate, crude (per 1,000 people)":                                "birth_rate",
    "Death rate, crude (per 1,000 people)":                                "death_rate",
    "Life expectancy at birth, total (years)":                             "life_exp",
    "Fertility rate, total (births per woman)":                            "fertility",
    "GDP (current US$)":                                                   "gdp",
    "GDP growth (annual %)":                                               "gdp_growth",
    "GDP per capita (current US$)":                                        "gdp_pc",
    "Inflation, consumer prices (annual %)":                               "inflation",
    "Unemployment, total (% of total labor force) (national estimate)":    "unemployment",
    "Government expenditure on education, total (% of GDP)":               "edu_exp",
    "Current health expenditure (% of GDP)":                               "health_exp",
    "Urban population (% of total population)":                            "urban",
    "Access to electricity (% of population)":                             "electricity",
    "School enrollment, secondary (% gross)":                              "school_enroll",
}

# Libellés français pour l'affichage dans les visualisations
INDICATOR_FR = {
    "pop":           "Population totale",
    "pop_growth":    "Croissance démographique (%)",
    "birth_rate":    "Taux de natalité (‰)",
    "death_rate":    "Taux de mortalité (‰)",
    "life_exp":      "Espérance de vie (ans)",
    "fertility":     "Indice de fécondité (naiss./femme)",
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

# Codes ISO3 des pays africains (pour cartographie)
ISO3_CODES = {
    "Algeria": "DZA", "Angola": "AGO", "Benin": "BEN", "Botswana": "BWA",
    "Burkina Faso": "BFA", "Burundi": "BDI", "Cabo Verde": "CPV",
    "Cameroon": "CMR", "Central African Republic": "CAF", "Chad": "TCD",
    "Comoros": "COM", "Congo, Dem. Rep.": "COD", "Congo, Rep.": "COG",
    "Cote d'Ivoire": "CIV", "Djibouti": "DJI", "Egypt, Arab Rep.": "EGY",
    "Equatorial Guinea": "GNQ", "Eritrea": "ERI", "Eswatini": "SWZ",
    "Ethiopia": "ETH", "Gabon": "GAB", "Gambia, The": "GMB",
    "Ghana": "GHA", "Guinea": "GIN", "Guinea-Bissau": "GNB",
    "Kenya": "KEN", "Lesotho": "LSO", "Liberia": "LBR", "Libya": "LBY",
    "Madagascar": "MDG", "Malawi": "MWI", "Mali": "MLI",
    "Mauritania": "MRT", "Mauritius": "MUS", "Morocco": "MAR",
    "Mozambique": "MOZ", "Namibia": "NAM", "Niger": "NER",
    "Nigeria": "NGA", "Rwanda": "RWA", "Sao Tome and Principe": "STP",
    "Senegal": "SEN", "Seychelles": "SYC", "Sierra Leone": "SLE",
    "Somalia, Fed. Rep.": "SOM", "South Africa": "ZAF",
    "South Sudan": "SSD", "Sudan": "SDN", "Tanzania": "TZA",
    "Togo": "TGO", "Tunisia": "TUN", "Uganda": "UGA",
    "Zambia": "ZMB", "Zimbabwe": "ZWE",
}


# ---------------------------------------------------------------------------
# FONCTION PRINCIPALE DE CHARGEMENT
# ---------------------------------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    """
    Charge les données brutes depuis le fichier Excel World Bank.

    Le fichier contient une feuille 'Data' avec :
      - Country Name  : nom du pays
      - Country Code  : code ISO3
      - Series Name   : nom de l'indicateur
      - Colonnes 2010–2025 : valeurs annuelles

    Returns
    -------
    pd.DataFrame
        Données brutes non transformées.
    """
    df = pd.read_excel(DATA_PATH, sheet_name="Data")
    return df


def clean_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et filtre le DataFrame brut :
      1. Supprime les lignes parasites (métadonnées World Bank)
      2. Remplace les valeurs manquantes '..'' par NaN
      3. Filtre uniquement les pays et indicateurs pertinents
      4. Convertit les valeurs numériques

    Parameters
    ----------
    df : pd.DataFrame
        Données brutes issues de load_raw_data().

    Returns
    -------
    pd.DataFrame
        DataFrame nettoyé, prêt pour l'analyse.
    """
    # --- Suppression des lignes parasites (métadonnées World Bank) ---
    valid_mask = (
        df["Country Name"].notna() &
        df["Country Name"].isin(ISO3_CODES.keys())
    )
    df = df[valid_mask].copy()

    # --- Renommage des colonnes années (ex: '2016 [YR2016]' → '2016') ---
    year_col_map = {}
    for col in df.columns:
        for y in range(2010, 2026):
            if str(y) in str(col) and "YR" in str(col):
                year_col_map[col] = str(y)
    df = df.rename(columns=year_col_map)

    # --- Remplacement des valeurs manquantes '..' par NaN ---
    df = df.replace("..", np.nan)

    # --- Filtrage des indicateurs pertinents ---
    df = df[df["Series Name"].isin(INDICATOR_LABELS.keys())].copy()

    # --- Remplacement du nom de l'indicateur par son code court ---
    df["indicator"] = df["Series Name"].map(INDICATOR_LABELS)

    # --- Conversion des colonnes numériques ---
    for y in YEARS:
        if y in df.columns:
            df[y] = pd.to_numeric(df[y], errors="coerce")

    return df


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme le DataFrame en format panel (long format) :
    chaque ligne = (pays, indicateur, année, valeur).

    Ce format est optimal pour les analyses statistiques et visualisations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame nettoyé (format large).

    Returns
    -------
    pd.DataFrame
        DataFrame en format long avec colonnes :
        [country, iso3, indicator, year, value]
    """
    id_vars = ["Country Name", "Country Code", "indicator"]
    year_cols = [y for y in YEARS if y in df.columns]

    panel = df[id_vars + year_cols].melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name="year",
        value_name="value"
    )
    panel = panel.rename(columns={"Country Name": "country", "Country Code": "iso3"})
    panel["year"] = panel["year"].astype(int)
    panel["value"] = pd.to_numeric(panel["value"], errors="coerce")
    panel = panel.dropna(subset=["value"])

    return panel.reset_index(drop=True)


def build_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un DataFrame en format large :
    chaque ligne = (pays, année), chaque colonne = un indicateur.

    Utile pour les analyses de corrélation et la modélisation.

    Parameters
    ----------
    panel : pd.DataFrame
        DataFrame en format long.

    Returns
    -------
    pd.DataFrame
        DataFrame en format large.
    """
    wide = panel.pivot_table(
        index=["country", "iso3", "year"],
        columns="indicator",
        values="value",
        aggfunc="mean"
    ).reset_index()
    wide.columns.name = None
    return wide


def get_last_value(panel: pd.DataFrame, country: str, indicator: str) -> float | None:
    """
    Retourne la dernière valeur disponible pour un pays et un indicateur donnés.

    Parameters
    ----------
    panel : pd.DataFrame
    country : str
    indicator : str

    Returns
    -------
    float | None
    """
    sub = panel[(panel["country"] == country) & (panel["indicator"] == indicator)]
    if sub.empty:
        return None
    return sub.sort_values("year").iloc[-1]["value"]


def load_all_data():
    """
    Fonction principale exportée : charge et prépare toutes les données.
    Mise en cache pour optimiser les performances de l'application.

    Returns
    -------
    dict avec clés :
      - 'panel'  : DataFrame long (pays × indicateur × année × valeur)
      - 'wide'   : DataFrame large (pays × année → colonnes indicateurs)
      - 'countries' : liste triée des pays disponibles
    """
    raw   = load_raw_data()
    clean = clean_and_filter(raw)
    panel = build_panel(clean)
    wide  = build_wide(panel)

    countries = sorted(panel["country"].unique().tolist())

    return {
        "panel":     panel,
        "wide":      wide,
        "countries": countries,
    }
