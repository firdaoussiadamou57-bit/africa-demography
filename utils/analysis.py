"""
=============================================================================
MODULE : analysis.py
=============================================================================
Rôle    : Analyses statistiques des dynamiques démographiques et économiques.
          Inclut : statistiques descriptives, corrélations, régression,
          projection démographique, et classification des pays.
Auteur  : Projet académique – Démographie Africaine
=============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------------
# 1. STATISTIQUES DESCRIPTIVES
# ---------------------------------------------------------------------------

def compute_continental_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les statistiques continentales agrégées par année.

    Pour chaque indicateur et chaque année, calcule :
    - moyenne, médiane, écart-type, min, max sur l'ensemble des pays.

    Parameters
    ----------
    panel : pd.DataFrame
        Données en format long.

    Returns
    -------
    pd.DataFrame
        Statistiques agrégées par (indicator, year).
    """
    stats_df = panel.groupby(["indicator", "year"])["value"].agg(
        mean="mean",
        median="median",
        std="std",
        min_val="min",
        max_val="max",
        count="count"
    ).reset_index()
    return stats_df


def compute_total_population(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la population totale africaine par année.

    Parameters
    ----------
    panel : pd.DataFrame

    Returns
    -------
    pd.DataFrame avec colonnes ['year', 'total_population']
    """
    pop_df = panel[panel["indicator"] == "pop"].groupby("year")["value"].sum().reset_index()
    pop_df.columns = ["year", "total_population"]
    return pop_df


def compute_country_summary(wide: pd.DataFrame, year: int = 2023) -> pd.DataFrame:
    """
    Construit un tableau récapitulatif des indicateurs clés pour tous les pays
    pour une année donnée.

    Parameters
    ----------
    wide  : pd.DataFrame  – format large
    year  : int           – année de référence

    Returns
    -------
    pd.DataFrame trié par population décroissante
    """
    year_data = wide[wide["year"] == year].copy()

    cols = ["country", "iso3", "pop", "gdp_pc", "life_exp",
            "fertility", "birth_rate", "gdp_growth", "inflation"]
    available = [c for c in cols if c in year_data.columns]

    summary = year_data[available].copy()
    if "pop" in summary.columns:
        summary = summary.sort_values("pop", ascending=False)

    return summary.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. ANALYSE DE CORRÉLATION
# ---------------------------------------------------------------------------

def compute_correlation_matrix(wide: pd.DataFrame,
                                year: int = 2022) -> pd.DataFrame:
    """
    Calcule la matrice de corrélation de Pearson entre tous les indicateurs
    pour une année donnée.

    La corrélation de Pearson mesure la relation linéaire entre deux variables
    (r ∈ [-1, 1]) :
      r > 0.7  → forte corrélation positive
      r < -0.7 → forte corrélation négative
      |r| < 0.3 → corrélation faible

    Parameters
    ----------
    wide : pd.DataFrame
    year : int

    Returns
    -------
    pd.DataFrame – matrice de corrélation (n_indicateurs × n_indicateurs)
    """
    year_data = wide[wide["year"] == year]
    numeric_cols = year_data.select_dtypes(include=[np.number]).columns
    # Exclure les colonnes non-indicateurs
    exclude = {"year"}
    indicator_cols = [c for c in numeric_cols if c not in exclude]

    corr_matrix = year_data[indicator_cols].corr(method="pearson")
    return corr_matrix


def compute_bivariate_correlation(panel: pd.DataFrame,
                                   ind_x: str,
                                   ind_y: str,
                                   year: int = 2022) -> dict:
    """
    Calcule la corrélation bivariée entre deux indicateurs avec tests statistiques.

    Retourne :
    - coefficient de Pearson (r) et p-value
    - coefficient de Spearman (ρ) et p-value (plus robuste aux outliers)
    - droite de régression linéaire (OLS)

    Parameters
    ----------
    panel : pd.DataFrame
    ind_x : str – indicateur sur l'axe X
    ind_y : str – indicateur sur l'axe Y
    year  : int

    Returns
    -------
    dict avec clés : x_vals, y_vals, labels, pearson_r, pearson_p,
                     spearman_r, spearman_p, slope, intercept, r_squared
    """
    # Pivot pour obtenir les deux indicateurs par pays
    sub = panel[panel["indicator"].isin([ind_x, ind_y]) & (panel["year"] == year)]
    pivoted = sub.pivot_table(index="country", columns="indicator", values="value")

    if ind_x not in pivoted.columns or ind_y not in pivoted.columns:
        return {}

    clean = pivoted[[ind_x, ind_y]].dropna()
    if len(clean) < 5:
        return {}

    x_vals = clean[ind_x].values
    y_vals = clean[ind_y].values
    labels = clean.index.tolist()

    # Corrélation de Pearson
    pearson_r, pearson_p = pearsonr(x_vals, y_vals)

    # Corrélation de Spearman (non-paramétrique)
    spearman_r, spearman_p = spearmanr(x_vals, y_vals)

    # Régression linéaire OLS (Ordinary Least Squares)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)

    return {
        "x_vals":     x_vals,
        "y_vals":     y_vals,
        "labels":     labels,
        "pearson_r":  round(pearson_r, 4),
        "pearson_p":  round(pearson_p, 4),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 4),
        "slope":      slope,
        "intercept":  intercept,
        "r_squared":  round(r_value**2, 4),
        "n_obs":      len(clean),
    }


# ---------------------------------------------------------------------------
# 3. MODÈLE DE PROJECTION DÉMOGRAPHIQUE
# ---------------------------------------------------------------------------

def project_population(panel: pd.DataFrame,
                        country: str,
                        horizon: int = 2040,
                        scenario: str = "baseline") -> pd.DataFrame:
    """
    Projette la population d'un pays jusqu'à une année horizon.

    Méthode : modèle exponentiel calibré sur la tendance historique.
    Le taux de croissance est ajusté selon le scénario :
      - 'baseline'   : tendance actuelle maintenue
      - 'optimiste'  : forte croissance économique → baisse de fécondité
      - 'pessimiste' : stagnation économique → fécondité persistante

    Formule : P(t+1) = P(t) × (1 + r_ajusté)
    avec r_ajusté = r_historique × facteur_scénario

    Parameters
    ----------
    panel    : pd.DataFrame
    country  : str
    horizon  : int – année finale de projection
    scenario : str – 'baseline', 'optimiste', 'pessimiste'

    Returns
    -------
    pd.DataFrame avec colonnes ['year', 'population', 'type']
    ('type' = 'historique' ou 'projection')
    """
    # --- Extraction des données historiques ---
    hist = panel[(panel["country"] == country) & (panel["indicator"] == "pop")]
    hist = hist.sort_values("year")[["year", "value"]].rename(columns={"value": "population"})
    hist["type"] = "historique"

    if hist.empty:
        return pd.DataFrame()

    # --- Calcul du taux de croissance moyen historique ---
    growth_data = panel[
        (panel["country"] == country) & (panel["indicator"] == "pop_growth")
    ].dropna(subset=["value"])

    if not growth_data.empty:
        avg_growth_rate = growth_data["value"].mean() / 100  # en décimal
    else:
        # Estimation par régression log-linéaire si les données de croissance manquent
        if len(hist) >= 2:
            log_pop = np.log(hist["population"].values)
            years_arr = hist["year"].values - hist["year"].min()
            slope, _, _, _, _ = stats.linregress(years_arr, log_pop)
            avg_growth_rate = slope
        else:
            avg_growth_rate = 0.025  # valeur par défaut africaine

    # --- Facteur d'ajustement selon le scénario ---
    scenario_factors = {
        "baseline":  1.00,   # tendance maintenue
        "optimiste": 0.82,   # éducation ↑ → fécondité ↓ → croissance ↓
        "pessimiste": 1.12,  # stagnation → fécondité persistante
    }
    factor = scenario_factors.get(scenario, 1.00)
    adjusted_rate = avg_growth_rate * factor

    # --- Projection année par année ---
    last_year = int(hist["year"].max())
    last_pop = hist.loc[hist["year"] == last_year, "population"].values[0]

    projection_rows = []
    current_pop = last_pop

    for year in range(last_year + 1, horizon + 1):
        # Légère décroissance progressive du taux (transition démographique)
        dynamic_rate = adjusted_rate * (1 - 0.002 * (year - last_year))
        current_pop = current_pop * (1 + max(0.005, dynamic_rate))
        projection_rows.append({
            "year":       year,
            "population": round(current_pop),
            "type":       "projection"
        })

    proj_df = pd.DataFrame(projection_rows)

    # --- Combinaison historique + projection ---
    result = pd.concat([hist, proj_df], ignore_index=True)
    result["scenario"] = scenario
    result["country"] = country

    return result


# ---------------------------------------------------------------------------
# 4. CLASSEMENT ET SEGMENTATION DES PAYS
# ---------------------------------------------------------------------------

def rank_countries(wide: pd.DataFrame,
                   indicator: str,
                   year: int = 2023,
                   ascending: bool = False) -> pd.DataFrame:
    """
    Classe les pays selon un indicateur pour une année donnée.

    Parameters
    ----------
    wide      : pd.DataFrame
    indicator : str
    year      : int
    ascending : bool – True pour classer du plus bas au plus haut

    Returns
    -------
    pd.DataFrame avec colonne 'rank' ajoutée, trié par rang
    """
    if indicator not in wide.columns:
        return pd.DataFrame()

    year_data = wide[wide["year"] == year][["country", "iso3", indicator]].dropna()
    year_data = year_data.sort_values(indicator, ascending=ascending).reset_index(drop=True)
    year_data["rank"] = year_data.index + 1
    return year_data


def classify_development_stage(wide: pd.DataFrame, year: int = 2022) -> pd.DataFrame:
    """
    Classifie les pays selon leur stade de transition démographique.

    Classification basée sur les travaux de Thompson (1929) et Notestein (1945) :
    - Phase 1 : Natalité élevée + Mortalité élevée (pré-transition)
    - Phase 2 : Natalité élevée + Mortalité en baisse (début transition)
    - Phase 3 : Natalité en baisse + Mortalité faible (transition avancée)
    - Phase 4 : Natalité faible + Mortalité faible (post-transition)

    Parameters
    ----------
    wide : pd.DataFrame
    year : int

    Returns
    -------
    pd.DataFrame avec colonnes supplémentaires 'demo_stage', 'stage_label'
    """
    year_data = wide[wide["year"] == year].copy()

    # Vérification de la disponibilité des colonnes requises
    required = {"birth_rate", "death_rate", "fertility"}
    if not required.issubset(set(year_data.columns)):
        return year_data

    def classify(row):
        br = row.get("birth_rate", np.nan)
        dr = row.get("death_rate", np.nan)
        fe = row.get("fertility",  np.nan)

        if any(pd.isna([br, dr, fe])):
            return 0, "Données insuffisantes"

        if br > 38 and dr > 10:
            return 1, "Phase 1 – Pré-transition"
        elif br > 30 and dr <= 10:
            return 2, "Phase 2 – Début de transition"
        elif 20 < br <= 30 and dr <= 8 and fe > 2.5:
            return 3, "Phase 3 – Transition avancée"
        else:
            return 4, "Phase 4 – Post-transition"

    stages = year_data.apply(classify, axis=1, result_type="expand")
    year_data["demo_stage"] = stages[0]
    year_data["stage_label"] = stages[1]

    return year_data[["country", "iso3", "birth_rate", "death_rate",
                       "fertility", "demo_stage", "stage_label"]].dropna(subset=["demo_stage"])


# ---------------------------------------------------------------------------
# 5. ANALYSE ÉCONOMIE–DÉMOGRAPHIE
# ---------------------------------------------------------------------------

def compute_economic_demographic_index(wide: pd.DataFrame, year: int = 2022) -> pd.DataFrame:
    """
    Calcule un Indice Composite Économie-Démographie (ICED) pour chaque pays.

    L'ICED synthétise les performances sur 6 dimensions normalisées (0–1) :
      - PIB/habitant       (poids : 25%)
      - Espérance de vie   (poids : 20%)
      - Scolarisation      (poids : 15%)
      - Accès électricité  (poids : 15%)
      - Stabilité prix     (poids : 15%)  ← [100 - inflation] normalisé
      - Emploi             (poids : 10%)  ← [100 - chômage] normalisé

    Parameters
    ----------
    wide : pd.DataFrame
    year : int

    Returns
    -------
    pd.DataFrame avec colonne 'ICED' (0–100), trié par score décroissant
    """
    year_data = wide[wide["year"] == year].copy()

    def normalize(series):
        """Min-Max normalization → [0, 1]"""
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    weights = {
        "gdp_pc":      0.25,
        "life_exp":    0.20,
        "school_enroll": 0.15,
        "electricity": 0.15,
        "inflation":   0.15,  # inversé
        "unemployment": 0.10, # inversé
    }

    score = pd.Series(0.0, index=year_data.index)
    total_weight = 0.0

    for col, w in weights.items():
        if col in year_data.columns:
            series = year_data[col].copy()
            # Inversion pour inflation et chômage (moins = mieux)
            if col in {"inflation", "unemployment"}:
                series = series.max() - series
            normalized = normalize(series.fillna(series.median()))
            score += normalized * w
            total_weight += w

    if total_weight > 0:
        score = score / total_weight * 100

    year_data["ICED"] = score.round(1)
    result = year_data[["country", "iso3", "ICED"]].dropna()
    return result.sort_values("ICED", ascending=False).reset_index(drop=True)
