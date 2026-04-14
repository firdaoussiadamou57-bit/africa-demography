"""
=============================================================================
MODULE : game_engine.py
=============================================================================
Rôle    : Moteur de simulation du jeu "Demography Manager: Africa Edition".
          Implémente la logique de simulation, les effets des politiques
          économiques sur les indicateurs démographiques, et le système de score.

Concept pédagogique :
    Le jeu illustre les relations causales entre politiques économiques
    et évolution démographique, basées sur les modèles académiques de :
    - Transition démographique (Notestein, 1945)
    - Capital humain et fécondité (Becker & Lewis, 1973)
    - Investissement en santé publique (Preston, 1975)

Auteur  : Projet académique – Démographie Africaine
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# STRUCTURES DE DONNÉES
# ---------------------------------------------------------------------------

@dataclass
class CountryState:
    """
    État complet du pays à un instant t dans la simulation.

    Tous les indicateurs sont modifiables par les décisions du joueur.
    """
    # Identification
    country:       str
    year:          int

    # Indicateurs démographiques
    population:    float   # Nombre d'habitants
    birth_rate:    float   # Taux de natalité (‰)
    death_rate:    float   # Taux de mortalité (‰)
    life_exp:      float   # Espérance de vie (années)
    fertility:     float   # Indice de fécondité (naissances/femme)

    # Indicateurs économiques
    gdp_pc:        float   # PIB par habitant (USD)
    inflation:     float   # Taux d'inflation (%)
    unemployment:  float   # Taux de chômage (%)

    # Indicateurs sociaux
    electricity:   float   # Accès à l'électricité (%)
    urban:         float   # Taux d'urbanisation (%)
    education_lvl: float   # Niveau d'éducation normalisé (0–100)

    # Indicateurs de performance
    score:         int = 0
    turn:          int = 0
    budget:        float = 0.0

    # Historique (listes pour traçage)
    history:       dict = field(default_factory=dict)


@dataclass
class PolicyAllocation:
    """
    Répartition budgétaire du joueur en pourcentage du budget total.
    La somme doit être égale à 100%.
    """
    education:      float = 25.0   # Investissement en éducation
    health:         float = 25.0   # Investissement en santé
    infrastructure: float = 20.0   # Infrastructures (énergie, transport)
    economy:        float = 20.0   # Développement économique (soutien PME, investissement)
    social:         float = 10.0   # Programmes sociaux (aide aux familles, retraites)

    @property
    def total(self) -> float:
        return self.education + self.health + self.infrastructure + self.economy + self.social

    def is_valid(self, tolerance: float = 2.0) -> bool:
        return abs(self.total - 100.0) <= tolerance


# ---------------------------------------------------------------------------
# INITIALISATION DE LA SIMULATION
# ---------------------------------------------------------------------------

def initialize_game(panel: pd.DataFrame,
                    country: str,
                    difficulty: str = "medium") -> CountryState:
    """
    Initialise l'état du pays à partir des données réelles 2016.

    Le niveau de difficulté modifie le budget disponible :
    - easy   : budget × 1.5  → marge de manœuvre confortable
    - medium : budget × 1.0  → équilibre réaliste
    - hard   : budget × 0.6  → contraintes sévères

    Parameters
    ----------
    panel      : pd.DataFrame – données historiques
    country    : str
    difficulty : str

    Returns
    -------
    CountryState initialisé avec les données réelles 2016
    """
    budget_multipliers = {"easy": 1.5, "medium": 1.0, "hard": 0.6}
    mult = budget_multipliers.get(difficulty, 1.0)

    def get_val(indicator, default):
        """Extrait la valeur 2016 ou la dernière disponible."""
        data = panel[(panel["country"] == country) & (panel["indicator"] == indicator)]
        if data.empty:
            return default
        row_2016 = data[data["year"] == 2016]
        if not row_2016.empty:
            v = row_2016["value"].values[0]
            return v if not np.isnan(v) else default
        v = data.sort_values("year").iloc[-1]["value"]
        return v if not np.isnan(v) else default

    pop       = get_val("pop",         15_000_000)
    gdp_pc    = get_val("gdp_pc",      1_500)
    birth     = get_val("birth_rate",  32.0)
    death     = get_val("death_rate",  8.0)
    life      = get_val("life_exp",    63.0)
    fertility = get_val("fertility",   4.0)
    inflation = get_val("inflation",   6.0)
    unemp     = get_val("unemployment",8.0)
    elec      = get_val("electricity", 50.0)
    urban     = get_val("urban",       40.0)
    edu_exp   = get_val("edu_exp",     4.0)

    # Budget = 2% du PIB total, ajusté par la difficulté
    budget = pop * gdp_pc * 0.02 * mult

    # Niveau d'éducation initial estimé à partir des dépenses éducation
    education_lvl = min(100, max(10, edu_exp * 15))

    state = CountryState(
        country=country,
        year=2016,
        population=pop,
        birth_rate=birth,
        death_rate=death,
        life_exp=life,
        fertility=fertility,
        gdp_pc=gdp_pc,
        inflation=inflation,
        unemployment=unemp,
        electricity=elec,
        urban=urban,
        education_lvl=education_lvl,
        budget=budget,
    )

    # Initialisation de l'historique
    state.history = {
        "years":       [2016],
        "population":  [pop],
        "gdp_pc":      [gdp_pc],
        "birth_rate":  [birth],
        "life_exp":    [life],
        "fertility":   [fertility],
        "inflation":   [inflation],
        "unemployment":[unemp],
        "electricity": [elec],
        "score":       [0],
    }

    return state


# ---------------------------------------------------------------------------
# MOTEUR DE SIMULATION (TOUR PAR TOUR)
# ---------------------------------------------------------------------------

def simulate_turn(state: CountryState,
                  policy: PolicyAllocation) -> tuple[CountryState, dict]:
    """
    Simule un tour de jeu (1 année) en appliquant les politiques choisies.

    MODÈLE ÉCONOMIQUE-DÉMOGRAPHIQUE
    ================================
    Les effets sont basés sur les élasticités estimées dans la littérature
    académique. Un résumé des mécanismes :

    Éducation → fécondité ↓ (Becker, 1981)
        +10% budget éducation → -0.05 fécondité, -1.5 natalité/10 ans
    Santé → mortalité ↓, espérance de vie ↑ (Preston, 1975)
        +10% budget santé → -0.3 mortalité, +0.5 ans espérance de vie
    Économie → PIB ↑, chômage ↓ (Harrod-Domar implicite)
        +10% budget économie → +2.5% croissance PIB/hab
    Infrastructure → électricité ↑, productivité ↑
        +10% budget infra → +3% accès électricité
    Social → inflation modérée, cohésion sociale
        +10% budget social → -0.5% inflation

    Parameters
    ----------
    state  : CountryState – état actuel du pays
    policy : PolicyAllocation – décisions du joueur (%)

    Returns
    -------
    tuple : (nouveau_state, events_dict)
        events_dict : événements survenus ce tour
    """
    events = []

    # Normalisation des effets de politique (écart à la baseline 20%)
    edu_eff   = (policy.education      - 20) / 100
    health_eff= (policy.health         - 20) / 100
    econ_eff  = (policy.economy        - 15) / 100
    infra_eff = (policy.infrastructure - 18) / 100
    social_eff= (policy.social         - 10) / 100

    rng = np.random.default_rng(seed=state.year + state.turn)  # reproductibilité

    # --- DÉMOGRAPHIE ---

    # Effet de l'éducation sur la fécondité (relation inverse documentée)
    d_fertility    = -0.04 - edu_eff * 0.25 + rng.normal(0, 0.02)
    new_fertility  = max(1.1, state.fertility + d_fertility)

    # Natalité liée à la fécondité et à l'urbanisation
    d_birth_rate   = -0.35 - edu_eff * 1.0 + (state.urban - 50) * (-0.01)
    new_birth_rate = max(8.0, state.birth_rate + d_birth_rate + rng.normal(0, 0.15))

    # Mortalité réduite par la santé
    d_death_rate   = -0.12 - health_eff * 0.4 + rng.normal(0, 0.08)
    new_death_rate = max(3.5, state.death_rate + d_death_rate)

    # Espérance de vie (santé + niveau de vie)
    d_life_exp     = 0.18 + health_eff * 0.65 + econ_eff * 0.15 + rng.normal(0, 0.05)
    new_life_exp   = min(85.0, state.life_exp + d_life_exp)

    # Croissance démographique naturelle
    natural_growth = (new_birth_rate - new_death_rate) / 1000
    new_population = state.population * (1 + natural_growth + rng.normal(0, 0.001))

    # --- ÉCONOMIE ---

    # Croissance du PIB/hab : économie + éducation (capital humain) + infrastructure
    gdp_growth_rate = (
        0.025
        + econ_eff  * 0.07
        + edu_eff   * 0.025
        + infra_eff * 0.035
        + rng.normal(0, 0.008)
    )
    new_gdp_pc = state.gdp_pc * (1 + gdp_growth_rate)

    # Inflation (économie de marché et politique budgétaire)
    d_inflation = (
        -econ_eff  * 1.5
        -social_eff * 0.8
        + (policy.education + policy.health > 65) * 1.2  # sur-dépense publique
        + rng.normal(0, 0.5)
    )
    new_inflation = max(0.5, min(60, state.inflation + d_inflation))

    # Chômage
    d_unemployment = (
        -econ_eff  * 2.5
        -infra_eff * 0.8
        -edu_eff   * 0.5
        + rng.normal(0, 0.4)
    )
    new_unemployment = max(0.5, min(50, state.unemployment + d_unemployment))

    # --- SOCIAL ---

    # Accès à l'électricité
    d_electricity = infra_eff * 3.5 + econ_eff * 1.0 + rng.normal(0, 0.3)
    new_electricity = min(100, max(0, state.electricity + d_electricity))

    # Urbanisation (tendance structurelle + économie)
    new_urban = min(100, state.urban + 0.5 + infra_eff * 0.4 + rng.normal(0, 0.1))

    # Niveau d'éducation cumulé
    new_edu_lvl = min(100, state.education_lvl + edu_eff * 5 + 0.5)

    # Budget pour le prochain tour (indexé sur la croissance du PIB)
    new_budget = state.budget * (1 + gdp_growth_rate)

    # --- SCORE ---
    # Le score récompense un développement équilibré
    score_delta = int(
        (new_gdp_pc / 100) * 1.5
        + (new_life_exp - 60) * 8
        + (100 - new_inflation) * 3
        + (100 - new_unemployment) * 2
        + new_electricity * 1.5
        + (100 - new_fertility * 20) * 0.5
        - (new_birth_rate > 40) * 30   # Pénalité natalité très élevée
        - (new_inflation > 25) * 50    # Pénalité inflation excessive
        - (new_unemployment > 20) * 40 # Pénalité chômage élevé
    )
    new_score = state.score + max(0, score_delta)

    # --- ÉVÉNEMENTS ALÉATOIRES ---
    events = _check_events(state, policy, rng)

    # Application des effets des événements
    for evt in events:
        if "gdp_shock" in evt:
            new_gdp_pc *= evt["gdp_shock"]
        if "unemployment_shock" in evt:
            new_unemployment = min(50, new_unemployment + evt["unemployment_shock"])
        if "fertility_shock" in evt:
            new_fertility = max(1.1, new_fertility + evt["fertility_shock"])
        if "life_exp_shock" in evt:
            new_life_exp = min(85, max(40, new_life_exp + evt["life_exp_shock"]))

    # --- CONSTRUCTION DU NOUVEL ÉTAT ---
    new_state = CountryState(
        country=state.country,
        year=state.year + 1,
        population=round(new_population),
        birth_rate=round(new_birth_rate, 2),
        death_rate=round(new_death_rate, 2),
        life_exp=round(new_life_exp, 2),
        fertility=round(new_fertility, 3),
        gdp_pc=round(new_gdp_pc, 1),
        inflation=round(new_inflation, 2),
        unemployment=round(new_unemployment, 2),
        electricity=round(new_electricity, 1),
        urban=round(new_urban, 1),
        education_lvl=round(new_edu_lvl, 1),
        score=new_score,
        turn=state.turn + 1,
        budget=round(new_budget),
        history=state.history,
    )

    # Mise à jour de l'historique
    h = new_state.history
    h["years"].append(new_state.year)
    h["population"].append(new_state.population)
    h["gdp_pc"].append(new_state.gdp_pc)
    h["birth_rate"].append(new_state.birth_rate)
    h["life_exp"].append(new_state.life_exp)
    h["fertility"].append(new_state.fertility)
    h["inflation"].append(new_state.inflation)
    h["unemployment"].append(new_state.unemployment)
    h["electricity"].append(new_state.electricity)
    h["score"].append(new_state.score)

    return new_state, {"events": events, "gdp_growth": round(gdp_growth_rate * 100, 2)}


def _check_events(state: CountryState,
                  policy: PolicyAllocation,
                  rng: np.random.Generator) -> list[dict]:
    """
    Génère des événements aléatoires conditionnels selon l'état du pays.

    Les événements simulent des chocs exogènes (crises, opportunités)
    probabilistes, dont la fréquence dépend de la vulnérabilité du pays.

    Returns
    -------
    list[dict] – liste d'événements survenus (peut être vide)
    """
    events = []

    # Probabilités conditionnelles selon l'état
    p_hyperinflation = 0.08 if state.inflation > 20 else 0.02
    p_edu_boom       = 0.15 if policy.education > 35 and state.turn % 3 == 0 else 0.0
    p_health_prog    = 0.12 if policy.health > 35 and state.turn % 4 == 0 else 0.0
    p_invest_influx  = 0.10 if state.gdp_pc > 2000 and policy.economy > 30 else 0.03
    p_drought        = 0.06  # choc climatique indépendant
    p_conflict_risk  = 0.04 if state.unemployment > 22 and state.inflation > 15 else 0.01

    event_pool = [
        (p_hyperinflation, {
            "type": "danger",
            "icon": "🔴",
            "title": "Crise inflationniste",
            "message": "La forte inflation érode le pouvoir d'achat et freine l'investissement.",
            "gdp_shock": 0.88,
            "unemployment_shock": 2.5,
        }),
        (p_edu_boom, {
            "type": "success",
            "icon": "🎓",
            "title": "Boom éducatif",
            "message": "L'investissement en éducation porte ses fruits : hausse du capital humain.",
            "fertility_shock": -0.08,
            "gdp_shock": 1.02,
        }),
        (p_health_prog, {
            "type": "success",
            "icon": "❤️",
            "title": "Progrès sanitaire",
            "message": "Les dépenses de santé réduisent la mortalité infantile.",
            "life_exp_shock": 0.4,
            "fertility_shock": -0.05,
        }),
        (p_invest_influx,  {
            "type": "success",
            "icon": "💹",
            "title": "Afflux d'investissements",
            "message": "Les réformes économiques attirent des capitaux étrangers.",
            "gdp_shock": 1.045,
            "unemployment_shock": -1.5,
        }),
        (p_drought, {
            "type": "warning",
            "icon": "alerte",
            "title": "Sécheresse régionale",
            "message": "Un épisode de sécheresse affecte la production agricole.",
            "gdp_shock": 0.97,
            "unemployment_shock": 1.0,
        }),
        (p_conflict_risk, {
            "type": "danger",
            "icon": "⚠️",
            "title": "Instabilité sociale",
            "message": "Chômage élevé et inflation créent des tensions sociales.",
            "gdp_shock": 0.93,
            "unemployment_shock": 3.0,
        }),
    ]

    for prob, evt in event_pool:
        if rng.random() < prob:
            events.append(evt)
            break  # Maximum 1 événement par tour pour la lisibilité

    return events


# ---------------------------------------------------------------------------
# ÉVALUATION FINALE
# ---------------------------------------------------------------------------

def evaluate_final_score(state: CountryState) -> dict:
    """
    Évalue les performances finales du joueur à la fin de la simulation.

    Calcule des scores détaillés par dimension et attribue un grade.

    Parameters
    ----------
    state : CountryState – état final du pays

    Returns
    -------
    dict avec score, grade, détail par dimension, et message de conclusion
    """
    # Scores par dimension (0–100 chacun)
    scores = {
        "Développement éco.": min(100, state.gdp_pc / 100),
        "Santé publique":      min(100, (state.life_exp - 55) / 20 * 100),
        "Stabilité des prix":  max(0, 100 - state.inflation * 2.5),
        "Emploi":              max(0, 100 - state.unemployment * 3),
        "Infrastructure":      state.electricity,
        "Transition démo.":    max(0, min(100, (7 - state.fertility) / 5 * 100)),
    }

    overall = round(np.mean(list(scores.values())), 1)
    score   = state.score

    # Attribution du grade académique
    if score > 120_000:   grade, mention = "S",  "Exceptionnel"
    elif score > 80_000:  grade, mention = "A",  "Excellent"
    elif score > 50_000:  grade, mention = "B",  "Bien"
    elif score > 25_000:  grade, mention = "C",  "Satisfaisant"
    elif score > 10_000:  grade, mention = "D",  "Passable"
    else:                 grade, mention = "F",  "Insuffisant"

    messages = {
        "S": " Performance exceptionnelle ! Votre pays est un modèle de développement durable.",
        "A": " Excellente gestion ! Vous avez réussi à allier croissance et bien-être social.",
        "B": " Bonne performance ! Quelques déséquilibres persistent mais le bilan est positif.",
        "C": " Résultat satisfaisant. Des politiques plus équilibrées auraient amélioré les résultats.",
        "D": " Résultat passable. La gestion a manqué d'équilibre entre dimensions économiques et sociales.",
        "F": " Résultat insuffisant. Les politiques adoptées n'ont pas permis un développement durable.",
    }

    return {
        "total_score":    score,
        "grade":          grade,
        "mention":        mention,
        "overall_pct":    overall,
        "dim_scores":     scores,
        "message":        messages.get(grade, ""),
        "final_pop":      state.population,
        "final_gdp_pc":   state.gdp_pc,
        "final_life_exp": state.life_exp,
        "final_fertility":state.fertility,
        "turns_played":   state.turn,
    }


# ---------------------------------------------------------------------------
# HISTORIQUE POUR GRAPHIQUES
# ---------------------------------------------------------------------------

def history_to_dataframe(state: CountryState) -> pd.DataFrame:
    """
    Convertit l'historique de simulation en DataFrame pour les graphiques.

    Parameters
    ----------
    state : CountryState

    Returns
    -------
    pd.DataFrame
    """
    h = state.history
    df = pd.DataFrame({
        "year":       h["years"],
        "population": h["population"],
        "gdp_pc":     h["gdp_pc"],
        "birth_rate": h["birth_rate"],
        "life_exp":   h["life_exp"],
        "fertility":  h["fertility"],
        "inflation":  h["inflation"],
        "unemployment":h["unemployment"],
        "electricity":h["electricity"],
        "score":      h["score"],
    })
    return df
