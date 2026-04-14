"""
=============================================================================
MODULE : charts.py
=============================================================================
Rôle    : Fonctions de visualisation interactive avec Plotly.
          Chaque fonction retourne un objet go.Figure prêt à afficher
          dans Streamlit ou un notebook Jupyter.
Auteur  : Projet académique – Démographie Africaine
=============================================================================
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.data_loader import INDICATOR_FR


# ---------------------------------------------------------------------------
# PALETTE ET THÈME GLOBAL
# ---------------------------------------------------------------------------

# Palette de couleurs cohérente pour toutes les visualisations
PALETTE = {
    "primary":   "#F5A623",   # Orange chaud
    "secondary": "#4A9EFF",   # Bleu vif
    "success":   "#00D4AA",   # Vert aqua
    "danger":    "#FF4D6A",   # Rouge
    "accent":    "#E84393",   # Rose
    "neutral":   "#8B9BB4",   # Gris-bleu
}

COLORS_SCALE = ["#1a2235", "#1e4a6e", "#1a7abf", "#00d4aa", "#f5a623"]

LAYOUT_BASE = dict(
    font=dict(family="Inter, sans-serif", size=12, color="#E8EEF8"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.6)",
    margin=dict(t=40, b=40, l=50, r=20),
    legend=dict(
        bgcolor="rgba(21,30,46,0.8)",
        bordercolor="#1F2D45",
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(
        gridcolor="rgba(31,45,69,0.5)",
        linecolor="#1F2D45",
        tickfont=dict(size=10),
    ),
    yaxis=dict(
        gridcolor="rgba(31,45,69,0.5)",
        linecolor="#1F2D45",
        tickfont=dict(size=10),
    ),
)


def apply_layout(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Applique le thème global à une figure Plotly."""
    layout = LAYOUT_BASE.copy()
    layout["title"] = dict(text=title, font=dict(size=14, color=PALETTE["primary"]),
                           x=0.02, xanchor="left")
    layout["height"] = height
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 1. GRAPHIQUES TEMPORELS
# ---------------------------------------------------------------------------

def plot_time_series(panel: pd.DataFrame,
                     countries: list[str],
                     indicator: str,
                     title: str = "") -> go.Figure:
    """
    Graphique linéaire de l'évolution temporelle d'un indicateur
    pour un ou plusieurs pays.

    Parameters
    ----------
    panel     : pd.DataFrame – données en format long
    countries : list[str]    – liste des pays à afficher
    indicator : str          – code de l'indicateur
    title     : str          – titre du graphique

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["success"],
              PALETTE["accent"], PALETTE["danger"], PALETTE["neutral"]]

    for i, country in enumerate(countries):
        data = panel[
            (panel["country"] == country) & (panel["indicator"] == indicator)
        ].sort_values("year")

        if data.empty:
            continue

        fig.add_trace(go.Scatter(
            x=data["year"],
            y=data["value"],
            name=country,
            mode="lines+markers",
            line=dict(color=colors[i % len(colors)], width=2.5),
            marker=dict(size=5),
            connectgaps=True,
            hovertemplate=f"<b>{country}</b><br>Année: %{{x}}<br>{INDICATOR_FR.get(indicator, indicator)}: %{{y:.2f}}<extra></extra>",
        ))

    label = INDICATOR_FR.get(indicator, indicator)
    return apply_layout(fig, title or f"Évolution — {label}")


def plot_continental_trend(stats_df: pd.DataFrame,
                            indicator: str) -> go.Figure:
    """
    Graphique de la tendance continentale avec intervalle de confiance.
    Affiche la moyenne ± écart-type pour montrer la dispersion inter-pays.

    Parameters
    ----------
    stats_df  : pd.DataFrame – statistiques agrégées (output de compute_continental_stats)
    indicator : str

    Returns
    -------
    go.Figure
    """
    data = stats_df[stats_df["indicator"] == indicator].sort_values("year")
    if data.empty:
        return go.Figure()

    fig = go.Figure()

    # Bande de dispersion (moyenne ± écart-type)
    fig.add_trace(go.Scatter(
        x=pd.concat([data["year"], data["year"].iloc[::-1]]),
        y=pd.concat([data["mean"] + data["std"], (data["mean"] - data["std"]).iloc[::-1]]),
        fill="toself",
        fillcolor=f"rgba(245,166,35,0.12)",
        line=dict(color="rgba(245,166,35,0)"),
        name="± Écart-type",
        showlegend=True,
    ))

    # Ligne de la moyenne
    fig.add_trace(go.Scatter(
        x=data["year"], y=data["mean"],
        name="Moyenne continentale",
        mode="lines+markers",
        line=dict(color=PALETTE["primary"], width=3),
        marker=dict(size=6),
    ))

    # Ligne de la médiane
    fig.add_trace(go.Scatter(
        x=data["year"], y=data["median"],
        name="Médiane",
        mode="lines",
        line=dict(color=PALETTE["secondary"], width=2, dash="dash"),
    ))

    label = INDICATOR_FR.get(indicator, indicator)
    return apply_layout(fig, f"Tendance africaine — {label}")


# ---------------------------------------------------------------------------
# 2. GRAPHIQUES DE COMPARAISON
# ---------------------------------------------------------------------------

def plot_bar_ranking(wide: pd.DataFrame,
                     indicator: str,
                     year: int = 2023,
                     top_n: int = 20,
                     highlight: list[str] | None = None) -> go.Figure:
    """
    Graphique à barres horizontales du classement des pays.

    Parameters
    ----------
    wide      : pd.DataFrame
    indicator : str
    year      : int
    top_n     : int – nombre de pays à afficher
    highlight : list[str] – pays à mettre en évidence (couleur distincte)

    Returns
    -------
    go.Figure
    """
    if indicator not in wide.columns:
        return go.Figure()

    data = wide[wide["year"] == year][["country", indicator]].dropna()
    data = data.sort_values(indicator, ascending=False).head(top_n)

    highlight = highlight or []
    colors = [
        PALETTE["accent"] if c in highlight else PALETTE["primary"]
        for c in data["country"]
    ]

    fig = go.Figure(go.Bar(
        x=data[indicator],
        y=data["country"],
        orientation="h",
        marker_color=colors,
        marker_line_color="rgba(0,0,0,0)",
        hovertemplate="<b>%{y}</b><br>Valeur: %{x:.2f}<extra></extra>",
    ))

    label = INDICATOR_FR.get(indicator, indicator)
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return apply_layout(fig, f"Classement — {label} ({year})", height=max(350, top_n * 22))


def plot_comparison_radar(wide: pd.DataFrame,
                           countries: list[str],
                           year: int = 2022) -> go.Figure:
    """
    Graphique radar (toile d'araignée) pour comparer plusieurs pays
    sur plusieurs indicateurs normalisés simultanément.

    Chaque dimension est normalisée entre 0 et 1 pour permettre
    la comparaison d'indicateurs d'unités différentes.

    Parameters
    ----------
    wide      : pd.DataFrame
    countries : list[str]
    year      : int

    Returns
    -------
    go.Figure
    """
    indicators = ["gdp_pc", "life_exp", "electricity", "school_enroll",
                  "urban", "health_exp", "edu_exp"]
    labels_fr = [INDICATOR_FR.get(i, i) for i in indicators]

    year_data = wide[wide["year"] == year]
    fig = go.Figure()

    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["success"],
              PALETTE["accent"], PALETTE["danger"]]

    for i, country in enumerate(countries[:5]):
        row = year_data[year_data["country"] == country]
        if row.empty:
            continue

        values = []
        for ind in indicators:
            if ind not in wide.columns:
                values.append(0)
                continue
            col_data = year_data[ind].dropna()
            val = row[ind].values[0] if ind in row.columns else np.nan
            if np.isnan(val) or col_data.empty:
                values.append(0)
            else:
                mn, mx = col_data.min(), col_data.max()
                norm = (val - mn) / (mx - mn) if mx > mn else 0.5
                values.append(round(norm * 100, 1))

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels_fr + [labels_fr[0]],
            fill="toself",
            name=country,
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)].replace(")", ",0.15)").replace("rgb", "rgba")
                        if "rgb" in colors[i % len(colors)] else colors[i % len(colors)] + "26",
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(17,24,39,0.8)",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        **{k: v for k, v in LAYOUT_BASE.items() if k != "xaxis" and k != "yaxis"},
        height=420,
        title=dict(text="Profil multi-indicateurs (normalisé 0–100)",
                   font=dict(size=14, color=PALETTE["primary"])),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. CARTE CHOROPLÈTHE
# ---------------------------------------------------------------------------

def plot_choropleth_map(wide: pd.DataFrame,
                         indicator: str,
                         year: int = 2023) -> go.Figure:
    """
    Carte choroplèthe de l'Afrique colorée selon la valeur d'un indicateur.

    Utilise les codes ISO-3 pour identifier les pays.
    Seule l'Afrique est affichée grâce à la configuration du scope.

    Parameters
    ----------
    wide      : pd.DataFrame – format large avec colonne 'iso3'
    indicator : str
    year      : int

    Returns
    -------
    go.Figure
    """
    if indicator not in wide.columns:
        return go.Figure()

    data = wide[wide["year"] == year][["country", "iso3", indicator]].dropna()
    label = INDICATOR_FR.get(indicator, indicator)

    fig = px.choropleth(
        data,
        locations="iso3",
        locationmode="ISO-3",
        color=indicator,
        hover_name="country",
        hover_data={indicator: ":.2f", "iso3": False},
        color_continuous_scale=["#1a2235", "#1e4a6e", "#1a7abf", "#00d4aa", "#f5a623"],
        labels={indicator: label},
        scope="africa",
    )

    fig.update_geos(
        bgcolor="rgba(10,14,26,1)",
        landcolor="rgba(26,34,53,1)",
        coastlinecolor="#1F2D45",
        countrycolor="#1F2D45",
        showocean=True,
        oceancolor="rgba(10,14,26,0.8)",
        showframe=False,
        projection_type="natural earth",
    )

    fig.update_coloraxes(
        colorbar=dict(
            title=dict(text=label, font=dict(color="#8B9BB4", size=11)),
            tickfont=dict(color="#8B9BB4"),
            bgcolor="rgba(21,30,46,0.8)",
            bordercolor="#1F2D45",
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=40, b=0, l=0, r=0),
        height=500,
        title=dict(text=f"Carte — {label} ({year})",
                   font=dict(size=14, color=PALETTE["primary"])),
        font=dict(color="#E8EEF8"),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. GRAPHIQUES DE CORRÉLATION
# ---------------------------------------------------------------------------

def plot_scatter_correlation(result: dict,
                              ind_x: str,
                              ind_y: str) -> go.Figure:
    """
    Nuage de points avec droite de régression et annotation des statistiques.

    Chaque point représente un pays africain.
    La droite OLS visualise la tendance linéaire.

    Parameters
    ----------
    result : dict – output de compute_bivariate_correlation()
    ind_x  : str  – indicateur X
    ind_y  : str  – indicateur Y

    Returns
    -------
    go.Figure
    """
    if not result or "x_vals" not in result:
        return go.Figure()

    x = result["x_vals"]
    y = result["y_vals"]
    labels = result["labels"]

    fig = go.Figure()

    # Nuage de points
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers+text",
        text=[c[:3] for c in labels],
        textposition="top center",
        textfont=dict(size=8, color=PALETTE["neutral"]),
        marker=dict(
            size=10,
            color=PALETTE["primary"],
            opacity=0.75,
            line=dict(color=PALETTE["primary"], width=0.5),
        ),
        name="Pays",
        hovertemplate="<b>%{customdata}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        customdata=labels,
    ))

    # Droite de régression linéaire
    x_line = np.linspace(min(x), max(x), 100)
    y_line = result["slope"] * x_line + result["intercept"]

    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        name=f"Régression OLS (R²={result['r_squared']:.3f})",
        line=dict(color=PALETTE["danger"], width=2, dash="dash"),
    ))

    # Annotation statistique
    r = result["pearson_r"]
    p = result["pearson_p"]
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    interpretation = (
        "Forte" if abs(r) > 0.7 else "Modérée" if abs(r) > 0.4 else "Faible"
    )
    direction = "positive" if r > 0 else "négative"

    annotation_text = (
        f"r = {r:.3f} {sig}<br>"
        f"R² = {result['r_squared']:.3f}<br>"
        f"n = {result['n_obs']}<br>"
        f"Corrélation {interpretation} {direction}"
    )

    fig.add_annotation(
        x=0.97, y=0.97,
        xref="paper", yref="paper",
        text=annotation_text,
        showarrow=False,
        font=dict(size=10, color="#E8EEF8"),
        bgcolor="rgba(21,30,46,0.85)",
        bordercolor=PALETTE["primary"],
        borderwidth=1,
        align="left",
    )

    x_label = INDICATOR_FR.get(ind_x, ind_x)
    y_label = INDICATOR_FR.get(ind_y, ind_y)

    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
    return apply_layout(fig, f"Corrélation : {x_label} ↔ {y_label}", height=420)


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """
    Heatmap de la matrice de corrélation inter-indicateurs.

    Couleurs :
      Bleu   → corrélation négative forte
      Blanc  → absence de corrélation
      Orange → corrélation positive forte

    Parameters
    ----------
    corr_matrix : pd.DataFrame – matrice carrée de corrélations

    Returns
    -------
    go.Figure
    """
    from utils.data_loader import INDICATOR_FR

    # Traduction des labels
    labels = [INDICATOR_FR.get(c, c)[:20] for c in corr_matrix.columns]

    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale=[
            [0.0,  "#1a7abf"],   # Corrélation négative forte → bleu
            [0.5,  "#1a2235"],   # Pas de corrélation → fond sombre
            [1.0,  "#f5a623"],   # Corrélation positive forte → orange
        ],
        zmid=0,
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=8),
        colorbar=dict(
            title=dict(text="r de Pearson", font=dict(color="#8B9BB4")),
            tickfont=dict(color="#8B9BB4"),
        ),
        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
        **{k: v for k, v in LAYOUT_BASE.items() if k != "xaxis" and k != "yaxis"},
        height=520,
        title=dict(text="Matrice de corrélation inter-indicateurs",
                   font=dict(size=14, color=PALETTE["primary"])),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. PROJECTION DÉMOGRAPHIQUE
# ---------------------------------------------------------------------------

def plot_population_projection(proj_data: pd.DataFrame,
                                country: str,
                                scenario: str) -> go.Figure:
    """
    Graphique de projection démographique combinant historique et projection.

    Parameters
    ----------
    proj_data : pd.DataFrame – output de project_population()
    country   : str
    scenario  : str

    Returns
    -------
    go.Figure
    """
    hist = proj_data[proj_data["type"] == "historique"]
    proj = proj_data[proj_data["type"] == "projection"]

    # Pont pour continuité visuelle
    if not hist.empty and not proj.empty:
        bridge = pd.DataFrame([hist.iloc[-1]])
        bridge["type"] = "bridge"
    else:
        bridge = pd.DataFrame()

    fig = go.Figure()

    # Données historiques
    fig.add_trace(go.Scatter(
        x=hist["year"], y=hist["population"] / 1e6,
        name="Historique (World Bank)",
        mode="lines+markers",
        line=dict(color=PALETTE["secondary"], width=3),
        marker=dict(size=5),
        hovertemplate="Année: %{x}<br>Population: %{y:.2f}M<extra></extra>",
    ))

    # Projection
    color_map = {
        "baseline":  PALETTE["primary"],
        "optimiste": PALETTE["success"],
        "pessimiste": PALETTE["danger"],
    }
    proj_color = color_map.get(scenario, PALETTE["primary"])

    # Continuité
    x_proj = pd.concat([hist.tail(1), proj])["year"] if not bridge.empty else proj["year"]
    y_proj = pd.concat([hist.tail(1), proj])["population"] / 1e6 if not bridge.empty else proj["population"] / 1e6

    fig.add_trace(go.Scatter(
        x=x_proj, y=y_proj,
        name=f"Projection ({scenario})",
        mode="lines",
        line=dict(color=proj_color, width=2.5, dash="dot"),
        fill="tonexty" if scenario == "optimiste" else "none",
        fillcolor=f"rgba(0,212,170,0.05)",
        hovertemplate="Année: %{x}<br>Projection: %{y:.2f}M<extra></extra>",
    ))

    # Marqueur de la rupture historique/projection
    if not hist.empty:
        cutoff = int(hist["year"].max())
        fig.add_vline(
            x=cutoff,
            line_dash="dash",
            line_color="#1F2D45",
            annotation_text=f"Données réelles ← | → Projection",
            annotation_font_size=10,
            annotation_font_color=PALETTE["neutral"],
        )

    fig.update_layout(yaxis_title="Population (millions)")
    scen_fr = {"baseline": "Baseline", "optimiste": "Optimiste", "pessimiste": "Pessimiste"}
    return apply_layout(
        fig,
        f"Projection démographique — {country} (scénario {scen_fr.get(scenario, scenario)})",
        height=400
    )


# ---------------------------------------------------------------------------
# 6. DASHBOARD — GRAPHIQUES SYNTHÉTIQUES
# ---------------------------------------------------------------------------

def plot_kpi_sparklines(panel: pd.DataFrame, country: str) -> go.Figure:
    """
    Mini-graphiques (sparklines) pour les indicateurs clés d'un pays.
    Présentation compacte en grille 2×3.

    Parameters
    ----------
    panel   : pd.DataFrame
    country : str

    Returns
    -------
    go.Figure
    """
    indicators = [
        ("pop",        "Population",        PALETTE["secondary"]),
        ("gdp_pc",     "PIB/hab.",          PALETTE["primary"]),
        ("birth_rate", "Natalité",          PALETTE["accent"]),
        ("life_exp",   "Esp. de vie",       PALETTE["success"]),
        ("inflation",  "Inflation",         PALETTE["danger"]),
        ("electricity","Électricité",       PALETTE["neutral"]),
    ]

    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=[i[1] for i in indicators],
                        vertical_spacing=0.2,
                        horizontal_spacing=0.1)

    for idx, (ind, name, color) in enumerate(indicators):
        row, col = divmod(idx, 3)
        data = panel[(panel["country"] == country) & (panel["indicator"] == ind)].sort_values("year")
        if data.empty:
            continue
        fig.add_trace(
            go.Scatter(x=data["year"], y=data["value"],
                       mode="lines+markers",
                       line=dict(color=color, width=2),
                       marker=dict(size=4),
                       showlegend=False,
                       name=name),
            row=row+1, col=col+1
        )

    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.4)",
        font=dict(color="#8B9BB4", size=10),
        margin=dict(t=40, b=20, l=30, r=20),
        title=dict(text=f"Évolution des indicateurs clés — {country}",
                   font=dict(size=13, color=PALETTE["primary"])),
    )
    fig.update_annotations(font_size=10, font_color="#8B9BB4")
    return fig
