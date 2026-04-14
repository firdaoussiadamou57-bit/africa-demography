"""
=============================================================================
APPLICATION : app.py  —  Point d'entrée principal
=============================================================================
Titre   : Analyse, simulation et gamification de l'évolution démographique
          en Afrique sous contraintes économiques
Framework: Streamlit
Auteur  : Projet académique

Pour lancer l'application :
    streamlit run app.py

Dépendances :
    pip install streamlit plotly pandas openpyxl scipy numpy
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURATION DE LA PAGE (doit être le premier appel Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Démographie Africaine",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# IMPORT DES MODULES INTERNES
# ---------------------------------------------------------------------------
from utils.data_loader import (
    load_raw_data, clean_and_filter, build_panel, build_wide,
    INDICATOR_FR, ISO3_CODES, YEARS
)
from utils.analysis import (
    compute_continental_stats, compute_total_population,
    compute_country_summary, compute_correlation_matrix,
    compute_bivariate_correlation, rank_countries,
    classify_development_stage, compute_economic_demographic_index,
    project_population,
)
from utils.charts import (
    plot_time_series, plot_continental_trend, plot_bar_ranking,
    plot_comparison_radar, plot_choropleth_map, plot_scatter_correlation,
    plot_correlation_heatmap, plot_population_projection,
    plot_kpi_sparklines, PALETTE,
)
from utils.game_engine import (
    initialize_game, simulate_turn, evaluate_final_score,
    history_to_dataframe, PolicyAllocation,
)


# ---------------------------------------------------------------------------
# STYLE CSS PERSONNALISÉ
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Thème sombre personnalisé */
:root {
    --bg:      #0a0e1a;
    --card:    #111827;
    --accent:  #F5A623;
    --text2:   #8b9bb4;
}

/* En-tête de section */
.section-header {
    font-family: 'Segoe UI', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #F5A623;
    border-left: 4px solid #F5A623;
    padding-left: 12px;
    margin-bottom: 4px;
}
.section-sub {
    color: #8b9bb4;
    font-size: 0.85rem;
    margin-bottom: 20px;
}

/* Cartes KPI */
.kpi-card {
    background: linear-gradient(135deg, #111827, #1a2235);
    border: 1px solid #1f2d45;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: #F5A623;
    line-height: 1;
}
.kpi-label {
    font-size: 0.75rem;
    color: #8b9bb4;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 6px;
}
.kpi-delta-pos { color: #00D4AA; font-size: 0.8rem; }
.kpi-delta-neg { color: #FF4D6A; font-size: 0.8rem; }

/* Boîte d'événement jeu */
.event-success { background: rgba(0,212,170,0.1); border-left: 3px solid #00D4AA; padding: 10px 14px; border-radius: 0 6px 6px 0; }
.event-danger  { background: rgba(255,77,106,0.1); border-left: 3px solid #FF4D6A; padding: 10px 14px; border-radius: 0 6px 6px 0; }
.event-warning { background: rgba(245,166,35,0.1); border-left: 3px solid #F5A623; padding: 10px 14px; border-radius: 0 6px 6px 0; }

/* Tag badge */
.tag { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; letter-spacing: 1px; }
.tag-blue   { background: rgba(74,158,255,0.15); color: #4A9EFF; }
.tag-green  { background: rgba(0,212,170,0.15);  color: #00D4AA; }
.tag-orange { background: rgba(245,166,35,0.15); color: #F5A623; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CHARGEMENT DES DONNÉES (mis en cache)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Chargement unique des données au démarrage de l'application.
    Le décorateur @st.cache_data évite de recharger à chaque interaction.
    """
    raw   = load_raw_data()
    clean = clean_and_filter(raw)
    panel = build_panel(clean)
    wide  = build_wide(panel)
    countries = sorted(panel["country"].unique().tolist())
    stats_df  = compute_continental_stats(panel)
    return panel, wide, countries, stats_df


panel, wide, COUNTRIES, stats_df = load_data()


# ---------------------------------------------------------------------------
# BARRE LATÉRALE — NAVIGATION
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌍 AfricaData")
    st.markdown("*Analyse démographique interactive*")
    st.divider()

    page = st.radio(
        "Navigation",
        options=[
            "📊 Dashboard",
            "🔍 Explorateur de données",
            "🌍 Carte choroplèthe",
            "📈 Simulation & Corrélations",
            "🎮 Jeu interactif",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Source des données**")
    st.markdown(
        "🏦 [World Bank WDI](https://databank.worldbank.org/)\n\n"
        "📅 Période : 2016–2023\n\n"
        "🌍 54 pays africains\n\n"
        "📊 16 indicateurs"
    )
    st.divider()
    st.caption("Projet académique · Démographie Africaine")


# ===========================================================================
# PAGE 1 — DASHBOARD
# ===========================================================================
if page == "📊 Dashboard":
    st.markdown('<div class="section-header">Tableau de bord continental</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Vue d\'ensemble des dynamiques démographiques et économiques africaines (2016–2023)</div>', unsafe_allow_html=True)

    st.markdown(
        '<span class="tag tag-blue">54 Pays</span> '
        '<span class="tag tag-green">16 Indicateurs</span> '
        '<span class="tag tag-orange">2016–2023</span>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # --- KPIs CONTINENTAUX ---
    st.subheader("📌 Indicateurs clés continentaux")

    pop_total = wide[wide["year"] == 2023]["pop"].sum() if "pop" in wide.columns else 0
    pop_2016  = wide[wide["year"] == 2016]["pop"].sum() if "pop" in wide.columns else 0
    pop_growth_pct = (pop_total - pop_2016) / pop_2016 * 100 if pop_2016 > 0 else 0

    avg_life  = wide[wide["year"] == 2023]["life_exp"].mean() if "life_exp" in wide.columns else 0
    avg_fert  = wide[wide["year"] == 2023]["fertility"].mean() if "fertility" in wide.columns else 0
    avg_gdpg  = wide[wide["year"] == 2023]["gdp_growth"].mean() if "gdp_growth" in wide.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{pop_total/1e9:.2f}B</div>
            <div class="kpi-label">Population totale 2023</div>
            <div class="kpi-delta-pos">▲ +{pop_growth_pct:.1f}% depuis 2016</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_life:.1f} ans</div>
            <div class="kpi-label">Espérance de vie moy.</div>
            <div style="color:#8b9bb4;font-size:0.75rem;">Moyenne africaine 2023</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_fert:.2f}</div>
            <div class="kpi-label">Fécondité moyenne</div>
            <div style="color:#8b9bb4;font-size:0.75rem;">Naissances par femme</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        delta_class = "kpi-delta-pos" if avg_gdpg > 0 else "kpi-delta-neg"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_gdpg:.1f}%</div>
            <div class="kpi-label">Croissance PIB moy.</div>
            <div class="{delta_class}">{"✓ Dynamique" if avg_gdpg > 3 else "⚠ Modérée"}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- GRAPHIQUES DE TENDANCE CONTINENTALE ---
    col1, col2 = st.columns(2)
    with col1:
        fig_pop = plot_continental_trend(stats_df, "pop")
        st.plotly_chart(fig_pop, use_container_width=True)
    with col2:
        fig_gdp = plot_continental_trend(stats_df, "gdp_growth")
        st.plotly_chart(fig_gdp, use_container_width=True)

    col3, col4, col5 = st.columns(3)
    with col3:
        fig_birth = plot_continental_trend(stats_df, "birth_rate")
        st.plotly_chart(fig_birth, use_container_width=True)
    with col4:
        fig_death = plot_continental_trend(stats_df, "death_rate")
        st.plotly_chart(fig_death, use_container_width=True)
    with col5:
        fig_life = plot_continental_trend(stats_df, "life_exp")
        st.plotly_chart(fig_life, use_container_width=True)

    # --- CLASSEMENTS ---
    st.subheader("🏆 Classements continentaux (2023)")
    r1, r2 = st.columns(2)
    with r1:
        fig_pop_rank = plot_bar_ranking(wide, "pop", 2023, top_n=15)
        st.plotly_chart(fig_pop_rank, use_container_width=True)
    with r2:
        fig_gdp_rank = plot_bar_ranking(wide, "gdp_pc", 2023, top_n=15)
        st.plotly_chart(fig_gdp_rank, use_container_width=True)

    # --- INDICE COMPOSITE ---
    st.subheader("🧮 Indice Composite Économie-Démographie (ICED)")
    st.caption("Score composite normalisé (0–100) combinant : PIB/hab, espérance de vie, scolarisation, électricité, stabilité des prix et emploi.")
    iced_df = compute_economic_demographic_index(wide, year=2022)
    if not iced_df.empty:
        st.dataframe(
            iced_df.rename(columns={"country": "Pays", "iso3": "Code", "ICED": "Score ICED"}),
            use_container_width=True,
            height=300,
        )


# ===========================================================================
# PAGE 2 — EXPLORATEUR
# ===========================================================================
elif page == "🔍 Explorateur de données":
    st.markdown('<div class="section-header">Explorateur de données</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Comparez les pays sur n\'importe quel indicateur</div>', unsafe_allow_html=True)

    # --- CONTRÔLES ---
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        country_a = st.selectbox("Pays A", COUNTRIES, index=COUNTRIES.index("Nigeria"))
    with c2:
        country_b = st.selectbox("Pays B", COUNTRIES, index=COUNTRIES.index("Ethiopia"))
    with c3:
        ind_options = list(INDICATOR_FR.keys())
        ind_labels  = list(INDICATOR_FR.values())
        ind_sel = st.selectbox("Indicateur", ind_labels, index=0)
        indicator = ind_options[ind_labels.index(ind_sel)]

    st.markdown("---")

    # --- GRAPHIQUE COMPARATIF ---
    col1, col2 = st.columns([3, 2])
    with col1:
        fig_cmp = plot_time_series(panel, [country_a, country_b], indicator)
        st.plotly_chart(fig_cmp, use_container_width=True)

    with col2:
        st.subheader("Tableau comparatif")
        rows = []
        for ind_key, ind_lbl in INDICATOR_FR.items():
            sub = panel[(panel["indicator"] == ind_key) & (panel["year"] == 2023)]
            vA = sub[sub["country"] == country_a]["value"].values
            vB = sub[sub["country"] == country_b]["value"].values
            rows.append({
                "Indicateur": ind_lbl,
                country_a[:10]: f"{vA[0]:.2f}" if len(vA) else "—",
                country_b[:10]: f"{vB[0]:.2f}" if len(vB) else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=380)

    # --- GRAPHIQUE RADAR ---
    st.subheader("🕸️ Profil multi-indicateurs (normalisé)")
    extra_countries = st.multiselect(
        "Ajouter des pays pour comparaison radar",
        [c for c in COUNTRIES if c not in [country_a, country_b]],
        default=[],
        max_selections=3,
    )
    all_sel = list({country_a, country_b} | set(extra_countries))
    fig_radar = plot_comparison_radar(wide, all_sel)
    st.plotly_chart(fig_radar, use_container_width=True)

    # --- CLASSEMENT CONTINENTAL ---
    st.subheader(f"🌐 Classement continental — {INDICATOR_FR.get(indicator, indicator)} (2023)")
    year_rank = st.slider("Année de référence", 2016, 2023, 2023)
    fig_rank = plot_bar_ranking(wide, indicator, year_rank, top_n=30,
                                 highlight=[country_a, country_b])
    st.plotly_chart(fig_rank, use_container_width=True)

    # --- PROFIL SPARKLINES ---
    st.subheader(f"📈 Évolution détaillée — {country_a}")
    fig_spark = plot_kpi_sparklines(panel, country_a)
    st.plotly_chart(fig_spark, use_container_width=True)


# ===========================================================================
# PAGE 3 — CARTE CHOROPLÈTHE
# ===========================================================================
elif page == "🌍 Carte choroplèthe":
    st.markdown('<div class="section-header">Carte géographique interactive</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Visualisation spatiale des indicateurs africains</div>', unsafe_allow_html=True)

    mc1, mc2 = st.columns([3, 1])
    with mc1:
        ind_sel_map = st.selectbox(
            "Indicateur cartographié",
            options=list(INDICATOR_FR.keys()),
            format_func=lambda x: INDICATOR_FR[x],
        )
    with mc2:
        year_map = st.selectbox("Année", list(range(2016, 2024)), index=7)

    fig_map = plot_choropleth_map(wide, ind_sel_map, year_map)
    st.plotly_chart(fig_map, use_container_width=True)

    # --- TOP & BOTTOM 5 ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🥇 Top 5")
        top5 = rank_countries(wide, ind_sel_map, year_map, ascending=False).head(5)
        if not top5.empty:
            for _, row in top5.iterrows():
                st.markdown(f"**{int(row['rank'])}. {row['country']}** — `{row[ind_sel_map]:.2f}`")

    with col2:
        st.subheader("🔻 Bottom 5")
        bot5 = rank_countries(wide, ind_sel_map, year_map, ascending=True).head(5)
        if not bot5.empty:
            for _, row in bot5.iterrows():
                st.markdown(f"**{int(row['rank'])}. {row['country']}** — `{row[ind_sel_map]:.2f}`")

    # --- CLASSIFICATION DÉMOGRAPHIQUE ---
    st.markdown("---")
    st.subheader("🔬 Classification par stade de transition démographique")
    st.caption(
        "Basée sur la théorie de la transition démographique (Notestein, 1945) : "
        "4 phases selon les niveaux de natalité, mortalité et fécondité."
    )

    demo_class = classify_development_stage(wide, year=2022)
    if not demo_class.empty:
        colors = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 0: "⚪"}
        for stage_num in sorted(demo_class["demo_stage"].unique()):
            sub = demo_class[demo_class["demo_stage"] == stage_num]
            if sub.empty or stage_num == 0:
                continue
            label = sub["stage_label"].iloc[0]
            countries_list = ", ".join(sub["country"].tolist())
            st.markdown(f"{colors.get(stage_num,'⚪')} **{label}** ({len(sub)} pays)  \n{countries_list}")


# ===========================================================================
# PAGE 4 — SIMULATION & CORRÉLATIONS
# ===========================================================================
elif page == "📈 Simulation & Corrélations":
    st.markdown('<div class="section-header">Modèle de simulation & Corrélations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Analyse statistique des relations économie–démographie et projections</div>', unsafe_allow_html=True)

    tabs = st.tabs(["🔗 Corrélations bivariées", "🌡️ Matrice de corrélation", "🔮 Projection démographique"])

    # --- TAB 1 : CORRÉLATIONS ---
    with tabs[0]:
        st.markdown("#### Analyse des corrélations économie ↔ démographie")
        st.caption(
            "Le coefficient de Pearson (r) mesure la force de la relation linéaire entre deux variables. "
            "La régression OLS (moindres carrés ordinaires) estime la droite de tendance."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Corrélation 1 : PIB/hab ↔ Fécondité**")
            res1 = compute_bivariate_correlation(panel, "gdp_pc", "fertility", year=2022)
            fig_s1 = plot_scatter_correlation(res1, "gdp_pc", "fertility")
            st.plotly_chart(fig_s1, use_container_width=True)
            if res1:
                st.info(
                    f"**r = {res1['pearson_r']}** · R² = {res1['r_squared']} · n = {res1['n_obs']} pays  \n"
                    f"{'Corrélation forte négative' if abs(res1['pearson_r'])>0.6 else 'Corrélation modérée'} "
                    f"→ Plus le PIB/hab est élevé, plus la fécondité tend à être faible."
                )

        with col2:
            st.markdown("**Corrélation 2 : Éducation ↔ Natalité**")
            res2 = compute_bivariate_correlation(panel, "edu_exp", "birth_rate", year=2020)
            fig_s2 = plot_scatter_correlation(res2, "edu_exp", "birth_rate")
            st.plotly_chart(fig_s2, use_container_width=True)
            if res2:
                st.info(
                    f"**r = {res2['pearson_r']}** · R² = {res2['r_squared']} · n = {res2['n_obs']} pays  \n"
                    f"L'investissement en éducation (% PIB) est associé à une natalité plus basse."
                )

        # Corrélation personnalisée
        st.markdown("#### Analyse personnalisée")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            x_ind = st.selectbox("Variable X", list(INDICATOR_FR.keys()), format_func=lambda x: INDICATOR_FR[x], key="cx")
        with sc2:
            y_ind = st.selectbox("Variable Y", list(INDICATOR_FR.keys()), format_func=lambda x: INDICATOR_FR[x], index=4, key="cy")
        with sc3:
            year_corr = st.selectbox("Année", list(range(2016, 2024)), index=6)

        res_custom = compute_bivariate_correlation(panel, x_ind, y_ind, year=year_corr)
        fig_custom = plot_scatter_correlation(res_custom, x_ind, y_ind)
        st.plotly_chart(fig_custom, use_container_width=True)

    # --- TAB 2 : MATRICE ---
    with tabs[1]:
        st.markdown("#### Matrice de corrélation inter-indicateurs")
        st.caption(
            "La diagonale vaut toujours 1 (auto-corrélation). "
            "Orange = corrélation positive, Bleu = corrélation négative."
        )
        year_matrix = st.selectbox("Année de référence", list(range(2016, 2024)), index=6)
        corr_mat = compute_correlation_matrix(wide, year=year_matrix)
        if not corr_mat.empty:
            fig_heat = plot_correlation_heatmap(corr_mat)
            st.plotly_chart(fig_heat, use_container_width=True)

            # Table des corrélations les plus fortes
            st.subheader("Top 10 corrélations les plus fortes")
            pairs = []
            cols = corr_mat.columns.tolist()
            for i in range(len(cols)):
                for j in range(i+1, len(cols)):
                    r = corr_mat.iloc[i, j]
                    pairs.append({
                        "Variable X": INDICATOR_FR.get(cols[i], cols[i]),
                        "Variable Y": INDICATOR_FR.get(cols[j], cols[j]),
                        "r de Pearson": round(r, 3),
                        "|r|": round(abs(r), 3),
                    })
            pairs_df = pd.DataFrame(pairs).sort_values("|r|", ascending=False).head(10)
            st.dataframe(pairs_df.drop(columns=["|r|"]), use_container_width=True)

    # --- TAB 3 : PROJECTION ---
    with tabs[2]:
        st.markdown("#### Projection démographique 2024–2040")
        st.caption(
            "Modèle de projection exponentielle calibré sur les taux de croissance historiques. "
            "Trois scénarios selon les hypothèses de croissance économique."
        )

        p1, p2 = st.columns(2)
        with p1:
            proj_country = st.selectbox("Pays à projeter", COUNTRIES, index=COUNTRIES.index("Nigeria"))
        with p2:
            proj_scenarios = st.multiselect(
                "Scénarios",
                ["baseline", "optimiste", "pessimiste"],
                default=["baseline", "optimiste"],
            )

        for scen in proj_scenarios:
            proj_data = project_population(panel, proj_country, horizon=2040, scenario=scen)
            if not proj_data.empty:
                fig_proj = plot_population_projection(proj_data, proj_country, scen)
                st.plotly_chart(fig_proj, use_container_width=True)

                # Statistiques de projection
                hist_pop = proj_data[proj_data["type"] == "historique"]["population"].iloc[-1]
                proj_pop = proj_data[proj_data["type"] == "projection"]["population"].iloc[-1]
                pct = (proj_pop - hist_pop) / hist_pop * 100
                st.markdown(
                    f"**Scénario {scen}** : Population 2023 → {hist_pop/1e6:.1f}M "
                    f"| Projection 2040 → {proj_pop/1e6:.1f}M "
                    f"(**+{pct:.1f}%**)"
                )


# ===========================================================================
# PAGE 5 — JEU INTERACTIF
# ===========================================================================
elif page == "🎮 Jeu interactif":
    st.markdown('<div class="section-header">🎮 Demography Manager: Africa Edition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Incarnez le ministre de l\'économie d\'un pays africain et gérez son développement de 2016 à 2035</div>',
        unsafe_allow_html=True
    )

    # --- INITIALISATION DU STATE STREAMLIT ---
    if "game_state" not in st.session_state:
        st.session_state.game_state = None
        st.session_state.game_events = []
        st.session_state.game_started = False
        st.session_state.game_over = False

    # ───────────────────────────────────────────
    # ÉCRAN DE CONFIGURATION
    # ───────────────────────────────────────────
    if not st.session_state.game_started:
        st.markdown("---")
        col_setup, col_rules = st.columns([1, 1])

        with col_setup:
            st.subheader("⚙️ Configuration")
            game_country = st.selectbox("Choisissez votre pays", COUNTRIES, index=COUNTRIES.index("Kenya"))
            game_diff    = st.selectbox(
                "Niveau de difficulté",
                options=["easy", "medium", "hard"],
                format_func=lambda x: {"easy": "🟢 Facile — Budget généreux", "medium": "🟡 Normal — Équilibré", "hard": "🔴 Difficile — Budget serré"}[x],
                index=1,
            )
            if st.button("🚀 Démarrer la simulation", use_container_width=True, type="primary"):
                st.session_state.game_state   = initialize_game(panel, game_country, game_diff)
                st.session_state.game_events  = []
                st.session_state.game_started = True
                st.session_state.game_over    = False
                st.rerun()

        with col_rules:
            st.subheader("📋 Règles du jeu")
            st.markdown("""
**Rôle** : Ministre de l'économie et de la planification nationale.

**Objectif** : Trouver un équilibre durable entre :
- 📈 Croissance économique
- 👶 Stabilité démographique
- 🧠 Bien-être social

**Mécanisme** :
1. Chaque tour = 1 an de simulation
2. Allouez votre budget (total = 100%) entre 5 secteurs
3. Les effets se cumulent sur 20 ans (2016–2035)

**Secteurs** :
| Secteur | Effet principal |
|---|---|
| 🎓 Éducation | ↓ Fécondité, ↑ PIB long terme |
| 🏥 Santé | ↓ Mortalité, ↑ Espérance de vie |
| ⚡ Infrastructure | ↑ Électricité, ↑ Productivité |
| 📈 Économie | ↑ PIB/hab, ↓ Chômage |
| 🤝 Social | Modère l'inflation, cohésion sociale |

**Score** : Calculé en fonction du niveau de développement global.
            """)

    # ───────────────────────────────────────────
    # ÉCRAN DE JEU
    # ───────────────────────────────────────────
    elif st.session_state.game_started and not st.session_state.game_over:
        gs = st.session_state.game_state

        if gs.year >= 2035:
            st.session_state.game_over = True
            st.rerun()

        # --- BARRE D'EN-TÊTE ---
        h1, h2, h3, h4, h5 = st.columns(5)
        with h1:
            st.metric("🌍 Pays",   gs.country)
        with h2:
            st.metric("📅 Année",  str(gs.year))
        with h3:
            st.metric("👥 Population", f"{gs.population/1e6:.1f}M")
        with h4:
            st.metric("💰 PIB/hab.", f"${gs.gdp_pc:,.0f}")
        with h5:
            st.metric("🏆 Score", f"{gs.score:,}")

        st.markdown("---")

        # --- PANNEAUX GAUCHE / DROITE ---
        left_col, right_col = st.columns([1, 2])

        # --- PANNEAU GAUCHE : MÉTRIQUES + SLIDERS ---
        with left_col:
            st.subheader("📊 État du pays")
            metrics_data = [
                ("Espérance de vie",      f"{gs.life_exp:.1f} ans",     gs.life_exp,       55, 85,  "#00D4AA"),
                ("Fécondité",             f"{gs.fertility:.2f} n/F",     7-gs.fertility,    0,  6,   "#E84393"),
                ("Taux de natalité",      f"{gs.birth_rate:.1f} ‰",     50-gs.birth_rate,  0,  50,  "#4A9EFF"),
                ("Accès électricité",     f"{gs.electricity:.0f}%",      gs.electricity,    0,  100, "#F5A623"),
                ("Inflation",             f"{gs.inflation:.1f}%",        30-gs.inflation,   0,  30,  "#FF4D6A"),
                ("Chômage",               f"{gs.unemployment:.1f}%",     30-gs.unemployment,0,  30,  "#8B9BB4"),
            ]
            for name, val_str, progress_val, mn, mx, color in metrics_data:
                pct = max(0, min(100, (progress_val - mn) / (mx - mn) * 100)) if mx > mn else 50
                st.markdown(f"**{name}** : {val_str}")
                st.progress(int(pct))

            st.markdown("---")
            st.subheader("💰 Allocation budgétaire")
            st.caption(f"Budget disponible : ${gs.budget/1e9:.2f}B | Total = 100%")

            # Sliders d'allocation
            edu_alloc   = st.slider("🎓 Éducation (%)",      0, 60, 25, key="sl_edu")
            health_alloc= st.slider("🏥 Santé (%)",          0, 60, 25, key="sl_health")
            infra_alloc = st.slider("⚡ Infrastructure (%)", 0, 60, 20, key="sl_infra")
            econ_alloc  = st.slider("📈 Économie (%)",       0, 60, 20, key="sl_econ")
            social_alloc= st.slider("🤝 Social (%)",         0, 60, 10, key="sl_social")

            total_alloc = edu_alloc + health_alloc + infra_alloc + econ_alloc + social_alloc
            if total_alloc != 100:
                st.warning(f"⚠️ Total : {total_alloc}% (doit être égal à 100%)")
            else:
                st.success("✅ Budget alloué à 100%")

            policy = PolicyAllocation(
                education=edu_alloc,
                health=health_alloc,
                infrastructure=infra_alloc,
                economy=econ_alloc,
                social=social_alloc,
            )

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                next_btn = st.button("▶ Tour suivant", use_container_width=True,
                                     type="primary", disabled=(total_alloc != 100))
            with btn_col2:
                if st.button("↺ Recommencer", use_container_width=True):
                    for key in ["game_state", "game_events", "game_started", "game_over"]:
                        del st.session_state[key]
                    st.rerun()

            # Exécution du tour
            if next_btn and total_alloc == 100:
                new_state, turn_result = simulate_turn(gs, policy)
                st.session_state.game_state  = new_state
                st.session_state.game_events = turn_result.get("events", [])
                st.rerun()

        # --- PANNEAU DROIT : GRAPHIQUES + ÉVÉNEMENTS ---
        with right_col:
            # Événements du tour
            if st.session_state.game_events:
                st.subheader("📰 Événements du tour")
                for evt in st.session_state.game_events:
                    css_class = f"event-{evt.get('type', 'warning')}"
                    st.markdown(
                        f'<div class="{css_class}">'
                        f'<strong>{evt["icon"]} {evt["title"]}</strong><br>'
                        f'{evt["message"]}</div>',
                        unsafe_allow_html=True
                    )

            # Graphiques d'historique
            hist_df = history_to_dataframe(gs)

            g1, g2 = st.columns(2)
            with g1:
                fig_pop = plot_time_series(
                    hist_df.rename(columns={"population": "value"}).assign(
                        indicator="pop", country=gs.country),
                    [gs.country], "pop",
                    title="📈 Évolution de la population (M)"
                )
                # Remplacement rapide pour les données du jeu
                import plotly.graph_objects as go
                fig_game_pop = go.Figure(go.Scatter(
                    x=hist_df["year"], y=hist_df["population"]/1e6,
                    fill="tozeroy", fillcolor="rgba(74,158,255,0.1)",
                    line=dict(color="#4A9EFF", width=2.5),
                    mode="lines+markers", name="Population (M)"
                ))
                fig_game_pop.update_layout(
                    height=250, title="Population (millions)",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.6)",
                    font=dict(color="#E8EEF8", size=11), margin=dict(t=30, b=30, l=40, r=10),
                )
                st.plotly_chart(fig_game_pop, use_container_width=True)

            with g2:
                import plotly.graph_objects as go
                fig_game_gdp = go.Figure(go.Scatter(
                    x=hist_df["year"], y=hist_df["gdp_pc"],
                    fill="tozeroy", fillcolor="rgba(245,166,35,0.1)",
                    line=dict(color="#F5A623", width=2.5),
                    mode="lines+markers", name="PIB/hab (USD)"
                ))
                fig_game_gdp.update_layout(
                    height=250, title="PIB par habitant (USD)",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.6)",
                    font=dict(color="#E8EEF8", size=11), margin=dict(t=30, b=30, l=40, r=10),
                )
                st.plotly_chart(fig_game_gdp, use_container_width=True)

            # Graphique multi-indicateurs
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            fig_multi = make_subplots(rows=2, cols=2,
                subplot_titles=["Natalité (‰)", "Espérance de vie (ans)", "Inflation (%)", "Score cumulé"])
            series = [
                (hist_df["birth_rate"], "#E84393", 1, 1),
                (hist_df["life_exp"],   "#00D4AA", 1, 2),
                (hist_df["inflation"],  "#FF4D6A", 2, 1),
                (hist_df["score"],      "#F5A623", 2, 2),
            ]
            for vals, color, row, col in series:
                fig_multi.add_trace(
                    go.Scatter(x=hist_df["year"], y=vals, mode="lines+markers",
                               line=dict(color=color, width=2), marker=dict(size=4),
                               showlegend=False),
                    row=row, col=col
                )
            fig_multi.update_layout(
                height=300, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.6)",
                font=dict(color="#8B9BB4", size=10),
                margin=dict(t=40, b=20, l=30, r=10),
            )
            st.plotly_chart(fig_multi, use_container_width=True)

    # ───────────────────────────────────────────
    # ÉCRAN DE FIN DE JEU
    # ───────────────────────────────────────────
    elif st.session_state.game_over:
        gs = st.session_state.game_state
        result = evaluate_final_score(gs)

        st.markdown("---")
        st.markdown(f"## 🏆 Simulation terminée — {gs.country}")

        # Grade
        grade_colors = {"S":"#F5A623","A":"#00D4AA","B":"#4A9EFF","C":"#8B9BB4","D":"#FF4D6A","F":"#FF1744"}
        color = grade_colors.get(result["grade"], "#8B9BB4")
        st.markdown(
            f'<h1 style="color:{color};text-align:center;font-size:5rem;">{result["grade"]}</h1>'
            f'<p style="text-align:center;color:#8b9bb4;">{result["mention"]}</p>',
            unsafe_allow_html=True
        )
        st.markdown(f"> {result['message']}")

        # KPIs finaux
        f1, f2, f3, f4 = st.columns(4)
        with f1: st.metric("🏆 Score final",      f"{result['total_score']:,}")
        with f2: st.metric("👥 Population finale", f"{result['final_pop']/1e6:.1f}M")
        with f3: st.metric("💰 PIB/hab. final",    f"${result['final_gdp_pc']:,.0f}")
        with f4: st.metric("❤️ Espérance de vie",  f"{result['final_life_exp']:.1f} ans")

        # Scores par dimension
        st.subheader("📊 Scores détaillés par dimension")
        dim_df = pd.DataFrame(
            [(k, f"{v:.1f}/100") for k, v in result["dim_scores"].items()],
            columns=["Dimension", "Score"]
        )
        st.dataframe(dim_df, use_container_width=True)

        # Graphiques finaux
        hist_df = history_to_dataframe(gs)
        import plotly.graph_objects as go
        fig_final = go.Figure()
        fig_final.add_trace(go.Scatter(x=hist_df["year"], y=hist_df["score"],
            fill="tozeroy", fillcolor="rgba(245,166,35,0.15)",
            line=dict(color="#F5A623", width=3), name="Score cumulé"))
        fig_final.update_layout(
            title="Évolution du score au cours de la simulation",
            height=300, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,24,39,0.6)", font=dict(color="#E8EEF8"),
        )
        st.plotly_chart(fig_final, use_container_width=True)

        if st.button("🔄 Nouvelle partie", type="primary"):
            for key in ["game_state", "game_events", "game_started", "game_over"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
