import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import skfuzzy as fuzz
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales – Clustering Dashboard",
    page_icon="🛒",
    layout="wide",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #0071CE 0%, #FFC220 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.2rem;
    }
    .subtitle { color: #555; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #0071CE;
        border-bottom: 2px solid #FFC220; padding-bottom: 4px;
        margin-top: 1rem; margin-bottom: 0.8rem;
    }
    .algo-box {
        background: #f0f8ff; border: 1px solid #b8d4f0; border-radius: 8px;
        padding: 12px 16px; font-size: 0.88rem; line-height: 1.6; color: #333;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛒 Walmart Sales — Clustering Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Análise de agrupamento com DBSCAN e Fuzzy K-Médias • Fatec / Prof. Davi Albuquerque</div>', unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Walmart_Sales.csv")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df

df = load_data()

@st.cache_data
def build_store_features(df):
    agg = df.groupby("Store").agg(
        Avg_Weekly_Sales=("Weekly_Sales", "mean"),
        Std_Weekly_Sales=("Weekly_Sales", "std"),
        Total_Sales=("Weekly_Sales", "sum"),
        Avg_Temperature=("Temperature", "mean"),
        Avg_Fuel_Price=("Fuel_Price", "mean"),
        Avg_CPI=("CPI", "mean"),
        Avg_Unemployment=("Unemployment", "mean"),
        Holiday_Rate=("Holiday_Flag", "mean"),
    ).reset_index()
    return agg

store_df = build_store_features(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Walmart_Spark.svg/120px-Walmart_Spark.svg.png", width=50)
    st.markdown("## ⚙️ Configurações")

    all_features = ["Avg_Weekly_Sales", "Std_Weekly_Sales", "Avg_Temperature",
                    "Avg_Fuel_Price", "Avg_CPI", "Avg_Unemployment", "Holiday_Rate"]
    selected_features = st.multiselect(
        "Variáveis para clustering",
        all_features,
        default=["Avg_Weekly_Sales", "Avg_Unemployment", "Avg_CPI", "Avg_Fuel_Price"]
    )

    st.divider()
    st.markdown("### 🔵 DBSCAN")
    eps = st.slider("Epsilon (ε)", 0.1, 3.0, 0.8, 0.05)
    min_samples = st.slider("MinPts", 2, 10, 3, 1)

    st.divider()
    st.markdown("### 🟡 Fuzzy K-Médias")
    n_clusters_fuzzy = st.slider("Número de grupos (K)", 2, 8, 3, 1)
    m_fuzz = st.slider("Coeficiente de fuzzificação (m)", 1.1, 3.0, 2.0, 0.1)

if not selected_features:
    st.warning("Selecione pelo menos 2 features.")
    st.stop()

# ── Preprocessing ─────────────────────────────────────────────────────────────
X_raw = store_df[selected_features].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# PCA 3D
pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_scaled)
store_df["PC1"] = X_pca3[:, 0]
store_df["PC2"] = X_pca3[:, 1]
store_df["PC3"] = X_pca3[:, 2]

var_exp = pca3.explained_variance_ratio_
pc_labels = [f"PC{i+1} ({var_exp[i]*100:.1f}%)" for i in range(3)]

# ── DBSCAN ────────────────────────────────────────────────────────────────────
db_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
store_df["DBSCAN_Cluster"] = db_labels.astype(str)
n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = list(db_labels).count(-1)

try:
    mask = db_labels != -1
    db_sil = silhouette_score(X_scaled[mask], db_labels[mask]) if mask.sum() > 1 and len(set(db_labels[mask])) > 1 else None
except:
    db_sil = None

color_map = {str(i): px.colors.qualitative.Bold[i % 10] for i in range(n_db_clusters + 1)}
color_map["-1"] = "#bbbbbb"

# ── Fuzzy K-Médias ────────────────────────────────────────────────────────────
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_scaled.T, n_clusters_fuzzy, m_fuzz, error=1e-5, maxiter=1000
)
fuzzy_labels = np.argmax(u, axis=0)
store_df["Fuzzy_Cluster"] = fuzzy_labels.astype(str)
store_df["Max_Membership"] = u.max(axis=0)

membership_df = pd.DataFrame(u.T, columns=[f"Grupo {i+1}" for i in range(n_clusters_fuzzy)])
membership_df.insert(0, "Store", store_df["Store"].values)

try:
    fuzzy_sil = silhouette_score(X_scaled, fuzzy_labels) if len(set(fuzzy_labels)) > 1 else None
except:
    fuzzy_sil = None

# ── KPIs ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Visão Geral</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🏪 Lojas", 45)
c2.metric("📅 Semanas", df["Date"].nunique())
c3.metric("💰 Total de Vendas", f"${df['Weekly_Sales'].sum()/1e9:.2f}B")
c4.metric("🔵 Grupos DBSCAN", n_db_clusters, f"{n_noise} ruídos")
c5.metric("🟡 Grupos Fuzzy", n_clusters_fuzzy, f"Sil: {fuzzy_sil:.3f}" if fuzzy_sil else "—")

# ── Explanations ──────────────────────────────────────────────────────────────
with st.expander("📚 Sobre os Algoritmos", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**🔵 DBSCAN**")
        st.markdown("""<div class="algo-box">O número de grupos é definido <b>automaticamente</b>. 
        Um <em>objeto de núcleo</em> tem pelo menos <code>minPts</code> vizinhos dentro do raio <code>ε</code>. 
        O grupo cresce agregando vizinhos recursivamente. Objetos sem grupo = <b>ruído</b>.</div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("**🟡 Fuzzy K-Médias**")
        st.markdown("""<div class="algo-box">Cada objeto tem um <b>grau de pertinência</b> a <em>todos</em> os grupos. 
        O valor <code>u_k(x)</code> indica o quanto o objeto pertence ao grupo <code>k</code>. 
        A soma dos graus de pertinência por objeto é sempre <b>1</b>.</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DBSCAN 3D
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔵 DBSCAN — Visualização 3D</div>', unsafe_allow_html=True)

col_db1, col_db2 = st.columns([3, 1])

with col_db1:
    fig_db3d = px.scatter_3d(
        store_df, x="PC1", y="PC2", z="PC3",
        color="DBSCAN_Cluster",
        hover_name="Store",
        color_discrete_map=color_map,
        title=f"DBSCAN 3D — ε={eps}, minPts={min_samples} → {n_db_clusters} grupo(s) + {n_noise} ruído(s)",
        labels={
            "DBSCAN_Cluster": "Grupo",
            "PC1": pc_labels[0], "PC2": pc_labels[1], "PC3": pc_labels[2]
        },
        symbol="DBSCAN_Cluster",
    )
    fig_db3d.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
    fig_db3d.update_layout(
        height=550,
        legend_title_text="Grupo DBSCAN",
        scene=dict(
            xaxis_title=pc_labels[0],
            yaxis_title=pc_labels[1],
            zaxis_title=pc_labels[2],
            bgcolor="rgba(240,248,255,1)",
            xaxis=dict(backgroundcolor="rgba(220,235,250,0.5)", gridcolor="white"),
            yaxis=dict(backgroundcolor="rgba(220,235,250,0.5)", gridcolor="white"),
            zaxis=dict(backgroundcolor="rgba(200,220,245,0.5)", gridcolor="white"),
        ),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_db3d, use_container_width=True)

with col_db2:
    st.markdown("**Composição dos grupos**")
    counts = store_df["DBSCAN_Cluster"].value_counts().reset_index()
    counts.columns = ["Grupo", "Lojas"]
    counts["Tipo"] = counts["Grupo"].apply(lambda x: "🔇 Ruído" if x == "-1" else "✅ Núcleo")
    st.dataframe(counts, use_container_width=True, hide_index=True)

    if db_sil is not None:
        st.metric("Silhouette Score", f"{db_sil:.3f}")
        st.progress(float(np.clip((db_sil + 1) / 2, 0, 1)))

    noise_stores = store_df[store_df["DBSCAN_Cluster"] == "-1"]["Store"].tolist()
    st.markdown("**Lojas ruído:**")
    st.write(noise_stores if noise_stores else "Nenhuma")

    # 2D Pie of cluster distribution
    fig_pie_db = px.pie(
        counts[counts["Grupo"] != "-1"], values="Lojas", names="Grupo",
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="Distribuição (sem ruído)",
        hole=0.4,
    )
    fig_pie_db.update_layout(height=250, margin=dict(t=35, b=0, l=0, r=0), showlegend=False)
    st.plotly_chart(fig_pie_db, use_container_width=True)

# DBSCAN 3D with actual features (not PCA) — axes selectable
st.markdown("**Explorar features reais em 3D (DBSCAN)**")
feat_options = [f for f in selected_features if f in store_df.columns]
if len(feat_options) >= 3:
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        ax_x = st.selectbox("Eixo X", feat_options, index=0, key="db_x")
    with col_y:
        ay_y = st.selectbox("Eixo Y", feat_options, index=1, key="db_y")
    with col_z:
        az_z = st.selectbox("Eixo Z", feat_options, index=2, key="db_z")

    fig_db_feat = px.scatter_3d(
        store_df, x=ax_x, y=ay_y, z=az_z,
        color="DBSCAN_Cluster",
        hover_name="Store",
        color_discrete_map=color_map,
        title=f"DBSCAN — {ax_x} × {ay_y} × {az_z}",
        labels={"DBSCAN_Cluster": "Grupo"},
    )
    fig_db_feat.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
    fig_db_feat.update_layout(
        height=500,
        scene=dict(
            bgcolor="rgba(240,248,255,1)",
            xaxis=dict(backgroundcolor="rgba(220,235,250,0.5)", gridcolor="white"),
            yaxis=dict(backgroundcolor="rgba(220,235,250,0.5)", gridcolor="white"),
            zaxis=dict(backgroundcolor="rgba(200,220,245,0.5)", gridcolor="white"),
        ),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_db_feat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# FUZZY 3D
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🟡 Fuzzy K-Médias — Visualização 3D</div>', unsafe_allow_html=True)

col_fz1, col_fz2 = st.columns([3, 1])

with col_fz1:
    fig_fz3d = px.scatter_3d(
        store_df, x="PC1", y="PC2", z="PC3",
        color="Fuzzy_Cluster",
        hover_name="Store",
        size="Max_Membership",
        size_max=18,
        title=f"Fuzzy K-Médias 3D — K={n_clusters_fuzzy}, m={m_fuzz}",
        labels={
            "Fuzzy_Cluster": "Grupo Dominante",
            "Max_Membership": "Pertinência Máx.",
            "PC1": pc_labels[0], "PC2": pc_labels[1], "PC3": pc_labels[2]
        },
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig_fz3d.update_traces(marker=dict(line=dict(width=0.5, color="white")))
    fig_fz3d.update_layout(
        height=550,
        scene=dict(
            bgcolor="rgba(255,252,235,1)",
            xaxis_title=pc_labels[0],
            yaxis_title=pc_labels[1],
            zaxis_title=pc_labels[2],
            xaxis=dict(backgroundcolor="rgba(255,245,200,0.5)", gridcolor="white"),
            yaxis=dict(backgroundcolor="rgba(255,245,200,0.5)", gridcolor="white"),
            zaxis=dict(backgroundcolor="rgba(255,235,180,0.5)", gridcolor="white"),
        ),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_fz3d, use_container_width=True)

with col_fz2:
    st.markdown("**Grupos (rótulo dominante)**")
    fz_counts = store_df["Fuzzy_Cluster"].value_counts().reset_index()
    fz_counts.columns = ["Grupo", "Lojas"]
    fz_counts["Grupo"] = fz_counts["Grupo"].apply(lambda x: f"Grupo {int(x)+1}")
    st.dataframe(fz_counts, use_container_width=True, hide_index=True)

    if fuzzy_sil is not None:
        st.metric("Silhouette Score", f"{fuzzy_sil:.3f}")
        st.progress(float(np.clip((fuzzy_sil + 1) / 2, 0, 1)))

    fig_hist = px.histogram(
        store_df, x="Max_Membership", nbins=15,
        color_discrete_sequence=["#FFC220"],
        title="Pertinência Máxima",
        template="plotly_white"
    )
    fig_hist.update_layout(height=220, margin=dict(t=35, b=30))
    st.plotly_chart(fig_hist, use_container_width=True)

# Fuzzy 3D with real features
st.markdown("**Explorar features reais em 3D (Fuzzy)**")
if len(feat_options) >= 3:
    col_x2, col_y2, col_z2 = st.columns(3)
    with col_x2:
        fax_x = st.selectbox("Eixo X", feat_options, index=0, key="fz_x")
    with col_y2:
        fay_y = st.selectbox("Eixo Y", feat_options, index=1, key="fz_y")
    with col_z2:
        faz_z = st.selectbox("Eixo Z", feat_options, index=2, key="fz_z")

    fig_fz_feat = px.scatter_3d(
        store_df, x=fax_x, y=fay_y, z=faz_z,
        color="Fuzzy_Cluster",
        hover_name="Store",
        size="Max_Membership",
        size_max=18,
        title=f"Fuzzy — {fax_x} × {fay_y} × {faz_z}",
        labels={"Fuzzy_Cluster": "Grupo", "Max_Membership": "Pertinência"},
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig_fz_feat.update_traces(marker=dict(line=dict(width=0.5, color="white")))
    fig_fz_feat.update_layout(
        height=500,
        scene=dict(
            bgcolor="rgba(255,252,235,1)",
            xaxis=dict(backgroundcolor="rgba(255,245,200,0.5)", gridcolor="white"),
            yaxis=dict(backgroundcolor="rgba(255,245,200,0.5)", gridcolor="white"),
            zaxis=dict(backgroundcolor="rgba(255,235,180,0.5)", gridcolor="white"),
        ),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_fz_feat, use_container_width=True)

# ── Membership heatmap ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔥 Matriz de Pertinência — Fuzzy</div>', unsafe_allow_html=True)
heatmap_data = membership_df.set_index("Store").sort_index()
fig_heat = px.imshow(
    heatmap_data.T,
    color_continuous_scale="YlOrRd",
    aspect="auto",
    title="Grau de Pertinência u_k(x) — soma por loja = 1",
    labels={"x": "Loja", "y": "Grupo", "color": "Pertinência"},
    template="plotly_white"
)
fig_heat.update_layout(height=300)
st.plotly_chart(fig_heat, use_container_width=True)

# ── 3D Membership surface for Fuzzy (store × group × membership) ──────────────
st.markdown("**Superfície 3D de Pertinência por Loja**")
stores_sorted = heatmap_data.index.tolist()
groups = heatmap_data.columns.tolist()
Z = heatmap_data.values.T  # shape: (n_groups, n_stores)

fig_surf = go.Figure(data=[
    go.Surface(
        z=Z,
        x=stores_sorted,
        y=list(range(1, len(groups) + 1)),
        colorscale="YlOrRd",
        showscale=True,
        colorbar=dict(title="Pertinência"),
        opacity=0.9,
    )
])
fig_surf.update_layout(
    title="Superfície de Graus de Pertinência — Fuzzy K-Médias",
    scene=dict(
        xaxis_title="Loja",
        yaxis_title="Grupo",
        zaxis_title="Grau de Pertinência",
        bgcolor="rgba(255,252,235,1)",
        xaxis=dict(backgroundcolor="rgba(255,245,200,0.5)", gridcolor="white"),
        yaxis=dict(backgroundcolor="rgba(255,245,200,0.5)", gridcolor="white"),
        zaxis=dict(backgroundcolor="rgba(255,235,180,0.5)", gridcolor="white"),
    ),
    height=520,
    paper_bgcolor="white",
)
st.plotly_chart(fig_surf, use_container_width=True)

# ── Comparison table ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚖️ Comparação DBSCAN vs Fuzzy K-Médias</div>', unsafe_allow_html=True)
comp_df = store_df[["Store", "DBSCAN_Cluster", "Fuzzy_Cluster", "Max_Membership",
                     "Avg_Weekly_Sales", "Avg_Unemployment", "Avg_CPI"]].copy()
comp_df["Fuzzy_Cluster"] = comp_df["Fuzzy_Cluster"].apply(lambda x: f"Grupo {int(x)+1}")
comp_df["DBSCAN_Cluster"] = comp_df["DBSCAN_Cluster"].apply(
    lambda x: "Ruído" if x == "-1" else f"Grupo {int(x)+1}"
)
comp_df["Max_Membership"] = comp_df["Max_Membership"].round(3)
comp_df["Avg_Weekly_Sales"] = comp_df["Avg_Weekly_Sales"].round(0).astype(int)
comp_df.columns = ["Loja", "DBSCAN", "Fuzzy", "Pertinência Máx.", "Vendas Médias ($)", "Desemprego Médio", "CPI Médio"]
st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ── Time series ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Série Temporal de Vendas por Loja</div>', unsafe_allow_html=True)
selected_stores = st.multiselect("Selecione lojas:", sorted(df["Store"].unique()), default=[1, 2, 3])
if selected_stores:
    ts_df = df[df["Store"].isin(selected_stores)].copy()
    fig_ts = px.line(
        ts_df, x="Date", y="Weekly_Sales", color="Store",
        title="Vendas Semanais por Loja",
        labels={"Weekly_Sales": "Vendas ($)", "Date": "Data", "Store": "Loja"},
        template="plotly_white"
    )
    fig_ts.update_layout(height=380)
    st.plotly_chart(fig_ts, use_container_width=True)

st.divider()
st.caption("Dashboard desenvolvido com Streamlit · DBSCAN (sklearn) · Fuzzy C-Means (skfuzzy) · Dados: Walmart Sales Dataset")
