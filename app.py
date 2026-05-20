import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import skfuzzy as fuzz
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales – Clustering Dashboard",
    page_icon="🛒",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0071CE 0%, #FFC220 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #555;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-left: 4px solid #0071CE;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0071CE;
        border-bottom: 2px solid #FFC220;
        padding-bottom: 4px;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }
    .algo-box {
        background: #f0f8ff;
        border: 1px solid #b8d4f0;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🛒 Walmart Sales — Clustering Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Análise de agrupamento com DBSCAN e Fuzzy K-Médias • Fatec / Prof. Davi Albuquerque</div>', unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Walmart_Sales.csv")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df

df = load_data()

# ── Feature engineering: aggregate per store ──────────────────────────────────
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

    st.markdown("### Features para Clustering")
    all_features = ["Avg_Weekly_Sales", "Std_Weekly_Sales", "Avg_Temperature",
                    "Avg_Fuel_Price", "Avg_CPI", "Avg_Unemployment", "Holiday_Rate"]
    selected_features = st.multiselect(
        "Selecione as variáveis",
        all_features,
        default=["Avg_Weekly_Sales", "Avg_Unemployment", "Avg_CPI", "Avg_Fuel_Price"]
    )

    st.divider()
    st.markdown("### 🔵 DBSCAN")
    eps = st.slider("Epsilon (ε) — raio de vizinhança", 0.1, 3.0, 0.8, 0.05)
    min_samples = st.slider("MinPts — objetos mínimos no raio", 2, 10, 3, 1)

    st.divider()
    st.markdown("### 🟡 Fuzzy K-Médias")
    n_clusters_fuzzy = st.slider("Número de grupos (K)", 2, 8, 3, 1)
    m_fuzz = st.slider("Coeficiente de fuzzificação (m)", 1.1, 3.0, 2.0, 0.1)
    fuzzy_threshold = st.slider("Limiar de pertinência para rótulo", 0.3, 0.9, 0.5, 0.05)

    st.divider()
    run_btn = st.button("▶ Executar Clustering", use_container_width=True, type="primary")

if not selected_features:
    st.warning("Selecione pelo menos 2 features no painel lateral.")
    st.stop()

# ── Prepare data matrix ───────────────────────────────────────────────────────
X_raw = store_df[selected_features].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ── PCA for 2-D vis ───────────────────────────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
store_df["PC1"] = X_pca[:, 0]
store_df["PC2"] = X_pca[:, 1]

# ── DBSCAN ────────────────────────────────────────────────────────────────────
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
db_labels = dbscan.fit_predict(X_scaled)
store_df["DBSCAN_Cluster"] = db_labels.astype(str)

n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = list(db_labels).count(-1)

try:
    if len(set(db_labels)) > 1 and n_noise < len(db_labels):
        mask = db_labels != -1
        db_sil = silhouette_score(X_scaled[mask], db_labels[mask]) if mask.sum() > 1 and len(set(db_labels[mask])) > 1 else None
    else:
        db_sil = None
except:
    db_sil = None

# ── Fuzzy K-Médias ────────────────────────────────────────────────────────────
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_scaled.T, n_clusters_fuzzy, m_fuzz, error=1e-5, maxiter=1000
)
fuzzy_labels = np.argmax(u, axis=0)
store_df["Fuzzy_Cluster"] = fuzzy_labels.astype(str)

# Membership matrix
membership_df = pd.DataFrame(u.T, columns=[f"Grupo {i+1}" for i in range(n_clusters_fuzzy)])
membership_df.insert(0, "Store", store_df["Store"].values)

# Dominant membership
store_df["Max_Membership"] = u.max(axis=0)

try:
    fuzzy_sil = silhouette_score(X_scaled, fuzzy_labels) if len(set(fuzzy_labels)) > 1 else None
except:
    fuzzy_sil = None

# ── KPIs Row ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Visão Geral</div>', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🏪 Lojas analisadas", 45)
col2.metric("📅 Semanas de dados", df["Date"].nunique())
col3.metric("💰 Vendas totais (M)", f"${df['Weekly_Sales'].sum()/1e9:.2f}B")
col4.metric("🔵 Grupos DBSCAN", n_db_clusters, f"{n_noise} ruídos")
col5.metric("🟡 Grupos Fuzzy", n_clusters_fuzzy, f"Sil: {fuzzy_sil:.3f}" if fuzzy_sil else "—")

# ── Algorithm explanations ────────────────────────────────────────────────────
with st.expander("📚 Sobre os Algoritmos", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🔵 DBSCAN** — Density-Based Spatial Clustering")
        st.markdown("""<div class="algo-box">
O número de grupos é definido <b>automaticamente</b> pelo algoritmo, sendo que cada grupo 
possui pelo menos um <em>objeto de núcleo</em>. Um objeto de núcleo é definido como um 
objeto com uma quantidade mínima (<code>minPts</code>) de objetos em seu raio de 
vizinhança <code>ε</code>. O grupo é formado agregando os objetos da vizinhança do 
objeto de núcleo. Objetos que não são adicionados a nenhum grupo são definidos como 
<b>ruído</b>.
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("**🟡 Fuzzy K-Médias** — Soft Clustering")
        st.markdown("""<div class="algo-box">
Extensão do K-Médias onde cada objeto possui um <b>grau de pertinência</b> em relação 
a <em>todos</em> os grupos. Um objeto pode pertencer a mais de um grupo, com variados 
graus de pertinência. Para cada objeto <code>x</code> há um valor <code>u<sub>k</sub>(x)</code> 
correspondente ao grau de pertinência ao grupo <code>k</code>. Por convenção, 
a <b>soma dos graus de pertinência</b> de um objeto a todos os grupos deve ser <b>1</b>.
</div>""", unsafe_allow_html=True)

# ── DBSCAN Section ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔵 DBSCAN</div>', unsafe_allow_html=True)
col_db1, col_db2 = st.columns([2, 1])

with col_db1:
    color_map = {str(i): px.colors.qualitative.Bold[i % 10] for i in range(-1, n_db_clusters+1)}
    color_map["-1"] = "#aaaaaa"

    fig_db = px.scatter(
        store_df, x="PC1", y="PC2",
        color="DBSCAN_Cluster",
        hover_name="Store",
        hover_data={c: ":.3f" for c in selected_features if c in store_df.columns},
        color_discrete_map=color_map,
        title=f"DBSCAN — ε={eps}, minPts={min_samples} → {n_db_clusters} grupo(s) + {n_noise} ruído(s)",
        labels={"DBSCAN_Cluster": "Grupo", "PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"},
        size_max=14,
    )
    fig_db.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    fig_db.update_layout(height=420, legend_title_text="Grupo DBSCAN", template="plotly_white")
    st.plotly_chart(fig_db, use_container_width=True)

with col_db2:
    st.markdown("**Composição dos grupos**")
    counts = store_df["DBSCAN_Cluster"].value_counts().reset_index()
    counts.columns = ["Grupo", "Lojas"]
    counts["Tipo"] = counts["Grupo"].apply(lambda x: "🔇 Ruído" if x == "-1" else "✅ Núcleo")
    st.dataframe(counts, use_container_width=True, hide_index=True)

    if db_sil is not None:
        st.metric("Silhouette Score (sem ruído)", f"{db_sil:.3f}")
        st.progress(float(np.clip((db_sil + 1) / 2, 0, 1)))

    st.markdown("**Lojas marcadas como ruído:**")
    noise_stores = store_df[store_df["DBSCAN_Cluster"] == "-1"]["Store"].tolist()
    st.write(noise_stores if noise_stores else "Nenhuma (ajuste ε ou minPts)")

# DBSCAN feature distributions
st.markdown("**Distribuição das features por grupo DBSCAN**")
feat_options = [f for f in selected_features if f in store_df.columns]
fig_box = px.box(
    store_df, x="DBSCAN_Cluster", y=feat_options[0] if feat_options else "Avg_Weekly_Sales",
    color="DBSCAN_Cluster", color_discrete_map=color_map,
    title="Distribuição por grupo",
    template="plotly_white"
)
feat_col, _ = st.columns([1, 2])
with feat_col:
    chosen_feat = st.selectbox("Feature para boxplot DBSCAN", feat_options)
fig_box2 = px.box(
    store_df, x="DBSCAN_Cluster", y=chosen_feat,
    color="DBSCAN_Cluster", color_discrete_map=color_map,
    template="plotly_white",
    labels={"DBSCAN_Cluster": "Grupo"}
)
fig_box2.update_layout(height=350, showlegend=False)
st.plotly_chart(fig_box2, use_container_width=True)

# ── Fuzzy Section ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🟡 Fuzzy K-Médias</div>', unsafe_allow_html=True)
col_fz1, col_fz2 = st.columns([2, 1])

fuzzy_colors = px.colors.qualitative.Vivid

with col_fz1:
    fig_fz = px.scatter(
        store_df, x="PC1", y="PC2",
        color="Fuzzy_Cluster",
        hover_name="Store",
        size="Max_Membership",
        size_max=20,
        title=f"Fuzzy K-Médias — K={n_clusters_fuzzy}, m={m_fuzz}",
        labels={"Fuzzy_Cluster": "Grupo Dominante", "Max_Membership": "Pertinência Máx.",
                "PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"},
        template="plotly_white"
    )
    fig_fz.update_traces(marker=dict(line=dict(width=1, color="white")))
    fig_fz.update_layout(height=420)
    st.plotly_chart(fig_fz, use_container_width=True)

with col_fz2:
    st.markdown("**Composição dos grupos (rótulo dominante)**")
    fz_counts = store_df["Fuzzy_Cluster"].value_counts().reset_index()
    fz_counts.columns = ["Grupo", "Lojas"]
    fz_counts["Grupo"] = fz_counts["Grupo"].apply(lambda x: f"Grupo {int(x)+1}")
    st.dataframe(fz_counts, use_container_width=True, hide_index=True)

    if fuzzy_sil is not None:
        st.metric("Silhouette Score", f"{fuzzy_sil:.3f}")
        st.progress(float(np.clip((fuzzy_sil + 1) / 2, 0, 1)))

    st.markdown("**Distribuição de pertinência máxima**")
    fig_hist = px.histogram(
        store_df, x="Max_Membership", nbins=15,
        color_discrete_sequence=["#FFC220"],
        template="plotly_white"
    )
    fig_hist.update_layout(height=200, margin=dict(t=10, b=30))
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Membership heatmap ────────────────────────────────────────────────────────
st.markdown("**Matriz de graus de pertinência por loja**")
heatmap_data = membership_df.set_index("Store")

# sort by store
heatmap_data = heatmap_data.sort_index()

fig_heat = px.imshow(
    heatmap_data.T,
    color_continuous_scale="YlOrRd",
    aspect="auto",
    title="Grau de Pertinência u_k(x) — cada célula soma 1 por loja",
    labels={"x": "Loja", "y": "Grupo", "color": "Pertinência"},
    template="plotly_white"
)
fig_heat.update_layout(height=280)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Side-by-side comparison ───────────────────────────────────────────────────
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

store_options = sorted(df["Store"].unique())
selected_stores = st.multiselect("Selecione lojas para visualizar:", store_options, default=[1, 2, 3])

if selected_stores:
    ts_df = df[df["Store"].isin(selected_stores)].copy()
    fig_ts = px.line(
        ts_df, x="Date", y="Weekly_Sales", color="Store",
        title="Vendas Semanais por Loja",
        labels={"Weekly_Sales": "Vendas Semanais ($)", "Date": "Data", "Store": "Loja"},
        template="plotly_white"
    )
    fig_ts.update_layout(height=380)
    st.plotly_chart(fig_ts, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Dashboard desenvolvido com Streamlit · DBSCAN (sklearn) · Fuzzy C-Means (skfuzzy) · Dados: Walmart Sales Dataset")