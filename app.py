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

st.set_page_config(page_title="Walmart Sales – Clustering", page_icon="🛒", layout="wide")

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
        margin-top: 1.2rem; margin-bottom: 0.8rem;
    }
    .algo-box {
        background: #f0f8ff; border: 1px solid #b8d4f0; border-radius: 8px;
        padding: 12px 16px; font-size: 0.88rem; line-height: 1.6; color: #333;
    }
    .metric-row {
        display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px;
    }
    .mcard {
        background: linear-gradient(135deg,#f8f9fa,#e9ecef);
        border-left: 4px solid #0071CE; border-radius: 8px;
        padding: 10px 16px; min-width: 140px; flex: 1;
    }
    .mcard-yellow { border-left-color: #FFC220; }
    .mcard label { font-size: 0.75rem; color: #666; display: block; }
    .mcard span { font-size: 1.4rem; font-weight: 700; color: #222; }
    .mcard small { font-size: 0.78rem; color: #888; display: block; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛒 Walmart Sales — Clustering Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Análise de agrupamento com DBSCAN e Fuzzy K-Médias • Fatec / Prof. Davi Albuquerque</div>', unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Walmart_Sales.csv")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df

df = load_data()

@st.cache_data
def build_store_features(df):
    return df.groupby("Store").agg(
        Avg_Weekly_Sales=("Weekly_Sales","mean"),
        Std_Weekly_Sales=("Weekly_Sales","std"),
        Total_Sales=("Weekly_Sales","sum"),
        Avg_Temperature=("Temperature","mean"),
        Avg_Fuel_Price=("Fuel_Price","mean"),
        Avg_CPI=("CPI","mean"),
        Avg_Unemployment=("Unemployment","mean"),
        Holiday_Rate=("Holiday_Flag","mean"),
    ).reset_index()

store_base = build_store_features(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Walmart_Spark.svg/120px-Walmart_Spark.svg.png", width=50)
    st.markdown("## ⚙️ Configurações")

    all_features = ["Avg_Weekly_Sales","Std_Weekly_Sales","Avg_Temperature",
                    "Avg_Fuel_Price","Avg_CPI","Avg_Unemployment","Holiday_Rate"]
    selected_features = st.multiselect("Variáveis para clustering", all_features,
        default=["Avg_Weekly_Sales","Avg_Unemployment","Avg_CPI","Avg_Fuel_Price"])

    st.divider()
    st.markdown("### 🔵 DBSCAN")
    eps        = st.slider("Epsilon (ε)", 0.1, 3.0, 0.8, 0.05)
    min_samples= st.slider("MinPts", 2, 10, 3, 1)

    st.divider()
    st.markdown("### 🟡 Fuzzy K-Médias")
    n_clusters_fuzzy = st.slider("Número de grupos (K)", 2, 8, 3, 1)
    m_fuzz           = st.slider("Coeficiente de fuzzificação (m)", 1.1, 3.0, 2.0, 0.1)

if not selected_features:
    st.warning("Selecione pelo menos 2 features.")
    st.stop()

# ── Prep ──────────────────────────────────────────────────────────────────────
store_df = store_base.copy()
X_raw    = store_df[selected_features].fillna(0).values
X_scaled = StandardScaler().fit_transform(X_raw)

pca3   = PCA(n_components=min(3, X_scaled.shape[1]))
X_pca3 = pca3.fit_transform(X_scaled)
# pad to 3 cols if features < 3
while X_pca3.shape[1] < 3:
    X_pca3 = np.hstack([X_pca3, np.zeros((X_pca3.shape[0],1))])

var_exp  = list(pca3.explained_variance_ratio_) + [0]*(3-len(pca3.explained_variance_ratio_))
pc_labels= [f"PC{i+1} ({var_exp[i]*100:.1f}%)" for i in range(3)]

store_df["PC1"] = X_pca3[:,0]
store_df["PC2"] = X_pca3[:,1]
store_df["PC3"] = X_pca3[:,2]

# ── DBSCAN ────────────────────────────────────────────────────────────────────
db_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
store_df["DBSCAN_Cluster"] = db_labels.astype(str)
n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise       = int((db_labels == -1).sum())

try:
    mask   = db_labels != -1
    db_sil = silhouette_score(X_scaled[mask], db_labels[mask]) \
             if mask.sum() > 1 and len(set(db_labels[mask])) > 1 else None
except:
    db_sil = None

bold_colors = px.colors.qualitative.Bold
color_map   = {str(i): bold_colors[i % len(bold_colors)] for i in range(n_db_clusters+1)}
color_map["-1"] = "#cccccc"

# ── Fuzzy ─────────────────────────────────────────────────────────────────────
cntr, u, _, _, _, n_iter, _ = fuzz.cluster.cmeans(
    X_scaled.T, n_clusters_fuzzy, m_fuzz, error=1e-5, maxiter=1000)
fuzzy_labels = np.argmax(u, axis=0)
store_df["Fuzzy_Cluster"]  = fuzzy_labels.astype(str)
store_df["Max_Membership"] = u.max(axis=0)

membership_df = pd.DataFrame(u.T, columns=[f"Grupo {i+1}" for i in range(n_clusters_fuzzy)])
membership_df.insert(0,"Store", store_df["Store"].values)

try:
    fuzzy_sil = silhouette_score(X_scaled, fuzzy_labels) if len(set(fuzzy_labels)) > 1 else None
except:
    fuzzy_sil = None

# ── KPIs ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Visão Geral</div>', unsafe_allow_html=True)
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("🏪 Lojas",        45)
k2.metric("📅 Semanas",      df["Date"].nunique())
k3.metric("💰 Total Vendas", f"${df['Weekly_Sales'].sum()/1e9:.2f}B")
k4.metric("🔵 Grupos DBSCAN", n_db_clusters, f"{n_noise} ruídos")
k5.metric("🟡 Grupos Fuzzy",  n_clusters_fuzzy, f"Sil: {fuzzy_sil:.3f}" if fuzzy_sil else "—")

with st.expander("📚 Sobre os Algoritmos"):
    ea, eb = st.columns(2)
    with ea:
        st.markdown("**🔵 DBSCAN**")
        st.markdown("""<div class="algo-box">O número de grupos é definido <b>automaticamente</b>.
        Um <em>objeto de núcleo</em> tem pelo menos <code>minPts</code> vizinhos dentro do raio <code>ε</code>.
        Grupos crescem agregando vizinhos. Objetos isolados = <b>ruído</b>.</div>""", unsafe_allow_html=True)
    with eb:
        st.markdown("**🟡 Fuzzy K-Médias**")
        st.markdown("""<div class="algo-box">Cada objeto tem <b>grau de pertinência</b> a <em>todos</em> os grupos.
        O valor <code>u_k(x)</code> indica o quanto pertence ao grupo <code>k</code>.
        Soma dos graus por objeto = <b>1</b>.</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DBSCAN 3D
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔵 DBSCAN — Visualização 3D</div>', unsafe_allow_html=True)

# ── métricas DBSCAN ──
m1,m2,m3,m4 = st.columns(4)
m1.metric("Grupos encontrados",    n_db_clusters)
m2.metric("Lojas como ruído",      n_noise)
m3.metric("Lojas em grupos",       45 - n_noise)
m4.metric("Silhouette Score",      f"{db_sil:.3f}" if db_sil else "—")

col_db1, col_db2 = st.columns([3,1])
with col_db1:
    scene_blue = dict(
        bgcolor="rgba(240,248,255,1)",
        xaxis=dict(title=pc_labels[0], backgroundcolor="rgba(220,235,250,0.6)", gridcolor="white", showbackground=True),
        yaxis=dict(title=pc_labels[1], backgroundcolor="rgba(210,228,248,0.6)", gridcolor="white", showbackground=True),
        zaxis=dict(title=pc_labels[2], backgroundcolor="rgba(200,220,245,0.6)", gridcolor="white", showbackground=True),
    )
    fig_db3d = px.scatter_3d(
        store_df, x="PC1", y="PC2", z="PC3",
        color="DBSCAN_Cluster",
        hover_name="Store",
        hover_data={f: ":.2f" for f in selected_features if f in store_df.columns},
        color_discrete_map=color_map,
        title=f"PCA 3D — ε={eps}, minPts={min_samples} → {n_db_clusters} grupo(s) + {n_noise} ruído(s)",
        labels={"DBSCAN_Cluster":"Grupo"},
    )
    fig_db3d.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
    fig_db3d.update_layout(height=520, scene=scene_blue, paper_bgcolor="white",
                           legend_title_text="Grupo DBSCAN")
    st.plotly_chart(fig_db3d, use_container_width=True)

with col_db2:
    counts = store_df["DBSCAN_Cluster"].value_counts().reset_index()
    counts.columns = ["Grupo","Lojas"]
    counts["Tipo"] = counts["Grupo"].apply(lambda x: "🔇 Ruído" if x=="-1" else "✅ Cluster")
    st.dataframe(counts, use_container_width=True, hide_index=True)

    noise_list = store_df[store_df["DBSCAN_Cluster"]=="-1"]["Store"].tolist()
    st.markdown(f"**Lojas ruído:** {noise_list if noise_list else 'Nenhuma'}")

    fig_pie = px.pie(
        counts[counts["Grupo"]!="-1"], values="Lojas", names="Grupo",
        color_discrete_sequence=bold_colors, hole=0.4,
        title="Distribuição (sem ruído)"
    )
    fig_pie.update_layout(height=260, margin=dict(t=35,b=0,l=0,r=0), showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

# 3D features reais DBSCAN
feat_options = [f for f in selected_features if f in store_df.columns]
if len(feat_options) >= 3:
    st.markdown("**Explorar features reais em 3D**")
    cx, cy, cz = st.columns(3)
    ax_x = cx.selectbox("Eixo X", feat_options, index=0, key="db_x")
    ax_y = cy.selectbox("Eixo Y", feat_options, index=1, key="db_y")
    ax_z = cz.selectbox("Eixo Z", feat_options, index=2, key="db_z")

    fig_db_f = px.scatter_3d(
        store_df, x=ax_x, y=ax_y, z=ax_z,
        color="DBSCAN_Cluster", hover_name="Store",
        color_discrete_map=color_map,
        title=f"DBSCAN — {ax_x} × {ax_y} × {ax_z}",
        labels={"DBSCAN_Cluster":"Grupo"},
    )
    fig_db_f.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
    fig_db_f.update_layout(height=480, scene=scene_blue, paper_bgcolor="white")
    st.plotly_chart(fig_db_f, use_container_width=True)

# Boxplot
st.markdown("**Distribuição de features por grupo**")
bc, _ = st.columns([1,2])
chosen_feat = bc.selectbox("Feature", feat_options, key="db_box")
fig_box = px.box(store_df, x="DBSCAN_Cluster", y=chosen_feat,
                 color="DBSCAN_Cluster", color_discrete_map=color_map,
                 labels={"DBSCAN_Cluster":"Grupo"}, template="plotly_white")
fig_box.update_layout(height=340, showlegend=False)
st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# FUZZY 3D
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🟡 Fuzzy K-Médias — Visualização 3D</div>', unsafe_allow_html=True)

# ── métricas Fuzzy ──
avg_mem  = float(store_df["Max_Membership"].mean())
low_mem  = int((store_df["Max_Membership"] < 0.6).sum())
high_mem = int((store_df["Max_Membership"] >= 0.8).sum())

f1,f2,f3,f4,f5 = st.columns(5)
f1.metric("Grupos (K)",            n_clusters_fuzzy)
f2.metric("Silhouette Score",      f"{fuzzy_sil:.3f}" if fuzzy_sil else "—")
f3.metric("Pertinência média",     f"{avg_mem:.3f}")
f4.metric("Lojas ambíguas (<0.6)", low_mem)
f5.metric("Lojas nítidas (≥0.8)",  high_mem)

scene_yellow = dict(
    bgcolor="rgba(255,252,235,1)",
    xaxis=dict(title=pc_labels[0], backgroundcolor="rgba(255,245,200,0.6)", gridcolor="white", showbackground=True),
    yaxis=dict(title=pc_labels[1], backgroundcolor="rgba(255,240,180,0.6)", gridcolor="white", showbackground=True),
    zaxis=dict(title=pc_labels[2], backgroundcolor="rgba(255,232,160,0.6)", gridcolor="white", showbackground=True),
)

col_fz1, col_fz2 = st.columns([3,1])
with col_fz1:
    fig_fz3d = px.scatter_3d(
        store_df, x="PC1", y="PC2", z="PC3",
        color="Fuzzy_Cluster",
        size="Max_Membership",
        size_max=18,
        hover_name="Store",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        title=f"PCA 3D — K={n_clusters_fuzzy}, m={m_fuzz} | tamanho ∝ pertinência",
        labels={"Fuzzy_Cluster":"Grupo","Max_Membership":"Pertinência"},
    )
    fig_fz3d.update_traces(marker=dict(line=dict(width=0.5, color="white")))
    fig_fz3d.update_layout(height=520, scene=scene_yellow, paper_bgcolor="white",
                           legend_title_text="Grupo Fuzzy")
    st.plotly_chart(fig_fz3d, use_container_width=True)

with col_fz2:
    fz_cnt = store_df["Fuzzy_Cluster"].value_counts().reset_index()
    fz_cnt.columns = ["Grupo","Lojas"]
    fz_cnt["Grupo"] = fz_cnt["Grupo"].apply(lambda x: f"Grupo {int(x)+1}")
    st.dataframe(fz_cnt, use_container_width=True, hide_index=True)

    fig_hist = px.histogram(store_df, x="Max_Membership", nbins=15,
                            color_discrete_sequence=["#FFC220"],
                            title="Pertinência máxima", template="plotly_white")
    fig_hist.add_vline(x=avg_mem, line_dash="dash", line_color="red",
                       annotation_text=f"média {avg_mem:.2f}")
    fig_hist.update_layout(height=240, margin=dict(t=35,b=30))
    st.plotly_chart(fig_hist, use_container_width=True)

# 3D features reais Fuzzy
if len(feat_options) >= 3:
    st.markdown("**Explorar features reais em 3D**")
    fx, fy, fz = st.columns(3)
    fax = fx.selectbox("Eixo X", feat_options, index=0, key="fz_x")
    fay = fy.selectbox("Eixo Y", feat_options, index=1, key="fz_y")
    faz = fz.selectbox("Eixo Z", feat_options, index=2, key="fz_z")

    fig_fz_f = px.scatter_3d(
        store_df, x=fax, y=fay, z=faz,
        color="Fuzzy_Cluster", size="Max_Membership", size_max=18,
        hover_name="Store",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        title=f"Fuzzy — {fax} × {fay} × {faz}",
        labels={"Fuzzy_Cluster":"Grupo","Max_Membership":"Pertinência"},
    )
    fig_fz_f.update_traces(marker=dict(line=dict(width=0.5, color="white")))
    fig_fz_f.update_layout(height=480, scene=scene_yellow, paper_bgcolor="white")
    st.plotly_chart(fig_fz_f, use_container_width=True)

# ── Heatmap ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔥 Matriz de Pertinência — Fuzzy</div>', unsafe_allow_html=True)
hm = membership_df.set_index("Store").sort_index()
fig_heat = px.imshow(hm.T, color_continuous_scale="YlOrRd", aspect="auto",
                     title="Grau u_k(x) — cada coluna (loja) soma 1",
                     labels={"x":"Loja","y":"Grupo","color":"Pertinência"})
fig_heat.update_layout(height=300, template="plotly_white")
st.plotly_chart(fig_heat, use_container_width=True)

# ── Superfície 3D ─────────────────────────────────────────────────────────────
st.markdown("**Superfície 3D de Pertinência**")
Z = hm.values.T
fig_surf = go.Figure(go.Surface(
    z=Z, x=hm.index.tolist(),
    y=list(range(1, n_clusters_fuzzy+1)),
    colorscale="YlOrRd", showscale=True,
    colorbar=dict(title="Pertinência"), opacity=0.92,
))
fig_surf.update_layout(
    title="Superfície de Graus de Pertinência — Fuzzy K-Médias",
    scene=dict(
        xaxis_title="Loja", yaxis_title="Grupo", zaxis_title="Pertinência",
        **{k:v for k,v in scene_yellow.items() if k!="xaxis" and k!="yaxis" and k!="zaxis"},
        xaxis=dict(title="Loja",  backgroundcolor="rgba(255,245,200,0.6)", gridcolor="white", showbackground=True),
        yaxis=dict(title="Grupo", backgroundcolor="rgba(255,240,180,0.6)", gridcolor="white", showbackground=True),
        zaxis=dict(title="Pertinência", backgroundcolor="rgba(255,232,160,0.6)", gridcolor="white", showbackground=True),
    ),
    height=500, paper_bgcolor="white"
)
st.plotly_chart(fig_surf, use_container_width=True)

# ── Comparação ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚖️ Comparação DBSCAN vs Fuzzy K-Médias</div>', unsafe_allow_html=True)
comp = store_df[["Store","DBSCAN_Cluster","Fuzzy_Cluster","Max_Membership",
                  "Avg_Weekly_Sales","Avg_Unemployment","Avg_CPI"]].copy()
comp["Fuzzy_Cluster"]   = comp["Fuzzy_Cluster"].apply(lambda x: f"Grupo {int(x)+1}")
comp["DBSCAN_Cluster"]  = comp["DBSCAN_Cluster"].apply(lambda x: "Ruído" if x=="-1" else f"Grupo {int(x)+1}")
comp["Max_Membership"]  = comp["Max_Membership"].round(3)
comp["Avg_Weekly_Sales"]= comp["Avg_Weekly_Sales"].round(0).astype(int)
comp.columns = ["Loja","DBSCAN","Fuzzy","Pertinência Máx.","Vendas Médias ($)","Desemprego Médio","CPI Médio"]
st.dataframe(comp, use_container_width=True, hide_index=True)

# ── Série temporal ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Série Temporal de Vendas por Loja</div>', unsafe_allow_html=True)
sel_stores = st.multiselect("Selecione lojas:", sorted(df["Store"].unique()), default=[1,2,3])
if sel_stores:
    ts = df[df["Store"].isin(sel_stores)]
    fig_ts = px.line(ts, x="Date", y="Weekly_Sales", color="Store",
                     labels={"Weekly_Sales":"Vendas ($)","Date":"Data","Store":"Loja"},
                     template="plotly_white", title="Vendas Semanais por Loja")
    fig_ts.update_layout(height=380)
    st.plotly_chart(fig_ts, use_container_width=True)

st.divider()
st.caption("Dashboard • Streamlit • DBSCAN (sklearn) • Fuzzy C-Means (skfuzzy) • Plotly 5 • Dados: Walmart Sales")
