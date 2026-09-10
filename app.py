import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==========================================
# CONFIGURAÇÃO DE SEGURANÇA E PADRÃO CORPORATIVO MULTINACIONAL
# ==========================================
st.set_page_config(
    page_title="Global Enterprise Intelligence Suite",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .stMetric { background-color: #ffffff; padding: 16px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e2e8f0; }
        h1, h2, h3 { color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { background-color: #ffffff; border-radius: 8px; padding: 10px 20px; border: 1px solid #e2e8f0; color: #475569; font-weight: 600; }
        .stTabs [aria-selected="true"] { background-color: #0f172a !important; color: #ffffff !important; border: 1px solid #0f172a; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# MÓDULOS DE MACHINE LEARNING E PREVISÃO
# ==========================================
def run_clustering(df, n_clusters=3):
    numeric_df = df.select_dtypes(include=np.number).dropna()
    if len(numeric_df) < n_clusters:
        return df, None
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    result_df = numeric_df.copy()
    result_df['Segmento_Cluster'] = [f"Cluster {c+1}" for c in clusters]
    return df.merge(result_df[['Segmento_Cluster']], left_index=True, right_index=True, how='left')

def run_forecast(df, date_col, value_col, periods=12):
    try:
        temp_df = df[[date_col, value_col]].dropna().copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col])
        temp_df = temp_df.sort_values(by=date_col).set_index(date_col)
        temp_df = temp_df.resample('ME').sum()
        
        if len(temp_df) < 6:
            return None
            
        model = ExponentialSmoothing(temp_df[value_col], trend='add', seasonal=None).fit()
        forecast = model.forecast(periods)
        forecast_dates = pd.date_range(start=temp_df.index[-1] + pd.offsets.MonthEnd(1), periods=periods, freq='ME')
        
        return pd.DataFrame({'Data': forecast_dates, 'Projecao': forecast.values})
    except Exception:
        return None

# ==========================================
# PAINEL DE CONTROLE LATERAL (MULTINATIONAL SETTINGS)
# ==========================================
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=150&auto=format&fit=crop&q=80", width=80)
    st.title("Global Ops Suite")
    st.markdown("---")
    
    source_type = st.radio("Fonte de Dados Corporativa", ["Google Sheets (Live/Auto-Sync)", "Upload Manual (CSV/Excel/Parquet)"])
    
    df = None
    
    if source_type == "Google Sheets (Live/Auto-Sync)":
        st.markdown("Insira o link público ou configure as credenciais via secrets para atualização em tempo real.")
        gsheet_url = st.text_input("URL da Planilha Google Sheets")
        
        if gsheet_url:
            try:
                @st.cache_data(ttl=10) # Atualização e cache otimizado a cada 10 segundos
                化物 = gsheet_url.replace("/edit?usp=sharing", "/export?format=csv").replace("/edit#gid=0", "/export?format=csv")
                df = pd.read_csv(化物)
                st.success("✅ Sincronizado com Google Sheets com sucesso!")
            except Exception as e:
                try:
                    df = pd.read_csv(gsheet_url)
                    st.success("✅ Sincronizado com sucesso!")
                except Exception as err:
                    st.error(f"Erro ao conectar na planilha. Verifique se o link está público para leitura. Detalhes: {err}")
    else:
        uploaded_file = st.file_uploader("Carregar Base de Dados Local", type=["csv", "xlsx", "parquet"])
        if uploaded_file is not None:
            @st.cache_data(show_spinner=False)
            def load_local_data(file):
                if file.name.endswith('.csv'):
                    return pd.read_csv(file)
                elif file.name.endswith('.parquet'):
                    return pd.read_parquet(file)
                else:
                    return pd.read_excel(file)
            df = load_local_data(uploaded_file)

    st.markdown("---")
    st.markdown("**Governança e Parâmetros**")
    currency_symbol = st.selectbox("Moeda Global de Exibição", ["USD ($)", "EUR (€)", "BRL (R$)", "GBP (£)"])
    enable_ml = st.toggle("Ativar Machine Learning (K-Means)", value=False)
    clusters_count = st.slider("Segmentos de Cluster", 2, 6, 3) if enable_ml else 3

if df is None:
    st.info("👋 **Ambiente Global Pronto.** Conecte uma planilha do Google Sheets (com atualização automática em tempo real) ou faça o upload de um arquivo para iniciar a automação corporativa.")
    st.stop()

if enable_ml:
    df = run_clustering(df, n_clusters=clusters_count)

# ==========================================
# HEADER EXECUTIVO E INDICADORES GLOBAIS
# ==========================================
st.title("Painel de Automação e Inteligência Analítica")
st.markdown(f"Monitoramento global multi-região integrado. Moeda base: **{currency_symbol}**.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total de Registros", f"{df.shape[0]:,}".replace(",", "."))
c2.metric("Atributos Monitorados", f"{df.shape[1]}")
c3.metric("Valores Omissos (Nulos)", f"{df.isna().sum().sum():,}".replace(",", "."))
c4.metric("Duplicidades Detectadas", f"{df.duplicated().sum():,}")
c5.metric("Pegada de Memória", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

st.markdown("---")

# ==========================================
# ABAS DE NAVEGAÇÃO ESTRUTURADA
# ==========================================
tab_visao, tab_eda, tab_ml_view, tab_pred, tab_export = st.tabs([
    "📈 Visão Executiva", 
    "🔍 Governança & EDA", 
    "🤖 Inteligência e Segmentação", 
    "🔮 Projeções Futuras", 
    "📥 Exportação & Compliance"
])

with tab_visao:
    col_v1, col_v2 = st.columns(2)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    with col_v1:
        if cat_cols and numeric_cols:
            dim_sel = st.selectbox("Dimensão Regional/Global", cat_cols, key="dim_v1")
            met_sel = st.selectbox("Métrica de Desempenho", numeric_cols, key="met_v1")
            agg_df = df.groupby(dim_sel)[met_sel].sum().reset_index().sort_values(by=met_sel, ascending=False).head(10)
            
            fig = px.bar(agg_df, x=dim_sel, y=met_sel, title=f"Top 10 {dim_sel} por {met_sel}",
                         template="plotly_white", color=met_sel, color_continuous_scale="Viridis")
            fig.update_layout(xaxis_title=dim_sel, yaxis_title=f"{met_sel} ({currency_symbol})", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("A base de dados requer colunas categóricas e numétricas combinadas.")
            
    with col_v2:
        if len(numeric_cols) >= 2:
            x_ax = st.selectbox("Eixo X de Correlação", numeric_cols, index=0, key="corr_x")
            y_ax = st.selectbox("Eixo Y de Correlação", numeric_cols, index=min(1, len(numeric_cols)-1), key="corr_y")
            color_dim = st.selectbox("Filtro Categórico Cruzado", ["Nenhum"] + cat_cols, key="corr_color")
            
            color_arg = None if color_dim == "Nenhum" else color_dim
            fig_scatter = px.scatter(df, x=x_ax, y=y_ax, color=color_arg, title=f"Dispersão Estratégica: {x_ax} vs {y_ax}", template="plotly_white")
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Necessário pelo menos duas variáveis numéricas.")

with tab_eda:
    st.subheader("Auditoria Estatística de Dados")
    st.dataframe(df.describe(include='all'), use_container_width=True)
    
    st.subheader("Matriz de Correlação Global")
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", origin="lower", template="plotly_white")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Insira variáveis numéricas adicionais para gerar a matriz.")

with tab_ml_view:
    if not enable_ml:
        st.warning("⚠️ Ative o **Machine Learning (K-Means)** na barra lateral para segmentação automática de portfólio/clientes.")
    else:
        st.subheader("Clusterização e Machine Learning Não Supervisionado")
        if 'Segmento_Cluster' in df.columns:
            cluster_counts = df['Segmento_Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Contagem']
            
            c_m1, c_m2 = st.columns([1, 2])
            with c_m1:
                st.dataframe(cluster_counts, use_container_width=True)
            with c_m2:
                fig_pie = px.pie(cluster_counts, names='Cluster', values='Contagem', title="Participação dos Segmentos", hole=0.4, template="plotly_white")
                st.plotly_chart(fig_pie, use_container_width=True)

with tab_pred:
    st.subheader("Projeção Preditiva de Séries Temporais")
    date_candidates = [c for c in df.columns if 'data' in c.lower() or 'date' in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])]
    
    if date_candidates and numeric_cols:
        p_date = st.selectbox("Variável Temporal", date_candidates)
        p_val = st.selectbox("Métrica de Projeção Alvo", numeric_cols)
        p_horizon = st.slider("Horizonte de Previsão (Meses)", 1, 24, 6)
        
        if st.button("Processar Projeção Estatística"):
            with st.spinner("Executando modelos de suavização exponencial..."):
                forecast_res = run_forecast(df, p_date, p_val, periods=p_horizon)
                if forecast_res is not None:
                    fig_f = go.Figure()
                    temp_orig = df.groupby(pd.to_datetime(df[p_date]))[p_val].sum().reset_index()
                    
                    fig_f.add_trace(go.Scatter(x=temp_orig[p_date], y=temp_orig[p_val], name="Histórico Real", mode='lines+markers', line=dict(color='#0f172a')))
                    fig_f.add_trace(go.Scatter(x=forecast_res['Data'], y=forecast_res['Projecao'], name="Projeção Futura", mode='lines+markers', line=dict(color='#2563eb', dash='dash')))
                    
                    fig_f.update_layout(title="Tendência e Previsão Consolidada", template="plotly_white", xaxis_title="Período", yaxis_title=p_val)
                    st.plotly_chart(fig_f, use_container_width=True)
                    st.dataframe(forecast_res, use_container_width=True)
                else:
                    st.error("Histórico insuficiente (mínimo de 6 períodos temporais necessários).")
    else:
        st.info("Certifique-se de que a base possui uma coluna de data e colunas numéricas.")

with tab_export:
    st.subheader("Central de Compliance e Exportação Corporativa")
    st.markdown("Baixe os dados processados e auditados em tempo real nos formatos oficiais da organização.")
    
    col_ex1, col_ex2 = st.columns(2)
    
    with col_ex1:
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Exportar Relatório CSV",
            data=csv_bytes,
            file_name="global_enterprise_export.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with col_ex2:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Global_Report')
        excel_data = output.getvalue()
        
        st.download_button(
            label="📊 Exportar Relatório Excel (.xlsx)",
            data=excel_data,
            file_name="global_enterprise_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
