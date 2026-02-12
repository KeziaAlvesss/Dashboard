# Dashboard de Assistência Técnica BonSono
# Este é um dashboard interativo para análise de dados de assistência técnica dos colchões BonSono

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import re

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title="Dashboard Assistência - BonSono",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS PERSONALIZADO - variaveis de cor 
primary_color = "#003399"   # Azul BonSono
secondary_color = "#FF6B00" # Laranja 
background_color = "#F8F9FA" # Cinza claro
text_color = "#333333" # Preto suave

# CSS PERSONALIZADO - aplica o estilo ao dashboard 
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {background_color};
        font-family: 'Segoe UI', sans-serif;
    }}
    .css-1d391kg p {{
        color: {text_color};
    }}
    h1, h2, h3, h4 {{
        color: {primary_color} !important;
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 8px;
        border: none;
    }}
    .stButton>button:hover {{
        background-color: #002277;
    }}
    .metric-container {{
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        color: {text_color};
    }}
    .metric-value {{
        font-size: 24px;
        font-weight: bold;
        color: {primary_color};
    }}
    .metric-label {{
        font-size: 14px;
        color: #666;
    }}
    </style>
""", unsafe_allow_html=True)

# LOGO E TÍTULO
col1, col2 = st.columns([1, 3])
with col1:
    try:
        st.image("logo-bonsono.png.webp", width=200)
    except:
        st.write("<div style='font-size:12px;color:#666'>Logo BonSono</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <h1 style='color: {primary_color}; margin: 0;'>📊 Dashboard de Assistência Técnica</h1>
    <p style='color: {text_color}; font-size: 16px;'>Colchões BonSono | Análise Dinâmica com Upload de Dados</p>
    """, unsafe_allow_html=True)

# --- CARREGAMENTO DE DADOS VIA UPLOAD ---
st.sidebar.markdown("## 📁 Upload de Dados")

uploaded_file = st.sidebar.file_uploader(
    "Faça upload do arquivo Excel ou CSV",
    type=["xlsx", "xls", "csv"],
    help="Carregue seu relatório de assistência técnica"
)

if uploaded_file is None:
    st.info("👆 Faça o upload de um arquivo Excel ou CSV para começar a análise.")
    st.image("https://via.placeholder.com/800x400?text=Faça+Upload+do+Relatório+de+Assistência", use_container_width=True)
    st.stop()

# Carregar dados com base na extensão
try:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.sidebar.success("✅ Arquivo carregado com sucesso!")

    # Limpar nomes das colunas
    df.columns = df.columns.str.strip()

    # Renomear REGIAO para GRUPO (padronização interna do dashboard)
    df['GRUPO'] = df['REGIAO'].fillna('OUTROS').astype(str)
    
    # Verificar colunas obrigatórias (com GRUPO agora criado)
    colunas_obrigatorias = ['Data Chamada', 'VENDEDOR', 'REGIAO', 'PRODUTO', 'Motivo Constatado', 'TOTAL']
    colunas_faltando = [col for col in colunas_obrigatorias if col not in df.columns]

    if colunas_faltando:
        st.error(f"❌ Colunas obrigatórias ausentes: {colunas_faltando}")
        st.info("O arquivo deve conter: " + ", ".join(colunas_obrigatorias))
        st.stop()

    # Converter 'Data Chamada' com formato explícito
    df['Data Chamada'] = pd.to_datetime(df['Data Chamada'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

    # Remover linhas onde a data é inválida
    if df['Data Chamada'].isna().all():
        st.error("❌ Todas as datas estão inválidas. Verifique a coluna 'Data Chamada'.")
        st.stop()

    df = df.dropna(subset=['Data Chamada'])

    # Criar coluna 'Data' como date
    df['Data'] = df['Data Chamada'].dt.date

    # Garantir que 'TOTAL' seja numérico (lidar com formatação brasileira)
    if not pd.api.types.is_numeric_dtype(df['TOTAL']):
        st.warning("⚠️ A coluna 'TOTAL' não é numérica. Tentando converter...")
        # Converter valores com vírgula como separador decimal e ponto como milhar
        df['TOTAL'] = df['TOTAL'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df['TOTAL'] = pd.to_numeric(df['TOTAL'], errors='coerce')
        if df['TOTAL'].isna().all():
            st.error("❌ Falha ao converter 'TOTAL' para número.")
            st.stop()

except Exception as e:
    st.error(f"❌ Erro ao carregar ou processar o arquivo: {e}")
    st.exception(e)
    st.stop()

#  FILTROS SIDEBAR
st.sidebar.header("🔍 Filtros")

if len(df) == 0:
    st.warning("⚠️ Nenhum dado válido para exibir.")
    st.stop()

data_min = df['Data'].min()
data_max = df['Data'].max()

date_range = st.sidebar.date_input(
    label="Selecione o período",
    value=[data_min, data_max],
    min_value=data_min,
    max_value=data_max
)

motivo_filter = st.sidebar.multiselect(
    "Motivo do Defeito",
    options=df['Motivo Constatado'].dropna().unique(),
    default=[]
)
produto_filter = st.sidebar.multiselect(
    "Produto",
    options=df['PRODUTO'].dropna().unique(),
    default=[]
)

grupo_filter = st.sidebar.multiselect(
    "Grupo (Região)",
    options=df['GRUPO'].dropna().unique(),
    default=[]
)

# Aplicar filtros
filtered_df = df.copy()
filtered_df = filtered_df[
    (filtered_df['Data'] >= date_range[0]) &
    (filtered_df['Data'] <= date_range[1])
]

if motivo_filter:
    filtered_df = filtered_df[filtered_df['Motivo Constatado'].isin(motivo_filter)]
if grupo_filter:
    filtered_df = filtered_df[filtered_df['GRUPO'].isin(grupo_filter)]
if produto_filter:
    filtered_df = filtered_df[filtered_df['PRODUTO'].isin(produto_filter)]

if len(filtered_df) == 0:
    st.warning("⚠️ Nenhum dado encontrado com os filtros aplicados.")
    st.stop()

#  MÉTRICAS GERAIS 
st.markdown("### 📈 Métricas Gerais")
col1, col2, col3, col4 = st.columns(4)

total_atendimentos = len(filtered_df)
valor_total = filtered_df['TOTAL'].sum()
ticket_medio = valor_total / total_atendimentos if total_atendimentos > 0 else 0
produtos_unicos = filtered_df['PRODUTO'].nunique()

col1.markdown(f"<div class='metric-container'><div class='metric-value'>{total_atendimentos}</div><div class='metric-label'>Atendimentos</div></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-container'><div class='metric-value'>R$ {valor_total:,.2f}</div><div class='metric-label'>Valor Total</div></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-container'><div class='metric-value'>R$ {ticket_medio:,.2f}</div><div class='metric-label'>Ticket Médio</div></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-container'><div class='metric-value'>{produtos_unicos}</div><div class='metric-label'>Produtos Únicos</div></div>", unsafe_allow_html=True)

# === GRÁFICOS INTERATIVOS ===

# 1. Mapa de Calor: Defeitos por Produto
st.markdown("### 🔥 Mapa de Calor: Defeitos por Produto")
heatmap_data = pd.crosstab(filtered_df['PRODUTO'], filtered_df['Motivo Constatado'])
if not heatmap_data.empty:
    fig_heatmap = px.imshow(
        heatmap_data,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Motivo", y="Produto", color="Quantidade")
    )
    fig_heatmap.update_layout(
        title="Frequência de Defeitos por Produto",
        xaxis_title="Motivo Constatado",
        yaxis_title="Produto",
        height=600
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    st.info("Nenhum dado disponível para o mapa de calor.")

# 2. Produtos Mais Atendidos
st.markdown("### 🏆 Produtos Mais Atendidos")
prod_count = filtered_df['PRODUTO'].value_counts().reset_index()
prod_count.columns = ['PRODUTO', 'Contagem']
fig_prod = px.bar(
    prod_count,
    x='Contagem',
    y='PRODUTO',
    orientation='h',
    color='Contagem',
    color_continuous_scale='Blues',
    labels={'Contagem': 'Quantidade', 'PRODUTO': 'Produto'},
    hover_data={'PRODUTO': True, 'Contagem': True}
)
fig_prod.update_layout(
    title="Quantidade de Atendimentos por Produto",
    yaxis={'categoryorder': 'total ascending'},
    height=500
)
st.plotly_chart(fig_prod, use_container_width=True)

# 3. Distribuição de Defeitos (Pizza)
st.markdown("### Distribuição de Motivos de Defeito")

defeito_count = filtered_df['Motivo Constatado'].value_counts().reset_index()
defeito_count.columns = ['Motivo', 'Quantidade']

# Paleta de azul personalizada para a empresa BonSono
cores_azul = [
    "#003399",   # Azul BonSono
    "#0052b3",
    "#0078d4",
    "#00a6ed",
    "#80cfff"
]

fig_defeito = px.pie(
    defeito_count,
    names='Motivo',
    values='Quantidade',
    color_discrete_sequence=cores_azul,
    hole=0.4
)

fig_defeito.update_traces(
    textinfo='percent',
    textposition='inside'
)

fig_defeito.update_layout(
    title="Distribuição de Motivos de Defeito",
    legend_title="Motivo",
    legend=dict(
        orientation="v",
        y=1,
        x=1.05
    ),
    height=600,
    width=900
)

st.plotly_chart(fig_defeito, use_container_width=True)

# 4. Evolução Diária de Atendimentos
st.markdown("### 📅 Evolução Diária de Atendimentos")

daily = filtered_df.groupby('Data').size().reset_index(name='Quantidade')
daily['Data'] = pd.to_datetime(daily['Data']).dt.strftime('%d/%m/%Y')

fig_daily = px.line(
    daily,
    x='Data',
    y='Quantidade',
    markers=True,
    line_shape='spline',
    color_discrete_sequence=[primary_color]
)

fig_daily.update_traces(line=dict(width=3))
fig_daily.update_layout(
    title="Quantidade de Atendimentos por Dia",
    xaxis_title="Data",
    yaxis_title="Atendimentos",
    hovermode="x unified"
)

st.plotly_chart(fig_daily, use_container_width=True)

# 5. Valor Total por Vendedor
st.markdown("### 💼 Valor Total por Vendedor")
vendedor_valor = filtered_df.groupby('VENDEDOR')['TOTAL'].sum().reset_index().sort_values('TOTAL', ascending=False)
fig_vendedor = px.bar(
    vendedor_valor,
    x='TOTAL',
    y='VENDEDOR',
    orientation='h',
    color='TOTAL',
    color_continuous_scale='Blues',
    labels={'TOTAL': 'Valor (R$)', 'VENDEDOR': 'Vendedor'}
)
fig_vendedor.update_layout(
    title="Valor Total de Atendimentos por Vendedor",
    height=400
)
st.plotly_chart(fig_vendedor, use_container_width=True)

# === TABELA DETALHADA - EXCEL ===
st.markdown("### 📄 Dados Detalhados")
columns_to_show = ['NUNOTA', 'Parceiro', 'PRODUTO', 'Motivo Constatado', 'GRUPO', 'VENDEDOR', 'TOTAL', 'Data']
# Verificar se todas as colunas existem
columns_to_show = [col for col in columns_to_show if col in filtered_df.columns]
st.dataframe(
    filtered_df[columns_to_show].sort_values('Data', ascending=False),
    use_container_width=True,
    hide_index=True,
    height=400
)

# RODAPÉ
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 12px; margin-top: 20px;'>
    Dashboard gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')} | © 2025 Colchões BonSono. Todos os direitos reservados.
</div>
""", unsafe_allow_html=True)