# Dashboard de Assistência Técnica BonSono
# Este é um dashboard interativo para análise de dados de assistência técnica dos colchões BonSono

import streamlit as st #cria dashboards interativos
import pandas as pd #manipula dados em tabelas
import plotly.express as px #cria gráficos interativos
import plotly.graph_objects as go #cria gráficos interativos avançados
from plotly.subplots import make_subplots #cria subgráficos
import seaborn as sns #cria gráficos estatísticos
import matplotlib.pyplot as plt #gera gráficos estáticos
from datetime import datetime # manipula datas e horas

# CONFIGURAÇÃO DA PÁGINA - define como a pagina será exibida 
st.set_page_config(
    page_title="Dashboard Assistência - BonSono",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS PERSONALIZADO  - variaveis de cor 
primary_color = "#003399"   # Azul BonSono
secondary_color = "#FF6B00" # Laranja 
background_color = "#F8F9FA" # Cinza claro
text_color = "#333333" # Preto suave

# CSS PERSONALIZADO - aplica o estilo ao dashboard 
# Define o estilo do dashboard com as cores e fontes escolhidas - INJETA html/css diretamente no Streamlit
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

# LOGO E TÍTULO - divide a tela em 2 colunas com proporção 1:3
col1, col2 = st.columns([1, 3])
with col1:
    try:
        st.image("logo-bonsono.png.webp", width=200)
    except:
        st.write("<div style='font-size:12px;color:#666'>Logo BonSono</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <h1 style='color: {primary_color}; margin: 0;'>📊 Dashboard de Assistência Técnica</h1>
    <p style='color: {text_color}; font-size: 16px;'>Colchões BonSono | Relatório: 02 de Maio a 28 de Agosto de 2025</p>
    """, unsafe_allow_html=True)

# CARREGAMENTO DE DADOS (CORRIGIDO) 
@st.cache_data # Armazena os dados em cache para melhorar a performance
def load_data():
    try:
        # Tenta carregar o arquivo
        df = pd.read_excel("RelatorioCompleto.xlsx")  #ler o arquivo Excel
        st.write("✅ Excel carregado com sucesso!")
        
        # Limpar nomes das colunas
        df.columns = df.columns.str.strip()
        
        # Converter 'Data Chamada' com formato explícito - dd/mm/yyyy HH:MM:SS
        df['Data Chamada'] = pd.to_datetime(df['Data Chamada'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        
        # Remover linhas onde a data é inválida (NaT)
        if df['Data Chamada'].isna().all():
            st.error("❌ Todas as datas estão inválidas. Verifique a coluna 'Data Chamada' no Excel.")
            st.stop()
            
        df = df.dropna(subset=['Data Chamada'])
        
        # Criar coluna 'Data' como date
        df['Data'] = df['Data Chamada'].dt.date
        
        return df

    except FileNotFoundError:
        st.error("❌ Arquivo 'relatorio.xlsx' não encontrado. Verifique se está na pasta correta.")
        st.info("📁 A pasta deve conter: `app.py`, `relatorio.xlsx` e `logo-bonsono.png.webp`")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        st.stop()

# Carregar dados
df = load_data()

#  FILTROS SIDEBAR - filtros laterais
st.sidebar.header("🔍 Filtros")

# Garantir que há dados para filtrar
if len(df) == 0:
    st.warning("⚠️ Nenhum dado válido para exibir.")
    st.stop()

data_min = df['Data'].min()
data_max = df['Data'].max()

#Cabeçalhos dos filtros
date_range = st.sidebar.date_input(
    label="Selecione o período",
    value=[data_min, data_max],
    min_value=data_min,
    max_value=data_max
)
#Cria um dropdown para selecionar vendedores
vendedor_filter = st.sidebar.multiselect(
    "Vendedor",
    options=df['VENDEDOR'].dropna().unique(),
    default=[]
)

grupo_filter = st.sidebar.multiselect(
    "Grupo",
    options=df['GRUPO'].dropna().unique(),
    default=[]
)

# Aplicar filtros
filtered_df = df.copy()
filtered_df = filtered_df[
    (filtered_df['Data'] >= date_range[0]) &
    (filtered_df['Data'] <= date_range[1])
]

if vendedor_filter:
    filtered_df = filtered_df[filtered_df['VENDEDOR'].isin(vendedor_filter)]
if grupo_filter:
    filtered_df = filtered_df[filtered_df['GRUPO'].isin(grupo_filter)]

if len(filtered_df) == 0:
    st.warning("⚠️ Nenhum dado encontrado com os filtros aplicados.")
    st.stop()

#  MÉTRICAS GERAIS 
st.markdown("### 📈 Métricas Gerais")
col1, col2, col3, col4 = st.columns(4) #4 colunas para as métricas

total_atendimentos = len(filtered_df)
valor_total = filtered_df['TOTAL'].sum()
ticket_medio = valor_total / total_atendimentos if total_atendimentos > 0 else 0
produtos_unicos = filtered_df['PRODUTO'].nunique()

# 4 colunas para exibir os cards
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

# Mostrar percentual dentro e nome na legenda
fig_defeito.update_traces(
    textinfo='percent',
    textposition='inside'
)

# Layout mais limpo com legenda ao lado
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

# Agrupar por data
daily = filtered_df.groupby('Data').size().reset_index(name='Quantidade')

# Converter 'Data' para string no formato dd/mm/yyyy
daily['Data'] = pd.to_datetime(daily['Data']).dt.strftime('%d/%m/%Y')

# Criar gráfico
fig_daily = px.line(
    daily,
    x='Data',
    y='Quantidade',
    markers=True,
    line_shape='spline',
    color_discrete_sequence=[primary_color]
)

# Atualizar estilo DA LINHA ANTES de exibir
fig_daily.update_traces(line=dict(width=3))

# Atualizar layout
fig_daily.update_layout(
    title="Quantidade de Atendimentos por Dia",
    xaxis_title="Data",
    yaxis_title="Atendimentos",
    hovermode="x unified"
)

# Exibir gráfico
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
st.dataframe(
    filtered_df[columns_to_show].sort_values('Data', ascending=False),
    use_container_width=True,
    hide_index=True,
    height=400
)
# --- PÁGINA: PREVISÃO DE DEFEITOS ---
st.markdown("---")
st.markdown("## 🔮 Previsão de Defeitos")

st.write("""
Selecione os dados do produto para prever o defeito mais provável com base em histórico.
""")

# Criar cópia segura do df
df_pred = df.copy()

# Garantir que não há valores nulos em colunas críticas
df_pred = df_pred.dropna(subset=['PRODUTO', 'GRUPO', 'VALOR_UNITARIO', 'Motivo Constatado'])

# Limpar motivos muito longos ou compostos
df_pred['Motivo Constatado'] = df_pred['Motivo Constatado'].astype(str) #converte para string
df_pred = df_pred[~df_pred['Motivo Constatado'].str.contains("AUTORIZADA", na=False)] #remove linhas com "AUTORIZADA"
df_pred = df_pred[df_pred['Motivo Constatado'] != 'nan'] #remove linhas com 'nan'

# Verificar quantidade de dados
if len(df_pred) < 5:
    st.warning("❌ Poucos dados para treinar o modelo. Atualmente: " + str(len(df_pred)) + " registros válidos.") #verifica se há poucos dados
else:
    try:
        # Importa o modelo de IA "Random Forest" (Floresta Aleatória), usado para prever qual defeito é mais provável com base nos dados históricos
        from sklearn.ensemble import RandomForestClassifier
        # Converte textos (como nomes de produtos ou vendedores) em números, porque modelos de IA só trabalham com números
        from sklearn.preprocessing import LabelEncoder
        # Divide os dados em conjuntos de treino e teste, para treinar o modelo e depois testar sua precisão - 80% treino e 20% teste
        from sklearn.model_selection import train_test_split

        # Selecionar colunas
        features = ['PRODUTO', 'GRUPO']
        target = 'Motivo Constatado'

        df_model = df_pred[features + [target]].dropna()

        # Verificar se há pelo menos 2 tipos diferentes de defeitos - Se todos os registros forem do mesmo defeito, o modelo não pode "aprender a diferenciar
        if df_model[target].nunique() < 2:
            st.warning("❌ É necessário ter pelo menos 2 tipos diferentes de defeitos para treinar o modelo.")
        else:
            # Codificar variáveis categóricas
            le_produto = LabelEncoder()
            le_grupo = LabelEncoder()
            le_motivo = LabelEncoder()

            df_model['PRODUTO'] = le_produto.fit_transform(df_model['PRODUTO']) # transforma os nomes dos produtos em números
            df_model['GRUPO'] = le_grupo.fit_transform(df_model['GRUPO']) # transforma os grupos (regiões) em números
            df_model['Motivo'] = le_motivo.fit_transform(df_model['Motivo Constatado']) # transforma os motivos de defeito em números

            X = df_model[['PRODUTO', 'GRUPO']] # Seleciona as colunas de entrada
            y = df_model['Motivo'] # Seleciona a coluna de saída (o que queremos prever)

            # Dividir dados com segurança - treino e teste
            # Se houver poucos dados, usa 20% para teste, mas garante que não seja maior que 50% do total
            test_size = min(0.2, 0.5) # 20% para teste, mas não mais que 50% do total
            if len(X) > 1:
                # Evitar erro de stratify com classes pequenas
                if y.nunique() < 2:
                 st.warning("❌ É necessário ter pelo menos 2 tipos diferentes de defeitos para treinar o modelo.")
                else:
                  if (y.value_counts() < 2).any():
                    st.warning("⚠️ Algum defeito tem apenas 1 registro. Treinando sem estratificação.")
                    X_train, X_test, y_train, y_test = train_test_split(
                   X, y, test_size=test_size, random_state=42
                   )
                  else:
                    X_train, X_test, y_train, y_test = train_test_split(
                     X, y, test_size=test_size, random_state=42, stratify=y
        )
            else:
                X_train, X_test, y_train, y_test = X, pd.DataFrame(), y, pd.Series()

            # Treinar modelo - algoritmo de Floresta Aleatória
            model = RandomForestClassifier(n_estimators=50, random_state=42, min_samples_split=2, min_samples_leaf=1)
            model.fit(X_train, y_train)

            # Mostrar acurácia se houver dados de teste
            if len(X_test) > 0:
                accuracy = model.score(X_test, y_test)
                st.success(f"🎯 Modelo treinado! Acurácia: {accuracy:.2f}")
            else:
                st.success("🎯 Modelo treinado com todos os dados (sem teste)")

            # Formulário de entrada
            st.markdown("### 📝 Insira os dados para previsão")

            produto_input = st.selectbox("Produto", options=le_produto.classes_)
            grupo_input = st.selectbox("Grupo (Região)", options=le_grupo.classes_)
    
            if st.button("🔍 Prever Defeito"):
                # Codificar entradas
                prod_cod = le_produto.transform([produto_input])[0]
                grupo_cod = le_grupo.transform([grupo_input])[0]

                # Fazer previsão
                pred = model.predict([[prod_cod, grupo_cod]])[0]
                proba = model.predict_proba([[prod_cod, grupo_cod]])[0]
                motivo_predito = le_motivo.inverse_transform([pred])[0]

                # Confiança
                confianca = max(proba) * 100

                # Exibir resultado
                st.markdown(f"""
                ### ✅ Previsão de Defeito
                - **Defeito mais provável:** `{motivo_predito}`
                - **Confiança da previsão:** `{confianca:.1f}%`
                """)

                # Mostrar detalhes
                with st.expander("Ver detalhes da previsão"):
                    probs = pd.DataFrame({
                        'Motivo': le_motivo.classes_,
                        'Probabilidade (%)': proba * 100
                    }).sort_values('Probabilidade (%)', ascending=False)
                    st.dataframe(probs)
    except Exception as e:
        st.error(f"❌ Erro ao treinar modelo: {str(e)}")
        st.info("💡 Isso pode acontecer se houver pouca variação nos tipos de defeito.")

    #  PÁGINA: LEAN SIX SIGMA (DMAIC)
st.markdown("---")
st.markdown("## 🧩 Lean Six Sigma: DMAIC")

st.write("""
Este módulo aplica a metodologia DMAIC para análise e melhoria contínua com base nos dados de assistência técnica.
""")

# Carregar dados para esta aba
df_dmaic = df.copy()

# === 1. DEFINE (Definir o Problema) ===
st.markdown("### 🔍 1. Define: Definir o Problema")

# Passo 1: Encontrar o par (Produto, Motivo) com mais ocorrências
contagem_defeitos = df_dmaic.groupby(['PRODUTO', 'Motivo Constatado']).size().reset_index(name='Contagem')
top_row = contagem_defeitos.loc[contagem_defeitos['Contagem'].idxmax()]

produto_prioritario = top_row['PRODUTO']
motivo_prioritario = top_row['Motivo Constatado']
frequencia = top_row['Contagem']

# Passo 2: Encontrar a região (GRUPO) onde esse produto tem mais defeitos
df_grupo = df_dmaic[df_dmaic['PRODUTO'] == produto_prioritario]
grupo_prioritario = df_grupo['GRUPO'].value_counts().index[0] if not df_grupo.empty else "Não disponível"

# Passo 3: Exibir o problema prioritário com base nos dados
st.write(f"""
**Problema Prioritário:**  
O defeito mais crítico é **'{motivo_prioritario}'** no produto **'{produto_prioritario}'**, com **{frequencia} ocorrências**, principalmente na região **'{grupo_prioritario}'**.

**Declaração do Problema:**  
"O produto **{produto_prioritario}** apresenta alto índice de **'{motivo_prioritario}'** na região **{grupo_prioritario}**, gerando retrabalho, custos elevados e insatisfação do cliente."
""")

# Campo para o usuário escrever seu problema (com valor sugerido)
st.text_area(
    "📝 Descreva o problema (opcional)",
    value=f"Reduzir o índice de '{motivo_prioritario}' no produto '{produto_prioritario}'",
    height=100
)
# === TOP 5 COMBINAÇÕES PRODUTO × DEFEITO ===
st.markdown("### 🔝 Top 5 Combinações Produto × Defeito")
st.write("As combinações de produto e defeito com maior frequência no período selecionado:")

# Mostrar tabela com top 5
contagem_defeitos = df_dmaic.groupby(['PRODUTO', 'Motivo Constatado']).size().reset_index(name='Contagem')
top_5 = contagem_defeitos.sort_values('Contagem', ascending=False).head(5)

# Formatar colunas para melhor visualização
st.dataframe(
    top_5,
    use_container_width=True,
    hide_index=True
)

# === 2. MEASURE (Medir) ===
st.markdown("### 📏 2. Measure: Medir o Problema")

# Métricas-chave
total_defeitos = len(df_dmaic)
defeitos_prioritarios = len(df_dmaic[df_dmaic['Motivo Constatado'] == motivo_prioritario])
taxa_defeito = (defeitos_prioritarios / total_defeitos) * 100

# Mostrar métricas
col1, col2, col3 = st.columns(3)
col1.metric("Total de Defeitos", total_defeitos)
col2.metric(f"Defeitos '{motivo_prioritario}'", defeitos_prioritarios)
col3.metric("Taxa de Ocorrência", f"{taxa_defeito:.1f}%")

# Gráfico de evolução
st.markdown(f"**Evolução de '{motivo_prioritario}' ao longo do mês**")
daily_top = df_dmaic[df_dmaic['Motivo Constatado'] == motivo_prioritario].groupby('Data').size().reset_index(name='Quantidade')
daily_top['Data'] = pd.to_datetime(daily_top['Data']).dt.strftime('%d/%m')
fig_measure = px.line(daily_top, x='Data', y='Quantidade', markers=True)
fig_measure.update_layout(title=f"Defeitos de '{motivo_prioritario}' por Dia", yaxis_title="Quantidade")
st.plotly_chart(fig_measure, use_container_width=True)

# === 3. ANALYZE (Analisar Causa Raiz) ===
st.markdown("### 🧠 3. Analyze: Analisar Causa Raiz")

# Gráfico de Pareto
st.markdown("#### 📊 Diagrama de Pareto")
defeitos = df_dmaic['Motivo Constatado'].value_counts().reset_index()
defeitos.columns = ['Motivo', 'Frequência']
defeitos['Acumulado %'] = (defeitos['Frequência'].cumsum() / defeitos['Frequência'].sum()) * 100

fig_pareto = go.Figure()
fig_pareto.add_trace(go.Bar(
    x=defeitos['Motivo'],
    y=defeitos['Frequência'],
    name="Frequência",
    marker_color=primary_color
))
fig_pareto.add_trace(go.Scatter(
    x=defeitos['Motivo'],
    y=defeitos['Acumulado %'],
    name="Acumulado %",
    yaxis='y2',
    mode='lines+markers',
    line=dict(color='red', width=2)
))
fig_pareto.update_layout(
    title="Pareto: 80% dos Defeitos vêm de 20% das Causas",
    yaxis=dict(title="Frequência"),
    yaxis2=dict(title="Acumulado %", overlaying='y', side='right'),
    xaxis=dict(title="Motivo do Defeito"),
    showlegend=True
)
st.plotly_chart(fig_pareto, use_container_width=True)

# Diagrama de Ishikawa (Sugestões)
# st.markdown("#### 🐟 Diagrama de Ishikawa (Causa e Efeito)")
# st.write("Possíveis causas para o defeito:")
# causas = {
#     "Mão de obra": "Treinamento insuficiente do técnico",
#     "Método": "Processo de montagem não padronizado",
#     "Material": "Qualidade da espuma ou tecido",
#     "Máquina": "Calibração incorreta da prensa",
#     "Meio ambiente": "Umidade no ambiente de armazenamento",
#     "Medida": "Falha no controle de qualidade"
# }
# for categoria, causa in causas.items():
#     st.markdown(f"- **{categoria}**: {causa}")


# === 3.0 ISHIKAWA INTERATIVO (SUNBURST) ===
st.markdown("### 🐟 Ishikawa Interativo: Cadastre as Causas-Raiz")

# Inicializar session_state para causas
if 'causas_ishikawa' not in st.session_state:
    st.session_state.causas_ishikawa = []

# Formulário para adicionar nova causa
with st.form("form_causa"):
    st.write("Adicione uma causa raiz identificada:")
    categoria = st.selectbox(
        "Categoria (6M)",
        ["Mão de Obra", "Método", "Material", "Máquina", "Meio Ambiente", "Medição"]
    )
    causa = st.text_input("Descrição da Causa")
    submitted = st.form_submit_button("➕ Adicionar Causa")

if submitted and causa.strip() != "":
    st.session_state.causas_ishikawa.append({
        "Categoria": categoria,
        "Causa": causa.strip()
    })
    st.success(f"Causa '{causa}' adicionada em '{categoria}'!")

# Botão para limpar causas (opcional)
if st.session_state.causas_ishikawa and st.button("🗑️ Limpar todas as causas"):
    st.session_state.causas_ishikawa = []
    st.rerun()

# Exibir lista de causas cadastradas
if st.session_state.causas_ishikawa:
    st.write("### Causas Cadastradas")
    df_causas = pd.DataFrame(st.session_state.causas_ishikawa)
    st.dataframe(df_causas, use_container_width=True)

    # Preparar dados para o sunburst
    # Adicionar nó raiz
    dados_sunburst = [
        {"Categoria": "Problema", "Causa": "Causas do Defeito", "Valor": len(st.session_state.causas_ishikawa)}
    ]
    
    for idx, row in df_causas.iterrows():
        dados_sunburst.append({
            "Categoria": row["Categoria"],
            "Causa": row["Causa"],
            "Valor": 1
        })

    df_sunburst = pd.DataFrame(dados_sunburst)

    # Criar gráfico sunburst
    fig_sunburst = px.sunburst(
        df_sunburst,
        path=['Categoria', 'Causa'],
        values='Valor',
        color='Categoria',
        color_discrete_map={
            "Mão de Obra": "#FF6B35",
            "Método": "#003366",
            "Material": "#2E8B57",
            "Máquina": "#D4AC0D",
            "Meio Ambiente": "#8E44AD",
            "Medição": "#3498DB",
            "Problema": "#000000"
        },
        title="Ishikawa Interativo (Sunburst) - Causas Cadastradas"
    )
    fig_sunburst.update_layout(height=600)
    st.plotly_chart(fig_sunburst, use_container_width=True)
else:
    st.info("Nenhuma causa cadastrada ainda. Use o formulário acima para começar.")

# === 4. IMPROVE (Melhorar) ===
st.markdown("### 🚀 4. Improve: Propor Melhorias")

melhorias_sugeridas = [
    f"Padronizar o processo de montagem do {produto_prioritario}",
    f"Treinar a equipe técnica sobre o defeito '{motivo_prioritario}'",
    f"Inspecionar mais rigorosamente a matéria-prima do {produto_prioritario}",
    f"Criar checklist de qualidade para {produto_prioritario}"
]

st.write("✅ **Melhorias sugeridas com base nos dados:**")
for i, melhoria in enumerate(melhorias_sugeridas, 1):
    st.markdown(f"{i}. {melhoria}")

# Campo para usuário adicionar ações
st.text_area("💡 Suas ações de melhoria", height=150)

# Botão para "Implementar"
if st.button("✅ Implementar Melhorias"):
    st.success("✅ Ações de melhoria foram registradas e podem ser acompanhadas na próxima etapa.")

# === 5. CONTROL (Controlar) ===
st.markdown("### 📉 5. Control: Controlar os Resultados")

st.write("""
Após implementar as melhorias, monitore os resultados com indicadores-chave.
""")

# Simulação de meta
meta_reducao = st.slider("Meta de redução de defeitos (%)", min_value=10, max_value=80, value=50)
reducao_atual = st.slider("Redução alcançada (%)", min_value=0, max_value=100, value=0)

# Medidor de progresso
progresso = reducao_atual / 100
st.progress(progresso)
st.write(f"Progresso: {reducao_atual}% da meta de {meta_reducao}%")

if reducao_atual >= meta_reducao:
    st.success("✅ Meta atingida! Processo sob controle.")
else:
    st.warning("⚠️ Continue monitorando. Ainda não atingiu a meta.")

# Link para gerar relatório em pdf
st.markdown("### 📥 Gerar Relatório Completo")
st.download_button(
    label="📄 Baixar Relatório DMAIC",
    data=df_dmaic.to_csv(index=False).encode('utf-8'),
    file_name="relatorio_dmaic.csv",
    mime="text/csv"
)
# RODAPÉ da pag. 
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 12px; margin-top: 20px;'>
    Dashboard gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')} | © 2025 Colchões BonSono. Todos os direitos reservados.
</div>
""", unsafe_allow_html=True)
