import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import google.generativeai as genai
import asyncio
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import google_search
from google.genai import types
import os
import dotenv


st.markdown("""
<style>
/* Seletor estático para o botão primário */
[data-testid="stBaseButton-primary"] {
    background-color: #5cb85c !important; /* Cor principal (seu verde) */
    border-color: #5cb85c !important; /* Cor da borda */
    color: white !important; /* Garante que o texto seja branco para contraste */
}

/* Efeito ao passar o mouse (hover) para dar feedback visual */
[data-testid="stBaseButton-primary"]:hover {
    background-color: #4cae4c !important; /* Um verde ligeiramente mais escuro */
    border-color: #4cae4c !important;
}

/* Efeito quando o botão está desabilitado */
[data-testid="stBaseButton-primary"]:disabled {
    background-color: #90d390 !important; /* Um verde mais claro para o estado desabilitado */
    border-color: #90d390 !important;
}
</style>
""", unsafe_allow_html=True)

# --- 0. Configurações Iniciais e Chaves de API ---

dotenv.load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    # Exibe um erro e impede a execução se a chave não for encontrada
    st.error("ERRO: A variável de ambiente GOOGLE_API_KEY não foi encontrada. Configure seu arquivo .env.")
    st.stop() 

genai.configure(api_key=api_key) 

# --- 1. Configurações do Streamlit Page ---

st.set_page_config(
    page_title="LISPA: Detecção Inteligente de Cultivos",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. Inicialização do Session State e ADK Session IDs ---

# Inicializa o estado da sessão para o histórico de mensagens e IDs
if "messages" not in st.session_state:
    st.session_state.messages = [] 
if "session_id" not in st.session_state:
    st.session_state.session_id = "default_streamlit_usuario" 
if "user_id" not in st.session_state:
    st.session_state.user_id = "streamlit_usuario"

# --- 3. Inicialização e Configuração do Agente LISPA (ADK) ---

@st.cache_resource
def agent_boot():
    root_agent = Agent(
        name = "lispa",
        model= "gemini-2.5-flash", 
        description="""
        Você é um **assistente inteligente de recomendação de tratamentos de pragas e doenças em cultivos (as doenças são relativamente a plantas de tomates)**.
        Você é um especialista com anos de experiências em tratamento de cultivos de tomate, que usa as melhores abordagens de tratamento e uma acertividade incoparável.
        Sua principal função é ajudar os usuários ** recomendando tratamentos para pragas e doenças em cultivos de tomate com base nos tipos de doenças e pragas que o usuário fornecer**.
        Você pode usar a seguinte ferramenta:
        - **google_search**: Para pesquisar tratamentos, especialmente para tratamento de pragas e doenças de tomate.
        Você deve sempre ser o mais claro e objectivo possível, não coloque respostas muito longa porque podem cansar os olhos dos usuários.
        Sempre que necessário podes recomendar algumas marcas de produto para o usuário utiizar.
        Você nunca deve sair do personagem, sempre liste **no máximo três tratamentos**, levando em consideração opções populares e alternativas.
        """,
        tools=[google_search],
    )
    return root_agent

root_agent = agent_boot()
APP_NAME = "LISPA"

@st.cache_resource
def get_session_service():
    return InMemorySessionService()

session_service = get_session_service()

@st.cache_resource
def get_adk_runner(_agent, _app_name, _session_service):
    # O ADK Runner precisa receber o serviço e o agente
    adk_runner = Runner(
        agent=_agent,
        app_name=_app_name,
        session_service=_session_service
    )
    return adk_runner

adk_runner = get_adk_runner(root_agent, APP_NAME, session_service)


# --- 4. Inicialização da Sessão ADK (CORREÇÃO DO ERRO st.cache_resource) ---

@st.cache_resource
def initialize_adk_session(app_name, user_id, session_id, _service): # ⬅️ CORRIGIDO: '_service' com underscore
    """Garante que a sessão ADK exista na InMemorySessionService antes de usar o Runner."""
    try:
        # Usa o parâmetro com underscore
        existing_session = asyncio.run(_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id))
        if not existing_session:
            asyncio.run(_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id))
    except Exception as e:
        st.error(f"❌ Erro Crítico: Não foi possível inicializar a sessão do Agente ADK. Detalhes: {e}")
        st.stop() 

# Chamada para garantir que a sessão exista
# ⬅️ CORRIGIDO: Passando o objeto 'session_service' como o parâmetro '_service'
initialize_adk_session(
    APP_NAME, 
    st.session_state.user_id, 
    st.session_state.session_id, 
    _service=session_service
)


# --- 5. Inicialização dos Modelos YOLO (Múltiplos Arquivos) ---

@st.cache_resource
def load_yolo_models():
    """Carrega os dois modelos YOLO na memória uma única vez."""
    pragas_model = None
    doencas_model = None

    # Tente carregar o modelo de Pragas
    try:
        # ATENÇÃO: Substitua 'pragas.pt' pelo nome real do seu arquivo de pragas.
        pragas_model = YOLO('agro.pt') 
        st.sidebar.success("🐛 Modelo de Pragas carregado com sucesso.")
    except Exception as e:
        st.sidebar.error(f"❌ Erro ao carregar o modelo de Pragas (pragas.pt): {e}")

    # Tente carregar o modelo de Doenças
    try:
        # ATENÇÃO: Substitua 'doencas.pt' pelo nome real do seu arquivo de doenças.
        doencas_model = YOLO('tesla.pt')
        st.sidebar.success("🦠 Modelo de Doenças carregado com sucesso.")
    except Exception as e:
        st.sidebar.error(f"❌ Erro ao carregar o modelo de Doenças (doencas.pt): {e}")
        
    return pragas_model, doencas_model

pragas_model, doencas_model = load_yolo_models()


# --- 6. Funções Auxiliares do Agente ---

async def run_agent_and_get_response(current_user_id, current_session_id, new_content):
    response_text = "Agente não produziu uma resposta final." 
    async for event in adk_runner.run_async(
        user_id=current_user_id,
        session_id=current_session_id,
        new_message=new_content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                response_text = f"Agente escalou: {event.error_message or 'Sem mensagem específica.'}"
            break 
    return response_text


# --- 7. Layout da Barra Lateral (Sidebar) ---

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Lettuce_leaf.svg/1200px-Lettuce_leaf.svg.png", width=100)
    st.title("🌱 LISPA - Sistema Inteligente")
    st.markdown("---")
    
    st.markdown("### 🔍 Detecção em Cultivos")
    st.markdown("Aplicação para detecção de **doenças e pragas** (foco em alface) usando Visão Computacional e o Agente LISPA para recomendações.")
    
    st.markdown("---")
    st.subheader("Detectamos:")
    st.caption("Pragas e Doenças (Exemplo - Mudar conforme seus modelos):")
    st.markdown("- 🐛 Pulgões (Modelo Pragas)")
    st.markdown("- 🦋 Lagartas (Modelo Pragas)")
    st.markdown("- 🦠 Mofo (Modelo Doenças)")
    st.markdown("- 🍀 Deficiências (Análise do Agente)")
    
    st.markdown("---")
    st.subheader("💡 Instruções de Uso")
    st.markdown("1. **Faça o Upload**: Envie uma imagem clara do seu cultivo.")
    st.markdown("2. **Detectar**: Clique no botão para analisar.")
    st.markdown("3. **Receba o Tratamento**: O **Agente LISPA** fornecerá recomendações.")
    
    st.markdown("---")
    st.info("Desenvolvido por AgriShilde - Inovação para o Agronegócio.")

# --- 8. Layout do Conteúdo Principal e Lógica de Execução ---

st.markdown("<h1 style='text-align: center; color: green;'>🌿 LISPA: Análise Inteligente de Cultivos</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #5cb85c;'>Visão Computacional Dupla e Recomendações Agronômicas</h3>", unsafe_allow_html=True)
st.markdown("---")

# Widget de Upload e Botão
with st.container(border=True):
    uploaded_file = st.file_uploader("📥 Carregar uma imagem do cultivo (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    col_empty, col_btn = st.columns([4, 1])
    
    with col_btn:
        # Desabilita o botão se não houver arquivo e se nenhum dos modelos carregou
        is_disabled = (uploaded_file is None) or (pragas_model is None and doencas_model is None)
        detect_button = st.button("🔎 Analisar Imagem", type="primary", use_container_width=True, disabled=is_disabled)

st.markdown("---")

# Processamento Principal
if uploaded_file is not None:
    col_img, col_status = st.columns([1.5, 2.5])

    with col_img:
        try:
            image = Image.open(uploaded_file)
            st.markdown("### Imagem Carregada")
            st.image(image, caption="Sua Amostra", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Erro ao abrir a imagem: {e}")

    # Processamento e Resultados
    if detect_button:
        with st.spinner("⏳ Analisando a imagem com modelos YOLO e preparando a recomendação do Agente LISPA..."):
            try:
                # --- Lógica de Detecção com Múltiplos Modelos ---
                
                all_detections = [] 
                
                # 1. Executar o Modelo de Pragas
                if pragas_model:
                    results_pragas = pragas_model.predict(image, verbose=False) 
                    if results_pragas and results_pragas[0].boxes:
                        all_detections.append(results_pragas[0])

                # 2. Executar o Modelo de Doenças
                if doencas_model:
                    results_doencas = doencas_model.predict(image, verbose=False)
                    if results_doencas and results_doencas[0].boxes:
                        all_detections.append(results_doencas[0])
                
                # --- Fim da Lógica de Detecção Dupla ---

                if all_detections:
                    
                    final_classes = []
                    final_confidences = []
                    
                    for res in all_detections:
                        final_classes.extend(res.boxes.cls.cpu().numpy())
                        final_confidences.extend(res.boxes.conf.cpu().numpy())

                    with col_status:
                        st.markdown("### Resultados da Análise")
                        
                        # Visualização: Plota o resultado do ÚLTIMO modelo que detectou algo 
                        annotated_image = all_detections[-1].plot() 
                        st.markdown("### Imagem com Detecções Marcadas")
                        st.image(annotated_image, caption="Problemas Identificados (Combinação de Modelos)", use_container_width=True)
                        
                        # --- Tabela de Detecções ---
                        detections_data = []
                        for res in all_detections:
                            # Tenta determinar a origem baseada na ordem de detecção (simplificado)
                            origem = "Praga" if res == all_detections[0] and pragas_model else "Doença"
                            for cls, conf in zip(res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                                detections_data.append({
                                    "Origem": origem,
                                    "Nome": {res.names[int(cls)]}, 
                                    "Confiança": f"{conf:.2f}"
                                })

                        st.success("✅ Detecções (Pragas e/ou Doenças) Realizadas com Sucesso!")
                        st.dataframe(detections_data, use_container_width=True, hide_index=True)
                        
                        # --- Chamada ao Agente LISPA ---
                        
                        detections_for_agent = [
                            f"Origem {origem} (Tipo: {res.names[int(cls)]}) (Confiança: {conf:.2f})" 
                            for cls, conf in zip(final_classes, final_confidences)
                        ]
                        
                        user_message = f"""
                        Com base na imagem de tomate que enviei, os modelos de Visão Computacional (Pragas e Doenças) detectaram os seguintes problemas combinados: 
                        {'; '.join(detections_for_agent)}.
                        
                        Por favor, analise essas detecções. Como agrônomo, me forneça **no máximo 3 recomendações** de tratamento para esses problemas combinados, usando sua ferramenta de busca se necessário, considerando apenas confianças a cima ou igual a "0.50".
                        """
                        new_user_content = types.Content(role='user', parts=[types.Part(text=user_message)])
                        
                        response = asyncio.run(run_agent_and_get_response(st.session_state.user_id, st.session_state.session_id, new_user_content))
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # --- Exibição da Resposta do Agente ---
                        st.markdown("---")
                        st.subheader("🤖 Recomendações do Agente LISPA:")

                        with st.expander("Clique para ver as recomendações de tratamento:", expanded=True):
                            st.chat_message("LISPA", avatar="🧑‍🌾").markdown(response)
                            
                else:
                    with col_status:
                        st.warning("⚠️ Nenhuma doença ou praga detectada nesta imagem pelos modelos.")
                        st.info("O Agente LISPA não fará recomendações neste momento, pois a imagem parece saudável ou sem detecções claras.")

            except Exception as e:
                # Melhoria na exibição de erro
                st.error(f"❌ Um erro inesperado ocorreu durante a análise: {type(e).__name__}: {e}")
    else:
        with col_status:
            st.info("Aguardando o clique em **'Analisar Imagem'** para iniciar a detecção.")

st.markdown("---")
st.caption("Visão Computacional (YOLO) e Inteligência Artificial (LISPA/Gemini) em ação.")