# ================================
# CONFIGURAÇÃO DO AMBIENTE
# ================================

# 1 - Criar ambiente virtual (venv)
# py -3.11 -m venv .venv
# -> Cria um ambiente isolado para evitar conflitos entre dependências

# 2 - Ativar o ambiente virtual (Windows)
# .venv\Scripts\activate
# -> Ativa o ambiente para instalar e usar bibliotecas localmente

# 3 - Instalar dependências do projeto
# pip install --upgrade pip
# pip install -U langchain langchain-core langchain-openai langchain-community chromadb pypdf langchain-experimental langchain-text-splitters langchain-huggingface python-dotenv faiss-cpu langchain-classic
# -> Instala bibliotecas necessárias para o pipeline RAG

# 4 - Criar arquivo .env com credenciais
# GROQ_API_KEY=sua_chave_de_api_aqui
# -> Armazena a chave de API com segurança (evita hardcode no código)

# ================================
# IMPORTAÇÕES
# ================================

import os

# Carrega variáveis do arquivo .env
from dotenv import load_dotenv

# Modelo LLM compatível com OpenAI
# Neste caso utilizando a API da Groq
from langchain_openai import ChatOpenAI

# Loader para leitura de arquivos texto
from langchain_community.document_loaders import TextLoader

# Splitters utilizados para dividir o documento
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Modelo de embeddings (transforma texto em vetores)
from langchain_huggingface import HuggingFaceEmbeddings

# Banco vetorial FAISS
from langchain_community.vectorstores import FAISS

# Templates de prompts
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate
)

# Parser de saída
from langchain_core.output_parsers import StrOutputParser

# Permite reutilizar a entrada original dentro da chain
from langchain_core.runnables import RunnablePassthrough

# Retriever avançado que cria múltiplas versões da pergunta
from langchain_classic.retrievers import MultiQueryRetriever


# ================================
# CONFIGURAÇÃO INICIAL
# ================================

# Carrega variáveis do arquivo .env
load_dotenv()


# ================================
# CONFIGURAÇÃO DO MODELO LLM
# ================================

# Inicializa o modelo via Groq
# temperature=0 deixa as respostas mais determinísticas
modelo = ChatOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.1-8b-instant",
    temperature=0
)


# ================================
# CARREGAMENTO DO DOCUMENTO
# ================================

# Carrega a base de conhecimento
documento = TextLoader(
    "documentos/GTB_gold_Nov23.txt",
    encoding="utf-8"
).load()


# ================================
# CONFIGURAÇÃO DOS EMBEDDINGS
# ================================

# Modelo multilíngue otimizado para similaridade semântica
embeddings_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)


# ================================
# CHUNKING DOS DOCUMENTOS
# ================================

# --------------------------------
# ETAPA 1 — Chunking estrutural
# --------------------------------
# Divide o texto em partes menores com overlap
# Isso evita perda de contexto entre os chunks

pre_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)

docs_pre = pre_splitter.split_documents(documento)


# --------------------------------
# ETAPA 2 — Chunking semântico
# --------------------------------
# Reorganiza os chunks baseado em semântica
# Isso melhora a coerência contextual do RAG

text_splitter = SemanticChunker(
    embeddings_model,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=0.5
)

# Chunks finais utilizados no sistema RAG
pedacos = text_splitter.split_documents(docs_pre)


# ================================
# DEBUG OPCIONAL
# ================================

# Quantidade total de chunks gerados
print(len(pedacos))

# Exibe o conteúdo do primeiro chunk
print(pedacos[0].page_content)


# ================================
# VECTOR STORE (FAISS)
# ================================

# Verifica se já existe um índice salvo localmente
if os.path.exists("faiss_index"):

    print("Carregando FAISS existente...")

    # Carrega índice vetorial já criado
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings_model,
        allow_dangerous_deserialization=True
    )

else:

    print("Criando novo FAISS...")

    # Cria o índice vetorial
    vectorstore = FAISS.from_documents(
        documents=pedacos,
        embedding=embeddings_model
    )

    # Salva localmente para reutilização futura
    vectorstore.save_local("faiss_index")


# ================================
# RETRIEVER BASE
# ================================

# Retriever tradicional por similaridade vetorial
# Ele será usado internamente pelo MultiQueryRetriever

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)


# ================================
# QUERY DO USUÁRIO
# ================================

# Pergunta enviada ao sistema
query = "como dar entrada em uma ocorrência de sinistro?"


# ================================
# PROMPT FINAL DO RAG
# ================================

# Prompt responsável pela resposta final
# Ele recebe:
# - contexto recuperado
# - pergunta do usuário

prompt = ChatPromptTemplate.from_messages([
    
    (
        "system",
        """
Você é um assistente de atendimento ao cliente para uma empresa de seguros.

Responda usando SOMENTE as informações do CONTEXTO.

REGRAS:
- Se a resposta NÃO estiver no contexto, responda exatamente:
  "Não encontrei essa informação na base de dados."
- NÃO use conhecimento externo
- NÃO faça suposições
- NÃO invente respostas
- Priorize informações mais relevantes
- Cite partes do contexto quando possível

CONTEXTO:
{contexto}
"""
    ),

    ("user", "{query}")

])


# ================================
# PROMPT DO MULTI QUERY
# ================================

# Este prompt é usado para gerar
# múltiplas versões da pergunta original

multi_prompt_template = """
Você é um assistente de modelo de linguagem de IA.

Sua tarefa é gerar cinco versões diferentes da pergunta do usuário
para recuperar documentos relevantes de um banco de dados vetorial.

Ao gerar múltiplas perspectivas sobre a pergunta do usuário,
seu objetivo é ajudar o usuário a superar algumas das limitações
da busca por similaridade baseada em distância.

Forneça estas perguntas alternativas separadas por quebras de linha.

Não forneça mais nada além das perguntas.

Pergunta original: {question}
"""

# Converte o template em um PromptTemplate
multi_prompt = PromptTemplate.from_template(
    multi_prompt_template
)


# ================================
# MULTI QUERY RETRIEVER
# ================================

# O MultiQueryRetriever:
#
# 1. Recebe a pergunta original
# 2. Gera múltiplas versões da pergunta
# 3. Executa várias buscas vetoriais
# 4. Combina os resultados encontrados
#
# Isso melhora recall e recuperação semântica

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=modelo,
    prompt=multi_prompt
)


# ================================
# PIPELINE RAG FINAL
# ================================

# Fluxo:
#
# Pergunta do usuário
#        ↓
# MultiQueryRetriever
#        ↓
# Recupera múltiplos contextos
#        ↓
# Envia contexto + pergunta ao LLM
#        ↓
# Gera resposta final

multi_rag_chain = (

    {
        "contexto": RunnablePassthrough() | multi_retriever,
        "query": RunnablePassthrough()
    }

    | prompt
    | modelo
    | StrOutputParser()

)


# ================================
# EXECUÇÃO FINAL
# ================================

# Executa toda a pipeline RAG
resposta = multi_rag_chain.invoke(query)

# Exibe resposta final
print(resposta)

# ================================
# LANGSMITH (DEBUG E MONITORAMENTO)
# ================================

# 1. Instalar:
# pip install langsmith

# 2. Criar conta e gerar API Key

# 3. Adicionar no .env:
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=sua_chave_aqui

# 4. Executar o código
# -> Permite visualizar:
#    - chunks recuperados
#    - prompt enviado
#    - resposta do modelo