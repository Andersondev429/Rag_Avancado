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

# Modelo de linguagem (LLM)
# Interface compatível com OpenAI usando Groq como backend
from langchain_openai import ChatOpenAI

# Carrega variáveis de ambiente do arquivo .env
from dotenv import load_dotenv

# Loader para leitura de arquivos .txt
from langchain_community.document_loaders import TextLoader

# Splitters utilizados para divisão inteligente dos documentos
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Modelo responsável por transformar textos em embeddings vetoriais
from langchain_huggingface import HuggingFaceEmbeddings

# Banco vetorial utilizado para armazenamento e busca semântica
from langchain_community.vectorstores import FAISS

# Criação de prompts estruturados para o LLM
from langchain_core.prompts import ChatPromptTemplate

# Parser que converte a saída do modelo para string simples
from langchain_core.output_parsers import StrOutputParser

# Permite reutilizar a entrada original dentro da chain
from langchain_core.runnables import RunnablePassthrough


# ================================
# CONFIGURAÇÃO INICIAL
# ================================

# Carrega variáveis do arquivo .env
# Exemplo:
# GROQ_API_KEY=xxxxxxxx
load_dotenv()


# ================================
# CONFIGURAÇÃO DO MODELO (LLM)
# ================================

# Inicializa o modelo de linguagem
# Utilizando Groq com API compatível OpenAI
modelo = ChatOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.1-8b-instant",

    # Temperature 0:
    # respostas mais determinísticas e objetivas
    # ideal para aplicações RAG
    temperature=0
)


# ================================
# CARREGAMENTO DOS DOCUMENTOS
# ================================

# Carrega o arquivo que será usado como base de conhecimento
documento = TextLoader(
    "documentos\GTB_gold_Nov23.txt",
    encoding="utf-8"
).load()


# ================================
# CONFIGURAÇÃO DOS EMBEDDINGS
# ================================

# Modelo multilíngue otimizado para busca semântica
# Excelente desempenho para português
embeddings_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)


# ================================
# CHUNKING (DIVISÃO DOS DOCUMENTOS)
# ================================

# Etapa 1:
# Divisão estrutural baseada em tamanho
#
# Objetivo:
# - evitar chunks gigantes
# - preservar contexto
# - melhorar performance do retrieval
pre_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)

docs_pre = pre_splitter.split_documents(documento)


# Etapa 2:
# Divisão semântica baseada em significado
#
# Objetivo:
# - criar chunks semanticamente coerentes
# - evitar cortes em mudanças de assunto
# - melhorar qualidade da recuperação
text_splitter = SemanticChunker(
    embeddings_model,

    # Usa percentil para identificar pontos de quebra semântica
    breakpoint_threshold_type="percentile",

    # Sensibilidade da quebra
    # quanto menor, mais divisões serão criadas
    breakpoint_threshold_amount=0.5
)

# Chunks finais utilizados no pipeline RAG
pedacos = text_splitter.split_documents(docs_pre)


# ================================
# DEBUG DOS CHUNKS
# ================================

# Quantidade total de chunks gerados
print(len(pedacos))

# Exibe o conteúdo do primeiro chunk
print(pedacos[0].page_content)


# ================================
# VECTOR STORE (FAISS)
# ================================

# Verifica se o índice vetorial já existe localmente
#
# Isso evita:
# - recriar embeddings toda execução
# - processamento desnecessário
if os.path.exists("faiss_index"):

    print("Carregando FAISS existente...")

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings_model,

        # Necessário para desserialização local do índice
        allow_dangerous_deserialization=True
    )

else:

    print("Criando novo FAISS...")

    # Cria o índice vetorial a partir dos chunks
    vectorstore = FAISS.from_documents(
        documents=pedacos,
        embedding=embeddings_model
    )

    # Salva o índice localmente
    vectorstore.save_local("faiss_index")


# ================================
# RETRIEVER (BUSCA SEMÂNTICA)
# ================================

# Configura mecanismo de recuperação de documentos
retriever = vectorstore.as_retriever(

    # Similarity:
    # retorna os chunks semanticamente mais próximos
    search_type="similarity",

    # k:
    # quantidade de chunks retornados
    search_kwargs={"k": 8}
)


# ================================
# QUERY DO USUÁRIO
# ================================

# Pergunta que será enviada ao pipeline RAG
query = "como dar entrada em uma ocorrência de sinistro?"


# ================================
# PROMPT DO SISTEMA
# ================================

# Prompt responsável por:
# - controlar comportamento do LLM
# - limitar hallucinations
# - forçar uso do contexto recuperado
prompt = ChatPromptTemplate.from_messages(
[
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

]
)


# ================================
# TESTE MANUAL DO RETRIEVER
# ================================

# Recupera os chunks mais relevantes
trechos = retriever.invoke(query)

# Une os chunks em um único contexto textual
contexto = "\n\n".join(
    [t.page_content for t in trechos]
)


# ================================
# PIPELINE RAG
# ================================

# Fluxo:
#
# Pergunta
#   ↓
# Retriever
#   ↓
# Contexto
#   ↓
# Prompt
#   ↓
# LLM
#   ↓
# Resposta final

rag_chain = (
    {
        "contexto": RunnablePassthrough() | retriever,
        "query": RunnablePassthrough()
    }

    | prompt
    | modelo
    | StrOutputParser()
)


# ================================
# EXECUÇÃO FINAL
# ================================

# Executa o pipeline completo
resposta = rag_chain.invoke(query)

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