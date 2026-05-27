# 🚀 Jornada RAG com LangChain — Projetos Avançados

````markdown
# 🧠 Jornada RAG com LangChain

Este repositório reúne projetos práticos de RAG (Retrieval-Augmented Generation) utilizando LangChain, FAISS, embeddings semânticos e modelos LLM via Groq.

Os projetos foram desenvolvidos com foco em:

- aplicações reais de IA Generativa
- busca semântica
- recuperação contextual
- engenharia de prompts
- arquiteturas modernas de RAG

---

# 📚 Projetos Disponíveis

## 1️⃣ RAG Tradicional com FAISS

Pipeline clássico de Retrieval-Augmented Generation utilizando:

- Chunking estrutural
- Chunking semântico
- Embeddings multilíngues
- Busca vetorial com FAISS
- Geração de respostas contextualizadas

### 🔥 Conceitos Aplicados

- Vector Store
- Similarity Search
- Semantic Search
- Embeddings
- Prompt Engineering

---

## 2️⃣ RAG Multi Query Retriever

Versão avançada utilizando `MultiQueryRetriever` do LangChain.

O sistema gera múltiplas reformulações da pergunta do usuário para aumentar a qualidade da recuperação semântica.

### 🔥 Benefícios

✅ Melhor recuperação de contexto

✅ Redução de ambiguidades

✅ Respostas mais precisas

✅ Melhor desempenho em perguntas complexas

---

# 🚀 Tecnologias Utilizadas

- Python 3.11+
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq API
- Semantic Chunking
- MultiQueryRetriever
- dotenv
- ChromaDB

---

# 📁 Estrutura do Repositório

```bash
.
├── documentos/
│   └── arquivos_base.txt
│
├── projeto-rag-tradicional/
│   ├── rag-tradicional.py
│   └── faiss_index/
│
├── projeto-rag-multi-query/
│   ├── rag-multi-query.py
│   └── faiss_index/
│
├── .env
├── requirements.txt
└── README.md
````

---

# ⚙️ Como Executar

## 1️⃣ Criar ambiente virtual

```bash
python -m venv .venv
```

---

## 2️⃣ Ativar ambiente virtual

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

## 3️⃣ Instalar dependências

```bash
pip install --upgrade pip

pip install -U \
langchain \
langchain-core \
langchain-openai \
langchain-community \
langchain-experimental \
langchain-text-splitters \
langchain-huggingface \
langchain-classic \
python-dotenv \
faiss-cpu \
chromadb \
pypdf
```

---

# 🔑 Configuração da API

Crie um arquivo `.env` na raiz do projeto:

```env
GROQ_API_KEY=sua_chave_aqui
```

---

# 🧩 Arquitetura dos Projetos

## 📄 Carregamento de Documentos

Leitura de arquivos utilizando:

```python
TextLoader
```

---

## ✂️ Chunking Estrutural

Separação inicial do texto utilizando:

```python
RecursiveCharacterTextSplitter
```

Com overlap para preservar contexto.

---

## 🧠 Chunking Semântico

Refinamento inteligente utilizando:

```python
SemanticChunker
```

---

## 🔎 Embeddings

Modelo utilizado:

```python
intfloat/multilingual-e5-base
```

Excelente desempenho para português.

---

## 🗂️ Banco Vetorial

Armazenamento vetorial utilizando:

```python
FAISS
```

---

## 🤖 Modelo LLM

Utilização do modelo:

```python
llama-3.1-8b-instant
```

via Groq API.

---

# ▶️ Executar os Projetos

## 🚀 RAG Tradicional

```bash
python rag-tradicional.py
```

---

## 🚀 RAG Multi Query Retriever

```bash
python rag-multi-query.py
```

---

# 📈 Conceitos Trabalhados

* Retrieval-Augmented Generation (RAG)
* Multi Query Retrieval
* Semantic Search
* Embeddings
* Vector Database
* Prompt Engineering
* LangChain Pipelines
* LLM Applications
* IA Generativa

---

# 📌 Melhorias Futuras

* Interface Web
* Upload de PDFs
* Histórico de conversas
* Streaming de respostas
* Persistência vetorial
* Integração com Streamlit
* Memória conversacional

---

# 👨‍💻 Autor

Anderson Pinheiro da Silva
