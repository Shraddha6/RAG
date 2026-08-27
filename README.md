# RAG

###### A collection of Retrieval-Augmented Generation (RAG) implementations using different libraries and approaches. All notebooks use the **AtliqAI HR Policies** document as the test corpus.

---

## Notebooks

### 1. `basic-rag.ipynb` — RAG from Scratch

A minimal RAG pipeline built without any high-level framework.

**Stack:** `sentence-transformers` · `qdrant-client` · `groq`

**Pipeline:**
- Fetches a plain-text document from a GitHub URL
- Splits text into fixed-size word chunks (50 words each)
- Embeds chunks using `all-MiniLM-L6-v2`
- Stores vectors in a local Qdrant collection
- Retrieves top-k chunks by cosine similarity
- Generates answers via Groq LLM

---

### 2. `langchain-rag.ipynb` — RAG with LangChain

RAG built using LangChain abstractions and components.

**Stack:** `langchain` · `langchain-huggingface` · `langchain-qdrant` · `langchain-groq` · `langchain-text-splitters` · `tiktoken`

**Pipeline:**
- Loads text as a LangChain `Document`
- Splits using `CharacterTextSplitter` with tiktoken tokenizer (`cl100k_base`, 100 tokens/chunk)
- Embeds using `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`)
- Stores in `QdrantVectorStore` (local path)
- Retrieves via LangChain retriever interface
- Generates answers with `ChatGroq`

---

### 3. `docling-rag.ipynb` — RAG with Docling

RAG using Docling for structure-aware PDF parsing and hierarchical chunking.

**Stack:** `docling` · `docling-hierarchical-pdf` · `sentence-transformers` · `qdrant-client` · `groq`

**Pipeline:**
- Converts a PDF using Docling's `DocumentConverter`
- Chunks hierarchically with `HierarchicalChunker` (preserves section headings)
- Embeds chunk text (breadcrumb + content) using `all-MiniLM-L6-v2`
- Stores vectors and heading metadata in a local Qdrant collection
- Retrieves top-k chunks and generates answers via Groq LLM

---

### 4. `rag_with_docling_and_langchain.ipynb` — RAG with Docling + LangChain

Combines Docling's PDF parsing with LangChain's LCEL chain pattern.

**Stack:** `langchain-docling` · `langchain-huggingface` · `langchain-qdrant` · `langchain-groq`

**Pipeline:**
- Loads and chunks a PDF in one step using `DoclingLoader` with `ExportType.DOC_CHUNKS`
- Converts Docling chunks to LangChain `Document` objects (with heading metadata)
- Embeds using `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`)
- Stores in `QdrantVectorStore`
- Runs an LCEL chain: `retriever → format_docs → prompt → ChatGroq → StrOutputParser`

---

## Setup

```bash
uv sync
```

Or install dependencies manually:

```bash
pip install qdrant-client sentence-transformers groq langchain langchain-huggingface langchain-qdrant langchain-groq langchain-text-splitters langchain-docling docling docling-hierarchical-pdf tiktoken
```

A `GROQ_API_KEY` environment variable is required to run any of the notebooks.
