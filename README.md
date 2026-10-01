# RAG Knowledge Assistant

A Streamlit-based Retrieval-Augmented Generation (RAG) application that enables users to interact with their own PDF and DOCX documents through a conversational question-answering interface.

The application combines document processing, semantic retrieval, vector search, conversation history, and large language model generation to provide context-grounded responses with source information.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Quick Start](#quick-start)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [RAG Pipeline](#rag-pipeline)
9. [Prompt Design](#prompt-design)
10. [Project Structure](#project-structure)
11. [Core Components](#core-components)
12. [Supported File Types](#supported-file-types)
13. [Knowledge Base Management](#knowledge-base-management)
14. [Troubleshooting](#troubleshooting)
15. [Development](#development)
16. [Requirements](#requirements)
17. [Limitations](#limitations)

---

# Overview

The RAG Knowledge Assistant allows users to upload documents and ask questions about their content using a conversational interface.

Instead of providing the complete document directly to the language model, the application follows a retrieval-augmented generation workflow:

```text
Documents
    |
    v
Document Loading
    |
    v
Text Splitting
    |
    v
Embedding Generation
    |
    v
FAISS Vector Store
    |
    |
User Question
    |
    v
Semantic Retrieval
    |
    v
Relevant Document Chunks
    |
    v
RAG Prompt
    |
    v
Groq LLM
    |
    v
Answer + Sources
```

This allows the language model to generate responses using relevant information retrieved from the user's documents.

---

# Architecture

The application is divided into five main layers:

```text
                ┌──────────────────────┐
                │    Streamlit UI      │
                └──────────┬───────────┘
                           |
                           v
                ┌──────────────────────┐
                │ Document Processing   │
                │ PDF / DOCX Loading   │
                └──────────┬───────────┘
                           |
                           v
                ┌──────────────────────┐
                │ Text Splitting       │
                │ + Embeddings         │
                └──────────┬───────────┘
                           |
                           v
                ┌──────────────────────┐
                │ FAISS Vector Store   │
                │ Semantic Retrieval   │
                └──────────┬───────────┘
                           |
                           v
                ┌──────────────────────┐
                │ RAG Chain            │
                │ Context + History    │
                └──────────┬───────────┘
                           |
                           v
                ┌──────────────────────┐
                │ Groq LLM             │
                └──────────┬───────────┘
                           |
                           v
                ┌──────────────────────┐
                │ Answer + Sources     │
                └──────────────────────┘
```

---

# Features

## Document Processing

* PDF document support
* DOCX document support
* Document text extraction
* Document metadata handling
* Configurable text chunking
* Configurable chunk overlap

## Semantic Retrieval

* HuggingFace embedding model
* FAISS vector similarity search
* Configurable Top-K retrieval
* Persistent vector index
* Relevant document chunk retrieval

## Conversational RAG

* Question answering over uploaded documents
* Conversation history
* Follow-up question support
* Context-aware responses
* Source information in responses

## Knowledge Base Management

* Build knowledge base
* Rebuild knowledge base
* Clear vector store
* Check vector-store status
* Track processed documents and chunks

## Application

* Streamlit interface
* Document upload
* Chat interface
* Sidebar controls
* Environment-based configuration

---

# Quick Start

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure Environment Variables

Create the environment file:

```bash
cp .env.example .env
```

Add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

## 3. Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# Installation

## Prerequisites

* Python 3.10 or higher
* Groq API key
* Internet access for the initial HuggingFace model download

The project supports Windows, macOS, and Linux. The original project documentation specifies Windows 10/11 as a tested environment.

## Install Dependencies

```bash
pip install -r requirements.txt
```

### Main Dependencies

| Package                    | Purpose                         |
| -------------------------- | ------------------------------- |
| `streamlit`                | Web application framework       |
| `langchain`                | RAG pipeline orchestration      |
| `langchain-community`      | Loaders and integrations        |
| `langchain-core`           | Core LangChain components       |
| `langchain-text-splitters` | Document chunking               |
| `langchain-groq`           | Groq LLM integration            |
| `faiss-cpu`                | Vector similarity search        |
| `pypdf`                    | PDF text extraction             |
| `docx2txt`                 | DOCX text extraction            |
| `python-docx`              | DOCX document handling          |
| `python-dotenv`            | Environment variable management |

---

# Configuration

The application uses environment variables for configuration.

Create:

```text
.env
```

from:

```text
.env.example
```

## Environment Variables

| Variable          | Default                                  | Description                           |
| ----------------- | ---------------------------------------- | ------------------------------------- |
| `GROQ_API_KEY`    | Required                                 | API key for Groq LLM inference        |
| `GROQ_MODEL`      | `openai/gpt-oss-120b`                    | Groq chat model                       |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model           |
| `TOP_K`           | `4`                                      | Number of document chunks retrieved   |
| `CHUNK_SIZE`      | `1000`                                   | Maximum characters per chunk          |
| `CHUNK_OVERLAP`   | `150`                                    | Overlapping characters between chunks |

These are the documented default configuration values for the project.

### Example

```env
GROQ_API_KEY=your_groq_api_key

GROQ_MODEL=openai/gpt-oss-120b

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

TOP_K=4

CHUNK_SIZE=1000

CHUNK_OVERLAP=150
```

Do not commit the `.env` file containing API credentials to version control.

---

# Usage

## Adding Documents

Documents can be added using either of the following methods.

### Streamlit Upload

Upload PDF or DOCX files using the file uploader in the application sidebar.

### Manual Placement

Place files directly inside:

```text
data/documents/
```

Supported extensions:

```text
.pdf
.docx
```

---

## Building the Knowledge Base

After adding documents:

1. Open the application.
2. Add the required PDF or DOCX files.
3. Select **Build Knowledge Base**.
4. Documents are loaded.
5. Text is split into chunks.
6. Embeddings are generated.
7. FAISS creates the vector index.
8. The index is stored in:

```text
data/vectorstore/
```

The application displays the number of processed documents and chunks after successful indexing.

---

## Chatting

Once the vector store status displays **Ready**, questions can be submitted through the chat interface.

The system performs the following workflow:

```text
User Question
      |
      v
Semantic Search
      |
      v
Top-K Relevant Chunks
      |
      v
Conversation History
      |
      v
RAG Prompt
      |
      v
Groq LLM
      |
      v
Generated Answer
      |
      v
Source Information
```

The retrieved document sources are included with the generated response.

---

# RAG Pipeline

The application implements a structured Retrieval-Augmented Generation pipeline.

## 1. Input Processing

The user submits a question through the Streamlit chat interface.

The question is validated before processing.

## 2. Knowledge Base Loading

The application loads the FAISS vector store from:

```text
data/vectorstore/
```

If the vector store does not exist, the user is prompted to build the knowledge base.

## 3. Document Retrieval

The retriever performs semantic similarity search against the FAISS index.

The default number of retrieved chunks is:

```text
TOP_K = 4
```

## 4. Context Formatting

Retrieved documents are converted into a structured context containing source information.

Example:

```text
[Source 1: document.pdf]

Relevant document content...

[Source 2: document.pdf]

Relevant document content...
```

## 5. Prompt Construction

The RAG prompt combines three components:

```text
Document Context
       +
Conversation History
       +
Current Question
```

## 6. LLM Invocation

The completed prompt is sent to the configured Groq chat model.

## 7. Response Extraction

The system extracts the generated answer and builds source citations using the retrieved document metadata.

---

# Prompt Design

The RAG prompt is designed to keep responses grounded in the retrieved document context.

The prompt instructs the model to:

* Use information from the provided document context.
* Avoid generating unsupported facts.
* Use conversation history to understand follow-up questions.
* Keep responses clear and concise.
* Indicate when the available context does not contain the required information.

If the required information cannot be found, the system responds:

```text
I could not find that information in the provided documents.
```

These rules are defined in:

```text
prompts/rag_prompt.txt
```

---

# Project Structure

```text
rag-knowledge-assistant/
│
├── app.py
├── requirements.txt
├── .env.example
├── .env
│
├── data/
│   ├── documents/
│   │   └── PDF / DOCX files
│   │
│   └── vectorstore/
│       └── FAISS index
│
├── src/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── loaders.py
│   ├── rag_chain.py
│   ├── schemas.py
│   └── vectorstore.py
│
├── prompts/
│   └── rag_prompt.txt
│
└── scripts/
    └── Future automation scripts
```

The project separates the user interface, document processing, embeddings, vector-store management, RAG logic, schemas, and prompt configuration into dedicated components.

---

# Core Components

## `app.py`

Main Streamlit application entry point.

Responsibilities:

* User interface
* Document upload
* Sidebar controls
* Chat interface
* Knowledge-base status
* Conversation history

## `src/embeddings.py`

Responsible for loading and configuring the HuggingFace embedding model.

The embedding model can be changed through:

```text
EMBEDDING_MODEL
```

## `src/loaders.py`

Handles document loading and preprocessing.

Responsibilities:

* PDF loading
* DOCX loading
* Text splitting
* Metadata enrichment
* Source-file counting

## `src/vectorstore.py`

Manages the FAISS vector store lifecycle.

Main functions include:

```text
build_vectorstore()
load_vectorstore()
rebuild_vectorstore()
clear_vectorstore()
get_retriever()
get_chunk_count()
format_docs()
vectorstore_exists()
```

## `src/rag_chain.py`

Contains the core RAG execution logic.

Main functions include:

```text
ask_question()
get_llm()
get_llm_display_name()
load_rag_prompt()
format_chat_history()
extract_sources()
```

## `src/schemas.py`

Defines structured response objects:

```text
SourceCitation
RAGResponse
```

## `prompts/rag_prompt.txt`

Contains the RAG prompt template controlling:

* Document context usage
* Conversation history
* Response behavior
* Source citation rules

---

# Supported File Types

| Extension | Loader           | Metadata                 |
| --------- | ---------------- | ------------------------ |
| `.pdf`    | `PyPDFLoader`    | Filename and page number |
| `.docx`   | `Docx2txtLoader` | Filename                 |

### PDF Processing

PDF documents are loaded page-by-page and include page metadata for source references.

### DOCX Processing

DOCX documents are loaded as continuous text. Page-level metadata is generally unavailable.

All supported documents contain source and filename metadata used for citation purposes.

---

# Knowledge Base Management

The application provides four knowledge-base operations.

| Action                     | Description                                      |
| -------------------------- | ------------------------------------------------ |
| **Build Knowledge Base**   | Creates a FAISS index from the current documents |
| **Rebuild Knowledge Base** | Removes the existing index and creates a new one |
| **Clear Vector Store**     | Removes the existing FAISS index                 |
| **Clear Chat**             | Clears the current conversation history          |

A knowledge base must be built again after the vector store has been cleared.

---

# Troubleshooting

| Problem                      | Possible Cause                           | Solution                                                             |
| ---------------------------- | ---------------------------------------- | -------------------------------------------------------------------- |
| No knowledge base found      | No documents available                   | Add PDF/DOCX files and build the knowledge base                      |
| `GROQ_API_KEY` missing       | Environment variable not configured      | Add the API key to `.env`                                            |
| Retrieval returns no results | FAISS index may be outdated or corrupted | Rebuild the knowledge base                                           |
| Poor retrieval quality       | Chunking or retrieval configuration      | Adjust `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `TOP_K`                     |
| Embedding creation fails     | Network/model configuration issue        | Check internet access and embedding model name                       |
| DOCX has no page numbers     | DOCX loader limitation                   | Page-level metadata is available for PDFs                            |
| First run is slow            | HuggingFace model download               | Wait for the initial model download; later runs use the cached model |

These troubleshooting cases are based on the project's documented failure conditions and configuration behavior.

---

# Development

The application follows a modular structure where each major component has a separate responsibility.

## Modify the RAG Prompt

Edit:

```text
prompts/rag_prompt.txt
```

Use this when changing:

* Answer instructions
* Context rules
* Citation behavior
* Response format

## Adjust Chunking

Modify:

```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

## Change Embedding Model

Modify:

```env
EMBEDDING_MODEL=your-model-name
```

or update:

```text
src/embeddings.py
```

## Add New File Types

Extend:

```text
SUPPORTED_EXTENSIONS
```

in:

```text
src/loaders.py
```

and add the corresponding document loader.

---

# Development Workflow

```bash
# Install dependencies
pip install -r requirements.txt

# Start the application
streamlit run app.py
```

During development:

```text
Modify source files
       |
       v
Restart application
       |
       v
Test changes
```

For document-related changes:

```text
Add / Update Documents
       |
       v
Rebuild Knowledge Base
       |
       v
Test Retrieval
       |
       v
Test Generated Responses
```

---

# Requirements

## System Requirements

| Requirement      | Specification                                                    |
| ---------------- | ---------------------------------------------------------------- |
| Operating System | Windows, macOS, Linux                                            |
| Python           | 3.10+                                                            |
| RAM              | 4 GB minimum                                                     |
| Recommended RAM  | 8 GB for larger document collections                             |
| Disk Space       | Approximately 1 GB+, depending on dependencies and document size |

The documented project requirements specify Python 3.10+, 4 GB minimum RAM, and approximately 1 GB of disk space, with higher RAM recommended for larger document collections.

## API Requirements

### Groq

A Groq API key is required:

```env
GROQ_API_KEY=your_api_key
```

### HuggingFace

The configured HuggingFace embedding model is downloaded automatically during the first run. No HuggingFace API key is required for the documented embedding workflow.

---

# Limitations

The current implementation has several practical limitations:

* Supported document formats are currently PDF and DOCX.
* Retrieval quality depends on the embedding model and chunking configuration.
* DOCX documents generally do not provide page-level source information.
* The quality of generated responses depends on the retrieved context.
* The application requires a Groq API key for LLM-based response generation.
* Embedding models may require an initial internet connection for download.
* The system should not be expected to answer questions that are not supported by the indexed documents.

---

# License

This project is intended for educational and personal use.

Refer to the repository license for the applicable licensing terms.

---

# Contact and Support

For questions, issues, or contributions, refer to the project repository and issue tracker.
