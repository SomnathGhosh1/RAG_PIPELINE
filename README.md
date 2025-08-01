# RAG_PIPELINE
RAG pipeline using LlamaParse, LangChain, Chroma, and Llama.cpp to parse PDFs, generate embeddings, retrieve relevant chunks, and answer questions locally using the LLaMA-Sentient-3B GGUF model.
🦙 RAG Pipeline using LlamaParse, LangChain & Llama.cpp
This project demonstrates a full Retrieval-Augmented Generation (RAG) pipeline for querying information from a PDF document using open-source tools like LlamaParse, LangChain, and a local LLaMA.cpp model.

🚀 Features
✅ PDF Parsing using LlamaParse API for accurate Markdown extraction

✅ Markdown Chunking via LangChain's MarkdownHeaderTextSplitter

✅ Semantic Embeddings using HuggingFace all-MiniLM-L6-v2 model

✅ Vector Store creation with Chroma for persistent similarity search

✅ Local LLM Inference using llama-cpp-python and Llama-Sentient-3.2-3B GGUF model

✅ Query Answering using semantic retrieval + prompt-injection-based answering

🔧 APIs and Models Used
LlamaParse API – for parsing PDFs to clean Markdown

all-MiniLM-L6-v2 (HuggingFace) – for generating embeddings

Chroma (LangChain Vector Store) – for similarity search

llama-cpp-python – for local model inference

LLaMA Model Used: prithivMLmods/Llama-Sentient-3.2-3B-Instruct.Q5_K_M.gguf

💡 How it works
Parse PDF → Extracts clean Markdown using LlamaParse

Chunk Document → Splits into hierarchical sections by markdown headings

Embed & Store → Generates vector embeddings and stores them in Chroma

Retrieve → Finds top-k relevant chunks for a given question

Answer → Uses a local LLaMA model to generate an answer from the chunk

📦 Installation
bash
Copy
Edit
pip install huggingface-hub langchain chromadb llama-cpp-python
⚠️ Make sure to set your LLAMAPARSE API key in your environment (e.g., via userdata.get() in Colab).

📄 Sample Query
Question: "Which detergent was used to clean the classrooms in the village Attayampatti?"
Answer: Retrieved from most relevant markdown chunk and answered using LLaMA-Sentient 3B locally.
