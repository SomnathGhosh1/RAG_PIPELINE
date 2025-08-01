!pip install llama-cloud-services llama-index-core llama-index-readers-file python-dotenv chromadb langchain-community sentence-transformers
!pip install nest_asyncio
# --- 1. Parse Document using LlamaParse ---
from llama_cloud_services import LlamaParse
from google.colab import userdata
import nest_asyncio

nest_asyncio.apply()

parser = LlamaParse(api_key=userdata.get('LLAMAPARSE'), result_type="markdown", premium_mode=True)

file_url = "https://raw.githubusercontent.com/sahajsoft/rag-workshop/main/Documents/September%202023%20Report.pdf"

parsed_docs = parser.load_data(file_url)

print(parsed_docs[5].text)
# --- 2. Chunking the Document ---
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.schema import Document
#Document is array of data in document converted to langchain form of document
docs = [Document(page_content=doc.text) for doc in parsed_docs]

#split document when you see #,##,###
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
    ("#", "Heading1"),
    ("##", "Heading2"),
    ("###", "Heading3")
])

#shows each page as different array
#d=markdown_splitter.split_text(docs[0].page_content)
#print(d)
#print(d[0].page_content)

#store the chunks
chunks = []
for doc in docs:
    chunks.extend(markdown_splitter.split_text(doc.page_content))

print(f"Number of markdown chunks: {len(chunks)}\n")
print(chunks[12].page_content)
print(chunks)
# --- 3. Generate Embeddings & Create Chroma Vector DB ---
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
#vector generate like 384 dim - mini Lm model

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#persist_directory="db" sores here in db
#gives standard component for llm etc-langchain
vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="db")
vector_store.persist()
#persistent is data is stored in disk and saved vector stored in memory to disk so that if power goes or crash it's still saved
#chroma uses sqllite to store metadeta
#Shows 384 dim of word water
word='water'
em=embedding_model.embed_query(word)
print(em)
print(len(em))
all_docs = vector_store.get()["documents"]


for i, doc in enumerate(all_docs[:5]):
    print(f"\n--- Document {i+1} ---")
    print(doc[:500])

#Metadata
all_metas = vector_store.get()["metadatas"]

for i in range(min(5, len(all_metas))):
    print(f"\n--- Metadata for Doc {i+1} ---")
    print(all_metas[i])

#{'Heading1': 'WEEKLY OUTBREAK REPORT'} is heading 1 is meta data stored by chromadatabase
# @title
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#k= number of chunks we need the answer to be
#query = "How many instances of chickenpox were observed"
query = "Which detergent was used to clean the classrooms in the village attayampatti ?"
relevant_chunks = retriever.get_relevant_documents(query) #converts question to vector and check similarity from it's data base

for i, chunk in enumerate(relevant_chunks):
    print(f"\nChunk {i+1}:")
    print(chunk.page_content)
!pip install huggingface-hub llama-cpp-python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="prithivMLmods/Llama-Sentient-3.2-3B-Instruct-GGUF",
    filename="Llama-Sentient-3.2-3B-Instruct.Q5_K_M.gguf",
    verbose=False
)

question = "Which detergent was used to clean the classrooms in the village attayampatti ?"

output = llm(
      prompt = f"""
      You are a helpful assistant. Use the context below to answer the question.

      Context:
      {relevant_chunks[:1]}

      Question:
      {question}

      Answer:
      """,
      max_tokens=32,
      stop=["Q:", "\n"],
)
print(output)
#pass chunks to llm then get the answer back like chatgpt  ,stop to stop at seeing the symbols   you are....is called role

from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="prithivMLmods/Llama-Sentient-3.2-3B-Instruct-GGUF",
    filename="Llama-Sentient-3.2-3B-Instruct.Q5_K_M.gguf",
    verbose=False
)

question = "Which detergent was used to clean the classrooms in the village attayampatti ?"

# Reduce the number of relevant chunks to 1
# or select the most relevant chunk based on similarity score or other criteria
relevant_chunk = relevant_chunks[0]  # Assuming the first chunk is the most relevant

output = llm(
      prompt = f"""
      You are a helpful assistant. Use the context below to answer the question.

      Context:
      {relevant_chunk[:1]} # Only include the content of the most relevant chunk

      Question:
      {question}

      Answer:
      """,
      max_tokens=32,
      stop=["Q:", "\n"],
)
print(output)
