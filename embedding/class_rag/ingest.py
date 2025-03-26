import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
import faiss
import chromadb
from chromadb.utils import embedding_functions

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# FAISS Index
faiss_index = None

# ChromaDB Client and Collection
chroma_client = chromadb.Client()
chroma_collection = None

# used to clear the redis vector store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")

# Create an HNSW index in Redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")

# Create index in FAISS
def create_faiss_index():
    global faiss_index
    faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
    print("FAISS index created successfully.")

# Create index in Chroma
def create_chroma_collection():
    global chroma_collection
    chroma_collection = chroma_client.create_collection(name="document_embeddings")
    print("ChromaDB collection created successfully.")


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# store the embedding in Redis
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {chunk}")

# Store embedding in FAISS
def store_embedding_faiss(file: str, page: str, chunk: str, embedding: list):
    global faiss_index
    vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
    faiss_index.add(vector)
    print(f"Stored embedding in FAISS for: {chunk}")

# Store embedding in Chroma
def store_embedding_chroma(file: str, page: str, chunk: str, embedding: list):
    global chroma_collection
    chroma_collection.add(
        documents=[chunk],
        metadatas=[{"file": file, "page": page}],
        embeddings=[embedding],
        ids=[f"{file}_page_{page}_chunk_{chunk}"]
    )
    print(f"Stored embedding in ChromaDB for: {chunk}")

# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

# Process all PDF files in a given directory
def process_pdfs(data_folder):
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)

                    # Store embeddings in all vector stores
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
                    store_embedding_faiss(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
                    store_embedding_chroma(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")

def query_redis(query_text: str):
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    query_text = "Efficient search in vector databases"
    embedding = get_embedding(query_text)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    # print(res.docs)

    for doc in res.docs:
        print(f"{doc.id} \n ----> {doc.vector_distance}\n")

def query_faiss(query_text: str, k=5):
    global faiss_index
    embedding = get_embedding(query_text)
    query_vector = np.array(embedding, dtype=np.float32).reshape(1, -1)

    D, I = faiss_index.search(query_vector, k)
    print(f"FAISS Results:\nIndices: {I}\nDistances: {D}")

def query_chroma(query_text: str, k=5):
    global chroma_collection
    embedding = get_embedding(query_text)
    results = chroma_collection.query(
        query_embeddings=[embedding],
        n_results=k
    )
    print("ChromaDB Results:")
    for i, result in enumerate(results["documents"]):
        print(f"{i+1}. {result[0]} -> Distance: {results['distances'][0][i]}")


def main():
    clear_redis_store()
    create_hnsw_index()
    create_faiss_index()
    create_chroma_collection()

    process_pdfs(data_folder)
    print("\n---Done processing PDFs---\n")

    # Query Redis, FAISS, and ChromaDB
    query_redis("What is the capital of France?")
    query_faiss("What is the capital of France?")
    query_chroma("What is the capital of France?")



if __name__ == "__main__":
    main()
