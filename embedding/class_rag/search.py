import redis
import faiss
import chromadb
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# Initialize models
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)

# FAISS Index
faiss_index = faiss.IndexFlatL2(VECTOR_DIM)

# ChromaDB Client and Collection
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(name="document_embeddings")

# def cosine_similarity(vec1, vec2):
#     """Calculate cosine similarity between two vectors."""
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def search_embeddings(query, top_k=3):
    """Search across Redis, FAISS, and ChromaDB for similar embeddings."""
    query_embedding = get_embedding(query)

    # Search Redis
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
    try:
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )
        redis_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": float(result.vector_distance),
            }
            for result in results.docs
        ][:top_k]
    except Exception as e:
        print(f"Redis search error: {e}")
        redis_results = []

    # Search FAISS
    faiss_results = search_faiss(query_embedding, top_k)

    # Search ChromaDB
    chroma_results = search_chroma(query_embedding, top_k)

    return redis_results


def search_faiss(query_embedding, top_k=3):
    """Search FAISS for similar embeddings."""
    query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

    if faiss_index.ntotal == 0:
        print("FAISS index is empty. No results found.")
        return []

    D, I = faiss_index.search(query_vector, top_k)
    results = []
    for i, index in enumerate(I[0]):
        if index == -1:
            continue
        results.append(
            {
                "file": f"faiss_file_{index}",
                "page": str(index),
                "chunk": f"chunk_{index}",
                "similarity": float(D[0][i]),
            }
        )
    return results

def search_chroma(query_embedding, top_k=3):
    """Search ChromaDB for similar embeddings."""
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    
    # Check if results or metadatas are empty
    if not results["documents"] or not results["metadatas"]:
        return []

    chroma_results = []
    for i, doc in enumerate(results["documents"]):
        # Check if metadata exists for the index
        if i >= len(results["metadatas"][0]):
            continue

        chroma_results.append(
            {
                "file": results["metadatas"][0][i].get("file", "Unknown file"),
                "page": results["metadatas"][0][i].get("page", "Unknown page"),
                "chunk": doc[0],
                "similarity": results["distances"][0][i],
            }
        )
    return chroma_results

def generate_rag_response(query, context_results):
    """Generate a RAG response using results from multiple vector stores."""
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.4f}"
            for result in context_results
        ]
    )

    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant.
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = search_embeddings(query)

        # Generate RAG response
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)


# def store_embedding(file, page, chunk, embedding):
#     """
#     Store an embedding in Redis using a hash with vector field.

#     Args:
#         file (str): Source file name
#         page (str): Page number
#         chunk (str): Chunk index
#         embedding (list): Embedding vector
#     """
#     key = f"{file}_page_{page}_chunk_{chunk}"
#     redis_client.hset(
#         key,
#         mapping={
#             "embedding": np.array(embedding, dtype=np.float32).tobytes(),
#             "file": file,
#             "page": page,
#             "chunk": chunk,
#         },
#     )


if __name__ == "__main__":
    interactive_search()
