import redis
import json
import numpy as np
import fitz  # PyMuPDF for PDF text extraction
from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField
import os

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)

# Constants
VECTOR_DIM = 384  # Dimension of the embeddings for all-MiniLM-L6-v2
INDEX_NAME = "embedding_index"  # Name of the Redis index
DOC_PREFIX = "doc:"  # Prefix for document keys in Redis
DISTANCE_METRIC = "COSINE"  # Distance metric for vector similarity
data_folder = "c:/Users/hifoo/OneDrive/Documents/Spring25/DS 4300/practical 2/DS4300-Practical-2/data/"  # Path to the folder containing PDFs

# Function to get sentence embeddings
def get_embedding(text: str) -> list:
    # Encode the text into a list of embeddings
    return embedding_model.encode(text).tolist()

# Create Redis index
def create_index():
    try:
        # Drop the existing index if it exists
        redis_client.ft(INDEX_NAME).dropindex(delete_documents=False)
        print("Existing index dropped")  # Debug print
    except Exception as e:
        print(f"Index drop error (if it didn't exist, this is expected): {e}")

    try:
        # Create a new index with the specified fields
        redis_client.ft(INDEX_NAME).create_index(
            [
                TextField("file"),  # Text field for the file name
                TextField("page"),  # Text field for the page number
                TextField("chunk"),  # Text field for the chunk number
                VectorField("embedding", "FLAT", {
                    "TYPE": "FLOAT32",
                    "DIM": VECTOR_DIM,
                    "DISTANCE_METRIC": DISTANCE_METRIC
                })  # Vector field for the embeddings
            ]
        )
        print("Index created successfully")  # Debug print
    except Exception as e:
        print(f"Index creation error: {e}")

# Search embeddings using Redis
def search_embeddings(query, top_k=3):
    # Get the embedding for the query text
    query_embedding = get_embedding(query)
    # Convert the embedding to a byte array
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        # Create a query to search for the top K nearest neighbors
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")  # Sort by vector distance
            .return_fields("file", "page", "chunk", "vector_distance")  # Return specified fields
            .dialect(2)
        )

        print("Executing search query...")  # Debug print
        # Execute the search query
        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        print(f"Number of results: {len(results.docs)}")  # Debug print

        # Extract the top results
        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

        # Print the top results
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []

# Extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # Open the PDF file
    text_by_page = []
    for page_num, page in enumerate(doc):
        text = page.get_text()  # Extract text from the page
        # print(f"Page {page_num} text: {text[:10]}")  # Print the first 500 characters of each page
        text_by_page.append((page_num, text))  # Append the text to the list
    return text_by_page

# Split text into chunks
def split_text_into_chunks(text, chunk_size=100, overlap=30):
    words = text.split()  # Split the text into words
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])  # Create a chunk of text
        chunks.append(chunk)  # Append the chunk to the list
    return chunks

# Store embeddings in Redis
def store_embedding(file, page, chunk, embedding):
    key = f"{DOC_PREFIX}{file}_page_{page}_chunk_{chunk}"  # Create a unique key for the chunk
    # print(f"Storing embedding for key: {key}")  # Debug print
    # Store the embedding in Redis
    redis_client.hset(
        key,
        mapping={
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
            "file": file,
            "page": page,
            "chunk": chunk,
        },
    )

# Process PDFs and store embeddings
def process_pdfs(data_folder):
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".pdf"):  # Check if the file is a PDF
            pdf_path = os.path.join(data_folder, file_name)  # Get the full path to the PDF file
            text_by_page = extract_text_from_pdf(pdf_path)  # Extract text from the PDF
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)  # Split the text into chunks
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)  # Get the embedding for the chunk
                    store_embedding(file_name, str(page_num), str(chunk_index), embedding)  # Store the embedding in Redis
            print(f"Processed: {file_name}")  # Debug print

# Main interactive search loop
def interactive_search():
    print("RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")  # Get the search query from the user

        if query.lower() == "exit":
            break

        context_results = search_embeddings(query)  # Search for the query
        print("\n--- Retrieved Chunks ---")
        for result in context_results:
            file_path = os.path.join(data_folder, result['file'])  # Get the full path to the PDF file
            text_by_page = extract_text_from_pdf(file_path)  # Extract text from the PDF
            page_num = int(result['page'])
            chunk_index = int(result['chunk'])
            chunks = split_text_into_chunks(text_by_page[page_num][1])  # Split the text into chunks
            print(f"Chunk index: {chunk_index}, Number of chunks: {len(chunks)}")  # Debug print
            if chunk_index < len(chunks):
                print(f"Answer: {chunks[chunk_index]} (Similarity: {result['similarity']})")  # Print the answer
            else:
                print(f"Invalid chunk index: {chunk_index} for file: {result['file']}, page: {result['page']}")  # Handle invalid chunk index
            
if __name__ == "__main__":
    create_index()  # Create the index in Redis
    process_pdfs(data_folder)  # Process the PDFs and store embeddings
    interactive_search()  # Start the interactive search loop


