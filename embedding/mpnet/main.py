import sys
import os
import time
import psutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from embedding.mpnet.mpnet import setup_redis, load_model, index_documents, query_documents
from ollama_client import generate_rag_response

def measure_performance(func, *args, **kwargs):
    """
    Measure performance of a function
    
    Args:
        func: Function to measure
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        Dictionary with performance metrics and function result
    """
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Measure time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    # Get final memory usage
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate metrics
    metrics = {
        "execution_time": end_time - start_time,
        "memory_usage": mem_after - mem_before,
        "total_memory": mem_after,
        "result": result
    }
    
    return metrics

def main():
    sample_documents = [
        {
            "content": "Redis is an open source, in-memory data structure store, used as a database, cache, and message broker. Redis provides data structures such as strings, hashes, lists, sets, sorted sets with range queries, bitmaps, hyperloglogs, geospatial indexes, and streams.",
            "metadata": "source: redis_documentation.txt"
        },
        {
            "content": "Vector databases are specialized database systems designed to store and query vector embeddings efficiently. They are optimized for similarity search operations such as k-nearest neighbors (KNN) and are commonly used in machine learning applications.",
            "metadata": "source: vector_databases.txt"
        },
        {
            "content": "MPNet is a pre-trained language model that combines the advantages of both BERT and XLNet. It uses a permuted language modeling objective and incorporates position information, making it effective for various NLP tasks including text classification, question answering, and sequence labeling.",
            "metadata": "source: mpnet_documentation.txt"
        }
    ]

    # Set up Redis client
    redis_client = setup_redis()
    
    # Load the model
    model = load_model()
    
    # Index the documents
    doc_keys = index_documents(sample_documents, redis_client, model)
    if not doc_keys:
        print("Failed to index documents. Exiting.")
        return
    
    # Query the documents
    query = "How does Redis support vector search?"
    print(f"\nQuery: {query}")
    results = query_documents(query, redis_client, model)
    
    # Print results
    print("\nResults:")
    if not results:
        print("No results found.")
    else:
        for i, result in enumerate(results):
            print(f"{i+1}. {result['id']} (Score: {result['score']:.4f})")
            print(f"   Content: {result['content'][:100]}...")
            print(f"   Metadata: {result['metadata']}")
            print()
    
    # Generate RAG response using Ollama
    print("\nGenerating RAG response using Ollama...")
    rag_response = generate_rag_response(
        query,
        redis_client,
        model,
        query_documents,
        llm_model="llama2" # MODIFY THIS LINE TO CHANGE WHAT LLM MODEL IS BEING USED
    )
    
    print("\nRAG Response:")
    print(f"Query: {rag_response['query']}")
    print(f"LLM Model: {rag_response['model']}")
    print(f"Response:\n{rag_response['response']}")

if __name__ == "__main__":
    main()