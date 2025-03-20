import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the vector database abstraction
from embedding.mpnet.database import get_vector_db, load_model
from embedding.mpnet.ollama_client import measure_performance, generate_rag_response

def main():
    # Sample documents
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
    
    # Configuration
    db_type = "faiss"  # Options: "redis", "chroma", "faiss"
    index_name = "document_index"
    llm_model = "mistral"  # Change to modify the LLM model
    
    # Get vector database instance based on type
    db_config = {}
    if db_type == "redis":
        db_config = {"host": "localhost", "port": 6379, "db": 0}
    elif db_type == "chroma":
        db_config = {"persist_directory": "./chroma_db"}
    
    try:
        # Initialize vector database
        vector_db = get_vector_db(db_type, **db_config)
        print(f"Successfully initialized {db_type} vector database")
        
        # Load embedding model
        model = load_model()
        
        # Create index/collection
        vector_dimensions = 768  # MPNet embedding dimension
        vector_db.create_index(index_name, vector_dimensions)
        
        # Index documents
        doc_keys = vector_db.index_documents(sample_documents, index_name, model=model)
        
        if not doc_keys:
            print("Failed to index documents. Exiting.")
            return
        
        # Query the database
        query = "How does Redis support vector search?"
        print(f"\nQuery: {query}")
        
        # Generate query vector
        query_vector = model.encode(query).tolist()
        
        # Search for similar documents
        results = vector_db.query(query_vector, index_name)
        
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
            vector_db,
            model,
            llm_model=llm_model,
            index_name=index_name
        )
        
        print("\nRAG Response:")
        print(f"Query: {rag_response['query']}")
        print(f"LLM Model: {rag_response['model']}")
        print(f"Response:\n{rag_response['response']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up
        if 'vector_db' in locals():
            vector_db.close()


if __name__ == "__main__":
    # To use a different vector database, modify the db_type variable in main()
    main()