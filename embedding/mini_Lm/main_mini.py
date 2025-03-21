import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from database_mini import get_vector_db, load_model
from ollama_mini import ingest_documents, measure_performance, generate_rag_response

def interactive_search(vector_db, model, llm_model, index_name):
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")  # Get the search query from the user

        if query.lower() == "exit":
            break

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

def main():
    # Configuration
    db_type = "redis"  # Options: "redis", "chroma", "faiss"
    index_name = "document_index"
    llm_model = "llama2"  # Change to modify the LLM model
    
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
        vector_dimensions = 384  # miniLM embedding dimension
        vector_db.create_index(index_name, vector_dimensions)

        # Ingest documents
        ingest_documents(vector_db, model, "./data")
        
        # Start interactive search loop
        interactive_search(vector_db, model, llm_model, index_name)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up
        if 'vector_db' in locals():
            vector_db.close()


if __name__ == "__main__":
    # To use a different vector database, modify the db_type variable in main()
    main()