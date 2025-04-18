import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from embedding.mpnet.database import get_vector_db, load_model
from embedding.mpnet.ollama_client import ingest_documents, measure_performance, generate_rag_response

def main():
    # Configuration
    db_type = "faiss"  # Options: "redis", "chroma", "faiss"
    index_name = "document_index"
    llm_model = "mistral"  # Options: "llama2", "mistral"
    
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

        ingest_documents(vector_db, model, "./data")
        
        # Query the database
        while True:
            query = input("\nEnter your search query: ")
            if query == "exit":
                break
        
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
            print(f"Response:\n{rag_response['response']}\n")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up
        if 'vector_db' in locals():
            vector_db.close()


if __name__ == "__main__":
    # To use a different vector database, modify the db_type variable in main()
    main()