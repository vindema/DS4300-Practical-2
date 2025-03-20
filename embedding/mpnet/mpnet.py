import numpy as np
import time
from redis import Redis
import redis
from sentence_transformers import SentenceTransformer
from redis.commands.search.field import TextField, TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition

def setup_redis(host='localhost', port=6379, db=0):
    """
    Set up Redis client connection
    
    Args:
        host (str): Redis host
        port (int): Redis port
        db (int): Redis database number
        
    Returns:
        Redis client
    """
    return Redis(host=host, port=port, db=db)

def load_model(model_name="all-mpnet-base-v2"):
    """
    Load the SentenceTransformers model
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        SentenceTransformer model
    """
    start_time = time.time()
    model = SentenceTransformer(model_name)
    duration = time.time() - start_time
    print(f"Loaded model in {duration:.2f} seconds.")
    return model

def index_documents(documents, client: redis.StrictRedis, model, index_name="document_index", vector_field="embedding", vector_dimensions=768):
    """
    Index documents in Redis with vector embeddings.
    
    Args:
        documents (list): List of document dictionaries with 'content' and 'metadata' fields
        client: Redis client
        model: The embedding model
        index_name (str): Name of the Redis index
        vector_field (str): Name of the vector field in Redis
        vector_dimensions (int): Dimensions of the vector embedding
        
    Returns:
        list: List of document keys in Redis
    """
    # Check if index already exists and delete if it does
    try:
        client.ft(index_name).info()
        client.ft(index_name).dropindex(delete_documents=True)
        print(f"Dropped existing index: {index_name}")
    except:
        print(f"No existing index found: {index_name}")
    
    # Create index with appropriate schema for vector search
    try:
        # Create the index
        client.ft(index_name).create_index(
            [
                TextField("content"),
                TagField("metadata"),
                VectorField(vector_field, 
                            "FLAT", 
                            {
                                "TYPE": "FLOAT32",
                                "DIM": vector_dimensions,
                                "DISTANCE_METRIC": "COSINE"
                            })
            ],
            definition=IndexDefinition(prefix=["doc:"])
        )
        print(f"Created index: {index_name}")
    except Exception as e:
        print(f"Error creating index: {e}")
        return []
    
    # Index documents
    doc_keys = []
    start_time = time.time()
    
    for i, doc in enumerate(documents):
        # Generate document key
        doc_key = f"doc:{i}"
        doc_keys.append(doc_key)
        
        # Generate embedding
        content = doc.get("content", "")
        embedding = model.encode(content).tolist()
        
        # Convert to bytes for Redis
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        
        # Index document in Redis
        client.hset(
            doc_key,
            mapping={
                "content": content,
                "metadata": doc.get("metadata", ""),
                vector_field: embedding_bytes
            }
        )
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Indexed {len(documents)} chunks in {duration:.2f} seconds.")
    
    return doc_keys

def query_documents(query, client, model, top_k=3, index_name="document_index"):
    """
    Query the Redis database for documents similar to the input query.
    
    Args:
        query (str): The query string
        client: Redis client
        model: The embedding model
        top_k (int): Number of results to return
        index_name (str): Name of the Redis index
        
    Returns:
        list: Top k most similar documents
    """
    import numpy as np
    import redis
    
    # Print Redis and redis-py version information for debugging
    info = client.info()
    # redis_version = info.get("redis_version", "unknown")
    # print(f"Redis server version: {redis_version}")
    # print(f"Redis-py version: {redis.__version__}")
    
    print(f"Searching for: {query}")
    
    # Generate embedding for the query
    query_embedding = model.encode(query).tolist()
    
    # Convert embedding to a Redis vector
    redis_query = np.array(query_embedding, dtype=np.float32).tobytes()
    
    # Try multiple query formats based on Redis version
    formatted_results = []
    
    try:
        # Basic query that should work with RediSearch 2.2+
        query_str = f"*"
        params = {
            "vector": redis_query,
            "num": top_k,
        }
        
        results = client.ft(index_name).search(query_str, params)
        
        # Format results
        for doc in results.docs:
            formatted_results.append({
                "id": doc.id,
                "content": getattr(doc, "content", "No content available"),
                "metadata": getattr(doc, "metadata", "No metadata available"),
                "score": float(getattr(doc, "score", 0.0))
            })
    except Exception as e:
        print(f"Basic vector query failed: {e}")
        
        # Fallback to basic text search as last resort
        try:
            print("Attempting direct content search...")
            # Find documents that might contain keywords from the query
            query_words = query.lower().split()
            matching_docs = []
            
            # Scan through all documents with the doc: prefix
            for key in client.scan_iter(match="doc:*"):
                doc_data = client.hgetall(key)
                
                # Convert byte data to strings
                content = doc_data.get(b"content", b"").decode("utf-8")
                metadata = doc_data.get(b"metadata", b"").decode("utf-8")
                
                # Check if any query word is in the content
                if any(word in content.lower() for word in query_words):
                    matching_docs.append({
                        "id": key.decode("utf-8"),
                        "content": content,
                        "metadata": metadata,
                        "score": 1.0  # Default score for text match
                    })
            
            # Sort by basic relevance (just counts occurrences of query words)
            for doc in matching_docs:
                score = sum(doc["content"].lower().count(word) for word in query_words)
                doc["score"] = score / len(query_words)
            
            formatted_results = sorted(matching_docs, key=lambda x: x["score"], reverse=True)[:top_k]
            
        except Exception as e:
            print(f"Fallback text search failed: {e}")
    
    return formatted_results