import numpy as np
import time
import abc
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer

class VectorDatabase(abc.ABC):
    """
    Abstract base class for vector database implementations
    """
    
    @abc.abstractmethod
    def setup(self, **kwargs):
        """Initialize the database connection with optional parameters"""
        pass
    
    @abc.abstractmethod
    def create_index(self, index_name: str, vector_dimensions: int, **kwargs):
        """Create a new index or collection for storing vectors"""
        pass
    
    @abc.abstractmethod
    def index_documents(self, documents: List[Dict[str, Any]], index_name: str, **kwargs) -> List[str]:
        """
        Index documents into the database
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata' fields
            index_name: Name of the index
            **kwargs: Additional parameters
            
        Returns:
            List of document IDs
        """
        pass
    
    @abc.abstractmethod
    def query(self, query_vector: List[float], index_name: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the database for similar vectors
        
        Args:
            query_vector: The query vector
            index_name: Name of the index
            top_k: Number of results to return
            
        Returns:
            List of results
        """
        pass
    
    @abc.abstractmethod
    def delete_index(self, index_name: str):
        """Delete an index"""
        pass
    
    @abc.abstractmethod
    def close(self):
        """Close the database connection"""
        pass


class RedisVectorDB(VectorDatabase):
    """
    Redis vector database implementation
    """
    
    def setup(self, host='localhost', port=6379, db=0):
        """
        Set up Redis client connection
        
        Args:
            host (str): Redis host
            port (int): Redis port
            db (int): Redis database number
        """
        from redis import Redis
        self.client = Redis(host=host, port=port, db=db)
        return self
    
    def create_index(self, index_name: str, vector_dimensions: int, prefix="doc:", distance_metric="COSINE"):
        """
        Create a new index in Redis
        
        Args:
            index_name: Name of the index
            vector_dimensions: Dimensions of the vectors
            prefix: Prefix for document keys
            distance_metric: Distance metric for vector similarity
        """
        from redis.commands.search.field import TextField, TagField, VectorField
        from redis.commands.search.indexDefinition import IndexDefinition
        
        try:
            self.client.ft(index_name).info()
            self.client.ft(index_name).dropindex(delete_documents=True)
            print(f"Dropped existing index: {index_name}")
        except:
            print(f"No existing index found: {index_name}")
        
        try:
            self.client.ft(index_name).create_index(
                [
                    TextField("content"),
                    TagField("metadata"),
                    VectorField("embedding", 
                                "FLAT", 
                                {
                                    "TYPE": "FLOAT32",
                                    "DIM": vector_dimensions,
                                    "DISTANCE_METRIC": distance_metric
                                })
                ],
                definition=IndexDefinition(prefix=[prefix])
            )
            print(f"Created index: {index_name}")
            self.prefix = prefix
        except Exception as e:
            print(f"Error creating index: {e}")
            raise e
    
    def index_documents(self, documents: List[Dict[str, Any]], index_name: str, vector_field="embedding", model=None) -> List[str]:
        """
        Index documents in Redis with vector embeddings
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata' fields
            index_name: Name of the index
            vector_field: Name of the vector field
            model: The embedding model to use (optional if embeddings are provided)
            
        Returns:
            List of document IDs
        """
        doc_keys = []
        start_time = time.time()
        
        for i, doc in enumerate(documents):
            # Generate document key
            doc_key = f"{self.prefix}{i}"
            doc_keys.append(doc_key)
            
            # Get or generate embedding
            if "embedding" in doc:
                embedding = doc["embedding"]
            elif model:
                content = doc.get("content", "")
                embedding = model.encode(content).tolist()
            else:
                raise ValueError("Either document must include embedding or model must be provided")
            
            # Convert to bytes for Redis
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            
            # Index document in Redis
            self.client.hset(
                doc_key,
                mapping={
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", ""),
                    vector_field: embedding_bytes
                }
            )
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Indexed {len(documents)} chunks in {duration:.2f} seconds.")
        
        return doc_keys
    
    def query(self, query_vector: List[float], index_name: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the Redis database for similar vectors
        
        Args:
            query_vector: The query vector
            index_name: Name of the index
            top_k: Number of results to return
            
        Returns:
            List of results
        """
        # Convert embedding to a Redis vector
        redis_query = np.array(query_vector, dtype=np.float32).tobytes()
        
        formatted_results = []
        
        try:
            # Basic query that should work with RediSearch 2.2+
            query_str = "*"
            params = {
                "vector": redis_query,
                "num": top_k,
            }
            
            results = self.client.ft(index_name).search(query_str, params)
            
            # Format results
            for doc in results.docs:
                formatted_results.append({
                    "id": doc.id,
                    "content": getattr(doc, "content", "No content available"),
                    "metadata": getattr(doc, "metadata", "No metadata available"),
                    "score": float(getattr(doc, "score", 0.0))
                })
        except Exception as e:
            print(f"Vector query failed: {e}")
            # Implement fallback methods if needed
        
        return formatted_results
    
    def delete_index(self, index_name: str):
        """Delete an index in Redis"""
        try:
            self.client.ft(index_name).dropindex(delete_documents=True)
            print(f"Deleted index: {index_name}")
        except Exception as e:
            print(f"Error deleting index: {e}")
    
    def close(self):
        """Close the Redis connection"""
        self.client.close()


class ChromaVectorDB(VectorDatabase):
    """
    Chroma vector database implementation
    """
    
    def setup(self, persist_directory=None, **kwargs):
        """
        Set up Chroma client connection
        
        Args:
            persist_directory: Directory to persist the database
            **kwargs: Additional parameters for Chroma
        """
        try:
            import chromadb
            self.client = chromadb.Client(
                settings=chromadb.config.Settings(
                    persist_directory=persist_directory,
                    **kwargs
                )
            )
            return self
        except ImportError:
            raise ImportError("ChromaDB package not installed. Install with 'pip install chromadb'")
    
    def create_index(self, index_name: str, vector_dimensions: int, **kwargs):
        """
        Create a new collection in ChromaDB
        
        Args:
            index_name: Name of the collection
            vector_dimensions: Not used in ChromaDB but kept for API consistency
            **kwargs: Additional parameters for collection creation
        """
        try:
            # Check if collection exists and get or create it
            try:
                self.collection = self.client.get_collection(name=index_name)
                print(f"Using existing collection: {index_name}")
            except:
                self.collection = self.client.create_collection(
                    name=index_name,
                    metadata={"dimension": vector_dimensions}
                )
                print(f"Created collection: {index_name}")
            
            self.index_name = index_name
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise e
    
    def index_documents(self, documents: List[Dict[str, Any]], index_name: str, model=None) -> List[str]:
        """
        Index documents in ChromaDB
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata' fields
            index_name: Name of the collection (unused, uses the collection from create_index)
            model: The embedding model to use (optional if embeddings are provided)
            
        Returns:
            List of document IDs
        """
        # Ensure collection is set
        if not hasattr(self, 'collection'):
            self.create_index(index_name, 0)
        
        start_time = time.time()
        
        # Prepare data for batch insertion
        ids = []
        documents_content = []
        metadatas = []
        embeddings = []
        
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            ids.append(doc_id)
            documents_content.append(doc.get("content", ""))
            metadatas.append({"source": doc.get("metadata", "")})
            
            # Get or generate embedding
            if "embedding" in doc:
                embeddings.append(doc["embedding"])
            elif model:
                content = doc.get("content", "")
                embedding = model.encode(content).tolist()
                embeddings.append(embedding)
            else:
                # If no embeddings provided, let ChromaDB generate them
                embeddings = None
                break
        
        # Add documents to collection
        self.collection.add(
            ids=ids,
            documents=documents_content,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Indexed {len(documents)} chunks in {duration:.2f} seconds.")
        
        return ids
    
    def query(self, query_vector: List[float], index_name: str = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the ChromaDB for similar vectors
        
        Args:
            query_vector: The query vector
            index_name: Name of the collection (unused, uses the collection from create_index)
            top_k: Number of results to return
            
        Returns:
            List of results
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )
            
            formatted_results = []
            
            # Format results to match the common interface
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]["source"],
                    "score": float(results["distances"][0][i]) if "distances" in results else 0.0
                })
            
            return formatted_results
        except Exception as e:
            print(f"ChromaDB query failed: {e}")
            return []
    
    def delete_index(self, index_name: str):
        """Delete a collection in ChromaDB"""
        try:
            self.client.delete_collection(name=index_name)
            print(f"Deleted collection: {index_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def close(self):
        """Close the ChromaDB connection"""
        pass  # ChromaDB doesn't require explicit closure


class FaissVectorDB(VectorDatabase):
    """
    FAISS vector database implementation
    """
    
    def setup(self, **kwargs):
        """
        Set up FAISS
        
        Args:
            **kwargs: Additional parameters for FAISS
        """
        try:
            import faiss
            self.faiss = faiss
            self.document_store = {}  # Store documents in memory
            return self
        except ImportError:
            raise ImportError("FAISS package not installed. Install with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
    
    def create_index(self, index_name: str, vector_dimensions: int, index_type="L2"):
        """
        Create a new FAISS index
        
        Args:
            index_name: Name of the index
            vector_dimensions: Dimensions of the vectors
            index_type: Type of FAISS index (L2, IP, etc.)
        """
        try:
            if index_type == "L2":
                index = self.faiss.IndexFlatL2(vector_dimensions)
            elif index_type == "IP":
                index = self.faiss.IndexFlatIP(vector_dimensions)
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            self.indexes = getattr(self, 'indexes', {})
            self.indexes[index_name] = index
            self.document_store[index_name] = []
            print(f"Created FAISS index: {index_name}")
        except Exception as e:
            print(f"Error creating FAISS index: {e}")
            raise e
    
    def index_documents(self, documents: List[Dict[str, Any]], index_name: str, model=None) -> List[str]:
        """
        Index documents in FAISS
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata' fields
            index_name: Name of the index
            model: The embedding model to use (optional if embeddings are provided)
            
        Returns:
            List of document IDs
        """
        if index_name not in self.indexes:
            raise ValueError(f"Index {index_name} does not exist")
        
        start_time = time.time()
        doc_ids = []
        
        # Store original documents
        current_id = len(self.document_store[index_name])
        
        # Create a batch of vectors for indexing
        vectors = []
        
        for i, doc in enumerate(documents):
            doc_id = f"doc_{current_id + i}"
            doc_ids.append(doc_id)
            
            # Get or generate embedding
            if "embedding" in doc:
                embedding = doc["embedding"]
            elif model:
                content = doc.get("content", "")
                embedding = model.encode(content).tolist()
            else:
                raise ValueError("Either document must include embedding or model must be provided")
            
            vectors.append(embedding)
            
            # Store the document
            self.document_store[index_name].append({
                "id": doc_id,
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", ""),
                "vector_id": current_id + i
            })
        
        # Convert to numpy array and add to index
        vectors_np = np.array(vectors).astype('float32')
        self.indexes[index_name].add(vectors_np)
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Indexed {len(documents)} chunks in {duration:.2f} seconds.")
        
        return doc_ids
    
    def query(self, query_vector: List[float], index_name: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the FAISS index for similar vectors
        
        Args:
            query_vector: The query vector
            index_name: Name of the index
            top_k: Number of results to return
            
        Returns:
            List of results
        """
        if index_name not in self.indexes:
            raise ValueError(f"Index {index_name} does not exist")
        
        try:
            # Convert to numpy array
            query_np = np.array([query_vector]).astype('float32')
            
            # Query the index
            distances, indices = self.indexes[index_name].search(query_np, top_k)
            
            formatted_results = []
            
            # Format results
            for i, idx in enumerate(indices[0]):
                if idx < len(self.document_store[index_name]) and idx >= 0:
                    doc = self.document_store[index_name][idx]
                    formatted_results.append({
                        "id": doc["id"],
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        # Convert distance to similarity score (1 - normalized distance)
                        "score": float(1.0 - distances[0][i] / (distances[0][0] + 1e-6)) if distances[0][0] > 0 else 0.0
                    })
            
            return formatted_results
        except Exception as e:
            print(f"FAISS query failed: {e}")
            return []
    
    def delete_index(self, index_name: str):
        """Delete a FAISS index"""
        if index_name in self.indexes:
            del self.indexes[index_name]
            if index_name in self.document_store:
                del self.document_store[index_name]
            print(f"Deleted FAISS index: {index_name}")
    
    def close(self):
        """Close FAISS connection"""
        self.indexes = {}
        self.document_store = {}


# Factory for creating vector databases
def get_vector_db(db_type: str, **kwargs) -> VectorDatabase:
    """
    Factory function to create a vector database instance
    
    Args:
        db_type: Type of vector database ('redis', 'chroma', 'faiss')
        **kwargs: Additional parameters for the database
        
    Returns:
        VectorDatabase instance
    """
    if db_type.lower() == 'redis':
        return RedisVectorDB().setup(**kwargs)
    elif db_type.lower() == 'chroma':
        return ChromaVectorDB().setup(**kwargs)
    elif db_type.lower() == 'faiss':
        return FaissVectorDB().setup(**kwargs)
    else:
        raise ValueError(f"Unsupported vector database type: {db_type}")


# Helper functions
def load_model():
    """
    Load the all-mini-lm model for sentence embeddings.
    
    Returns:
        SentenceTransformer: Model for generating embeddings
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the all-mini-lm model
    return model