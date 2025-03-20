import requests
import json

def query_ollama(prompt, model_name):
    """
    Query Ollama with a prompt
    
    Args:
        prompt: Prompt to send to Ollama
        model_name: Name of the model to use
    
    Returns:
        Response from Ollama
    """
    url = "http://localhost:11434/api/generate"
    
    # Map common model shortnames to their full names in Ollama
    model_mapping = {
        "mistral": "mistral:7b",  # Use the full model name
        "llama2": "llama3.2",  # Keep as is
        "ollama/llama2": "llama2"  # Handle the case with prefix
    }
    
    # Get the correct model name or use the provided one as fallback
    actual_model = model_mapping.get(model_name, model_name)
    
    payload = {
        "model": actual_model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            return response.json()["response"]
        elif response.status_code == 404:
            # Special handling for model not found
            error_msg = f"Model '{actual_model}' not found. Available models can be listed with 'ollama list'"
            print(error_msg)
            return error_msg
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

def generate_rag_response(query_text, redis_client, embedding_model, query_documents_func, llm_model, top_k=3):
    """
    Generate a response using RAG
    
    Args:
        query_text: Query text
        redis_client: Redis client
        embedding_model: Embedding model
        query_documents_func: Function to query documents
        llm_model: Name of the LLM model to use
        top_k: Number of documents to retrieve
    
    Returns:
        Generated response with metadata
    """
    # Retrieve relevant documents
    relevant_docs = query_documents_func(query_text, redis_client, embedding_model, top_k=top_k)
    
    # Prepare context
    context = "\n\n".join([doc["content"] for doc in relevant_docs])
    
    # Create prompt
    system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
    If you don't know the answer based on the context, say that you don't know. 
    Don't make up information that isn't in the context."""
    
    full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"
    
    # Query Ollama
    response = query_ollama(full_prompt, model_name=llm_model)
    
    return {
        "query": query_text,
        "response": response,
        "context": context,
        "documents": relevant_docs,
        "model": llm_model
    }