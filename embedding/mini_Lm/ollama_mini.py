import sys
import os
import time
import psutil
import json
import requests
from database_mini import get_vector_db
import PyPDF2
import re

# Measure Performance of a function
def measure_performance(func, *args, **kwargs):
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

# Query Ollama with the given prompt
def query_ollama(prompt, model_name):
    url = "http://localhost:11434/api/generate"
    
    # Map common model shortnames to their full names in Ollama
    model_mapping = {
        "mistral": "mistral:7b",
        "llama2": "llama2"
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

# Generate a response using RAG
def generate_rag_response(query_text, vector_db, embedding_model, llm_model, index_name="document_index", top_k=3):
    # Generate embedding for the query
    query_vector = embedding_model.encode(query_text).tolist()
    
    # Retrieve relevant documents
    relevant_docs = vector_db.query(query_vector, index_name, top_k=top_k)
    
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

# Process a PDF document and split it into chunks
def process_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    print(f"Processing PDF: {pdf_path}")
    
    # Extract text from PDF
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return []
    
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split text into chunks
    chunks = []
    start = 0
    
    while start < len(text):
        # Define chunk end with overlap
        end = min(start + chunk_size, len(text))
        
        # Extract chunk
        chunk = text[start:end]
        
        # Add chunk to list
        chunks.append({
            "content": chunk,
            "metadata": f"source: {os.path.basename(pdf_path)}, position: {start}-{end}"
        })
        
        # Move start position for next chunk (with overlap)
        start = start + chunk_size - chunk_overlap
    
    print(f"Created {len(chunks)} chunks from {pdf_path}")
    return chunks

# Ingest documents from a directory into a vector database
def ingest_documents(vector_db, embedding_model, directory_path, index_name="document_index"):
    # Ensure vector database is set up
    if not hasattr(vector_db, 'indexes') and not hasattr(vector_db, 'collection'):
        vector_dimensions = 768  # MPNet embedding dimension
        vector_db.create_index(index_name, vector_dimensions)
    
    all_chunks = []
    
    # Process each file in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Check if file is a PDF
        if filename.lower().endswith('.pdf'):
            chunks = process_pdf(file_path)
            all_chunks.extend(chunks)
        # Add support for other file types if needed
        
    # Index all chunks
    if all_chunks:
        doc_keys = vector_db.index_documents(all_chunks, index_name, model=embedding_model)
        print(f"Indexed {len(doc_keys)} chunks in vector database")
        return len(doc_keys)
    else:
        print("No documents were processed")
        return 0