import shutil
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from bs4 import BeautifulSoup
import requests
import os
import chromadb
import pathlib
from weave import Dataset
from openai import OpenAI


# Crawl docs at https://github.com/wandb/weave/tree/master/docs/docs
# use llama index to put in a vector db

temp_dir = 'temp_docs'
chroma_path = "./chroma_db"


def crawl_github_docs(url, docs_dir):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all links to .md files
    md_links = [a['href'] for a in soup.find_all(
        'a', href=True) if a['href'].endswith('.md')]

    # sort and unique md_links
    md_links = sorted(set(md_links))

    # print("Links found:")
    # print(md_links)

    # Create a temporary directory to store the .md files
    os.makedirs(docs_dir, exist_ok=True)

    # Download and save each .md file
    for link in md_links:
        # Extract just the filename from the link
        filename = os.path.basename(link)
        # Construct the raw content URL
        file_url = f"https://raw.githubusercontent.com/{link}"
        file_content = requests.get(file_url).text

        # Save the file using just the filename
        with open(os.path.join(docs_dir, filename), 'w', encoding='utf-8') as f:
            f.write(file_content)

    return md_links


def create_vector_index(docs_dir, index_options=None, chunk_size=1024, chunk_overlap=20):
    if index_options is None:
        index_options = {}
    
    db = chromadb.PersistentClient(path=chroma_path)
    
    # Configure document loading with chunk size and overlap
    documents = SimpleDirectoryReader(docs_dir).load_data()
    
    # Create text splitter with the specified chunk size and overlap
    from llama_index.core.node_parser import SentenceSplitter
    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = node_parser.get_nodes_from_documents(documents)

    embed_model = OpenAIEmbedding(
        embed_batch_size=index_options.get('embed_batch_size', 10)
    )

    collection_name = index_options.get('collection_name', 'quickstart')
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes, 
        storage_context=storage_context, 
        embed_model=embed_model,
        **{k: v for k, v in index_options.items() if k not in ['embed_batch_size', 'collection_name']}
    )

    return index


def load_vector_index(index_options=None):
    if index_options is None:
        index_options = {}
    
    db = chromadb.PersistentClient(path=chroma_path)
    embed_model = OpenAIEmbedding(
        embed_batch_size=index_options.get('embed_batch_size', 10)
    )

    collection_name = index_options.get('collection_name', 'quickstart')
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
        **{k: v for k, v in index_options.items() if k not in ['embed_batch_size', 'collection_name']}
    )
    return index


def query(user_query, query_engine, model_name, n=1):
    context = query_engine.query(user_query).response

    # Use OpenAI to generate a response based on the context
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about Weave based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_query}\n\nPlease answer the question based on the given context."}
        ],
    )

    # Extract the generated answer
    answer = response.choices[0].message.content
    return answer


def query_multiple(user_query, query_engine, model_name, n):
    context = query_engine.query(user_query).response

    # Use OpenAI to generate a response based on the context
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about Weave based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_query}\n\nPlease answer the question based on the given context."}
        ],
        n=n
    )

    # Extract the generated answer
    answers = []
    for choice in response.choices:
        answers.append(choice.message.content)
    return answers


def load_files_as_dataset(directory_path):
    """
    Load all text files from a directory and create a Weave dataset.

    Args:
        directory_path (str): Path to the directory containing text files

    Returns:
        Dataset: Weave dataset containing the files' content
    """
    files_data = []

    # Convert string path to Path object
    path = pathlib.Path(directory_path)

    # Iterate through all files in the directory
    for file_path in path.glob('*.*'):
        if file_path.is_file():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    files_data.append({
                        'filename': file_path.name,
                        'content': content,
                        'path': str(file_path)
                    })
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    # Create and return the dataset
    return Dataset(rows=files_data)
