# Install library if not installed yet.
# pip install -U chromadb

# Load libraries
import os
import json
import requests
from urllib.parse import urlparse
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from openai import OpenAI
from tiktoken import encoding_for_model
from pydantic import BaseModel
import json

# Path setting
dir_path = '/content/drive/MyDrive/Colab_Notebooks/'

# API keys
openai_file = "googlecolab_openai_key.txt"
with open(dir_path + openai_file, "r") as file:
    openai_api_key = file.read()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Set OpenAI client and Chorma client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path=dir_path+"+repo_search/chroma_db")
# chroma_client.delete_collection("example_collection_2")

# Embedding model
default_ef = embedding_functions.DefaultEmbeddingFunction() # by default, all-MiniLM-L6-v2

# Information on repo to search. 
# In the function 'get_contents', need the info for api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
owner="woo1976"
repo="jenkins-python-test"
path=""
branch="master"

# Functions
def get_repo_info(repo_url):
    """
    Extract the owner and repository name from the GitHub URL.
    """
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) < 2:
        raise ValueError("Invalid GitHub repository URL.")
    owner = path_parts[0]
    repo = path_parts[1].replace('.git', '')
    return owner, repo

def get_file_content(download_url):
    """
    Retrieve the raw content of a file.
    """
    response = requests.get(download_url)
    response.raise_for_status()
    return response.text

def get_contents(owner="woo1976", repo="jenkins-python-test", path="", branch="master"):
    """
    Recursively retrieve the contents of the repository, excluding image and media files.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": branch}  # You can change the branch if needed
    response = requests.get(api_url, params=params)
    response.raise_for_status()
    items = response.json()
    contents = []

    if not isinstance(items, list):
        items = [items]

    # Exclude image and media formats
    excluded_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.mp4', '.mp3', '.wav', '.avi', '.mov']

    for item in items:
        if item['type'] == 'file' and any(item['path'].endswith(ext) for ext in excluded_extensions):
            continue

        if item['type'] == 'file':
            file_content = get_file_content(item['download_url'])
            contents.append({
                'type': 'file',
                'path': item['path'],
                'content': file_content
            })
        elif item['type'] == 'dir':
            dir_contents = get_contents(owner, repo, item['path'], branch)
            contents.append({
                'type': 'dir',
                'path': item['path'],
                'contents': dir_contents
            })
    return contents

def flatten_contents(contents):
    """
    Flatten the nested contents into a list of files with paths and contents.
    """
    flat_files = []

    def _flatten(items):
        for item in items:
            if item['type'] == 'file':
                flat_files.append({
                    'path': item['path'],
                    'content': item['content']
                })
            elif item['type'] == 'dir':
                _flatten(item['contents'])

    _flatten(contents)
    return flat_files

def truncate_to_token_limit(text, max_tokens, encoding='gpt-4o'):
    """
    Truncate the text to fit within the specified token limit.
    """
    tokenizer = encoding_for_model(encoding)
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens)

def summarize_text(text, max_tokens=200):
    """
    Summarize the text to fit within a limited number of tokens.
    """
    prompt = f"Summarize the following text to {max_tokens} tokens:\n\n{text}\n\nSummary:"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def generate_gpt4o_response(query, context_docs, max_context_tokens=6000):
    """
    Generate a response using OpenAI's GPT-4o based on the query and retrieved context documents.
    """
    # Summarize to reduce token size
    summarized_contexts = [summarize_text(doc, max_tokens=200) for doc in context_docs]

    # Combine into a single string
    combined_context = "\n\n".join(summarized_contexts)

    # Truncate
    context = truncate_to_token_limit(combined_context, max_context_tokens)

    prompt = f"""You are an assistant that provides detailed answers based on the following context.
Please generate 3 distinct answers in the order from the most relevant one to the least relevant one. 
Each answer should be different in terms of file names and contents.
Each answer should include a file name and its path. 
It would be also better for each asnwer to include summary of the file
Lastly, it would be best to have the most relevant code snippet from the script:

Context:
{context}

Question:
{query}

Answer:"""

    try:
       response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.2
        )
       answer = response.choices[0].message.content.strip()
       
       return answer
        # Save the below for later usage just in case.
        # response = client.beta.chat.completions.parse(
        #     model="gpt-4o",
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": prompt}
        #     ],
        #     max_tokens=1000,
        #     temperature=0.2,
        #     response_format=ResFormat,
        # )
        # answer = response.choices[0].message.content
        # json_answer = json.loads(answer)

        # return json_answer

    except Exception as e:
        print(f"Error generating GPT-4o response: {e}")
        return "I'm sorry, I couldn't process your request at the moment."

# Define a structured output for LLM results (save this for later usage just in case)
# class ResFormat(BaseModel):
#     file_name: str
#     file_path: str
#     summary: str
#     code_snippet: str

#  Extract repo contents
res = get_contents(owner=owner, repo=repo, path=path, branch=branch)
flat_files = flatten_contents(res)

texts = [file['content'] for file in flat_files]
metadatas = [{'path': file['path']} for file in flat_files]
ids = [file['path'] for file in flat_files]

# Create vector DB
collection = chroma_client.get_or_create_collection(name="example_collection", embedding_function=default_ef)
collection.add(documents=texts, metadatas=metadatas, ids=ids)

# Query results from vector DB
query = "Give me a script that is related to testing."
top_k = 5
results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

retrieved_docs_with_metadata = []
for i in range(len(results['documents'][0])):
  doc = results['documents'][0][i]
  metadata = results['metadatas'][0][i]
  retrieved_docs_with_metadata.append({'document': doc, 'metadata': metadata})
context_docs = [doc for doc in retrieved_docs_with_metadata]

# Generate LLM agent answers
answer = generate_gpt4o_response(query, context_docs)

print(answer)
# Save the below for later usage just in case.
# print(answer['file_name'])
# print(answer['file_path'])
# print(answer['summary'])
# print(answer['code_snippet'])

