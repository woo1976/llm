import os
import json
import uuid
import glob
from pathlib import Path 
import pprint 
import logging 
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, RequestError
from opensearchpy.helpers import bulk
import re

## Get repo contents
parent_path = "/your/path/"
json_files_path = "your_file_path/"

cloned_repo_path = parent_path + "cloned_repo_name"

def get_file_content(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f: 
            return f.read() 
    except UnicodeDecodeError:
        # print(f"UnicodeDecodeError: Could not read file as UTF-8: {file_path}") # These are found to be related to .git/objects contents.
        return None # Skip non-UTF-8 files

def get_contents(cloned_repo_path):
    """
    Recursively retrieve the contents of the repository, excluding image, media, and jupyter notebook files.
    """        
    cloned_repo_path = Path(cloned_repo_path) 

    items = [] 
    for path in cloned_repo_path.rglob("*"): 
        # Exclude hidden files and folders (starting with . like .git or .gitignore)
        if any(part.startswith('.') for part in path.parts): 
            continue
        item = {
            "name": path.name,
            "path": str(path), # str(path.relative_to(repo_path)),
            "type": "dir" if path.is_dir() else "file"
        }
        items.append(item)
    contents = []
    # Exclude image and media formats. Also, exclude ipynb files (for simplicity for now).
    excluded_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.mp4', '.mp3',
                           '.wav', '.avi', '.mov', '.ipynb']

    for item in items:
        if item['type'] == 'file' and any(item['path'].endswith(ext) for ext in excluded_extensions):
            continue

        if item['type'] == 'file':
            file_content = get_file_content(item['path'])
            contents.append({
                'type': 'file',
                'path': item['path'],
                'content': file_content
            })
        elif item['type'] == 'dir':
            dir_contents = get_contents(item['path']) # Recursion
            contents.append({
                'type': 'dir',
                'path': item['path'],
                'content': dir_contents
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
                _flatten(item['contents']) # Recursion 

    _flatten(contents) 
    return flat_files

# Prepare three lists: ids, metadatas, and texts
res = get_contents(cloned_repo_path)
flat_files = flatten_contents(res)
texts = [file['content'] for file in flat_files]
# metadatas = [{'path': file['path']} for file in flat_files]
metadatas = [file['path'] for file in flat_files]
ids = [file['path'] for file in flat_files]

len(ids)

# Combine the lists into a list of dictionaries
combined = []
for id_, meta, text in zip(ids, metadatas, texts):
    combined.append({
        "id": id_,
        "metadata": meta,
        # "metadata_orig": meta,
        "document": text
    })

# Set json files path
output_dir = parent_path + json_files_path

# Delete any existing files (if needed ONLY)
for file_path in glob.glob(os.path.join(output_dir, '*')):
    if os.path.isfile(file_path):
        os.remove(file_path)

# Save json files
common_prefix = "/your/common/prefix/path/you/want/to/remove/" # File path you want to delete when you save.
for idx, doc in enumerate(combined, 1):
    filename = f"file{idx}.json"
    file_path = os.path.join(output_dir, filename)
    # Save the main data file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    # Remove the common prefix from the original path
    original_path = doc['id'] if 'id' in doc else doc['metadata'] 
    if original_path.startswith(common_prefix):
        trimmed_path = original_path[len(common_prefix):]
    else:
        trimmed_path = original_path
    # Save the sidecar metadata file in the correct Bedrock format
    metadata = {
        "metadaataAttributes": {
            "original_path": {
                "value": {
                    "type": "STRING",
                    "stringValue": trimmed_path
                },
                "includeForEmbedding": True
            }
        }
    }
metadata_filename = f"{filename}.metadata.json"
metadata_file_path = os.path.join(output_dir, metadata_filename)
with open(metadata_file_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

# Count number of the saved json files
num_files = len([f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]) 
print(f"Number of files: {num_files}")

## Upload the JSON files into S3
s3_bucket = "your-bucket"
s3_bucket_arn = "arn:aws:s3:::your-bucket"
s3_prefix = "your/prefix-name/"

# Create S3 client
s3 = boto3.client("s3", region_name="us-east-2") # Change region if needed.

# Upload all .json and .json.metadata.json files in output_dir to s3
for filename in os.listdir(output_dir):
    file_path = os.path.join(output_dir, filename)
    if os.path.isfile(file_path) and (filename.endswith('.json') or filename.endswith('.json.metadata.json')):
        s3_key = s3_prefix + filename
        s3.upload_file(file_path, s3_bucket, s3_key)

# Check the count of files saved in s3
paginator = s3.get_paginator('list_objects_v2')
page_iterator = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)


