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

file_count = 0
for page in page_iterator:
    if "Contents" in page:
        file_count += len(page["Contents"])

print(f"Number of file s in S3 prefix: {file_count}")

# Delete all files in s3 (if needed ONLY)
objects_to_delete = []
for page in page_iterator:
    if "Contents" in page:
        for obj in page["Contents"]:
            objects_to_delete.append({'Key': obj["Key"]})

# Delete in batches of 1000 (s3 limit per request)
for i in range(0, len(objects_to_delete), 1000):
    s3.delete_objects(
        Bucket=s3_bucket,
        Delete=['Objects': objects_to_delete[i:i+1000]]
    )

print("All files in the S3 prefix have been deleted.")

## OpenSearch Service setup
credentials = boto3.Session().get_credentials()

host = "hostidnumber.region.aoss.amazonaws.com" # Get OpenSeasrch Serivce end point url from the console.
awsauth = AWSV4SignerAuth(credentials, region_name, 'aoss')
oss_client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=300
)

# Set index name for document loadings
index_name = "test-index-1" # Change name if needed.

# Use amanaon.titan-embed-text-v2:0 for embedding. Use other embedding model if needed.
region_name = "us=east-2" # Change if needed.
embedding_model = "amazon.titan-embed-text-v2:0"
embedding_model_arn = f"arn:aws:bedrock:{region_name}::foundation-model/{embedding_model}"
embedding_max_tokens = 8192 # Max input tokens for this model.
embedding_context_dimension = 1024 # Output dimension per aws web documentations.

body_json = {
    "settings": {
        "index.knn": "true",
        "number_of_shards": 1,
        "knn.algo+param.ef_search": 512,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "vector": {
                "type": "knn_vector",
                "dimension": embedding_context_dimension,
                "method": {
                    "name": "hnsw",
                    "engine": "faiss",
                    "space_type": "12"
                }
            },
            "ids": {
                "type": "text"
            },
            "metadata": {
                "type": "text"
            },
            "document": {
                "type": "text"
            }
        }
    }
}
try:
    index_response = oss_client.indices.create(index=index_name, body=json.dump(body_json))
except RequestError as e:
    print(f'Error while trying to create the index, with error {e.error}')

# Get existing index settings and mappings
index_info = oss_client.indices.get(index=index_name)
print(json.dumps(index_info, indent=2))

## Bedrock Knowledge Base setup
bedrock_agent_client = boto3.client(
    'bedrock-agent', region_name=region_name
)

# Get information about previously created knowledge bases (if any)
# List all knowledge bases
response = bedrock_agent_client.list_knowledge_bases()
knowledge_bases = response.get('knowledgeBaseSummaries', [])
 
# Filter by name containing "repo-searchbot" (if it was named as such)
filtered_kbs = [
    kb for kb in knowledge_bases
    if "repo" in kb.get('name', '')
]

# Print matching knowledge bases
for kb in filtered_kbs:
    print(f"ID: {kb['knowledgeBaseId']}, Name: {kb['name']}")

# Delete knowledge base (if needed ONLY)
for kb in filtered_kbs:
    kb_id = kb['KnowledgeBaseId']
    print(f"Deleting knowledge base: {kb_id}")
    bedrock_agent_client.delete_knowledge_base(knowledgeBasedId=kb_id)

# OpenSearch service storage configuration
storage_config = {
    "type": "OPENSEARCH SERVERLESS",
    "opensearchServerlessConfiguration": {
        "collectionArn": "arn:aws:aoss:region:numbers:collection/alphanumericids", # Get info about region, numbers, and alphanumericids.
        "vectorIndexName": index_name,
        "fieldMapping": {
            "textField": "document",
            "metadataField": "metadata",
            "vectorField": "vector"
        }
    }
}

# Set some parameters
kb_name = "your-kb-name"
kb_description = "Knowledge base for repo search agent" 
kb_configuration = {"type": "VECTOR", "vectorKnowledgeBaseConfiguration": {"embeddingModelArn": embedding_model_arn}}
# guardrail_arn = "arn:aws:bedrock:region:guardrailinfo"
# guardrail_version = "1"

# Create knowledge base
try:
    create_kb_response = bedrock_agent_client.create_knowledge_base(
        name=kb_name,
        description=kb_description,
        roleArn="arn:aws:iam::rolearninformation",
        knowledgeBaseConfiguration=kb_configuration,
        storageConfiguration=storage_config
    )
    print("knowledge base created successfully.")
    pprint.pprint(create_kb_response)
except bedrock_agent_client.exceptions.ConflictException:
    print(f"Knowledge base: '{kb_name}' already exists. Skipping creation.")
except Exception as e:
    print(f"Error: {e}")

# Create knowledge base guardrail and chunking
# Use chunking when creating dadta source after the knowledge base is creatd.
# bedrock_client = boto3.client("bedrock", region_name=region_name)
# bedrock_client.create_data_source(....)
# Guardrail can be used with bedrock agent-runtime
# See the email I sent.

# Set knowledge base id
knowledgeBaseId = "yourkbid" # Result from running the prvious code

## Create Data Source
# Search any data sources you created before (if any)
# List all data sources
response = bedrock_agent_client.list_data_sources(knowledgeBaseId=knowledgeBaseId)
data_sources = response.get('dataSourceSummaries', [])

# Filter data sources containing "repo" in the name (if you named it as such before).
filtered_sources = [
    ds for ds in data_sources
    if "repo" in ds.get('name', '').lower()
]

# Print matching data sources
for ds in filtered_sources:
    print(f"ID: {ds['dataSourceId']}, Name: {ds['name']}")

# Delete existing data sources (if needed ONLY).
data_source_id = "get-this-from-above-code-result" # ID to be deleted

try:
    response = bedrock_agent_client.delete_data_source(
        knowledgeBaseId=knowledgeBaseId,
        dataSourceId=data_source_id
    )
    print("Data source deleted successfully.")
    print(response)
except Exception as e:
    print(f"Error deleting data source: {e}")

# Create dat asource of knowledge base
data_source_name = "repo-searchbot-data-source" # Change if needed.
knowledgeBaseId = knowledgeBaseId

try:
    response = bedrock_agent_client.create_data_source(
        knowledgeBaseId=knowledgeBaseId,
        name=data_source_name,
        dataDeletionPolicy="DELETE",
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": {
                "bucketArn": s3_bucket_arn,
                "inclusionPrefixes": [s3_prefix]
            }
        },
        vectorIngestionConfiguration={
            "chunkingConfiguration": {
                "chunkingStrategy": "FIXED_SIZE",
                "fixedSizeChunkingConfiguration": {
                    "maxTokens": 1000, # or adjust as needed
                    "overlapPercentage": 20 # or adjust as needed
                }
            }
        }
    )
    pprint.pprint(response)
except bedrock_agent_client.exceptions.ConflictException:
    print("Data source already exists. Skipping creation.")
except Exception as e:
    print(f"Error: {e}")

# NOTE: Per https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_UpdateDataSource.html,
# chunking strategy cannot be updated after creation. Maybe need to delete data source and start over if needed.

# See information about created data source
response = bedrock_agent_client.list_data_sources(
    knowledgeBaseId=knowledgeBaseId
)
pprint.pprint(response)

## Ingest s3 dasta into OpenSearch Service DB via Knowledge Base and Data Source
# Ingest data into knowledge base (This may take a few minutes to hours depending on data size).
data_source_id = 'thesourceid' # Get this from the create_data_source response
response = bedrock_agent_client.start_ingestion_job(
    knowledgeBaseId=knowledgeBaseId,
    dataSourceId=data_source_id
)
pprint.pprint(response)

# Check ingesting job completion status
job = response["ingestionJob"]
job_details = bedrock_agent_client.get_ingestion_job(
    knowledgeBaseId=knowledgeBaseId,
    dataSourceId=data_source_id,
    ingestionJobId=job["ingestionJobId"]
)

ingestion_job = job_details["ingestionJob"]
pprint.pprint(ingestion_job)

# Fetch documents from the index
response = oss_client.search(
    index=index_name,
    body={
        "size": 2, # Number of documents to fetch
        "query": {
            "match_all": {}
        }
    }
)

# Print the ingested documents. The results may be long. Run the below when needed.
for hit in response['hits']['hits']:
    pprint.pprint(hit['_source'])

## Bedrock Agent
agent_name = "agent-repo-searchbot" # Change if needed.
agent_description = "Agent for codebase search using Bedrock KB"
foundation_model = "us.anthropic.claude-3-5-sonnet-20241022-v2:0" # Get this information from bedrock console. Use other models if needed.
agent_role_arn = "arn:aws:iam::role-name"
kms_key_arn = "arn:aws:kms:region:keyinformation"

instruction = (
    "You are an expert codebase assistant. For each user question, use the provided knowledge base to retrieve and present the most relevant scripts. "
    "For your response, follow these guidelines:\n"
    "- Return 3 to 5 scripts in order of relevance (if available).\n"
    "- For each script, provide:\n"
    "    - The file path (as found in the metadata, so users can locate the file in the codebase).\n"
    "    - A concise summary of what the script does, especially steps related to the question.\n"
    "    - Several relevant code snippets that demonstrate these steps, with the starting line number for each snippet.\n"
    "- If multiple scripts are found, summarize and provide snippets for each one separately.\n"
    "- If you do not have enough information to answer, reply with 'not enough information' or 'cannot find'.\n"
    "Always base your answers strictly on the content of the knowledge base. Do not fabricate information."
)

response = bedrock_agent_client.create_agent(
    agentCollaboration='DISABLED',
    agentName=agent_name,
    agentResourceRoleArb=agent_role_arn,
    customerEncryptionKeyArn=kms_key_arn,
    description=agent_description,
    foundationModel=foundation_model,
    instruction=instruction
)

response

# agent_id = response['agent']['agentId']
agent_id = 'youragentid'

try:
    # Prepare the agent after creation
    response = bedrock_agent_client.prepare_agent(agentId=agent_id)
    
    # Print the response details
    print(f"Agent preparation initiated for agent_ID: {response['agentId']}")
    print(f"Agent status: {response['agentStatus']}")
    print(f"Agent version: {response['agentVersion']}")
    print(f"Prepared at: {response['preparedAt']}")

except Exception as e:
    print(f"Error preparing agent: {e}")

# Note: Next, associate knowledge base with the DRAFT version (default) of the agent.
# Associate knowledge base with the agent
response = bedrock_agent_client.associate_agent_knowledge_base(
    agentId=agent_id,
    agentVersion='DRAFT', # Only this value is accepted at this time.
    description="Associate KB with DRAFT version",
    knowledgeBaseId=knowledgeBaseId,
    knowledgeBaseState="ENABLED"
)
print("Knowledge base associated with DRAFT version.")

# NOTE: then, create an alias from the bedrock agent console. Cannot be done via boto3.
# Alias creation will create a published version. The knowledge base associated with DRAFT
# version will be automatically associated with the published version.

agent_alias_name = "first-version-repo-searchbot" # Change if needed.

# Check agent version
response = bedrock_agent_client.list_agent_versions(agentId=agent_id)

for version in response["agentVersionSummaries"]:
    print(f"Version: {version['agentVersion']}, Status: {version['agentStatus']}, Description: {version.get('description', '')}")

# Test bedrock agent
# Get alias id
response = bedrock_agent_client.list_agent_aliases(agentId=agent_id)
for alias in response['agentAliasSummaries']:
    print(f"Alias name: {alias['agentAliasName']}, Alias ID: {alias['agentAliasId']}")

# Initialize the Bedrock Agent Runtime client
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime', region_name = "us-east-2") # Change region if needed.

# Define your agent's details
agent_alias_id = "agentaliasid" # Replace with your Agent Alias ID
session_id = "test-session-1" # A unique ID for the conversation session. Change if needed.

def extract_bedrock_agent_answer(response):
    """
    Extracts and returns only the answer text from a Bedrock Agent streaming response.
    Handles both 'chunk' and 'bytes' event types.
    """
    answer = ""
    for event in response['completion']:
        # If the event contains the answer as bytes (most common for final answer)
        if 'chunk' in event and 'bytes' in event['chunk']:
            answer += event['chunk']['bytes'].decode('utf-8')
        # Some agents may use just 'bytes' at the top level
        elif 'bytes' in event:
            answer += event['bytes'].decode('utf-8')
        # Some agents may use 'chunk' and 'completion'
        elif 'chunk' in event and 'completion' in event['chunk']:
            answer += event['chunk']['completion']
    return answer.strip()

def format_query_with_instructions(query):
    """
    Adds explicit formatting instructions to each query to ensure the agent follows
    the required response format.
    """
    format_instructions = (
        "CRITICAL INSTRUCTION - YOUR MUST FOLLOW THIS EXACT FORMAT FOR YOUR RESPONSE:\n\n"
        "For each relevant file (provide at least 3 if available), include ALL of the following:\n"
        "## [File Name]\n"
        "- **Full path**: [complete file path from metadata]\n"
        "- **Summary**: [concise summary of what the script does]\n"
        "- **Code Snippets**:\n"
        "```[language]\n"
        "# Line [line number]\n"
        "```\n\n"
        "EVERY file you mention MUST include ALL three elements above (path, summary, AND code snippets).\n"
        "DO NOT provide general information without specific files.\n"
        "If you cannot find specific files with paths and code, say 'I cannot find specific fles related to this query.'\n\n"
        "Now, regarding this question: "
    )
    return format_instructions + query

def validate_agent_response(response_text):
    """
    Validate that the agent's response includes the required elements:
    - File paths
    - Summary
    - Code Snippets

    Returns a tuple of (is_valid, missing_elements)
    """
    # First check if this is a "cannot find" response
    if re.seasrch(r'cannot find|not enough information', response_text, re.IGNORECASE):
        # This is a valid "no results" response
        return (True, [])

    missing_elements = []

    # Look for file path pattern (more specific than beore)
    file_paths = re.findall(r'\*\*Full Path\*\*:\s*([\/\\][^\n]+)', response_text)
    if not file_paths:
        missing_elements.append("file paths")
    
    # Look for summary section
    summaries = re.findall(r'\*\*Summary\*\*:\s*([^\n]+)', response_text)
    if not summaries:
        missing_elements.append("file summaries")

    # Look for code blocks with language and line numebrs
    code_blocks = re.findall(r'```[\w]*[\s\S]*?```', response_text)
    if not code_blocks:
        missing_elements.append("code snippets")
    else:
        # Check if code blocks have line numbers
        has_line_numbers = any(re.search(r'# Line \d+', block) for block in code_blocks)
        if not has_line_numbers:
            missing_elements.append("line numbers in code snippets")

    # Cound file headings (## [filename])
    file_headings = re.findall(r'##\s+\S+', response_text)

    # Check for proper structure - we should have matching counts of elements
    element_counts = {
        "file_headings": len(file_headings),
        "file_paths": len(file_paths),
        "summaries": len(summaries),
        "code blocks": len(code_blocks)
    }

    # If we have inconsistent counts, note it
    if len(set(element_counts.values())) > 1 and min(element_counts.values()) > 0:
        missing_elements.append(f"consistent structure (element counts: {element_counts})")

    is_valid = len(missing_elements) == 0
    return (is_valid, missing_elements)

def get_validated_agent_response(client, agent_id, agent_alias_id, session_id, query, max_return=3):
    """
    Gets a response from the agent and validates it contains the required elements.
    Will retry up to max retries times if the response is missing elemennts.
    """
    formatted_prompt = format_query_with_instructions(query)

    for attemp in range(max_retries + 1):
        if attempt == 0:
            prompt_to_send = formatted_prompt
        else:
            # More forceful retry instruction
            prompt_to_send = (
                f"CRITICAL: Your previous response did not follow the required format. Attempt {attempt}/{max_retries}.\n\n"
                f"You MUST structure your response like this for EACH file:\n"
                f"## [File Name]\n"
                f"- **Full Path**: [complete file path]\n"
                f"- **Summary**: [concise summary]\n"
                f"- **Code Snippets**:\n"
                f"```[language]\n"
                f"# Line [number]\n"
                f"[code snippet]\n"
                f"```\n\n"
                f"Original question: {query}\n\n"
                f"DO NOT respond with general information. Each file mentioned MUST include ALL three elements."
            )

        print(f"Attempt {attempt+1}/{max_retries+1} - Sending query to agent...")

        response = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=prompt_to_send,
            enableTrace=True
        )

        answer = extract_bedrock_agent_answer(response)
        is_valid, missing_elements = validate_agent_response(answer)

        if is_valid:
            print("Response validated successfully.")
            return answer

        print(f"Attempt {attempt+1}/{max_retries+1} - Response missing: {', '.join(missing_elements)}")

        if attempt == max_retries:
            print("WARNING: Maximum retries reached. Returning last response with a warning header.")
            # Add a warning header to the response
            return (
                "WARNING: This response doesn't follow the required format. It may be missing file paths, "
                "summaries, or code snippets. \n\n" + answer
            )

    # Should never reach here, but just in case
    return answer

# The prompt you want to send to the agent
prompt_text = "please find the scripts regarding how to track marketing performance metrics."

answer = get_validated_agent_response(
    bedrock_agent_runtime_client,
    agent_id,
    agent_alias_id,
    session_id,
    prompt_text
)

print("\n==== FIRST QUERY RESULTS ====")
print(answer)

# For debugging, - print a clear evaluation of response quality
is_valid, missing_elements = validate_agent_response(answer)
if not is_valid:
    print(f"\n Response still missing: {', '.join(missing_elements)}")
else:
    print("\n Response meets all format requirements")

prompt_text = "Can you show me scripts for auto or mortage other than other loans? Pleae give me the file names and their file paths, respectively."

answer = get_validated_agent_response(
    bedrock_agent_runtime_client,
    agent_id,
    agent_alias_id,
    session_id,
    prompt_text
)

print("\n==== SECOND QUERY RESULTS ====")
print(answer)

# For debugging, - print a clear evaluation of response quality
is_valid, missing_elements = validate_agent_response(answer)
if not is_valid:
    print(f"\n Response still missing: {', '.join(missing_elements)}")
else:
    print("\n Response meets all format requirements")

prompt_text = "Can you show useful code snippets for pulling customer profile data?"

answer = get_validated_agent_response(
    bedrock_agent_runtime_client,
    agent_id,
    agent_alias_id,
    session_id,
    prompt_text
)

print("\n==== THIRD QUERY RESULTS ====")
print(answer)

# For debugging, - print a clear evaluation of response quality
is_valid, missing_elements = validate_agent_response(answer)
if not is_valid:
    print(f"\n Response still missing: {', '.join(missing_elements)}")
else:
    print("\n Response meets all format requirements")

# TODO: May need to work on improving the bedrock agent performance. 
# It seems that the agent using langchain and langgraph performs better than this.