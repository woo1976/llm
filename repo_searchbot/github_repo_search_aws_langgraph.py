
from pathlib import Path
import boto3
import logging
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, RequestError
from opensearchpy.helpers import bulk
from langchain_core.documents import Document 
from langchain_aws.chat_models import ChatBedrock
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever
from langhchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, END 
from langchain.text_splitter import RecursiveCharacterTextSplitter 

# BEdrock client
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-2")

# Langchain Bedrock LLM and Embeddings
llm = ChatBedrock(
    client=bedrock_runtime,
    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0", # May need to check in Bedrock console for exact name.
    model_kwargs={"anthropic_version": "bedrock=2023-05-31"}
)
embeddings = BedrockEmbeddings(
    client=bedrock_runtime,
    model_id="amazon.titan=embed-text-V2:0"
)

# OpenSeaerch VectorStore
credentials = boto3.Session().get_credentials()

# Need to create the vectorstore collections in aws OpenSearch console first
host = "hosturl:port" # without https:// part. get this from opensearch console.
awsauth = AWSV4SignerAuth(credentials, 'us-east-2', 'aoss')
vectorstore = OpenSearchVectorSearch(
    opensearch_url=host,
    index_name="your-index-name" # create this yourself.
    embedding_function=embeddings,
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestHttpConnection,
    timeout=300
)

# Prepare data
cloned_repo_path = "/your/repo/path"

# Functions to process data
def get_file_content(file_path):
    try:
        with_open(file_path, "r", encoding="utf-8") as f:
            return f.read() 
    except UnicodeDecodeError:
        # print(f"UnicideDecodeError: Could not read file as UTF-8: (file_path)") # There were related to .git/objects contents.
        return None # Skip non-UTF-8 files 

def get_contents(cloned_repo_path):
    """
    Recursively retrieve the contents of the repository, excluding image, media, and ipynb files.
    """
    cloned_repo_path = Path(cloned_repo_path)

    items = []
    for path in cloned_repo_path.rglob("*"):
        item = {
            "name": path.name,
            "path": str(path), # str(path.relative_to(repo_path)),
            "type": "dir" if path.is_dir() else "file"
        }
        items.append(item)

    contents = []

    # Exclude image and media formats. Also exclude ipynb files (for simplicity for now).
    excluded_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg',
                           '.mp4', '.mp3', '.wav', '.avi', '.mov', '.ipynb']

    for item in items:
        if item['type'] == 'file' and any(item['path'].endswith(ext) in excluded_extensions):
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
                _flatten(item['contents']) # Recursion

    _flatten(contents)
    return flat_files

# Prepare three lists: ids, metadats, and texts
res = get_contents(cloned_repo_path)
flat_files = flatten_contents(res)
texts = [file['contents'] for file in flat_Files]
metadatas = [{'path': file['path']} for file in flat_files]
ids = [file['path'] for file in flat_files]

# len(ids) : 2218

embedding_max_tokens = 8192
multiplier = 1.5
chunk_size = int(embedding_max_tokens * multiplier)
chunk_overlap = 200 # You can adjust overlap as needed.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

docs = []
for id_, text, meta in zip(ids, texts, metadatas):
    if text is not None:
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={**meta, "id": f"{id_}_chunk{i}"}))

# len(docs) : 3295

# Add documents (with IDs in metadata). This may take a few minutes.
batch_size = 1000
for i in range(0, len(docs), batch_size):
    _ = vectorstore.add_documents(docs[i:i+batch_size], bulk_size=batch_size)

# Create a hybrid retriever
k = len(docs)

# SimilarityArithmetricError retriever
sim_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":k})

# Vector retriever (semantic)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k":k})

# Ensemble retriever (hybrid)
hybrid_retriever = EnsembleRetriever(
    retrievers=[sim_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# Test to see if the retriever works.
results = hybrid_retriever.get_relevant_documents("vantage")

# for doc in results:
#     print("ID:", doc.metadata.get("id"))
#     print("Path:", doc.metadata.get("path"))
#     print("Content:", doc.page_content)
#     print("---")

# len(results) : 452

# for doc in results[0:2]:
#     print("ID:", doc.metadata.get("id"))
#     print("Path:", doc.metadata.get("path"))
#     print("Content:", doc.page_content)
#     print("---")

# Build an agent
# Add memory for conversation
memory = ConversationBufferMemory()

# Prompt templates
prompt_template = ChatPromptTemplate.from_template(
    """You are an expert code analyst.
    Given the following scripts and their file paths, please provide several scripts in the top relevant order.
    Ideally three to five result scripts will be good. :
- A concise summary of what each script does, highlighting any steps related to {focust_phrase}.
- Several relevant code snippets that demonstrate these steps, and for each snippet, indicate the starting line number in the script.
- For each script, also include the file path so users can locate the file.
If there are multiple scripts, summarize and provide snippets for each one separately.
If you do not have enough information to answer, reply with 'not enough information' or 'cannot fine'.

Scripts and file paths:
{context}

Conversation history:
{history}

User request: {user_prompt}

Summary, relevant code snippets with line numbers, and file paths:"""
)

refine_prompt_template = (
    "Given the previous user query:\n"
    "{user_prompt}\n"
    "and the context:\n"
    "{context}\n"
    "and the answer:\n"
    "{answer}\n"
    "and the conversation history:\n"
    "{history}\n"
    "Suggest a more specific or reworded query to help find a better answer. "
    "If possible, use synonyms, clarify the intent, or add details."
)

# Define nodes for LangGraph
def retrieve_node(state):
    # Limit number and size of retrieved docs to avoid LLM input errors
    docs = hybrid_retriever.get_relevant_documents(state["user_prompt"])
    # print("Retrieved docs:")
    # for doc in docs:
    #     print(doc.metadata.get("path", "Unknown path"), doc.page_content[:100])
    docs = docs[:200] # Only use top 200 docs
    context = ""
    max_script_length = 2000 # Truncate each script to certain number of characters
    for doc in docs:
        file_path = doc.metadata.get("path", "Unknown path")
        script = doc.page_content[:max_script_length]
        context += f"File path: {file_path}\nScript:\n{script}\n\n"
    state["context"] = context
    return state

def llm_node(state):
    # Truncate conversation history to avoid prompt overflow
    max_history_chars = 2000
    limited_history = memory.buffer[-max_history_chars:]
    full_context = f"{limited_history}\n{state['context']}"
    prompt = prompt_template.format(
        user_prompt=state["user_prompt"],
        focus_phrase=state["focus_phrase"],
        context=full_context,
        history=limited_history
    )
    answer = llm.invoke(prompt)
    state["answer"] = answer
    state["attempts"] += 1
    memory.save_context({"user": state["user_prompt"]}, {"llm": getattr(answer, 'content', str(answer))})
    return state

def check_node(state):
    answer = state["answer"]
    if hasattr(answer, "content"):
        answer_text = answer.content.lower()
    else:
        answer_text = str(answer).lower()
    if ("not enough information" in answer_text or "cannot find" in answer_text) and state["attempts"] < 2:
        return {"next": "refine", **state}
    return {"next": END, **state}

def refine_node(state):
    # Include memory in refine prompt
    full_context = f"{memory.buffer}\n{state['context']}"
    refine_prompt = refine_prompt_template.format(
        user_prompt=state["user_prompt"],
        context=full_context,
        answer=getattr(state["answer"], 'content', str(state["answer"])),
        history=memory.buffer
    )
    new_query = llm.invoke(refine_prompt)
    state["user_prompt"] = getattr(new_query, 'content', str(new_query))
    return state

def llm_detect_flush_intent(user_input):
    """
    Uses LLM to determine if the user wants to flush memory.
    Return True if LLM detects intent to clear memory, False otherwise.
    """
    intent_prompt = (
        f"User input: '{user_input}'\n"
        "Does the user want to clear, reset, or flush the conversation memory/history? "
        "Reply only 'yes' or 'no'."
    )
    response = llm.invoke(intent_prompt)
    answer = getattr(response, 'content', str(response)).strip().lower()
    return answer.startswith("yes")

def maybe_flush_memory(user_input):
    """
    Flushes the conversation memory if the user requests it.
    Return True if memory was flushed, False otherwise.
    Now supports flexible keyword matching and LLM intent detection.
    """
    keywords = ["flush", "clear", "reset", "remove", "memory", "memories", "history"]
    user_input_lower = user_input.lower()
    match_count = sum(kw in user_input_lower for kw in keywords)
    if match_count >= 2:
        memory.clear() # or memory.buffer = ""
        print("Conversation memory flushed.")
        return True
    # Use LLM to detect intent if keyword match fails
    if llm_detect_flush_intent(user_input):
        memory.clear()
        print("Conversation memory flushed (LLM intent detected).")
        return True
    return False

# Build the LangGraph agent
entry_point = "retrieve"
graph = StateGraph(dict)
graph.add_node("retrieve", retrieve_node)
graph.add_node("llm", llm_node)
graph.add_node("check", check_node)
graph.add_node("refine", refine_node)
graph.add_edge("retrieve", "llm")
graph.add_edge("llm", "check")
graph.add_conditional_edges("check", check_node, lambda x: x["next"])
graph.add_edge("refine", "retrieve")
graph.set_entry_point(entry_point)

rag_agent = graph.compile()

# Visualize the graph in case draw_mermaid_png() do not work due to reasons like firewall, etc.
def print_graph_structure(graph):
    print("\LangGraph Structure:")
    print("Nodes:", list(graph.nodes.keys()))
    print("Edges:")
    for edge in graph.edges:
        print(f" {edge[0]} -> {edge[1]}")

def print_graph_tree(graph, start_node=None, indent=0, max_depth=10): # Recursion
    if start_node is None:
        start_node = "START"
        print("  " * indent + f"- {start_node}")
        print_graph_tree(graph, start+node=entry_point, indent=indent + 1, max_depth=max_depth)
        return
    if indent > max_depth:
        print("  " * indent + "... (cycle detected)")
        return
    print("  " * indent + f"- {start_node}")
    # Handle conditional transitions for 'check'
    if start_node == "check":
        print("  " * (indent + 1) + "- refine")
        print_graph_tree(graph, "refine", indent + 2, max_depth)
        print("  " * (indent + 1) + "- END")
    for edge in graph.edges:
        if edge[0] == start_node and edge[1] not in ["refine", END]:
            print_graph_tree(graph, edge[1], indent + 1, max_depth)

print("\nLangGraph Tree Structure:")
print_graph_tree(graph)
# LangGraph Tree Structure:
# - START
#   - retrieve
#     - llm
#       - check
#         - refine
#           - refine
#             - retrieve
#               - llm
#                 - check
#                   - refine
#                     - refine
#                       ... (cycle detected)
#                   - END
#           - END

# Another visual method
import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(graph):
    G = nx.DiGraph()
    # Add all nodes
    for node in graph.nodes:
        G.add_node(node)
    G.add_node("END") # Ensure END node is present
    # Add all edges
    for edge in graph.edges:
        G.add_edge(edge[0], edge[1])
    # Add conditional edges for 'check' node if not present
    if ("check", "refine") not in G.edges:
        G.add_edge("check", "refine")
    if ("check", "END") not in G.edges:
        G.add_edge("check", "END")
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True)
    plt.show()

plot_graph(graph) 

# Asking an initial question.
# Usage
user_prompt = "Please find the script regarding how to build XXX steps. Memory can be erased and start over." # Update accordingly
focus_phrase = "ABC, 123, XYZ" # Update accordingly

# Check if user wants to flush memory before running agent
maybe_flush_memory(user_prompt)

state = {
    "user_prompt": user_prompt,
    "focus_phrase": focus_phrase,
    "context": "",
    "answer": "",
    "attempts": 0
}

# To suppress verbose logging statements when printing final answers.
logging.getLogger("langgraph").setLevel(logging.ERROR)
logging.getLogger("langgraph.pregel").setLevel(logging.ERROR)

final_state = rag_agent.invoke(state)
print("Final Answer:\n", getattr(final_state["answer"], 'content', str(final_state["answer"])))

# To see conversation memory:
# print("\n=================================================")
# print("\nConversation history:")
# print(memory.buffer)

#### Answer ####
# Conversation memory flushed (LLM intent detected).
# Final Answer:
#  Based on the scripts provided, here are the most relevant files whoing XXX steps, ordered by relevance:

# 1. File. `/file_1/path/script.R`

# Summary: Shows XXXXYYYYZZZZ.

# Key code snippets:
# ```r
# # Lines 14-18: AAABBBBCCCC
# .....
# [content omitted for brevity]
# .....
#
# The common pattern across these scripts shows XXX typically involves:
# 1. Initial ABC section
# 2. Application of XYZ in a abc approach
# 3. Segmentation based on CDFGH scores.
# 4. Validation checks and LMNOP analysis.
# 5. Final segment creation and sizing

# Let me know if you would like additional details about any of these scripts or other aspects of XYZ implementations.
######

# Asking a subsequent question.
# Usage
user_prompt = "Other than the XXX related scripts you provided, I need ABCDE related scripts." # Update accordingly
focus_phrase = "efg, 456" # Update accordingly

# Check if user wants to flush memory before running agent
maybe_flush_memory(user_prompt)

state = {
    "user_prompt": user_prompt,
    "focus_phrase": focus_phrase,
    "context": "",
    "answer": "",
    "attempts": 0
}

# To suppress verbose logging statements when printing final answers.
logging.getLogger("langgraph").setLevel(logging.ERROR)
logging.getLogger("langgraph.pregel").setLevel(logging.ERROR)

final_state = rag_agent.invoke(state)
print("Final Answer:\n", getattr(final_state["answer"], 'content', str(final_state["answer"])))

# To see conversation memory:
# print("\n=================================================")
# print("\nConversation history:")
# print(memory.buffer)

#### Answer ####
# Answer format is similar to the first answer
# .....
# [content omitted for brevity]
# .....

# Asking another subsequent question.
# Usage
user_prompt = "Ok. Please give me some scripts that do this and that." # Update accordingly
focus_phrase = "HIJK, 5678" # Update accordingly

# Check if user wants to flush memory before running agent
maybe_flush_memory(user_prompt)

state = {
    "user_prompt": user_prompt,
    "focus_phrase": focus_phrase,
    "context": "",
    "answer": "",
    "attempts": 0
}

# To suppress verbose logging statements when printing final answers.
logging.getLogger("langgraph").setLevel(logging.ERROR)
logging.getLogger("langgraph.pregel").setLevel(logging.ERROR)

final_state = rag_agent.invoke(state)
print("Final Answer:\n", getattr(final_state["answer"], 'content', str(final_state["answer"])))

# To see conversation memory:
# print("\n=================================================")
# print("\nConversation history:")
# print(memory.buffer)

#### Answer ####
# Answer format is similar to the first answer
# .....
# [content omitted for brevity]
# .....

# LLM based evaluation:
# Use another LLM prompt to reate the answer for relevance, completness, and clarity.
# Example: "Given the user request and the answer, rate the answer from 1-5 for relevance."

eval_prompt = f"""
User request: {user_prompt}
LLM answer: {final_state['answer']}
Rate the answer for relevance and completeness (1-5) and explain your rating. 
"""
eval_result = llm.invoke(eval_prompt) 
print("LLM Evaluation:", getattr(eval_result, 'content', str(eval_result))) 

#### Answer ####
# LLM Evaluation: Rating: 4.5/5

# Explanation of rating:

# Relevance (4/5):
# Strengths: 
# - Provides clear summaries for each script
# - Includes relevant code snippets demonstrating key functionality
# - Outline common elements across the scripts
# - Shows different aspects of XXX (abc, edf, 123)

# Minor gaps that prevent a perfect 5/5:
# - Could include more details about the ABC methodology
# - Might benefit from explaining the business context/goals of these purposes
# - Could provide more information about how these scripts interact with each other

# Overall, the answer provides a compreshensive overview of YYY related scripts while maintaining focus on the most relevant aspects. The small gaps in completeness don't significantly impact the usefullness of the response.
####