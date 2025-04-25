import os
from openai import OpenAI
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

st.title("Toy chatbot for HR onboarding")

dir_path = "/content/drive/MyDrive/Colab_Notebooks/"

# API keys
openai_file = "googlecolab_openai_key.txt"
with open(dir_path + openai_file, "r") as file:
    openai_api_key = file.read()

langsmith_file = "langsmith_api_key.txt"
with open(dir_path + langsmith_file, "r") as file:
    langsmith_api_key = file.read()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = openai_api_key 

os.environ["LANGSMITH_TRACING"] = "true"
if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key

# LLM model load
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Embedding model load
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Vector DB setup
vector_store = InMemoryVectorStore(embeddings)

# Load data
directory_path = '/content/drive/MyDrive/Colab_Notebooks/chatbot_for_onboarding/RAG_data/'
all_texts = []
for filename in os.listdir(directory_path) :
  if filename.endswith(".csv"):
      file_path = os.path.join(directory_path, filename)
      loader = CSVLoader(file_path=file_path, encoding="utf-8")
      data = loader.load()

      text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
      texts = text_splitter.split_documents(data)
      all_texts.extend(texts)
  elif filename.endswith(".pdf"):
      file_path = os.path.join(directory_path, filename)
      loader = PyPDFLoader(file_path)
      data = loader.load()

      text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
      texts = text_splitter.split_documents(data)
      all_texts.extend(texts)

# Add documents to vector DB
filenames = [filename for filename in os.listdir(directory_path) if (filename.endswith(".pdf") or filename.endswith(".csv"))]
_ = vector_store.add_documents(documents=all_texts, metadatas=[{"source": filename} for filename in filenames])

# Retrieve tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Create agent
memory = MemorySaver()
agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)

config = {"configurable": {"thread_id": "abc543"}}

user_question = st.text_input("User Question: ", "")

preset_message = "Use the tools to answer the user's questions. \nAlwasys provide a download link."

input_message = (
       preset_message + "\n" + 
       "User's question: " + user_question
)

# Generate responses
for event in agent_executor.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    print("in progress")

st.write(event["messages"][-1].content)
