from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from models import State
from prompt import *


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vdb = Chroma(
    persist_directory="News",
    embedding_function=embedding
)
llm = ChatOllama(model="gemma2:2b", temperature=0.0)

def query_rewrite_agent(state: State) -> dict:
    user_input = state.get("query")
    chat_history = state.get("chat_history")

    messages = [
        SystemMessage(content=REWRITE_PROMPT),
        HumanMessage(content=query_rewrite_extend(user_input, chat_history))
    ]

    response = llm.invoke(messages).content
    return {
        "query_rewrite": response
    }

def retriever_agent(state: State):
    rewritten_query = state.get("query_rewrite")

    retriever = vdb.as_retriever(search_kwargs={"k": 10})
    results = retriever.invoke(rewritten_query)

    return {
        "context": results
    }

def response_agent(state: State) -> dict:
    query_rewrite = state.get("query")
    chat_history = state.get("chat_history")
    content = state.get("context")

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=system_prompt_extend(query_rewrite, chat_history, content))
    ]

    response = llm.invoke(messages).content
    return {
        "response": response
    }