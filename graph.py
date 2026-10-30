import os
import streamlit as st
from dotenv import load_dotenv
import urllib.request

# Load local .env for development
load_dotenv()

# If running on Streamlit Cloud, load keys from Streamlit Secrets
if hasattr(st, "secrets"):
    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        if "LANGCHAIN_API_KEY" in st.secrets:
            os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
        if "LANGCHAIN_TRACING_V2" in st.secrets:
            os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
        if "LANGCHAIN_PROJECT" in st.secrets:
            os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
        if "TAVILY_API_KEY" in st.secrets:
            os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
    except Exception:
        pass # Fallback to local .env

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from tavily import TavilyClient

# 1. Setup APIs
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# 2. Setup Vector Database (Load PDF)
pdf_path = "project pdf.pdf"
pdf_url = "https://arxiv.org/pdf/2401.00001"

if not os.path.exists(pdf_path):
    print("Downloading PDF...")
    urllib.request.urlretrieve(pdf_url, pdf_path)

loader = PyPDFLoader(pdf_path)
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Define State (Added 'route')
class AgentState(TypedDict):
    question: str
    route: str
    context: str
    analysis: str
    answer: str
    critique: str
    iterations: int

# 4. Define Nodes
def orchestrator_node(state: AgentState):
    print("--- ORCHESTRATOR: DECIDING ROUTE ---")
    prompt = f"""
    You are a strict routing agent. Your job is to decide where to search for the answer.
    
    Route to 'WEB' if the question is about:
    - Current events, sports, pop culture, or recent history.
    - General knowledge that would be found on Wikipedia or news sites.
    
    Route to 'PDF' ONLY if the question is about:
    - Technical AI concepts, RAG systems, LLMs, or the specific contents of a research paper.
    
    Question: {state['question']}
    
    Reply with EXACTLY ONE WORD: either 'WEB' or 'PDF'.
    """
    response = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    route = "web" if "WEB" in response else "pdf"
    print(f"Routing to: {route.upper()}")
    return {"route": route}

import time

def web_search_node(state: AgentState):
    print("--- RESEARCH AGENT: SEARCHING THE WEB ---")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            results = tavily.search(query=state["question"], search_depth="basic")
            context = "\n\n".join([f"Source: {r['url']}\nContent: {r['content']}" for r in results['results']])
            return {"context": context}
        except Exception as e:
            print(f"Tavily search failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Wait 1s, then 2s before retrying
            else:
                # If all retries fail, return an error message as context
                return {"context": "Web search is temporarily unavailable. Please try again."}

def rag_node(state: AgentState):
    print("--- KNOWLEDGE AGENT: SEARCHING PDF ---")
    retrieved_docs = retriever.invoke(state["question"])
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return {"context": context}

def analysis_node(state: AgentState):
    print("--- ANALYSIS AGENT: REASONING ---")
    prompt = f"Extract the key facts to answer this question:\nQ: {state['question']}\nContext: {state['context']}"
    return {"analysis": llm.invoke([HumanMessage(content=prompt)]).content}

def response_node(state: AgentState):
    print("--- FINAL RESPONSE AGENT: DRAFTING ---")
    prompt = f"Write a professional answer.\nQ: {state['question']}\nFacts: {state['analysis']}"
    response = llm.invoke([HumanMessage(content=prompt)]).content
    return {"answer": response, "iterations": state.get("iterations", 0) + 1}

def critique_node(state: AgentState):
    print("--- CRITIC AGENT: FACT-CHECKING ---")
    prompt = f"Does the answer rely on the context? Reply PASS or FAIL.\nContext: {state['context']}\nAnswer: {state['answer']}"
    grade = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    print(f"Critique: {grade}")
    return {"critique": grade}

# 5. Routing Logic
def route_decision(state: AgentState) -> Literal["web", "pdf"]:
    return state["route"]

def route_critique(state: AgentState) -> Literal["analysis", "end"]:
    if "FAIL" in state.get("critique", "") and state.get("iterations", 0) < 2:
        print("--- LOOPING BACK TO ANALYSIS ---")
        return "analysis"
    return "end"

# 6. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("rag_search", rag_node)
workflow.add_node("analysis_agent", analysis_node)
workflow.add_node("response_agent", response_node)
workflow.add_node("critic", critique_node)

workflow.set_entry_point("orchestrator")

workflow.add_conditional_edges(
    "orchestrator",
    route_decision,
    {
        "web": "web_search",
        "pdf": "rag_search"
    }
)

workflow.add_edge("web_search", "analysis_agent")
workflow.add_edge("rag_search", "analysis_agent")
workflow.add_edge("analysis_agent", "response_agent")
workflow.add_edge("response_agent", "critic")

workflow.add_conditional_edges(
    "critic",
    route_critique,
    {
        "analysis": "analysis_agent",
        "end": END
    }
)

app = workflow.compile()