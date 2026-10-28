import os
from typing import TypedDict, Literal
from dotenv import load_dotenv

# Load environment variables securely
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Setup API
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# 2. Setup Vector Database (Load PDF)
loader = PyPDFLoader("project pdf.pdf") # Change to your technical PDF later
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Define State
class AgentState(TypedDict):
    question: str
    context: str
    analysis: str
    answer: str
    critique: str
    iterations: int

# 4. Define Nodes
def retrieve_node(state: AgentState):
    print("--- 1. RETRIEVING CONTEXT ---")
    retrieved_docs = retriever.invoke(state["question"])
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return {"context": context}

def analysis_node(state: AgentState):
    print("--- 2. ANALYZING EVIDENCE ---")
    prompt = f"""
    You are a Research Analysis Agent. Read the provided context and extract the exact facts needed to answer the question. 
    Do not write a conversational answer. Just output the raw facts and reasoning.
    
    Question: {state['question']}
    Context: {state['context']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"analysis": response.content}

def response_node(state: AgentState):
    print("--- 3. GENERATING FINAL RESPONSE ---")
    prompt = f"""
    You are a Final Response Agent. Use the raw analysis below to write a clear, professional answer to the user's question.
    If the analysis says the context does not contain the answer, state that you don't know.
    
    Question: {state['question']}
    Raw Analysis: {state['analysis']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    
    current_iterations = state.get("iterations", 0) + 1
    return {"answer": response.content, "iterations": current_iterations}

def critique_node(state: AgentState):
    print("--- 4. CRITIQUING ANSWER ---")
    prompt = f"""
    You are a strict Fact-Checking Critic. Does the Final Answer rely ONLY on the provided Context? 
    If it hallucinates or ignores the context, you must FAIL it.
    
    Context: {state['context']}
    Final Answer: {state['answer']}
    
    Output exactly one word: PASS or FAIL.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    grade = response.content.strip().upper()
    print(f"Critique Result: {grade}")
    
    return {"critique": grade}

# 5. Define Routing Logic
def route_critique(state: AgentState) -> Literal["analysis", "end"]:
    if "FAIL" in state.get("critique", "") and state.get("iterations", 0) < 2:
        print("--- ROUTING: RE-ANALYZING (Looping Back) ---")
        return "analysis"
    print("--- ROUTING: FINISHING ---")
    return "end"

# 6. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("retriever", retrieve_node)
workflow.add_node("analysis_agent", analysis_node)
workflow.add_node("response_agent", response_node)
workflow.add_node("critic", critique_node)

workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "analysis_agent")
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