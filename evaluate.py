import time
from graph import app
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

# A separate LLM just for grading
evaluator_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# Test questions based on the research paper
test_cases = [
    "What is the main thesis of this paper?",
    "How does memory failure impact LLM group decisions?",
    "What methodologies did the researchers use?",
    "What are the key findings of the study?",
    "Who are the authors of this paper?" # This might trigger a FAIL in the critic if authors aren't explicitly in the text
]

def grade_answer(question, context, answer):
    prompt = f"""
    You are an objective evaluator. Grade the following answer based on the context.
    Question: {question}
    Context: {context}
    Answer: {answer}
    
    Return exactly two numbers separated by a comma:
    1. Relevance Score (1-10): How well does the answer address the question?
    2. Hallucination Score (0 or 1): 0 if the answer is fully supported by context, 1 if it makes things up.
    Example output: 9, 0
    """
    response = evaluator_llm.invoke([HumanMessage(content=prompt)])
    return response.content

print("Starting Evaluation...\n")
print(f"{'Question':<40} | {'Latency (s)':<12} | {'Grade (Rel, Hall)':<15}")
print("-" * 75)

for question in test_cases:
    start_time = time.time()
    
    # Run the graph
    inputs = {"question": question, "iterations": 0}
    result = app.invoke(inputs)
    
    end_time = time.time()
    latency = round(end_time - start_time, 2)
    
    # Grade the result
    grade = grade_answer(question, result["context"], result["answer"])
    
    print(f"{question[:37]:<40} | {latency:<12} | {grade:<15}")

print("\nEvaluation Complete.")