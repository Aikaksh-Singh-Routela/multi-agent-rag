import streamlit as st
import time
from graph import app

st.set_page_config(page_title="Agentic RAG System", page_icon="🤖")
st.title("🤖 Agentic RAG with Critic Loop")
st.write("This system uses Analysis, Generation, and Critique agents to ensure accurate answers.")

question = st.text_input("Ask a question about the document:")

if st.button("Generate Answer"):
    if question:
        # Create placeholders to update the UI in real-time
        status_text = st.empty()
        final_answer = st.empty()
        
        inputs = {"question": question, "iterations": 0}
        
        # We use app.stream() instead of app.invoke() to see the steps as they happen!
        for output in app.stream(inputs):
            for key, value in output.items():
                
                if key == "retriever":
                    status_text.info("🔍 Agent 1: Retrieving context from the document...")
                elif key == "analysis_agent":
                    status_text.info("🧠 Agent 2: Analyzing evidence and extracting facts...")
                elif key == "response_agent":
                    status_text.info("✍️ Agent 3: Drafting the final response...")
                elif key == "critic":
                    critique = value.get("critique", "")
                    if "PASS" in critique:
                        status_text.success("✅ Agent 4: Fact-check PASSED.")
                    else:
                        status_text.warning("⚠️ Agent 4: Fact-check FAILED. Looping back to fix...")
                
                time.sleep(0.5) # Just to make the UI transitions smooth
                
                # If this is the final response step, display it
                if key == "response_agent":
                    final_answer.markdown(f"### Final Answer:\n{value['answer']}")
        
        status_text.empty() # Clear the status bar when done
    else:
        st.warning("Please enter a question first.")