import streamlit as st
import time
from graph import app

st.set_page_config(page_title="Agentic RAG System", page_icon="🤖")
st.title("🤖 Agentic RAG with Critic Loop")
st.write("This system uses Analysis, Generation, and Critique agents to ensure accurate answers.")

question = st.text_input("Ask a question about the document:")

if st.button("Generate Answer"):
    if question:
        start_time = time.time() # <-- START TIMER
        status_text = st.empty()
        final_answer = st.empty()
        
        inputs = {"question": question, "iterations": 0}
        
        # We need a variable to store the final state for the metrics
        final_state = None 
        
        for output in app.stream(inputs):
            for key, value in output.items():
                
                # ... (Keep your existing status_text logic here) ...
                if key == "orchestrator":
                    status_text.info(f"🧭 Orchestrator: Routing to {value.get('route', '').upper()}...")
                elif key == "web_search":
                    status_text.info("🌐 Research Agent: Searching the live web...")
                elif key == "rag_search":
                    status_text.info("📄 Knowledge Agent: Searching the PDF...")
                elif key == "analysis_agent":
                    status_text.info("🧠 Analysis Agent: Extracting facts...")
                elif key == "response_agent":
                    status_text.info("✍️ Final Response Agent: Drafting...")
                elif key == "critic":
                    critique = value.get("critique", "")
                    if "PASS" in critique:
                        status_text.success("✅ Critic: Fact-check PASSED.")
                    else:
                        status_text.warning("⚠️ Critic: Fact-check FAILED. Looping back...")
                
                time.sleep(0.5)
                
                if key == "response_agent":
                    final_answer.markdown(f"### Final Answer:\n{value['answer']}")
                
                # Save the final state for metrics
                final_state = value 

        end_time = time.time() # <-- END TIMER
        latency = round(end_time - start_time, 2)
        
        # --- EVALUATOR DASHBOARD ---
        status_text.empty()
        st.divider()
        st.subheader("📊 Evaluation Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        # 1. Latency
        col1.metric(label="Total Latency", value=f"{latency} sec")
        
        # 2. Hallucination Check (from Critic)
        if final_state and "critique" in final_state:
            is_pass = "PASS" in final_state.get("critique", "")
            col2.metric(label="Hallucination Check", value="PASSED" if is_pass else "FAILED")
            
        # 3. Route Taken (Web vs PDF)
        if final_state and "route" in final_state:
            col3.metric(label="Agent Route", value=final_state.get("route", "Unknown").upper())

    else:
        st.warning("Please enter a question first.")