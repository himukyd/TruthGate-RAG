import streamlit as st
import requests
import json

st.set_page_config(page_title="TruthGate RAG", page_icon="🛡️")

st.title("🛡️ TruthGate: FastAPI Expert")
st.markdown("""
This RAG system is optimized to **refuse** answers not found in the documentation 
and detect **false premises**.
""")

query = st.text_input("Ask a question about FastAPI:", placeholder="e.g., How do I use path parameters?")

if st.button("Query TruthGate"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Consulting the documentation..."):
            try:
                response = requests.post(
                    "http://localhost:8000/query",
                    json={"query": query},
                    timeout=180
                )
                response.raise_for_status()
                data = response.json()
                
                # Display Status Badge
                status = data["status"]
                if status == "Success":
                    st.success("Answer Found")
                elif status == "Refused":
                    st.error("Not answerable from the docs")
                else:
                    st.warning("False Premise Detected")
                
                # Display Answer
                st.subheader("Answer")
                st.write(data["answer"])
                
                # Display Citations
                if data["citations"]:
                    with st.expander("Sources & Citations"):
                        for cite in data["citations"]:
                            st.markdown(f"**[{cite['section']}]({cite['source']})**")
                            st.caption(cite["content"])
                
                # Display Metrics
                st.divider()
                col1, col2 = st.columns(2)
                col1.metric("Latency", f"{data['latency']:.2f}s")
                col2.metric("Estimated Cost", f"${data['cost']:.5f}")
                
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
                st.info("Make sure the backend is running at http://localhost:8000")

st.sidebar.title("About TruthGate")
st.sidebar.info("""
Built for the AI Engineer Assessment.
- **LLM:** Qwen2.5-1.5B (Local)
- **Embeddings:** all-MiniLM-L6-v2 (Local)
- **Vector DB:** ChromaDB
- **Features:** False Premise Detection, Refusal Logic, Mandatory Citations.
""")
