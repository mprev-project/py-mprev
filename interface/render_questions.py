import streamlit as st

def render_questions_page():
    st.title("Vector Search Q&A Assistant")
    st.caption("Ask natural language questions to query community clusters and source chunks.")

    # get graphRAG from session state
    graphRAG = st.session_state.config["graphRAG"]

    with st.form(key="rag_form"):
        user_query = st.text_input(
            "Enter your query:", 
            value="What safety problems are associated to Neural Nets in Aviation?")
        top_k_global = st.number_input("Top K Global", min_value=1, max_value=20, value=10)
        top_k_local = st.number_input("Top K Local", min_value=1, max_value=20, value=10)
                
        submit_button = st.form_submit_button(label="Query RAG", type="primary")

    if submit_button and user_query:
        with st.spinner("Retrieving vector matches and generating answer..."):
            # call your backend GraphRAG
            result = graphRAG.query_RAG(
                user_query,
                top_k_global=top_k_global,
                top_k_local=top_k_local)

        st.markdown(f"### {result['title']}")
        st.success(result["answer"])

        st.subheader("Retrieved Context")
        with st.expander(f"Matched Cluster Summaries ({len(result['clusters'])})"):
            for cluster in result["clusters"]:
                st.markdown(f"• **[{cluster['name']} - Level {cluster['level']}]** *(Similarity: {cluster['score']:.2f})*")
                st.caption(f"\"{cluster['summary']}\"")

        with st.expander(f"Matched Text Chunks ({len(result['chunks'])})"):
            for chunk in result["chunks"]:
                st.markdown(f"• **Excerpt [`{chunk['index']}`] - Document: {chunk['source']}** *(Similarity: {chunk['score']:.2f})*")
                st.caption(f"\"{chunk['text']}\"")