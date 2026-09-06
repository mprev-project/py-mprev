import streamlit as st

# Initialize GraphRAG backend
from pymprev.util import GraphRAG
from pymprev.ontology import Ontology
from pymprev.connection import connect_llm, connect_embedder, connect_database

# credentials page
def render_credentials_page():
    st.title("Graph RAG Setup & Credentials")
    st.caption("Enter your model configurations and database connection details to initialize the system.")

    with st.form(key="llm_connection_form"):
        # LLM Settings
        st.subheader("LLM Configuration")
        llm_model = st.text_input(
            "LLM Model name",
            value="gemma-4-31b-it")
        llm_api_key = st.text_input(
            "LLM API Key", 
            type="password",
            help="Your LLM key/password")

        st.divider()

        # Embedder Settings
        st.subheader("Embedder Configuration")
        embedder_model = st.text_input(
            "Embedder Model", 
            value="google/embeddinggemma-300m")
        embedder_key = st.text_input(
            "Embedder Model key/password", 
            type="password",
            help="your Embedder Model key/password")

        st.divider()

        # Database Settings
        st.subheader("Neo4j Database Connection")
        database_uri = st.text_input(
            "Database URI",
            value="neo4j+s://312151bc.databases.neo4j.io")
        database_name = st.text_input(
            "Database name",
            value="312151bc")
        database_username = st.text_input(
            "Database Username",
            value="312151bc")
        database_password = st.text_input(
            "Database Password",
            type="password",
            help="Your database access key/password")

        submit_button = st.form_submit_button(label="Connect & Start Session", type="primary")

    if submit_button:
        # Validate missing fields
        if not llm_api_key or not embedder_key or not database_password:
            st.error("Please fill in all secret key and password fields.")
            return

        # Save configuration in session state
        st.session_state.config = {
            "llm_model": llm_model,
            "llm_api_key": llm_api_key,
            "embedder_model": embedder_model,
            "embedder_key": embedder_key,
            "database_uri": database_uri,
            "database_name": database_name,
            "database_username": database_username,
            "database_password": database_password
        }
        st.session_state.cluster_details = dict()

        with st.spinner("Connecting and initializing models..."):
            try:
                # connect to GraphRag
                ontology = Ontology()
                llm = connect_llm(
                    model_name=llm_model, api_key=llm_api_key)
                embedder = connect_embedder(
                    model_name=embedder_model, api_key=embedder_key) 
                graph = connect_database(
                    uri=database_uri, database=database_name,
                    username=database_username, password=database_password)

                graphRAG = GraphRAG(llm, embedder, graph, ontology)
                
                st.session_state.connected = True
                st.session_state.config["graphRAG"] = graphRAG
                st.success("Connection established successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"Failed to connect to database or initialize models: {e}")
