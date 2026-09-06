import streamlit as st

from render_questions import render_questions_page
from render_hierarchy import render_hierarchy_page
from render_ingestion import render_ingestion_page
from render_credentials import render_credentials_page


def welcome_page():
    st.write("# Welcome to Py-MPREV")

    st.markdown(
    """
    ## Python Preventive Maitenance Document Explorer
        
    This project focuses on the discussion of applicability, requirements, and employment of
    ML-Based techniques for preventive maintenance in the context of civil aviation, with a
    focus on Preventive and Health Management (PHM) tools. This is justified by the potential
    impact of Artificial Intelligence and Machine Learning in this sector, which requires strict
    regulations, currently not compatible with the most recent technologies.
    """)

# session initialization
if "connected" not in st.session_state:
    st.session_state.connected = False
if "config" not in st.session_state:
    st.session_state.config = {}

# routing map
page_routing = {
    "Welcome": welcome_page,
    "Ingestion & Configuration": render_ingestion_page,
    "Vector Similarity Q&A": render_questions_page,
    "Topic Hierarchy Navigator": render_hierarchy_page,
}

def main():
    st.sidebar.title("Navigation")

    if not st.session_state.connected:
        st.sidebar.warning("⚠️ Disconnected")
        render_credentials_page()
    else:
        st.sidebar.success("🟢 Connected")
        
        # Sidebar Menu Selection
        page_name = st.sidebar.selectbox(
            "Select Page:",
            page_routing
        )
        try:
            page_routing[page_name]()
        except:
            pass

        st.sidebar.divider()
        st.sidebar.caption(f"**LLM:** `{st.session_state.config['llm_model']}`")
        st.sidebar.caption(f"**Embedder:** `{st.session_state.config['embedder_model']}`")
        st.sidebar.caption(f"**DB URI:** `{st.session_state.config['database_uri']}`")
                
        if st.sidebar.button("Disconnect / Edit Keys"):
            st.session_state.connected = False
            st.session_state.config = {}
            st.rerun()


if __name__ == "__main__":
    main()