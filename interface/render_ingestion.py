import pymupdf
import streamlit as st

from pymprev.util import GraphRAG

def ingest_documents_visual(graphRAG:GraphRAG, input_path: str):
    filenames = graphRAG._get_filenames(input_path)

    for file_index, filename in enumerate(filenames, start=1):
        st.markdown(f"**Processing file ({file_index}/{len(filenames)})**: [`{filename}`]")

        started, ingested = graphRAG._get_source_ingested(filename)
        if started and ingested:
            st.caption("File already processed... skipping ingestion.")
            continue
        elif started and not ingested:
            st.caption("File ingestion interrupted... restarting ingestion.")
            graphRAG.remove_document(filename)

        doc = pymupdf.open(filename) # doc pages
        full_text = "\n".join([page.get_text() for page in doc])

        # add source file to graph
        graphRAG._add_source_node(filename, doc.metadata)

        # splits document in chunks
        chunks = graphRAG.splitter.split_text(full_text)
        progress_text = "Processing text chunks"
        my_bar = st.progress(0, text=progress_text)
        for chunk_index, chunk in enumerate(chunks, start=1):
            percent_complete = chunk_index/len(chunks)
            my_bar.progress(percent_complete, text=progress_text + f" ({chunk_index}/{len(chunks)})")
            graphRAG.ingest_text_chunk(chunk, chunk_index, filename)

        with st.spinner("Finaling file ingestion...", show_time=True):
            # add "NEXT_CHUNK" and "HAS_CHUNK" relations
            # graphRAG._add_next_chunk_relations(filename=filename) 
            graphRAG._add_source_has_chunk_relations(filename=filename)

            # set file status as completed
            graphRAG._set_source_ingested(filename=filename)

            # ensure nodes are overloaded with type Node
            graphRAG._set_entity_node_types()


def document_ingestion_form(graphRAG:GraphRAG):
    st.subheader("Document Ingestion")
    with st.form(key="ingestion_form"):
        input_path = st.text_input(
            label="Target Directory Path",
            placeholder="/path/to/documents",
            help="Enter the absolute or relative path to the folder containing your documents.")
        
        submit_button = st.form_submit_button(label="Start Ingestion")
    
    # ingestion pipeline
    if submit_button:
        with st.expander("Ingestion Console Output", expanded=True):
            ingest_documents_visual(graphRAG, input_path.strip())



def document_list_visual(graphRAG:GraphRAG):
    st.subheader("Document Manager")
    
    filenames = graphRAG.get_unique_documents()

    document_lookup = st.button("Load Documents in Database")
    if document_lookup:
        with st.expander(f"Total of Documents Found: {len(filenames)}", expanded=False):
            for filename in filenames:
                details = graphRAG.get_file_details(filename)[0]
                chunks = details["chunks"]
                ingestion = details["file"].get("ingested", None)
        
                st.markdown(f"**Document:** `{filename}`")
                st.markdown(f"text chunks: `{chunks}`, ingestion complete: `{ingestion}`")
                        
                with st.expander("see document details and options"):
                    st.caption(details)
                    st.button("Remove document", type="primary", key=filename,
                            on_click=graphRAG.remove_document, kwargs={"filename":filename})


def run_leiden_clusterization_visual(
    graphRAG:GraphRAG,
    max_cluster_size, resolution, ontology_weight,
    low_threshold, high_threshold, text_threshold, top_k_unify,
    score_threshold, top_k_connect):

    # delete any clustering
    graphRAG.delete_clustering()
    st.markdown("**Removed any previous Leiden clustering**")

    if "leiden_clustering" in st.session_state.config:
        results = st.session_state.config["leiden_clustering"]
    else:
        with st.spinner("Generating Leiden cluster structure", show_time=True):
            results = graphRAG.preview_leiden_clustering(
                max_cluster_size, resolution, ontology_weight,
                low_threshold, high_threshold, text_threshold, top_k_unify,
                score_threshold, top_k_connect)
    clusters, level_cluster_map, distribution = results
    st.markdown(f"**Leiden clustering determined:** {distribution}")

    # create clusters and summarizations
    with st.spinner("Saving Leiden cluster structure to database", show_time=True):
        graphRAG._add_cluster_node_relations(clusters)
        graphRAG._add_cluster_subcluster_relations()
    st.markdown("**Cluster structure added to database**")

    for level in range(len(level_cluster_map)-1, -1, -1):
        with st.spinner(f"Creating bottom-up cluster summarization: level {level}", show_time=True):
            progress_text = "Summarizing cluters"
            progress_bar = st.progress(0, text=progress_text)
            for cluster_index, cluster_id in enumerate(level_cluster_map[level], start=1):
                leaf_only = level==len(level_cluster_map)-1
                cluster_context = graphRAG._get_cluster_context(cluster_id, leaf_only)
                graphRAG._add_cluster_summary(cluster_id, cluster_context)
                graphRAG._add_cluster_embedding(cluster_id, cluster_context)

                # update progreess
                percent_complete = cluster_index/len(level_cluster_map[level])
                curr_progress_text = progress_text + f" ({cluster_index}/{len(level_cluster_map[level])})"
                progress_bar.progress(percent_complete, text=curr_progress_text)
            progress_bar.progress(percent_complete, text=curr_progress_text)
        # st.markdown(f"**Done summarizing level {level}**")

def leiden_clusterization(graphRAG:GraphRAG):
    st.subheader("Graph Configuration")

    with st.form(key="leiden_clustering_form"):
        
        st.markdown("**Entity Unification**")

        subcol011, subcol012 = st.columns(2)
        with subcol011:
            high_threshold = st.slider(
                "High Threshold", min_value=0.9, max_value=1.0, value=0.995, step=0.001,
                format="plain",
                help="Unifies entities if semantic similarity is above threshold.")
        with subcol012:
            low_threshold = st.slider(
                "Low Threshold", min_value=0.5, max_value=1.0, value=0.95, step=0.01,
                help="Unifies entities if semantic and text similarities are above this threshold.")

        subcol021, subcol022 = st.columns(2)
        with subcol021:
            text_threshold = st.slider(
                "Text Threshold", min_value=0.5, max_value=1.0, value=0.95, step=0.01,
                help="Unifies entities if semantic and text similarities are above this threshold.")
        with subcol022:
            top_k_unify = st.number_input(
                "K Nearest Neighbors", min_value=0, value=10, step=1,
                help="number of closest neighbors to unify by similarity")

        st.caption("Attention! Entity unification is NOT reversible.")

        st.divider()
        st.markdown("**Entity Connection**")

        col11, col12 = st.columns(2)
        with col11:
            score_threshold = st.slider(
                "Similarity Threshold", min_value=0.5, max_value=1.0, value=0.95, step=0.01,
                help="connects closest entities if semantic similarity is above this threshold.")
        with col12:
            top_k_connect = st.number_input(
                "K Nearest Neighbors", min_value=0, value=10, step=1,
                help="number of closest neighbors to connect by similarity")

        st.caption("Entity similarity connection is not definitive.")

        st.divider()
        st.markdown("**Leiden Clusterization**")

        col21, col22, col23 = st.columns(3)
        with col21:
            resolution = st.slider(
                "Leiden Resolution", min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                help="Controls the coarseness. Higher values produce more, smaller clusters.")
        with col22:
            ontology_weight = st.slider(
                "Ontology Weight", min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                help="Relative importance of extracted ontology versus similarity enrichment.")
        with col23:
            max_cluster_size = st.number_input(
                "Max Cluster Size", min_value=0, value=15, step=1,
                help="Maximun number of elements a cluster might have before broken down.")


        st.divider()
        preview_buttom_graph = st.form_submit_button("Precompute Connected Components")
        if preview_buttom_graph:
            with st.spinner(f"Precomputing components. This may take a while", show_time=True):
                results = graphRAG.preview_entity_components(
                    high_threshold, low_threshold, text_threshold, top_k_unify,
                    score_threshold, top_k_connect)
                components, component_number, component_distribution = results
        
            st.markdown(f"**Number of components:** {component_number}")
            st.markdown(f"**Component size distribution:** {component_distribution}")

        
        st.divider()
        preview_buttom_leiden = st.form_submit_button("Precompute Leiden Clusterization")
        if preview_buttom_leiden:
            with st.spinner(f"Precomputing Leiden components", show_time=True):
                results = graphRAG.preview_leiden_clustering(
                    max_cluster_size=max_cluster_size, resolution=resolution, ontology_weight=ontology_weight,
                    high_threshold=high_threshold, low_threshold=low_threshold, text_threshold=text_threshold, top_k_unify=10,
                    score_threshold=score_threshold, top_k_connect=top_k_connect)
                st.session_state.config["leiden_clustering"] = results
                clusters, level_cluster_map, distribution = results

            st.markdown(f"**Component Distribution:** {distribution}")

        st.divider()
        commit_buttom = st.form_submit_button("Commit Leiden Clusterization", type="primary")
    if commit_buttom:
        # unifying nodes
        # with st.expander("Commiting Entity Unification", expanded=True):
        #     for node_type in graphRAG.ontology.nodes:
        #         with st.spinner(f"Unifying nodes of type `{node_type}`", show_time=True):
        #             merged_count = graphRAG._unify_nodes_by_similarity_type(
        #                 node_type, low_threshold, high_threshold,
        #                 text_threshold, top_k_unify)
        #         st.markdown(f"Total nodes of type `{node_type}` merged: {merged_count}")

        run_leiden_clusterization_visual(
            graphRAG,
            max_cluster_size, resolution, ontology_weight,
            high_threshold, low_threshold, text_threshold, top_k_unify,
            score_threshold, top_k_connect)


def danger_zone(graphRAG:GraphRAG, safety_message="I am realy sure"):
    st.subheader("Danger Zone")
    
    with st.form(key="danger_zone"):
        safety_key_input = st.text_input(
            label=f"Type the safety message to confirm: {safety_message}",
            placeholder="safety message",
            help="Enter the safety message to confirme the changes.")
        
        reset_embeddings = st.form_submit_button("Reset Embeddings", type="primary")
        # delete_clustering = st.form_submit_button("Delete Clustering", type="primary")
        reset_databasis = st.form_submit_button("Reset Databasis", type="primary")

        if reset_embeddings and safety_key_input==safety_message:
            with st.spinner("Reseting graph node embeddings"):
                graphRAG.reset_node_embeddings(batch_size=10)
            st.markdown("**Done reseting graph node embeddings**")

            with st.spinner("Reseting text chunk embeddings"):
                graphRAG.reset_chunk_embeddings(batch_size=10)
            st.markdown("**Done reseting text chunk embeddings**")

            with st.spinner("Reseting cluster summary embeddings"):
                graphRAG.reset_cluster_embeddings(batch_size=10)
            st.markdown("**Done reseting cluster summary embeddings**")

        if reset_databasis and safety_key_input==safety_message:
            graphRAG.reset_databasis()
            st.markdown("**Done reseting entire databasis**")


def render_ingestion_page():
    st.title("Document Ingestion & Configuration")
    st.caption("Ingest files, entity unification and topic summarization.")

    # get graphRAG from session state
    graphRAG = st.session_state.config["graphRAG"]

    # filepath ingestion form
    document_ingestion_form(graphRAG)

    # Document List
    document_list_visual(graphRAG)

    # leiden clusterization
    leiden_clusterization(graphRAG)

    # danger zone
    danger_zone(graphRAG)
