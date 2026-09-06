import streamlit as st


def _load_cluster_information(cluster_id):
    # get graphRAG from session state
    graphRAG = st.session_state.config["graphRAG"]

    name, summary, level = graphRAG.get_cluster_details(cluster_id)
    childnodes = graphRAG.get_children_entities(cluster_id)
    subclusters = graphRAG.get_children_clusters(cluster_id)

    st.session_state.cluster_details[cluster_id] = {
        "name": name,
        "level":level,
        "summary": summary,
        "childnodes": childnodes,
        "subclusters": subclusters}


def _expand_cluster_tree_id(cluster_id):
    # get graphRAG from session state
    graphRAG = st.session_state.config["graphRAG"]

    if cluster_id not in st.session_state.cluster_details:
        name, summary, level = graphRAG.get_cluster_details(cluster_id)
    else:
        name = st.session_state.cluster_details[cluster_id]["name"]
        level = st.session_state.cluster_details[cluster_id]["level"]
        summary = st.session_state.cluster_details[cluster_id]["summary"]

    st.markdown(f"**Cluster {cluster_id}:** {name}")
    with st.expander(f"See More: summary and subnodes", expanded=False):
        st.markdown(f"**Summary:** {summary}")

        # load further details
        st.markdown("---")
        if cluster_id not in st.session_state.cluster_details:
            st.button(
                "Load Cluster Details",
                key=f"btn_{cluster_id}",
                on_click=_load_cluster_information,
                kwargs={"cluster_id":cluster_id})
        else:
            childnodes = st.session_state.cluster_details[cluster_id]["childnodes"]
            subclusters = st.session_state.cluster_details[cluster_id]["subclusters"]

            if childnodes:
                st.markdown(f"**Child Nodes: {len(childnodes)}**")
                for childnode in childnodes:
                    st.markdown(f" - **{childnode['node_id']} {childnode['labels']}**: {childnode['description']}")

            if subclusters:
                st.markdown(f"**Sub-Clusters: {len(subclusters)}**")
                for subcluster in subclusters:
                    subcluster_id = subcluster["child_id"]
                    _expand_cluster_tree_id(subcluster_id)


def render_hierarchy_page():
    st.title("Top-Bottom Summarization Hierarchy")
    st.caption("Explore how general community topics branch down into granular sub-summaries.")

    # get graphRAG from session state
    graphRAG = st.session_state.config["graphRAG"]

    # top-bottom tree
    cluster_ids = graphRAG.get_cluster_ids_by_level(0)
    for cluster_id in cluster_ids:
        _expand_cluster_tree_id(cluster_id)