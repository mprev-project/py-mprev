import os
import tqdm
import pymupdf
import networkx as nx

from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_neo4j.graph_transformers.llm import LLMGraphTransformer

# community detection
import graspologic_native

# querying
from langchain_neo4j import GraphCypherQAChain


class GraphRAG:
    def __init__(self, llm, embedder, graph, ontology):
        self.llm = llm
        self.embedder = embedder
        self.graph = graph
        self.ontology = ontology

        # get ontology-based subgraph
        # self.node_types = " | ".join(ontology.nodes)
        # self.relation_types = " | ".join(ontology.relations_schema.keys())

        # create index names
        self.entity_vector_index_name = "entity_vector_index"
        self.chunk_vector_index_name = "chunk_vector_index"
        self.cluster_vector_index_name = "cluster_vector_index"
        self.entity_property_index_name = "entity_property_index"
        self.chunk_property_index_name = "chunk_property_index"
        self.source_property_index_name = "source_property_index"
        self.cluster_property_index_name = "cluster_property_index"

        self.create_indices()

        # splits text into chunks
        self.splitter = RecursiveCharacterTextSplitter(
            # chunk_size=1000,
            # chunk_overlap=200,
            # separators=["\n\n", "\n", " ", ""]
        )

        # recovers structured data
        self.graph_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes = self.ontology.nodes,
            allowed_relationships = self.ontology.relations,
            node_properties = self.ontology.node_properties,
            relationship_properties = self.ontology.relationship_properties,
            additional_instructions = "\n\n".join([
                "Follow STRICTLY the Ontology:" + self.ontology.description,
                "Answers should be always given in english."
                ])
        )

    def create_indices(self):
        # entity
        self._create_vector_index("Node", self.entity_vector_index_name)
        self._create_property_index("Node", self.entity_property_index_name, "id")
        
        # Chunk
        self._create_vector_index("Chunk", self.chunk_vector_index_name)
        self._create_property_index("Chunk", self.chunk_property_index_name, ["index", "source"])

        # Source
        self._create_property_index("Source", self.source_property_index_name, "filename")

        # Cluster
        self._create_property_index("Cluster", self.cluster_property_index_name, "id")
        self._create_vector_index("Cluster", self.cluster_vector_index_name)


    def _get_filenames(self, path:str):
        # we assume only valid files in the directory
        if os.path.isfile(path):
            return [path]

        if os.path.isdir(path):
            filenames = []
            for subpath in os.listdir(path):
                subpath = os.path.join(path, subpath)
                filenames += self._get_filenames(subpath)

            return filenames

        # raise Exception("Invalid path type found.")


    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=4, max=60))
    def _transform_chunk(self, chunk:str):
        document = Document(page_content=chunk)
        return self.graph_transformer.convert_to_graph_documents([document])


    def _add_source_node(self, filename: str, metadata:dict = {}):
        query = """
        MERGE (f:Source {filename: $filename})
        SET f += $metadata
        SET f.ingested = $value
        """
        params = {"filename": filename, "metadata":metadata, "value":False}
        self.graph.query(query, params=params)


    def _set_source_ingested(self, filename: str):
        query = """
        MATCH (f:Source {filename: $filename})
        SET f.ingested = $value
        """
        params = {"filename": filename, "value":True}
        self.graph.query(query, params=params)


    def _get_source_ingested(self, filename: str):
        """
        Returns a pair informing if file was found, and if has been totally ingested
        """

        query = """
        MATCH (f:Source {filename: $filename})
        RETURN f.ingested as ingested
        """
        params = {"filename": filename}
        result = self.graph.query(query, params=params)
        if result:
            return True, result[0]["ingested"]
        return (False, False)

        
    def _add_source_node(
            self, chunk_text:str, chunk_index:int, filename:str):
        query = """
        MERGE (c:Chunk {text: $text, embedding:$embedding, index: $index, source: $filename})
        """

        chunk_prefix = "title: none | text: "
        chunk_embedding = self.embedder.embed_query(f"{chunk_prefix}{chunk_text}")
        
        params = {
            "text": chunk_text,
            "index": chunk_index,
            "embedding": chunk_embedding,
            "filename": filename}
        self.graph.query(query, params=params)


    def _add_chunk_mentions_node_relations(self, node_ids, chunk_index:int, filename:str):
        query = """
        MATCH (c:Chunk {source: $filename, index: $index})
        UNWIND $node_ids AS node_id
        MATCH (n) WHERE n.id = node_id
        MERGE (c)-[:MENTIONS]->(n)
        """
        params = {"node_ids": node_ids, "index": chunk_index, "filename": filename}
        self.graph.query(query, params=params)


    def _get_unifiable_edges(
            self, node_type,
            low_threshold=0.95, high_threshold=0.995, text_threshold=0.95, top_k=10):
        query = """
        MATCH (target:Node:{node_type})
        CALL (target) {{
            MATCH (duplicate:Node)
            SEARCH duplicate IN (
                VECTOR INDEX {index_name}
                FOR target.embedding
                LIMIT $top_k
            ) SCORE as vec_score
            WHERE
                duplicate:{node_type} AND
                target.id < duplicate.id AND
                vec_score >= $low_threshold
            RETURN duplicate, vec_score
        }}
        WITH target, duplicate, vec_score,
            apoc.text.levenshteinSimilarity(toLower(target.id), toLower(duplicate.id)) AS text_score
        WHERE (vec_score >= $high_threshold) OR (text_score >= $text_threshold)
        RETURN
            target.id AS target, 
            duplicate.id AS duplicate, 
            vec_score,
            text_score
        ORDER BY vec_score DESC
        """.format(node_type=node_type, index_name=self.entity_vector_index_name)

        params={"low_threshold":low_threshold,
                "high_threshold":high_threshold,
                "text_threshold":text_threshold,
                "top_k":top_k}
        
        unifiable = self.graph.query(query, params=params)
        return unifiable


    def _unify_nodes_by_similarity_type(
            self, node_type, low_threshold, high_threshold, text_threshold, top_k, batch_size=1000):
        query = """
        UNWIND $candidates AS row
        MATCH (target:Node:{node_type}) WHERE target.id = row.target_id
        MATCH (duplicate:Node:{node_type}) WHERE duplicate.id = row.duplicate_id
        MERGE (target)-[:TEMP_SAME_AS]-(duplicate)
        """

        merge_query = """
        MATCH (n:Node:{node_type})-[:TEMP_SAME_AS]-()
        CALL apoc.path.subgraphNodes(n, {{relationshipFilter: "TEMP_SAME_AS"}}) YIELD node AS m
        // group by n so each component is collected individually
        WITH n, collect(DISTINCT m) AS cluster
        WHERE size(cluster) > 1
        // identify the representative ID for each component
        WITH coll.min([x IN cluster | x.id]) AS rep_id, cluster
        // deduplicate to execute 1 merge component
        WITH DISTINCT rep_id, cluster
        WITH
            [x IN cluster WHERE x.id = rep_id][0] AS target,
            [x IN cluster WHERE x.id <> rep_id] AS duplicates
        CALL apoc.refactor.mergeNodes([target] + duplicates, {{
            properties: {{ name: "discard", description: "combine", `.*`: "override" }},
            mergeRels: true
        }}) YIELD node
        RETURN count(node) AS merged_count
        """

        cleanup_query = "MATCH ()-[r:TEMP_SAME_AS]-() DELETE r"

        unifiable = self._get_unifiable_edges(
            node_type, low_threshold, high_threshold, text_threshold, top_k)
            
        candidates = [{"target_id":row["target"],
                       "duplicate_id":row["duplicate"]} for row in unifiable]

        total_merged = 0
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            
            # creates temporary relations
            self.graph.query(query.format(node_type=node_type),
                            params={"candidates":batch})

            # merges temporary component
            res = self.graph.query(merge_query.format(node_type=node_type))
            total_merged += res[0]["merged_count"]

            # removes temporary relations
            self.graph.query(cleanup_query)

        return total_merged


    def unify_nodes_by_similarity(
            self,
            low_threshold=0.95, high_threshold=0.995, text_threshold=0.95, top_k=10):

        for node_type in self.ontology.nodes:
            merged_count = self._unify_nodes_by_similarity_type(
                node_type, low_threshold, high_threshold, text_threshold, top_k)
            print(f"total merged nodes of type {node_type}: {merged_count}")


    # def _add_knn_similarity_relations_type(
    #         self, target_type, score_threshold=0.8, top_k=10, batch_size=1000):
    #     link_query = """
    #     UNWIND $candidates AS group
    #     MATCH (a:Node:{target_type} {{id: group.target_id}})
    #     UNWIND group.neighbors AS nbr
    #     MATCH (b:Node:{duplicate_type} {{id: nbr.duplicate_id}})
    #     MERGE (a)-[r:SIMILAR_TO]-(b)
    #     SET r.score = nbr.score
    #     """

    #     edges = self._get_nearest_edges(
    #         top_k, score_threshold, target_type, target_type)

    #     candidates = dict()
    #     for row in edges:
    #         target_id, duplicate_id, score = row
    #         if target_id not in candidates:
    #             candidates[target_id] = []
    #         candidates[target_id].append({"duplicate_id": duplicate_id, "score":score})

    #     grouped_candidates = [
    #         {"target_id":target_id, "neighbors":neighbors}
    #          for target_id, neighbors in candidates.items()]

    #     query = link_query.format(
    #         target_type=target_type, duplicate_type=target_type)
        
    #     # process candidates in batches
    #     total_connected = 0
    #     for i in range(0, len(grouped_candidates), batch_size):
    #         batch = grouped_candidates[i:i + batch_size]
    #         self.graph.query(query, params={"candidates": batch})
    #         total_connected += len(batch)

    #     return total_connected
    

    # def add_knn_similarity_relations(self, score_threshold=0.8, top_k=10):
    #     for target_type in self.ontology.nodes:
    #         # get neighbors
    #         connected_cunt = self._add_knn_similarity_relations_type(
    #             target_type, score_threshold, top_k)

    #         print(f"total connected nodes from type {target_type}: {connected_cunt}")


    # def _add_next_chunk_relations(self, filename:str):
    #     query = """
    #     MATCH (c1:Chunk {source: $filename})
    #     MATCH (c2:Chunk {source: $filename, index: c1.index + 1})
    #     MERGE (c1)-[:NEXT_CHUNK]->(c2)
    #     """
    #     self.graph.query(query, params={"filename":filename})


    def _add_source_has_chunk_relations(self, filename:str):
        query = """
        MATCH (f:Source {filename: $filename})
        MATCH (c:Chunk {source: $filename})
        MERGE (f)-[:HAS_CHUNK]->(c)
        """
        self.graph.query(query, params={"filename":filename})


    def _set_entity_node_types(self):
        node_types = " | ".join(self.ontology.nodes)
        self.graph.query(f"MATCH (e: {node_types}) WHERE NOT e:Node SET e:Node")


    def ingest_documents(self, input_path:str):
        filenames = self._get_filenames(input_path)

        # ingest one document at a time
        for file_index, filename in enumerate(filenames, start=1):
            print(f"Processing file ({file_index}/{len(filenames)}):",  filename)

            # check if file already processed
            started, ingested = self._get_source_ingested(filename)
            if started and ingested:
                print("File already processed... skipping ingestion.")
                continue
            elif started and not ingested:
                print("File ingestion interrupted... restarting ingestion.")
                self.remove_document(filename)

            doc = pymupdf.open(filename) # doc pages
            full_text = "\n".join([page.get_text() for page in doc])

            # add source file to graph
            self._add_source_node(filename, doc.metadata)

            # splits document in chunks
            chunks = self.splitter.split_text(full_text)
            for chunk_index, chunk in enumerate(tqdm.tqdm(chunks)):
                self.ingest_text_chunk(chunk, chunk_index, filename)

            # add "NEXT_CHUNK" and "HAS_CHUNK" relations
            # self._add_next_chunk_relations(filename=filename) 
            self._add_source_has_chunk_relations(filename=filename)

            # set file status as completed
            self._set_source_ingested(filename=filename)
        
            # ensure nodes are overloaded with type Node
            self._set_entity_node_types()


    def ingest_text_chunk(self, chunk, chunk_index, filename):
        # add shource chunk to graph
        self._add_source_node(
            chunk_text=chunk,
            chunk_index=chunk_index,
            filename=filename)

        # get structured graph from chunk
        chunk_graph = self._transform_chunk(chunk)
        # Attach chunk properties to nodes + relationships
        for element in chunk_graph[0].nodes + chunk_graph[0].relationships:
            element.properties["chunk_index"] = chunk_index
            element.properties["chunk_source"] = filename

        for node in chunk_graph[0].nodes:
            # add node embeddings
            node_text = self._node_as_text(
                node.id,
                node.type,
                node.properties.get("name"),
                node.properties.get("description"))
            node_embedding = self.embedder.embed_query(node_text)
            node.properties["embedding"] = node_embedding
                    
        self.graph.add_graph_documents(chunk_graph)

        # crates chunk mentioning relation to nodes
        node_ids = [node.id for node in chunk_graph[0].nodes]
        self._add_chunk_mentions_node_relations(node_ids, chunk_index, filename)


    def _get_similarity_edges(self,
            low_threshold, high_threshold, text_threshold, top_k_unify,
            score_threshold, top_k_connect, unifiable_weight=100):

        edges = []
        for node_type in self.ontology.nodes:
            results_unify = self._get_unifiable_edges(
                node_type, low_threshold, high_threshold, text_threshold, top_k_unify)
            edges += [(row["target"], row["duplicate"], unifiable_weight) for row in results_unify]

            results_connect = self._get_nearest_edges(
                top_k_connect, score_threshold, node_type, node_type)
            edges += results_connect
        
        return edges
    

    def _get_leiden_clusters(self, edges, max_cluster_size=10, resolution=1.0):
        clusters = graspologic_native.hierarchical_leiden(
             edges, max_cluster_size=max_cluster_size, resolution=resolution)
                
        # find depth and width
        level_cluster_map = dict()
        for row in clusters:
            if row.level not in level_cluster_map:
                level_cluster_map[row.level] = set()
            level_cluster_map[row.level].add(int(row.cluster))

        return clusters, level_cluster_map
        

    def _add_cluster_node_relations(self, clusters, batch_size=1000):
        query_create = """
        UNWIND $clusters AS row
        CALL (row) {{
            MATCH (e:Node {{id: row.node_id}})
            MERGE (c:Cluster {{id: row.cluster_id}})
            SET c.level = row.level,
                c.parent = row.parent
            MERGE (e)-[r:IN_CLUSTER]->(c)
            SET r.final = row.final
        }} IN TRANSACTIONS OF {batch_size} ROWS
        """.format(batch_size = batch_size)

        params = {
            "clusters":[{
                "node_id": row.node,
                "cluster_id": int(row.cluster),
                "level": int(row.level),
                "parent": row.parent_cluster,
                "final": bool(row.is_final_cluster)
            } for row in clusters]}

        self.graph.query(query_create, params=params)


    def _add_cluster_subcluster_relations(self, batch_size=1000):
        query = """
        MATCH (c:Cluster)
        WHERE c.parent IS NOT NULL
        CALL (c) {{
            MATCH (p:Cluster {{id: c.parent}})
            MERGE (c)-[:IN_CLUSTER]->(p)
        }} IN TRANSACTIONS OF {batch_size} ROWS
        """.format(batch_size=batch_size)

        self.graph.query(query)


    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=4, max=60))
    def _get_cluster_summary(self, prompt):
        json_schema = {
            "title": "cluster_summary",
            "description": "Cluster name and summary",
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Short, descriptive title for the cluster (3-6 words)"
                },
                "summary": {
                    "type": "string",
                    "description": "Comprehensive synthesis of the cluster contents"
                }
            },
            "required": ["name", "summary"]}

        # Pass the dict schema directly
        structured_llm = self.llm.with_structured_output(json_schema)
        result = structured_llm.invoke(prompt)

        return result


    def _get_cluster_context(self, cluster_id, leaf_only):
        # non-cluster entity details
        entities = self.get_children_entities(cluster_id)
        entity_texts = [self._node_as_text(r['node_id'], r['labels'], r['name'], r['description']) for r in entities]
        
        # child cluster summaries
        if leaf_only:
            subcluster_texts = []
        else:
            subclusters = self.get_children_clusters(cluster_id)
            subcluster_texts = [self._cluster_as_text(r['child_id'], r['name'], r['summary']) for r in subclusters]

        # Combine into context
        context_parts = []
        if subcluster_texts:
            context_parts.append("### Sub-Community Summaries:\n- " + "\n\n - ".join(subcluster_texts))
        if entity_texts:
            context_parts.append("### Direct Entity Details:\n- " + "\n\n - ".join(entity_texts))
        
        context = "\n\n".join(context_parts)

        return context


    def _add_cluster_summary(self, cluster_id, context):
        prompt = f"""
        Synthesize the following information for Cluster {cluster_id} into a name and a cohesive summary. 
        It contains both broader sub-community summaries and direct entity/concept details.

        {context}

        When relevant, take in consideration the base ontology described in the following:

        {self.ontology.description}
        """

        # request cluster summaries
        try:
            result = self._get_cluster_summary(prompt)
        except Exception as e:
            print(f"Error while parsing cluster '{cluster_id}': ", e)
            print("Prompt: ", prompt)
        else:
            # Store summary on the Cluster node
            query_save = """
            MATCH (c:Cluster {id: $cluster_id})
            SET c.name = $name, c.summary = $summary
            """
            self.graph.query(query_save, params={
                "cluster_id": cluster_id,
                "name": result["name"],
                "summary": result["summary"]
            })


    def _add_cluster_embedding(self, cluster_id, context):
        title, summary, _ = self.get_cluster_details(cluster_id)
        text = f"title: {title} | text: {summary} | context: {context}"

        embedding = self.embedder.embed_query(text)
       
        # Atualização em massa no Neo4j
        update_query = """
        MATCH (c:Cluster) WHERE c.id = $cluster_id
        SET c.embedding = $embedding
        """
        params={"cluster_id": cluster_id, "embedding":embedding}
        self.graph.query(update_query, params=params)


    def _get_nearest_edges(self, top_k, score_threshold, target_type, duplicate_type):
        close_neighbors_query = """
        MATCH (target:Node:{target_type})
        CALL (target) {{
            MATCH (duplicate:Node)
            SEARCH duplicate IN (
                VECTOR INDEX {index_name}
                    FOR target.embedding
                    LIMIT $top_k
            ) SCORE as score
            WHERE
                duplicate:{duplicate_type} AND
                duplicate.id <> target.id AND 
                score >= $score_threshold
            RETURN duplicate, score
        }}
        RETURN
            target.id AS target_id,
            duplicate.id AS source_id,
            score
        """.format(
            target_type=target_type,
            duplicate_type=duplicate_type,
            index_name=self.entity_vector_index_name)

        params={"top_k":top_k, "score_threshold":score_threshold}
        results = self.graph.query(close_neighbors_query, params=params)
        edges = [(row["source_id"], row["target_id"], row["score"]) for row in results]
        
        return edges


    def _get_aiml_ontology_subgraph(self):
        relation_types = " | ".join(self.ontology.relations_schema.keys())
        query = f"""MATCH (source: Node)-[r: {relation_types}]->(target: Node)
        RETURN source.id AS source_id, target.id AS target_id """
        
        return self.graph.query(query)


    def _get_ontology_edges(self, ontology_weight=1):
        ontology_subgraph = self._get_aiml_ontology_subgraph()
        
        edges = []
        for entry in ontology_subgraph:
            edges.append((entry["source_id"], entry["target_id"], ontology_weight))
        
        return edges
    

    def run_leiden_clustering(
            self,
            max_cluster_size, resolution, ontology_weight,
            low_threshold, high_threshold, text_threshold, top_k_unify,
            score_threshold, top_k_connect):
        
        clusters, level_cluster_map, _ = self.preview_leiden_clustering(
            max_cluster_size, resolution, ontology_weight,
            low_threshold, high_threshold, text_threshold, top_k_unify,
            score_threshold, top_k_connect)

        print(f"Base clusters: {len(level_cluster_map[0])}, Hierarchy depth: {len(level_cluster_map)}")

        # create clusters and summarizations
        print("Adding Leiden cluster structure to Database")
        self._add_cluster_node_relations(clusters)
        self._add_cluster_subcluster_relations()

        for level in range(len(level_cluster_map)-1, -1, -1):
            print(f"Creating bottom-up cluster summarization: level {level}")
            for cluster_id in tqdm.tqdm(level_cluster_map[level]):
                leaf_only = level==len(level_cluster_map)-1

                cluster_context = self._get_cluster_context(cluster_id, leaf_only)
                self._add_cluster_summary(cluster_id, cluster_context)
                self._add_cluster_embedding(cluster_id, cluster_context)


    def preview_leiden_clustering(
            self,
            max_cluster_size, resolution, ontology_weight,
            low_threshold, high_threshold, text_threshold, top_k_unify,
            score_threshold, top_k_connect):
        
        # access only ontology, ignoring document structure
        edges = []
        edges += self._get_ontology_edges(ontology_weight)
        edges += self._get_similarity_edges(
            low_threshold, high_threshold, text_threshold, top_k_unify,
            score_threshold, top_k_connect)
        
        # determine cluster structure
        clusters, level_cluster_map = self._get_leiden_clusters(
            edges, max_cluster_size, resolution)

        # cluster distribution
        distribution = dict()
        for level, level_clusters in level_cluster_map.items():
            distribution[level] = len(level_clusters)

        return clusters, level_cluster_map, distribution

    
    def preview_entity_components(
            self,
            low_threshold, high_threshold, text_threshold, top_k_unify,
            score_threshold, top_k_connect):

        edges = []
        edges += self._get_ontology_edges()
        edges += self._get_similarity_edges(
            low_threshold, high_threshold, text_threshold, top_k_unify,
            score_threshold, top_k_connect)

        # create temporary graph
        G = nx.Graph()
        for e in edges:
            if not e[0] == e[1]:
                G.add_edge(e[0], e[1])

        components = list(nx.connected_components(G))
        component_number = len(components)
        component_distribution = dict()
        for component in components:
            component_size = len(component)
            if component_size not in component_distribution:
                component_distribution[component_size] = 0
            component_distribution[component_size] += 1

        return components, component_number, component_distribution


    def reset_databasis(self):
        self.graph.query("MATCH (n) DETACH DELETE n")


    def delete_clustering(self):
        self.graph.query("MATCH (c:Cluster) DETACH DELETE c")


    # def delete_similarity_relations(self):
    #     self.graph.query("MERGE ()-[r:SIMILAR_TO]-() DELETE r")


    def remove_document(self, filename: str):
        # removes relations based on file
        query_remove_relations = """
        MATCH ()-[r {chunk_source: $filename}]->() DELETE r """
            
        # removes nodes created based on file
        query_remove_nodes = """
        MATCH (n)
        WHERE n.chunk_source = $filename AND NOT n:Chunk AND NOT n:Source
        DETACH DELETE n """
            
        # removes chunks and connections
        query_remove_chunks = """
        MATCH (c:Chunk {source: $filename}) DETACH DELETE c """

        # removes source node
        query_remove_source = """
        MATCH (f:Source {filename: $filename}) DETACH DELETE f """

        params = {"filename": filename}
        self.graph.query(query_remove_relations, params=params)
        self.graph.query(query_remove_nodes, params=params)
        self.graph.query(query_remove_chunks, params=params)
        self.graph.query(query_remove_source, params=params)


    def reset_chunk_embeddings(self, batch_size=10):
        fetch_query = """
        MATCH (c:Chunk)
        RETURN elementId(c) AS id, c.text AS text
        """
        records = self.graph.query(fetch_query)
        
        # process in batches
        chunk_prefix = "title: none | text: "
        print("seting text chunk embeddings")
        for i in tqdm.tqdm(range(0, len(records), batch_size)):
            batch_records = records[i : i + batch_size]

            # prepares texts with document prefix
            texts = [f"{chunk_prefix}{r['text']}" for r in batch_records]

            # vectorize
            embeddings = self.embedder.embed_documents(texts)

            # 
            payload = [
                {"id": r["id"], "embedding": emb}
                for r, emb in zip(batch_records, embeddings)]

            # update database
            update_query = """
            UNWIND $batch AS row
            MATCH (c) WHERE elementId(c) = row.id
            SET c.embedding = row.embedding
            """
            self.graph.query(update_query, params={"batch": payload})

        
    def reset_cluster_embeddings(self, batch_size=10):
        fetch_query = """
        MATCH (c:Cluster)
        RETURN elementId(c) AS id, c.name as name, c.summary as summary
        """
        records = self.graph.query(fetch_query)
            
        # process in batches
        print("seting cluster embeddings")
        for i in tqdm.tqdm(range(0, len(records), batch_size)):
            batch_records = records[i : i + batch_size]
    
            # prepares texts with document prefix
            texts = []
            for r in batch_records:
                cluster_id = r["id"]
                name = r["name"]
                summary = r["summary"]
                context = self._get_cluster_context(cluster_id, False)
                text = f"title: {name} | text: {summary} | context: {context}"
                texts.append(text)
    
            # vectorize
            embeddings = self.embedder.embed_documents(texts)
    
            # 
            payload = [
                {"id": r["id"], "embedding": emb}
                for r, emb in zip(batch_records, embeddings)]
    
            # update database
            update_query = """
            UNWIND $batch AS row
            MATCH (c) WHERE elementId(c) = row.id
            SET c.embedding = row.embedding
            """
            self.graph.query(update_query, params={"batch": payload})
    

    def _create_property_index(self, element_type:str, index_name:str, property_names:str|list|tuple):
        # create index
        if type(property_names) == str:
            property_names = [property_names]
        key_property = ",".join([f"e.{property_name}" for property_name in property_names])

        query_index = """
        CREATE INDEX {index_name} IF NOT EXISTS
        FOR (e:{element_type}) ON ({key_property})
        """.format(
            element_type=element_type,
            index_name=index_name,
            key_property=key_property)

        self.graph.query(query_index)


    def _create_vector_index(self, element_type:str, index_name:str):
        # create vector index
        sample_vector = self.embedder.embed_query("sample")
        vector_dim = len(sample_vector)

        query_index = """
        CREATE VECTOR INDEX {} IF NOT EXISTS
        FOR (c:{}) ON (c.embedding)
        OPTIONS {{indexConfig: {{
            `vector.similarity_function`: 'cosine',
            `vector.dimensions`: $vector_dim
        }}}}
        """.format(index_name, element_type)
        self.graph.query(query_index, params={"vector_dim":vector_dim})


    def reset_node_embeddings(self, batch_size=10):
        fetch_query = """
        MATCH (n:Node: {node_type})
        RETURN elementId(n) AS id,
            n.id AS name_id,
            labels(n) as type,
            n.name AS name,
            n.description AS description
        """

        update_query = """
        UNWIND $batch AS row
        MATCH (n:Node) WHERE elementId(n) = row.id
        SET n.embedding = row.embedding
        """

        print("seting node embeddings")
        # gets rlevant nodes
        for node_type in self.ontology.nodes:
            print("seting node embeddings type: ", node_type)
            fetch_query_type = fetch_query.format(node_type=node_type)
            records = self.graph.query(fetch_query_type)
            node_prefix = "task: clustering | query: "
            for i in tqdm(range(0, len(records), batch_size)):
                batch_records = records[i : i + batch_size]

                # formats texts with preffix
                texts = []
                for r in batch_records:
                    text = self._node_as_text(r['name_id'], r['type'], r['name'], r['description'])
                    texts.append(f"{node_prefix}{text}")

                # vectorize in batches
                embeddings = self.embedder.embed_documents(texts)

                payload = [
                    {"id": r["id"], "embedding": emb}
                    for r, emb in zip(batch_records, embeddings)]

                self.graph.query(update_query, params={"batch": payload})


    def get_children_clusters(self, cluster_id):
        query_subclusters = """
        MATCH (child:Cluster)-[:IN_CLUSTER]->(parent:Cluster {id: $cluster_id})
        RETURN child.id AS child_id, child.name AS name, child.summary AS summary
        """
        subclusters = self.graph.query(query_subclusters, params={"cluster_id": cluster_id})
        return subclusters
    

    def get_children_entities(self, cluster_id):
        query_entities = """
        MATCH (e:Node)-[r:IN_CLUSTER]->(c:Cluster {id: $cluster_id})
        WHERE NOT e:Cluster AND r.final = true
        RETURN e.id as node_id, labels(e) AS labels, e.name as name, e.description as description
        """
        entities = self.graph.query(query_entities, params={"cluster_id": cluster_id})
        return entities


    def _cluster_as_text(self, cluster_id, name, descripion):
        return f"Cluster ID: {cluster_id} \n\t* name: {name}\n\t* description: {descripion}"


    def _chunk_as_text(self, chunk_index, text, source):
        return f"Chunk index {chunk_index} from document {source}:\n\t: text: {text}"


    def _node_as_text(self, node_id, labels, name, description):
        return f"Node ID/Type: {node_id}/{labels} \n\t* name: {name}\n\t* description: {description}"


    def get_cluster_details(self, cluster_id):
        query = """
        MATCH (c:Cluster {id: $cluster_id})
        RETURN c.name as name, c.summary as summary, c.level as level
        """
        result = self.graph.query(query, params={"cluster_id":cluster_id})

        name = result[0]["name"]
        summary = result[0]["summary"]
        level = result[0]["level"]

        return name, summary, level


    def get_unique_documents(self):
        query = """
        MATCH (f:Source)
        RETURN f.filename AS filename
        """
        results = self.graph.query(query)
        filenames = [row["filename"] for row in results]
        return filenames


    def get_file_details(self, filename):
        query = """
        MATCH (f:Source {filename: $filename})
        OPTIONAL MATCH (f)-[:HAS_CHUNK]->(c:Chunk)
        RETURN f AS file, count(c) AS chunks
        """
        results = self.graph.query(query, params={"filename":filename})
        return results
        

    def get_cluster_ids_by_level(self, level=0):
        query = """
        MATCH (c:Cluster {level: $level})
        RETURN c.id as id
        """
        results = self.graph.query(query, params={"level":level})
        return [row["id"] for row in results]


    def _retrieve_nearest_clusters(self, vector, top_k):
        cluster_query = """
        MATCH (c:Cluster)
        SEARCH c IN (
            VECTOR INDEX {}
            FOR $query_vec
            LIMIT $top_k
        ) SCORE as score
        RETURN c.id as id, c.name AS name, c.summary AS summary, c.level AS level, score
        """.format(self.cluster_vector_index_name)

        cluster_results = self.graph.query(
            cluster_query, params={"query_vec":vector, "top_k":top_k})

        # structured_results = [
        #     {"id":c["id"], 
        #      "name":c["name"],
        #      "summary":c["summary"],
        #      "level":c["level"],
        #      "score":c["score"]} for c in cluster_results]

        return cluster_results

    
    def _retrieve_nearest_nodes(self, vector, top_k):
        node_query = """
        MATCH (n: Node)
        SEARCH n IN (
            VECTOR INDEX {index_name}
            FOR $query_vec
            LIMIT $top_k
        ) SCORE as score
        RETURN
            n.id as id,
            labels(n) as type, 
            node.name as name, 
            node.description as description, 
            score
        """.format(index_name=self.chunk_vector_index_name)
    
        node_results = self.graph.query(
            node_query, params={"query_vec":vector, "top_k":top_k})
            
        return node_results


    def _retrieve_nearest_chunks(self, vector, top_k):
        chunk_query = """
        MATCH (c:Chunk)
        SEARCH c IN (
            VECTOR INDEX {}
            FOR $query_vec
            LIMIT $top_k
        ) SCORE as score
        RETURN c.text AS text, c.index AS index,  c.source AS source, score
        """.format(self.chunk_vector_index_name)

        chunk_results = self.graph.query(
            chunk_query, params={"query_vec":vector, "top_k":top_k})
        
        # structured_results = [
        #     {"text":c["text"],
        #      "index":c["index"],
        #      "source":c["source"],
        #      "score":c["score"]} for c in chunk_results]
        
        return chunk_results


    def _get_query_context(self, query_text, top_k_global, top_k_local):
        # embbed query
        query_prefix = "task: search result | query: "
        query_vector = self.embedder.embed_query(f"{query_prefix}{query_text}")

        # find nearest elements
        cluster_results = self._retrieve_nearest_clusters(query_vector, top_k=top_k_global)
        chunk_results = self._retrieve_nearest_chunks(query_vector, top_k=top_k_local)

        # mount context
        context_parts = []

        if cluster_results:
            context_parts.append("=== SUMMARIES ===")
            for c in cluster_results:
                cluster_relevance = f"Cluster relevance {c['score']}: "
                cluster_text = self._cluster_as_text(c['id'], c['name'], c['summary'])
                context_parts.append(cluster_relevance + cluster_text)

        if chunk_results:
            context_parts.append("=== SOURCE CHUNKS ===")
            for c in chunk_results:
                chunk_relevance = f"Text chunk relevance {c['score']}: "
                chunk_text = self._chunk_as_text(c["index"], c["text"], c["source"])
                context_parts.append(chunk_relevance + chunk_text)

        context = "\n".join(context_parts)

        return context, cluster_results, chunk_results


    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=4, max=60))
    def _get_query_answer(self, prompt):
        json_schema = {
            "title": "text_query_summary",
            "description": "Query with title and answer",
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short, descriptive title for the question and answer"
                },
                "answer": {
                    "type": "string",
                    "description": "Comprehensive answer to the question"
                }},
                "required": ["title", "answer"]}

        # Pass the dict schema directly
        structured_llm = self.llm.with_structured_output(json_schema)
        result = structured_llm.invoke(prompt)

        return result


    def query_RAG(self, query_text:str, top_k_global=10, top_k_local=10):
        # get relevant context
        context, clusters, chunks = self._get_query_context(
            query_text, top_k_global=top_k_global, top_k_local=top_k_local)

        prompt = """
        System: You are an expert assistant answering questions based on a Knowledge Graph context.
        Use the high-level community summaries for broad context and specific entity relationships for factual grounding.
        
        Context:
        {context}
        
        User Query: {query}

        Answer:
        """.format(context=context, query=query_text)

        result = self._get_query_answer(prompt=prompt)
        
        return {"title":result["title"],
                "answer":result["answer"],
                "clusters": clusters,
                "chunks": chunks}


    def custom_query_Graph(self, text_query:str):
        # # chain used for queries
        # self.chain = GraphCypherQAChain.from_llm(
        #     llm=self.llm,
        #     )
        pass


    def generate_reports(self):
        #   reports clarifying clusters of concepts: ML for Corrosion, etc
        #   also report how the documents are related
        #   answers to relevant questions already in documents
        pass