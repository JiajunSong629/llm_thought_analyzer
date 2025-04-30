from streamlit_agraph import Node, Edge


# Define a structure to hold merged node information
class MergedNode:
    def __init__(self, node_id, variable, expression, node_type="step"):
        self.node_id = node_id
        self.variable = variable
        self.expression = expression
        self.node_type = node_type  # 'step' or 'input'
        self.sources = set()  # Store sample_ids or other identifiers
        self.reasoning_paths = {}  # Map sample_id to its reasoning path
        # Relative topological level (0.0 to 1.0), -1 for inputs, None initially?
        self.relative_level = 1.1  # Use a value > 1 to ensure first assignment takes
        # Add other relevant attributes if needed, e.g., shape based on source mix?

    def get_agraph_node(self):
        """Converts the MergedNode to a streamlit_agraph Node."""
        label = self.variable
        title = (
            f"{self.expression} || freq: {len(self.sources)}"
            if self.node_type == "step"
            else self.expression
        )
        shape = "box" if self.node_type == "input" else "ellipse"

        # Determine color based on node type and source
        if self.node_type == "input":
            color = "lightblue"  # Blue for input variables
        elif "ground_truth" in self.sources:
            color = "lightgreen"  # Green for ground truth nodes
        else:
            # For sample nodes, check if it's the last node
            is_last_node = False
            for source in self.sources:
                if source.startswith("sample_"):
                    reasoning_path = self.reasoning_paths.get(source, [])
                    if reasoning_path:
                        max_level = max([l for l, _ in reasoning_path])
                        for level, steps in reasoning_path:
                            if level == max_level:
                                for step in steps:
                                    if (
                                        step.get("variable") == self.variable
                                        and step.get("expression") == self.expression
                                    ):
                                        is_last_node = True
                                        break
            color = (
                "orange" if is_last_node else "red"
            )  # Orange for last nodes, red for other sample nodes

        return Node(id=self.node_id, label=label, title=title, shape=shape, color=color)


def process_merged_graph_data(data: dict):
    """
    Processes the combined ground truth and results data from a JSON file
    to create a merged graph suitable for visualization.

    Args:
        data: A dictionary loaded from the JSON file, expected to contain:
              - 'factual_assignment': { 'var': value, ... }
              - 'ground_truth_function': { 'reasoning_path_topological_levels': [...] }
              - 'results': [ { 'sample_id': int, 'reasoning_path_topological_levels': [...] }, ... ]

    Returns:
        A tuple: (list[Node], list[Edge], dict[str, MergedNode])
        - list[Node]: List of streamlit_agraph Node objects for visualization.
        - list[Edge]: List of streamlit_agraph Edge objects for visualization.
        - dict[str, MergedNode]: Dictionary mapping unique node keys to MergedNode objects.
    """
    unique_nodes = {}
    edges_set = set()
    node_id_counter = 0

    global_inputs = data.get("factual_assignment", {})

    # Prepare a list of all reasoning paths to process (ground truth + samples)
    processing_list = []

    # 1. Add Ground Truth
    gt_function = data.get("ground_truth_function", {})
    if gt_function:
        processing_list.append(
            {
                "sample_id": "ground_truth",
                "reasoning_path": gt_function.get(
                    "reasoning_path_topological_levels", []
                ),
                "inputs": global_inputs,
            }
        )

    # 2. Add Results
    results = data.get("results", [])
    for result in results:
        sample_id_num = result.get("sample_id", "unknown")
        processing_list.append(
            {
                "sample_id": f"sample_{sample_id_num}",
                "reasoning_path": result.get("reasoning_path_topological_levels", []),
                "inputs": global_inputs,
            }
        )

    # --- Process all items (GT + Samples) ---
    for item in processing_list:
        sample_id = item["sample_id"]
        reasoning_path = item["reasoning_path"]
        inputs = item["inputs"]

        local_step_id_to_merged_node_key = {}

        # Find max level *in this specific path*
        max_level_in_path = 0
        for level_data in reasoning_path:
            if not isinstance(level_data, (list, tuple)) or len(level_data) != 2:
                continue
            level, _ = level_data
            if level > max_level_in_path:
                max_level_in_path = level

        # A. Process Input Variables
        for var_name in inputs.keys():
            node_key = ("input", var_name)
            if node_key not in unique_nodes:
                merged_node_id = f"node_{node_id_counter}"
                unique_nodes[node_key] = MergedNode(
                    node_id=merged_node_id,
                    variable=var_name,
                    expression=f"Input: {var_name}",
                    node_type="input",
                )
                node_id_counter += 1
                # Assign relative level -1 for inputs only when creating
                unique_nodes[node_key].relative_level = -1
            unique_nodes[node_key].sources.add(sample_id)

        # B. Process Steps for this reasoning path
        for level_data in reasoning_path:
            if not isinstance(level_data, (list, tuple)) or len(level_data) != 2:
                continue
            level, steps_in_level = level_data
            if not isinstance(steps_in_level, list):
                continue

            # Calculate relative level for this step
            current_relative_level = (
                (level / max_level_in_path) if max_level_in_path > 0 else 0.0
            )

            for step in steps_in_level:
                if not isinstance(step, dict):
                    continue

                step_id = step.get("step_id")
                variable = step.get("variable")
                expression = step.get("expression")

                if step_id is None or variable is None or expression is None:
                    continue

                node_key = (variable, expression)

                if node_key not in unique_nodes:
                    merged_node_id = f"node_{node_id_counter}"
                    unique_nodes[node_key] = MergedNode(
                        node_id=merged_node_id,
                        variable=variable,
                        expression=expression,
                        node_type="step",
                    )
                    node_id_counter += 1
                    # Initial relative level assigned here
                    unique_nodes[node_key].relative_level = current_relative_level

                unique_nodes[node_key].sources.add(sample_id)
                unique_nodes[node_key].reasoning_paths[sample_id] = reasoning_path
                local_step_id_to_merged_node_key[step_id] = node_key
                # Update relative level only if the current path places it relatively earlier
                if current_relative_level < unique_nodes[node_key].relative_level:
                    unique_nodes[node_key].relative_level = current_relative_level

        # C. Create Edges for this reasoning path
        for level_data in reasoning_path:
            if not isinstance(level_data, (list, tuple)) or len(level_data) != 2:
                continue
            level, steps_in_level = level_data
            if not isinstance(steps_in_level, list):
                continue

            for step in steps_in_level:
                if not isinstance(step, dict):
                    continue
                current_step_id = step.get("step_id")
                variable = step.get("variable")
                expression = step.get("expression")
                if current_step_id is None or variable is None or expression is None:
                    continue

                target_node_key = (variable, expression)
                if target_node_key not in unique_nodes:
                    continue
                target_merged_node = unique_nodes[target_node_key]

                # Edges from step dependencies
                for dep_step_id in step.get("dependencies", []):
                    source_node_key = local_step_id_to_merged_node_key.get(dep_step_id)
                    if source_node_key and source_node_key in unique_nodes:
                        source_merged_node = unique_nodes[source_node_key]
                        # Determine edge style based on source and target
                        if (
                            "ground_truth" in source_merged_node.sources
                            and "ground_truth" in target_merged_node.sources
                        ):
                            # Edge between ground truth nodes
                            edge_tuple = (
                                source_merged_node.node_id,
                                target_merged_node.node_id,
                                "green",
                                3,
                            )
                        else:
                            # Edge between sample nodes
                            edge_tuple = (
                                source_merged_node.node_id,
                                target_merged_node.node_id,
                                "red",
                                1,
                            )
                        edges_set.add(edge_tuple)

                # Edges from input dependencies
                for dep_input_var in step.get("dependencies_input", []):
                    source_node_key = ("input", dep_input_var)
                    if source_node_key in unique_nodes:
                        source_merged_node = unique_nodes[source_node_key]
                        # Edge from input to any node is gray
                        edge_tuple = (
                            source_merged_node.node_id,
                            target_merged_node.node_id,
                            "rgba(200, 200, 200, 0.5)",  # Lighter, semi-transparent gray
                            1,
                            "none",  # No arrows
                        )
                        edges_set.add(edge_tuple)

    # Convert MergedNode objects to streamlit_agraph Nodes
    agraph_nodes = [mn.get_agraph_node() for mn in unique_nodes.values()]

    # Convert edge tuples set back to list of Edge objects
    # Handle variable tuple lengths (some have arrows, some don't)
    agraph_edges = []
    for edge_data in edges_set:
        if len(edge_data) == 5:  # source, target, color, width, arrows
            s, t, c, w, a = edge_data
            agraph_edges.append(Edge(source=s, target=t, color=c, width=w, arrows=a))
        elif len(edge_data) == 4:  # source, target, color, width (default arrows)
            s, t, c, w = edge_data
            agraph_edges.append(Edge(source=s, target=t, color=c, width=w))
        # Add more conditions if other tuple structures exist

    # Return unique_nodes dictionary as well, it contains level info
    return agraph_nodes, agraph_edges, unique_nodes
