import streamlit as st
import json
from streamlit_agraph import agraph, Config
from llm_thought_analyzer.app_utils import process_merged_graph_data

GRAPH_WIDTH = 1000
GRAPH_HEIGHT = 600


def process_json_file(uploaded_file):
    """Processes the uploaded JSON file to extract graph data."""
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            # Get the unique_nodes dict containing level info
            agraph_nodes, edges, unique_nodes = process_merged_graph_data(data)

            # Calculate positions for better visualization
            graph_width = GRAPH_WIDTH
            graph_height = GRAPH_HEIGHT
            top_y = 50  # Y position for the start of steps (relative_level 0)
            bottom_y = graph_height - 50  # Y position for input nodes
            # Calculate the height available for drawing step nodes
            drawing_height = bottom_y - top_y - 100  # Leave margin above inputs
            if drawing_height < 50:  # Ensure a minimum height
                drawing_height = 50

            # Group agraph_nodes by relative level for positioning
            nodes_by_relative_level = {}
            for node_key, merged_node in unique_nodes.items():
                relative_level = merged_node.relative_level
                # Use a string key for dict as float levels might be imprecise
                level_key = f"{relative_level:.4f}"
                if level_key not in nodes_by_relative_level:
                    nodes_by_relative_level[level_key] = []

                agraph_node = next(
                    (n for n in agraph_nodes if n.id == merged_node.node_id), None
                )
                if agraph_node:
                    # Store the node and its original relative level for y-calc
                    nodes_by_relative_level[level_key].append(
                        (agraph_node, relative_level)
                    )

            # Position nodes based on relative levels
            input_level_key = f"{-1.0:.4f}"
            for level_key, nodes_with_level in nodes_by_relative_level.items():
                num_nodes_in_level = len(nodes_with_level)
                level_spacing = (
                    graph_width / (num_nodes_in_level + 1)
                    if num_nodes_in_level > 0
                    else graph_width / 2
                )

                # Get the actual relative level for Y calculation (same for all nodes in this group)
                # Take it from the first node in the list
                relative_level = nodes_with_level[0][1]

                if level_key == input_level_key:  # Input nodes
                    y_pos = top_y
                else:  # Step nodes (relative_level should be 0.0 to 1.0)
                    y_pos = top_y + (relative_level + 0.5) * drawing_height

                # Sort nodes in the level horizontally for consistent layout (e.g., by label)
                nodes_with_level.sort(key=lambda item: item[0].label)

                for i, (node, _) in enumerate(nodes_with_level):
                    node.x = (i + 1) * level_spacing
                    node.y = y_pos

            st.success("Graph data extracted successfully.")
            # Return the processed agraph_nodes and edges
            return agraph_nodes, edges
        except json.JSONDecodeError:
            st.error("Invalid JSON file.")
            return [], []
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            import traceback

            st.error(traceback.format_exc())
            return [], []
    return [], []


# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("Function Step Visualizer")

# Initialize session state for selected node if not exists
if "selected_node" not in st.session_state:
    st.session_state.selected_node = None

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a JSON file", type="json")
    show_samples = st.checkbox("Show Sample Nodes", value=True)
    process_button = st.button("Process File")

    if process_button and uploaded_file:
        nodes_res, edges_res = process_json_file(uploaded_file)
        st.session_state.nodes = nodes_res
        st.session_state.edges = edges_res
        st.session_state.selected_node = (
            None  # Reset selected node when new file is processed
        )


st.subheader("Visualization")
nodes_to_display = st.session_state.get("nodes", [])
edges_to_display = st.session_state.get("edges", [])

# Filter nodes based on visibility settings
if not show_samples:
    nodes_to_display = [n for n in nodes_to_display if n.color != "orange"]

config = Config(
    width=GRAPH_WIDTH,
    height=GRAPH_HEIGHT,
    directed=True,
    physics=False,
    hierarchical=False,
    collapsible=True,
    node={
        "labelProperty": "label",
        "font": {"size": 10, "face": "arial", "align": "center"},
    },
    link={"highlightColor": "#F7A7A6"},
)

# Use a container to prevent refresh
with st.container():
    selected_node = agraph(
        nodes=nodes_to_display, edges=edges_to_display, config=config
    )
    if selected_node:
        st.session_state.selected_node = selected_node
