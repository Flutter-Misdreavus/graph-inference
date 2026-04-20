"""Prompt templates for Graph-as-Code node classification."""


def build_graph_as_code_prompt(node_id: int, class_labels: dict[int, str]) -> str:
    """Return the initial system/task prompt for Graph-as-Code mode.

    Follows the paper's Template 3 with a pandas DataFrame schema.
    """
    class_lines = "\n".join(
        f"{cid}: {cname}" for cid, cname in sorted(class_labels.items())
    )

    prompt = f"""Task: You are solving a node-based reasoning task for node {node_id}. You have a pandas DataFrame df where each row corresponds to a node, indexed by its node_id.

Instructions: Always begin with reasoning. After your reasoning, provide a single, valid pandas command on a new line.

# Schema structure:

- The DataFrame index is the node id. Access a row by node id with: df.loc[node_id].   
- The column features stores each node’s textual description: df.loc[node_id, ’features’].   
- The column neighbors stores a list of neighbor node IDs: df.loc[node_id, ’neighbors’].   
- The column label contains the integer node label if it belongs to the training set; otherwise None.

You may query ANY column(s) of df using any valid pandas command that applies to a DataFrame named df. You may also use pd.* utilities with df as input. The dataframe can be long, so you may want to avoid commands that print the entire table.

# Response format:

- For intermediate steps: reason then on the final line output a single valid pandas expression.   
- To finish: reason then on the final line respond exactly as: Answer [class_id].

# Available class labels:
{class_lines}

Now begin your reasoning."""
    return prompt
