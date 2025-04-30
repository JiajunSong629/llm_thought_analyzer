"""
Models for representing reasoning steps and paths.
"""

import ast
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ReasoningStep:
    """Represents a single step in a reasoning process."""

    step_id: int
    variable: str
    dependencies: List[int]
    expression: str
    parent_path: Optional["ReasoningPath"] = None
    dependencies_input: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        deps = [f"Step {dep}" for dep in self.dependencies]
        deps_str = f" (depends on {', '.join(deps)})" if deps else ""
        input_deps_str = (
            f" (input deps: {', '.join(self.dependencies_input)})"
            if self.dependencies_input
            else ""
        )
        return f"Step {self.step_id}: Calculate {self.variable}: {deps_str}{input_deps_str}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the step to a dictionary representation."""
        return {
            "step_id": self.step_id,
            "variable": self.variable,
            "dependencies": self.dependencies,
            "expression": self.expression,
            "dependencies_input": self.dependencies_input,
        }


@dataclass
class ReasoningPath:
    """Represents a complete reasoning path with multiple steps."""

    steps: List[ReasoningStep] = field(default_factory=list)
    var_to_step: Dict[str, ReasoningStep] = field(default_factory=dict)
    return_vars: List[str] = field(
        default_factory=list
    )  # Stores names of returned variables

    def add_step(self, step: ReasoningStep) -> None:
        """Add a step to the reasoning path."""
        step.parent_path = self
        self.steps.append(step)
        self.var_to_step[step.variable] = step

    def get_step(self, step_id: int) -> Optional[ReasoningStep]:
        """Get a step by its ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_step_by_var(self, variable: str) -> Optional[ReasoningStep]:
        """Get a step by its variable name."""
        return self.var_to_step.get(variable)

    def __str__(self) -> str:
        return "\n".join(str(step) for step in self.steps)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert the path to a list of dictionaries."""
        return [step.to_dict() for step in self.steps]

    def get_topological_levels(self) -> Dict[int, List[ReasoningStep]]:
        """Performs a topological sort and returns steps grouped by level."""
        in_degree = {step.step_id: 0 for step in self.steps}
        adj = {step.step_id: [] for step in self.steps}
        step_map = {step.step_id: step for step in self.steps}

        for step in self.steps:
            for dep_id in step.dependencies:
                if dep_id in step_map:
                    adj[dep_id].append(step.step_id)
                    in_degree[step.step_id] += 1
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]

        levels: Dict[int, List[ReasoningStep]] = {}
        depth = 0

        while queue:
            current_level_nodes = []
            next_queue = []
            for step_id in queue:
                current_level_nodes.append(step_map[step_id])
                for neighbor_id in adj[step_id]:
                    in_degree[neighbor_id] -= 1
                    if in_degree[neighbor_id] == 0:
                        next_queue.append(neighbor_id)

            if current_level_nodes:
                levels[depth] = current_level_nodes

            queue = sorted(next_queue)
            depth += 1

        if sum(len(nodes) for nodes in levels.values()) != len(self.steps):
            print("Warning: Cycle detected or node missing in topological sort.")

        return [
            (level, [step.to_dict() for step in steps])
            for level, steps in levels.items()
        ]

    def simplify(self) -> "ReasoningPath":
        """Simplifies the reasoning path by removing alias steps and dead code."""
        if not self.steps or not self.return_vars:
            return self  # Cannot simplify if no steps or no return detected

        # 1. Alias Elimination
        alias_map: Dict[str, str] = {}
        redundant_step_ids: set[int] = set()
        steps_by_id = {step.step_id: step for step in self.steps}

        for step in self.steps:
            is_simple_alias = isinstance(
                ast.parse(step.expression).body[0], ast.Expr
            ) and isinstance(ast.parse(step.expression).body[0].value, ast.Name)
            if is_simple_alias:
                aliased_var = ast.parse(step.expression).body[0].value.id
                # Check if the aliased var is from a *previous* step or an input
                source_step = self.get_step_by_var(aliased_var)
                is_known_var = (
                    source_step is not None
                    or aliased_var in self.get_all_input_dependencies()
                )

                if is_known_var:
                    redundant_step_ids.add(step.step_id)
                    # Find the ultimate source (handling chains)
                    source = alias_map.get(aliased_var, aliased_var)
                    alias_map[step.variable] = source

        # Resolve transitive aliases
        for alias, source in alias_map.items():
            while source in alias_map:
                source = alias_map[source]
            alias_map[alias] = source

        # 2. Filter Aliases & Remap Dependencies
        intermediate_steps: List[ReasoningStep] = []
        old_to_intermediate_step: Dict[int, ReasoningStep] = {}

        for step in self.steps:
            if step.step_id in redundant_step_ids:
                continue

            new_deps = set()
            step_dependencies = list(step.dependencies)  # Use original deps for lookup

            # Include dependencies inherited through aliases
            temp_aliases_to_check = {
                dep_id
                for dep_id in step_dependencies
                if steps_by_id[dep_id].variable in alias_map
            }
            while temp_aliases_to_check:
                alias_dep_id = temp_aliases_to_check.pop()
                alias_var = steps_by_id[alias_dep_id].variable
                source_var = alias_map[alias_var]
                source_step = self.get_step_by_var(source_var)
                if source_step and source_step.step_id not in redundant_step_ids:
                    step_dependencies.extend(
                        source_step.dependencies
                    )  # Add deps of the source
                    # We might need to recursively check if the source's deps are also aliases
                    temp_aliases_to_check.update(
                        {
                            dep
                            for dep in source_step.dependencies
                            if steps_by_id[dep].variable in alias_map
                        }
                    )

            # Now map remaining dependencies
            for dep_id in step_dependencies:
                original_dep_step = steps_by_id.get(dep_id)
                if not original_dep_step:
                    continue  # Should not happen with valid paths

                final_var = alias_map.get(
                    original_dep_step.variable, original_dep_step.variable
                )
                final_step = self.get_step_by_var(final_var)

                if final_step and final_step.step_id not in redundant_step_ids:
                    new_deps.add(final_step.step_id)

            # Copy the step and update dependencies
            new_step = ReasoningStep(
                step_id=step.step_id,  # Keep old ID temporarily
                variable=step.variable,
                dependencies=sorted(list(new_deps)),
                dependencies_input=step.dependencies_input,  # Input deps remain the same
                expression=step.expression,
                parent_path=step.parent_path,  # Will be updated later
            )
            intermediate_steps.append(new_step)
            old_to_intermediate_step[step.step_id] = new_step

        # 3. Reachability Analysis (Backward Traversal)
        reachable_step_ids: set[int] = set()
        queue: List[int] = []

        # Find starting points (steps defining the ultimate return vars)
        for ret_var in self.return_vars:
            source_var = alias_map.get(ret_var, ret_var)
            source_step = self.get_step_by_var(source_var)  # Find in original path
            if source_step and source_step.step_id in old_to_intermediate_step:
                intermediate_step_id = source_step.step_id  # Use the old ID for lookup
                if intermediate_step_id not in reachable_step_ids:
                    reachable_step_ids.add(intermediate_step_id)
                    queue.append(intermediate_step_id)

        # Perform backward traversal
        visited_for_bfs = set(queue)
        while queue:
            current_id = queue.pop(0)
            if current_id not in old_to_intermediate_step:
                continue  # Skip if it was an alias step

            current_step = old_to_intermediate_step[current_id]
            for dep_id in current_step.dependencies:
                if dep_id in old_to_intermediate_step and dep_id not in visited_for_bfs:
                    reachable_step_ids.add(dep_id)
                    visited_for_bfs.add(dep_id)
                    queue.append(dep_id)

        # 4. Final Pruning & Path Construction
        final_steps_ordered = sorted(
            [old_to_intermediate_step[step_id] for step_id in reachable_step_ids],
            key=lambda s: s.step_id,
        )

        # 5. Re-indexing
        simplified_path = ReasoningPath(return_vars=self.return_vars)
        old_id_to_new_id: Dict[int, int] = {}
        new_step_counter = 1

        for old_step in final_steps_ordered:
            new_id = new_step_counter
            old_id_to_new_id[old_step.step_id] = new_id

            # Create the final step with new ID and remapped dependencies
            final_step = ReasoningStep(
                step_id=new_id,
                variable=old_step.variable,
                dependencies=[
                    old_id_to_new_id[old_dep_id]
                    for old_dep_id in old_step.dependencies
                    if old_dep_id in old_id_to_new_id
                ],
                dependencies_input=old_step.dependencies_input,
                expression=old_step.expression,
                parent_path=simplified_path,  # Set parent to the new path
            )
            simplified_path.add_step(final_step)
            new_step_counter += 1

        return simplified_path

    def get_all_input_dependencies(self) -> set[str]:
        """Helper to get all unique input dependencies across all steps."""
        all_inputs = set()
        for step in self.steps:
            all_inputs.update(step.dependencies_input)
        return all_inputs

    @classmethod
    def from_function_str(cls, function_str: str) -> "ReasoningPath":
        """Create a ReasoningPath from a function string."""
        ast_tree = ast.parse(function_str)
        return cls.from_ast(ast_tree)

    @classmethod
    def from_ast(cls, ast_tree: ast.AST) -> "ReasoningPath":
        """Create a ReasoningPath from an AST tree."""
        path = cls()
        step_counter = 1
        function_args = set()  # Keep track of function arguments

        def visit_node(current_node):
            nonlocal step_counter

            if isinstance(current_node, ast.Module):
                # Process each statement in the module
                for stmt in current_node.body:
                    visit_node(stmt)

            elif isinstance(current_node, ast.FunctionDef):
                # Store function arguments
                nonlocal function_args
                function_args = {arg.arg for arg in current_node.args.args}
                # Process each statement in the function body
                for stmt in current_node.body:
                    visit_node(stmt)
                # Find return variables after processing body
                for stmt in ast.walk(current_node):
                    if isinstance(stmt, ast.Return):
                        if isinstance(stmt.value, ast.Name):
                            path.return_vars.append(stmt.value.id)
                        # TODO: Handle more complex return expressions if needed
                        # elif isinstance(stmt.value, ast.Tuple): etc.
                # Deduplicate return vars
                path.return_vars = sorted(list(set(path.return_vars)))

            elif isinstance(current_node, ast.Assign):
                # This is a reasoning step
                var_name = (
                    current_node.targets[0].id
                    if isinstance(current_node.targets[0], ast.Name)
                    else "unknown"
                )

                dependencies = []
                dependencies_input = []  # Track input dependencies
                for child in ast.walk(current_node.value):
                    if isinstance(child, ast.Name) and child.id != var_name:
                        if child.id in function_args:
                            dependencies_input.append(child.id)
                        else:
                            dep_step = path.get_step_by_var(child.id)
                            if dep_step:
                                # Ensure dependency exists before adding
                                dependencies.append(dep_step.step_id)

                dependencies_input = sorted(
                    list(set(dependencies_input))
                )  # Deduplicate and sort

                # Create step
                step = ReasoningStep(
                    step_id=step_counter,
                    variable=var_name,
                    dependencies=sorted(
                        list(set(dependencies))
                    ),  # Deduplicate and sort
                    dependencies_input=dependencies_input,
                    expression=ast.unparse(current_node.value),
                )

                # Add to path
                path.add_step(step)
                step_counter += 1

        # Call visit_node on the input node
        visit_node(ast_tree)
        return path
