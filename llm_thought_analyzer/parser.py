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
    operation: str
    dependencies: List[int]
    expression: str
    parent_path: Optional["ReasoningPath"] = None

    def __str__(self) -> str:
        deps = [f"Step {dep}" for dep in self.dependencies]
        deps_str = f" (depends on {', '.join(deps)})" if deps else ""
        return f"Step {self.step_id}: Calculate {self.variable} using {self.operation}{deps_str}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the step to a dictionary representation."""
        return {
            "step_id": self.step_id,
            "variable": self.variable,
            "operation": self.operation,
            "dependencies": self.dependencies,
            "expression": self.expression,
        }


@dataclass
class ReasoningPath:
    """Represents a complete reasoning path with multiple steps."""

    steps: List[ReasoningStep] = field(default_factory=list)
    var_to_step: Dict[str, ReasoningStep] = field(default_factory=dict)

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

        def visit_node(current_node):
            nonlocal step_counter

            if isinstance(current_node, ast.Module):
                # Process each statement in the module
                for stmt in current_node.body:
                    visit_node(stmt)

            elif isinstance(current_node, ast.FunctionDef):
                # Process each statement in the function body
                for stmt in current_node.body:
                    visit_node(stmt)

            elif isinstance(current_node, ast.Assign):
                # This is a reasoning step
                var_name = (
                    current_node.targets[0].id
                    if isinstance(current_node.targets[0], ast.Name)
                    else "unknown"
                )

                # Determine operation type
                operation = "assignment"
                if isinstance(current_node.value, ast.BinOp):
                    if isinstance(current_node.value.op, ast.Add):
                        operation = "addition"
                    elif isinstance(current_node.value.op, ast.Sub):
                        operation = "subtraction"
                    elif isinstance(current_node.value.op, ast.Mult):
                        operation = "multiplication"
                    elif isinstance(current_node.value.op, ast.Div):
                        operation = "division"

                # Find dependencies (variables used in this step)
                dependencies = []
                for child in ast.walk(current_node.value):
                    if isinstance(child, ast.Name) and child.id != var_name:
                        dep_step = path.get_step_by_var(child.id)
                        if dep_step:
                            dependencies.append(dep_step.step_id)

                # Create step
                step = ReasoningStep(
                    step_id=step_counter,
                    variable=var_name,
                    operation=operation,
                    dependencies=dependencies,
                    expression=ast.unparse(current_node.value),
                )

                # Add to path
                path.add_step(step)
                step_counter += 1

        # Call visit_node on the input node
        visit_node(ast_tree)
        return path

    def find_last_difference(
        self, other: "ReasoningPath"
    ) -> Tuple[Optional[ReasoningStep], Optional[ReasoningStep]]:
        """
        Find the last difference between this path and another path.
        Returns a tuple of (this_path_step, other_path_step) where the difference occurs.
        """
        # Create mappings for easier comparison
        this_steps = {step.variable: step for step in self.steps}
        other_steps = {step.variable: step for step in other.steps}

        # Find common variables
        common_vars = set(this_steps.keys()) & set(other_steps.keys())

        # Track the last difference found
        last_diff = None

        # Compare steps for common variables
        for var in sorted(common_vars, key=lambda v: this_steps[v].step_id):
            this_step = this_steps[var]
            other_step = other_steps[var]

            # Check for differences in operation or dependencies
            if (
                this_step.operation != other_step.operation
                or this_step.dependencies != other_step.dependencies
                or this_step.expression != other_step.expression
            ):
                last_diff = (this_step, other_step)

        return last_diff if last_diff else (None, None)

    def compare_with(self, other: "ReasoningPath") -> Dict[str, Any]:
        """
        Compare this path with another path and return detailed differences.
        """
        differences = {
            "operation_changes": [],
            "dependency_changes": [],
            "unique_steps": {"this_path": [], "other_path": []},
        }

        # Create mappings for easier comparison
        this_steps = {step.variable: step for step in self.steps}
        other_steps = {step.variable: step for step in other.steps}

        # Find common variables
        common_vars = set(this_steps.keys()) & set(other_steps.keys())

        # Compare steps for common variables
        for var in common_vars:
            this_step = this_steps[var]
            other_step = other_steps[var]

            # Check for operation differences
            if this_step.operation != other_step.operation:
                differences["operation_changes"].append(
                    {
                        "variable": var,
                        "this_operation": this_step.operation,
                        "other_operation": other_step.operation,
                        "this_expression": this_step.expression,
                        "other_expression": other_step.expression,
                    }
                )

            # Check for dependency differences
            if this_step.dependencies != other_step.dependencies:
                differences["dependency_changes"].append(
                    {
                        "variable": var,
                        "this_dependencies": this_step.dependencies,
                        "other_dependencies": other_step.dependencies,
                    }
                )

        # Find unique steps
        unique_to_this = set(this_steps.keys()) - set(other_steps.keys())
        unique_to_other = set(other_steps.keys()) - set(this_steps.keys())

        differences["unique_steps"]["this_path"] = [
            this_steps[var].to_dict() for var in unique_to_this
        ]
        differences["unique_steps"]["other_path"] = [
            other_steps[var].to_dict() for var in unique_to_other
        ]

        return differences
